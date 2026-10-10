"""Frozen, checkpointed forward evaluation. No tuning on test outcomes."""
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from .checkpoints import Stages, Paused, atomic_json, exclusive_lock, freeze
from .data import load_archive, sha256
from .features import build_features, eligible
from .modeling import CANDIDATES, fit, predict, point_metrics, radius, interval_metrics

ROOT = Path(__file__).resolve().parents[2]
FOLDS = [
    {'id': '2012_Q2', 'tune': '2012-02-01', 'calibrate': '2012-03-01', 'test': '2012-04-01', 'end': '2012-07-01'},
    {'id': '2012_Q3', 'tune': '2012-05-01', 'calibrate': '2012-06-01', 'test': '2012-07-01', 'end': '2012-10-01'},
    {'id': '2012_Q4', 'tune': '2012-08-01', 'calibrate': '2012-09-01', 'test': '2012-10-01', 'end': '2013-01-01'},
]


def masks(index, fold):
    tune, cal, test, end = [pd.Timestamp(fold[key]) for key in ['tune', 'calibrate', 'test', 'end']]
    if not tune < cal < test < end:
        raise ValueError('Fold boundaries must be chronological')
    return {'train': index < tune, 'tune': (index >= tune) & (index < cal),
            'refit': index < cal, 'calibrate': (index >= cal) & (index < test),
            'test': (index >= test) & (index < end)}


def source_manifest():
    paths = sorted((ROOT / 'src/bike_forecast').glob('*.py'))
    paths += [ROOT / 'docs/PROTOCOL.md', ROOT / 'pyproject.toml']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def save_predictions(path, y, p):
    pd.DataFrame({'actual': y, 'prediction': p}, index=y.index).to_csv(path, float_format='%.17g')


def assert_replay(stage, x, filename='predictions.csv'):
    stored = pd.read_csv(stage / filename, index_col=0, parse_dates=True, float_precision='round_trip')
    model = joblib.load(stage / 'model.joblib')
    actual = predict(model, x.loc[stored.index])
    if not np.allclose(actual, stored.prediction.to_numpy(), rtol=0, atol=1e-10):
        raise ValueError('Saved-model prediction replay mismatch')


def run_frame(frame, audit, output, folds=None, candidates=None, iterations=200, max_stages=None):
    output = Path(output)
    folds = folds or FOLDS
    candidates = candidates or CANDIDATES
    if len(set(candidates)) != len(candidates) or any(c not in CANDIDATES for c in candidates):
        raise ValueError('Candidates must be a unique subset of the fixed bank')
    x = build_features(frame)
    valid = eligible(frame, x)
    x, y = x.loc[valid], frame.cnt.loc[valid]
    if x.empty:
        raise ValueError('Insufficient eligible hourly history')
    manifest = {'data': audit, 'sources': source_manifest(), 'folds': folds,
                'candidates': candidates, 'iterations': iterations, 'interval_alpha': .1,
                'features': list(x.columns), 'eligible_rows': len(x),
                'frame_sha256': __import__('hashlib').sha256(frame.to_csv().encode()).hexdigest(),
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                    'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
                    'scipy': scipy.__version__, 'joblib': joblib.__version__}}
    with exclusive_lock(output / '.lock'):
        freeze(output / 'manifest.json', manifest)
        stages = Stages(output / 'stages', max_stages=max_stages)
        all_predictions, summaries = [], []
        try:
            for fold in folds:
                partitions = masks(x.index, fold)
                if any(not mask.any() for mask in partitions.values()):
                    raise ValueError(f'Empty partition in {fold["id"]}')
                scores = []
                for name in candidates:
                    def build_tuned(path, name=name):
                        model = fit(name, x.loc[partitions['train']], y.loc[partitions['train']], iterations)
                        p = predict(model, x.loc[partitions['tune']])
                        joblib.dump(model, path / 'model.joblib')
                        save_predictions(path / 'predictions.csv', y.loc[partitions['tune']], p)
                        atomic_json(path / 'metrics.json', {'name': name, **point_metrics(y.loc[partitions['tune']], p)})
                    stage = stages.get(f'{fold["id"]}/tune_{name}', build_tuned)
                    assert_replay(stage, x)
                    scores.append(json.loads((stage / 'metrics.json').read_text()))
                chosen = min(scores, key=lambda score: score['mae'])['name']
                def build_calibration(path):
                    model = fit(chosen, x.loc[partitions['refit']], y.loc[partitions['refit']], iterations)
                    p = predict(model, x.loc[partitions['calibrate']])
                    q = radius(y.loc[partitions['calibrate']], p)
                    joblib.dump(model, path / 'model.joblib')
                    save_predictions(path / 'predictions.csv', y.loc[partitions['calibrate']], p)
                    boundaries = {name: {'rows': int(mask.sum()), 'first': str(x.index[mask].min()),
                                        'last': str(x.index[mask].max())} for name, mask in partitions.items()}
                    atomic_json(path / 'selection.json', {'selected': chosen, 'radius': q,
                        'candidate_scores': scores, 'partitions': boundaries})
                final = stages.get(f'{fold["id"]}/refit_and_calibrate', build_calibration)
                assert_replay(final, x)
                selection = json.loads((final / 'selection.json').read_text())
                if selection['selected'] != chosen:
                    raise ValueError('Selection disagrees with cached calibration')
                def build_test(path):
                    model = joblib.load(final / 'model.joblib')
                    p = predict(model, x.loc[partitions['test']])
                    q = selection['radius']
                    rows = pd.DataFrame({'actual': y.loc[partitions['test']], 'selected': p,
                                         'lower': np.maximum(p - q, 0), 'upper': p + q})
                    for name in ['naive_1', 'naive_24', 'naive_168']:
                        rows[name] = x.loc[partitions['test'], name]
                    rows.to_csv(path / 'predictions.csv', float_format='%.17g')
                test_stage = stages.get(f'{fold["id"]}/test', build_test)
                rows = pd.read_csv(test_stage / 'predictions.csv', index_col=0, parse_dates=True,
                                   float_precision='round_trip')
                replay = predict(joblib.load(final / 'model.joblib'), x.loc[rows.index])
                if not np.allclose(replay, rows.selected, rtol=0, atol=1e-10):
                    raise ValueError('Test prediction replay mismatch')
                if not np.array_equal(rows.actual.to_numpy(), y.loc[rows.index].to_numpy()):
                    raise ValueError('Cached target mismatch')
                rows['fold'] = fold['id']
                all_predictions.append(rows)
                summaries.append({'fold': fold['id'], **selection, **summarize(rows)})
            combined = pd.concat(all_predictions)
            if not combined.index.is_unique:
                raise ValueError('Overlapping test folds')
            result = {'data': audit, 'feature_count': len(x.columns), 'folds': summaries,
                      'pooled': summarize(combined), 'test_rows': len(combined),
                      'replay_max_tolerance': 1e-10, 'independent_rows_assumed': False}
            atomic_json(output / 'metrics.json', result)
            temp = output / 'predictions.csv.tmp'
            combined.to_csv(temp, float_format='%.17g')
            temp.replace(output / 'predictions.csv')
            state = {'status': 'complete', 'new_stages': stages.built, 'reused_stages': stages.reused,
                     'test_rows': len(combined)}
            atomic_json(output / 'state.json', state)
            return result, state
        except Paused as exc:
            state = {'status': 'paused', 'next_stage': str(exc),
                     'new_stages': stages.built, 'reused_stages': stages.reused}
            atomic_json(output / 'state.json', state)
            return None, state


def summarize(rows):
    return {'point': {name: point_metrics(rows.actual, rows[name])
                      for name in ['selected', 'naive_1', 'naive_24', 'naive_168']},
            'interval': interval_metrics(rows.actual, rows.lower, rows.upper)}


def run(archive, output, max_stages=None):
    frame, audit = load_archive(archive)
    return run_frame(frame, audit, output, max_stages=max_stages)
