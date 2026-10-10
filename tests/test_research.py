import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from bike_forecast.checkpoints import Stages, Paused, exclusive_lock, freeze
from bike_forecast.data import regularize
from bike_forecast.experiment import masks, run_frame
from bike_forecast.features import build_features
from bike_forecast.modeling import fit, predict, radius, interval_metrics


def fixture(days=75):
    dates = pd.date_range('2011-01-01', periods=24*days, freq='h')
    n = len(dates)
    count = np.rint(40 + 20*np.sin(np.arange(n)*2*np.pi/24) + np.arange(n)*.01)
    return pd.DataFrame({'dteday': dates.strftime('%Y-%m-%d'), 'hr': dates.hour,
        'cnt': count, 'holiday': 0, 'temp': .5, 'atemp': .45, 'hum': .6,
        'windspeed': .15, 'weathersit': 1})


class DataAndCausalityTests(unittest.TestCase):
    def test_regularized_column_order_is_stable_across_python_processes(self):
        import os
        import subprocess
        import sys
        program = "from test_research import fixture; from bike_forecast.data import regularize; import hashlib; print(hashlib.sha256(regularize(fixture(10)).to_csv().encode()).hexdigest())"
        hashes = []
        for seed in ['1', '2']:
            environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH='src' + os.pathsep + 'tests')
            hashes.append(subprocess.check_output([sys.executable, '-c', program], env=environment, text=True).strip())
        self.assertEqual(hashes[0], hashes[1])

    def test_missing_hour_stays_missing_and_lag_means_clock_hour(self):
        raw = fixture(10).drop(index=49)
        frame = regularize(raw)
        self.assertTrue(np.isnan(frame.cnt.iloc[49]))
        x = build_features(frame)
        self.assertTrue(np.isnan(x.count_lag_1.iloc[50]))
        self.assertEqual(x.naive_1.iloc[50], frame.cnt.iloc[48])
        self.assertEqual(x.count_lag_24.iloc[50], frame.cnt.iloc[26])

    def test_current_and_future_observations_cannot_change_prior_features(self):
        frame = regularize(fixture(12)); changed = frame.copy()
        cols = ['cnt', 'temp', 'atemp', 'hum', 'windspeed', 'weathersit']
        changed.loc[changed.index[200]:, cols] = 99999
        pd.testing.assert_frame_equal(build_features(frame).iloc[:201], build_features(changed).iloc[:201])

    def test_unsafe_fields_do_not_enter_features(self):
        frame = regularize(fixture(10))
        frame['casual'] = frame.cnt
        frame['registered'] = 0
        columns = build_features(frame).columns
        self.assertFalse({'cnt', 'casual', 'registered', 'temp', 'atemp', 'hum', 'windspeed'} & set(columns))

    def test_duplicate_dates_invalid_hours_and_counts_are_rejected(self):
        raw = fixture(10)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            regularize(pd.concat([raw, raw.iloc[:1]]))
        changed = raw.copy(); changed.loc[1, 'hr'] = 24
        with self.assertRaises(ValueError): regularize(changed)
        changed = raw.copy(); changed.loc[1, 'cnt'] = -1
        with self.assertRaises(ValueError): regularize(changed)

    def test_unsorted_grid_rejected_and_input_regularization_sorts(self):
        raw = fixture(10)
        pd.testing.assert_frame_equal(regularize(raw), regularize(raw.iloc[::-1]))
        with self.assertRaises(ValueError): build_features(regularize(raw).iloc[::-1])

    def test_partition_boundaries_are_strict_and_disjoint(self):
        index = pd.date_range('2011-01-01', periods=100*24, freq='h')
        f = {'tune':'2011-02-01','calibrate':'2011-02-15','test':'2011-03-01','end':'2011-03-15'}
        parts = masks(index, f)
        self.assertTrue(np.all(sum(parts[n].astype(int) for n in ['train','tune','calibrate','test']) <= 1))
        self.assertEqual(index[parts['test']].min(), pd.Timestamp('2011-03-01'))
        self.assertTrue(np.array_equal(parts['refit'], parts['train'] | parts['tune']))


class ModelsAndIntervalsTests(unittest.TestCase):
    def test_scaler_is_fit_on_training_only_and_schema_is_checked(self):
        x = build_features(regularize(fixture(12))).iloc[168:]
        y = np.arange(len(x)) + 1
        model = fit('ridge_1', x.iloc[:50], y[:50])
        scaler = model['model'].named_steps['standardscaler']
        before = scaler.mean_.copy()
        altered = x.iloc[50:].copy(); altered['elapsed_days'] = 1e8
        predict(model, altered)
        np.testing.assert_array_equal(before, scaler.mean_)
        with self.assertRaisesRegex(ValueError, 'schema'): predict(model, x.iloc[:, ::-1])

    def test_finite_sample_order_statistic(self):
        self.assertEqual(radius(np.arange(10), np.zeros(10)), 9)
        self.assertEqual(radius(np.ones(20), np.zeros(20)), 1)
        with self.assertRaises(ValueError): radius([], [])

    def test_interval_coverage_width_and_penalty(self):
        actual = interval_metrics([0, 4], [0, 1], [2, 3])
        self.assertEqual(actual['coverage'], .5)
        self.assertEqual(actual['mean_width'], 2)
        self.assertEqual(actual['interval_score'], 12)


class CheckpointTests(unittest.TestCase):
    def test_atomic_resume_does_not_repeat_completed_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            count = []
            def builder(path):
                count.append(1); (path/'model.txt').write_text('model')
            Stages(tmp).get('fold/model', builder)
            Stages(tmp).get('fold/model', builder)
            self.assertEqual(len(count), 1)

    def test_failed_stage_is_not_certified_and_rebuilds(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fail(path):
                (path/'partial.txt').write_text('incomplete'); raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError): Stages(tmp).get('model', fail)
            self.assertFalse((Path(tmp)/'model').exists())
            Stages(tmp).get('model', lambda p:(p/'ok').write_text('complete'))
            self.assertTrue((Path(tmp)/'model/complete.json').is_file())

    def test_corrupt_stage_and_changed_manifest_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Stages(tmp).get('model', lambda p:(p/'weights').write_text('correct'))
            (stage/'weights').write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                Stages(tmp).get('model', lambda p:None)
            freeze(Path(tmp)/'manifest.json', {'seed':42})
            with self.assertRaisesRegex(ValueError, 'manifest'):
                freeze(Path(tmp)/'manifest.json', {'seed':43})

    def test_duplicate_worker_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with exclusive_lock(Path(tmp)/'.lock'):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(Path(tmp)/'.lock'): pass

    def test_zero_budget_pauses_without_a_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Paused): Stages(tmp, 0).get('new', lambda p:None)


class PipelineTests(unittest.TestCase):
    def test_pause_resume_exact_replay_and_independent_metric_recomputation(self):
        frame = regularize(fixture())
        folds = [{'id':'fixture','tune':'2011-02-01','calibrate':'2011-02-15',
                  'test':'2011-03-01','end':'2011-03-15'}]
        kwargs = dict(folds=folds, candidates=['naive_1','ridge_1','hgb_15'], iterations=3)
        with tempfile.TemporaryDirectory() as tmp:
            _, paused = run_frame(frame, {'kind':'synthetic_unit_fixture'}, tmp, max_stages=1, **kwargs)
            self.assertEqual(paused['status'], 'paused')
            self.assertEqual(paused['new_stages'], 1)
            metrics, state = run_frame(frame, {'kind':'synthetic_unit_fixture'}, tmp, **kwargs)
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['reused_stages'], 1)
            rows = pd.read_csv(Path(tmp)/'predictions.csv')
            expected = np.mean(np.abs(rows.actual-rows.selected))
            self.assertAlmostEqual(metrics['pooled']['point']['selected']['mae'], expected)
            with patch('bike_forecast.experiment.fit', side_effect=AssertionError('Unexpected refit')):
                replayed, state = run_frame(frame, {'kind':'synthetic_unit_fixture'}, tmp, max_stages=0, **kwargs)
            self.assertEqual(state['new_stages'], 0)
            self.assertEqual(state['reused_stages'], 5)
            self.assertEqual(metrics, replayed)


if __name__ == '__main__':
    unittest.main()
