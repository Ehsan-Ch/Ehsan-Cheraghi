"""Generate readable reports and static figures from computed evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def report(output, destination):
    output, destination = Path(output), Path(destination)
    metrics = json.loads((output / 'metrics.json').read_text())
    predictions = pd.read_csv(output / 'predictions.csv', index_col=0, parse_dates=True)
    destination.mkdir(parents=True, exist_ok=True)
    figures = destination / 'figures'
    figures.mkdir(exist_ok=True)
    colors = ['#126e82', '#97a3b6', '#97a3b6', '#97a3b6']
    names = ['selected', 'naive_1', 'naive_24', 'naive_168']
    labels = ['Selected workflow', 'Last observation', 'Daily naive', 'Weekly naive']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'bike-demand-forecasting'})
    fig, ax = plt.subplots(figsize=(9, 4.2), layout='constrained')
    values = [metrics['pooled']['point'][n]['mae'] for n in names]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=.6)
    ax.bar_label(bars, fmt='%.2f', padding=6)
    ax.set_xlim(0, max(values)*1.2)
    ax.set_xlabel('Mean absolute error · rentals per hour · lower is better')
    ax.set_title('Forward tests on real bike-sharing demand', loc='left', fontweight='bold', pad=16)
    ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    fig.savefig(figures / 'benchmark.svg', metadata={'Date': None}); plt.close(fig)
    # Fixed display window: first seven days of the final test quarter.
    last_fold = metrics['folds'][-1]['fold']
    subset = predictions[predictions.fold == last_fold]
    subset = subset.loc[subset.index < subset.index.min() + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(11, 4.3), layout='constrained')
    ax.fill_between(subset.index, subset.lower, subset.upper, color='#126e82', alpha=.18,
                    label='Nominal 90% interval (empirically assessed)')
    ax.plot(subset.index, subset.actual, color='#263348', linewidth=1.1, label='Observed')
    ax.plot(subset.index, subset.selected, color='#0b8893', linewidth=1.2, label='Forecast')
    ax.set_ylabel('Rentals per hour'); ax.set_xlabel('Naive source timestamp')
    ax.set_title('First week of the final test quarter', loc='left', fontweight='bold')
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    fig.savefig(figures / 'forecast.svg', metadata={'Date': None}); plt.close(fig)
    (destination / 'metrics.json').write_bytes((output / 'metrics.json').read_bytes())
    (destination / 'manifest.json').write_bytes((output / 'manifest.json').read_bytes())
    (destination / 'predictions.csv').write_bytes((output / 'predictions.csv').read_bytes())
    lines = ['# Executed results', '', 'All values below are computed from the public UCI dataset.',
             'This is a historical forecast experiment, not a production or competition result.', '',
             f"Test observations: **{metrics['test_rows']:,}**. Features: **{metrics['feature_count']}**.", '',
             '| Workflow | Pooled MAE | Pooled RMSE |', '| --- | ---: | ---: |']
    for name, label in zip(names, labels):
        m = metrics['pooled']['point'][name]
        lines.append(f"| {label} | {m['mae']:.3f} | {m['rmse']:.3f} |")
    lines += ['', '![Baseline comparison](figures/benchmark.svg)', '',
              '| Test quarter | Selected recipe | MAE | Coverage | Mean interval width |',
              '| --- | --- | ---: | ---: | ---: |']
    for fold in metrics['folds']:
        lines.append(f"| {fold['fold']} | {fold['selected']} | {fold['point']['selected']['mae']:.3f} | "
                     f"{fold['interval']['coverage']:.1%} | {fold['interval']['mean_width']:.2f} |")
    interval = metrics['pooled']['interval']
    lines += ['', f"Pooled nominal-90% interval coverage: **{interval['coverage']:.1%}**. "
              f"Mean width: **{interval['mean_width']:.2f}** rentals; interval score: **{interval['interval_score']:.2f}**.",
              '', '![Forecast and interval](figures/forecast.svg)', '',
              '## Interpretation', '',
              'Point models are selected using earlier months only. The reported test outcomes do not select ',
              'hyperparameters, model families or the interval radius. Forecasts may use earlier observed ',
              'test-hour counts, as allowed by the one-hour forecasting contract.', '',
              'Temporal dependence and distribution shift prevent an exchangeability-based coverage ',
              'guarantee. Empirical coverage is evidence about these periods only; missed nominal ',
              'coverage must not be hidden or recalibrated using the test labels.', '',
              'Missing clock hours remain unobserved and are excluded from scoring, not treated as zero ',
              'demand. The absence mechanism, naive local timestamps, historical operating conditions ',
              'and unverified measurement latency limit practical transfer.', '',
              '## Reproduction evidence', '',
              '- [Exact metrics, selection scores and split boundaries](metrics.json)',
              '- [Frozen source, data and environment manifest](manifest.json)',
              '- [Every held-out observation, prediction, interval and baseline](predictions.csv)',
              '- [Protocol fixed before fitting](../docs/PROTOCOL.md)', '']
    (destination / 'RESULTS.md').write_text('\n'.join(lines))
