# Executed results

All values below are computed from the public UCI dataset.
This is a historical forecast experiment, not a production or competition result.

Test observations: **6,558**. Features: **32**.

| Workflow | Pooled MAE | Pooled RMSE |
| --- | ---: | ---: |
| Selected workflow | 39.007 | 65.891 |
| Last observation | 86.036 | 128.825 |
| Daily naive | 80.769 | 134.401 |
| Weekly naive | 62.703 | 107.590 |

![Baseline comparison](figures/benchmark.svg)

| Test quarter | Selected recipe | MAE | Coverage | Mean interval width |
| --- | --- | ---: | ---: | ---: |
| 2012_Q2 | hgb_31 | 54.282 | 84.9% | 207.94 |
| 2012_Q3 | hgb_31 | 35.576 | 87.5% | 142.76 |
| 2012_Q4 | hgb_15 | 27.128 | 92.7% | 140.49 |

Pooled nominal-90% interval coverage: **88.3%**. Mean width: **163.70** rentals; interval score: **305.94**.

![Forecast and interval](figures/forecast.svg)

## Interpretation

Point models are selected using earlier months only. The reported test outcomes do not select 
hyperparameters, model families or the interval radius. Forecasts may use earlier observed 
test-hour counts, as allowed by the one-hour forecasting contract.

Temporal dependence and distribution shift prevent an exchangeability-based coverage 
guarantee. Empirical coverage is evidence about these periods only; missed nominal 
coverage must not be hidden or recalibrated using the test labels.

Missing clock hours remain unobserved and are excluded from scoring, not treated as zero 
demand. The absence mechanism, naive local timestamps, historical operating conditions 
and unverified measurement latency limit practical transfer.

## Reproduction evidence

- [Exact metrics, selection scores and split boundaries](metrics.json)
- [Frozen source, data and environment manifest](manifest.json)
- [Every held-out observation, prediction, interval and baseline](predictions.csv)
- [Protocol fixed before fitting](../docs/PROTOCOL.md)
