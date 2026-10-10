# Continuation checkpoint

Project: **Bike Demand Forecasting**.
Current repository: `Ehsan-Ch/bike-demand-forecasting`.
Status: **Complete**, including repository rename and profile link.

## Completed as of 10 October 2026

- Official public UCI archive downloaded and hash-verified.
- Fixed forecasting protocol saved before training in commit `a87e640`.
- Package, CLI, atomic stages, reporting and 16 regression tests implemented.
- Input-column order made deterministic across separate Python processes.

The original `artifacts/official` attempt contains four completed stages. Its
restart failed because an unordered Python set changed input-column order and
therefore the frame fingerprint. That namespace is preserved. The corrected
experiment uses `artifacts/official_v2`; old manifests are not edited to force
reuse. No test-set performance was evaluated in the initial attempt.

## Resume the corrected experiment

```bash
python -m pip install -e .
bike-forecast download
# Linux/macOS, from the repository root:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bike-forecast run --output artifacts/official_v2
bike-forecast verify --output artifacts/official_v2
bike-forecast report --output artifacts/official_v2
```

On Windows PowerShell set `$env:OPENBLAS_NUM_THREADS="1"` and
`$env:OMP_NUM_THREADS="1"`, then run the same CLI commands without the leading
environment assignments. `--max-stages N` bounds newly completed stages; rerun
the same command to verify and reuse existing stages. Never alter a frozen
manifest or fabricate missing outputs. An interrupted, uncommitted stage is
repeated; completed verified stages are reused.

## Completed experiment — 10 October 2026

- All 27 stages completed; a separate replay reused all 27 with zero new fits.
- 6,558 held-out hourly observations; 32 features.
- MAE 39.007083; weekly naive MAE 62.702501; nominal-90% coverage 0.883196.
- Metrics, selections and temporal boundaries independently checked.
- Source snapshot published at commit `2786f00`.

## Publication and administration completed — 10 October 2026

The complete measured project was published and file hashes verified at commit
`c0710dbd57701d97924ba645f2b5dfd2a4162e9f`. Source used for the measured run is
commit `2786f00`; the report manifest records every code and protocol hash.

Following explicit user approval, the existing repository was renamed from
`Ehsan-Ch/Ehsan-Cheraghi` to `Ehsan-Ch/bike-demand-forecasting`. Its `main`
branch, history, source and reports were preserved. No duplicate repository was
created. The project README's rename-pending note has been removed.

The profile README in `Ehsan-Ch/Ehsan-Ch` now links to the renamed project and
summarizes the measured result. Profile update commit:
`e35c2d2a5804b70c02f5307fd0aa4c652c38bb8d`.

Repository: https://github.com/Ehsan-Ch/bike-demand-forecasting
Profile README: https://github.com/Ehsan-Ch/Ehsan-Ch/blob/main/README.md

## Recovery after interruption

Read this file and the remote main branch first. The project and administrative
steps are complete; do not retrain or repeat publication merely to resume this
task. Inspect remote files before retrying any interrupted future write.

Completed local stages can be reused using the commands above. If local artifacts
disappeared, the pinned public archive and frozen code can reproduce the run in a
fresh output directory; local checkpoint files cannot be guaranteed across a
workspace reset. Token resets cannot be detected by this code. An interrupted
stage repeats, while completed verified stages are reused.
