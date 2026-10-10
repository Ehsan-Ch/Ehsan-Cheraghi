# Continuation checkpoint

Project: **Bike Demand Forecasting**.
Current repository: `Ehsan-Ch/Ehsan-Cheraghi`.
Intended repository name: `bike-demand-forecasting`.

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

## Remaining publication steps

1. Finish all three forward test quarters and replay every saved prediction.
2. Export and independently verify reports, figures, and per-hour predictions.
3. Publish results and the final README; update the main profile link.
4. Rename the repository. The current connector does not expose repository
   renaming; browser fallback needs separate approval before use.

After an interruption, read this file and the remote main branch first. Inspect
remote files before retrying a publish. If local artifacts disappeared, download
the pinned public archive and rerun the frozen code in a fresh output directory;
local checkpoint files cannot be guaranteed across a workspace reset. Token
resets cannot be detected by this code. Continue when the user gives the signal.
