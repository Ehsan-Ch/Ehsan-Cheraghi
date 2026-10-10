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

## Completed experiment — 10 October 2026

- All 27 stages completed; a separate replay reused all 27 with zero new fits.
- 6,558 held-out hourly observations; 32 features.
- MAE 39.007083; weekly naive MAE 62.702501; nominal-90% coverage 0.883196.
- Metrics, selections and temporal boundaries independently checked.
- Source snapshot published at commit `2786f00`.

## Publication and remaining administration

The implemented package, tests, computed results, figures and verification
receipt are included in this repository. Source used for the measured run is
commit `2786f00`; the report manifest records every code and protocol hash.
The project README title is **Bike Demand Forecasting**.

Remaining action: rename `Ehsan-Cheraghi` to `bike-demand-forecasting`. The
connector does not expose repository renaming. Browser fallback needs approval
before use. After renaming, update the main profile project link and remove the
rename-pending note from the project README. Verify the new URL, its default
branch and existing content; do not create a duplicate repository.

After an interruption, read this file and the remote main branch first. Inspect
remote files before retrying a publish. If local artifacts disappeared, download
the pinned public archive and rerun the frozen code in a fresh output directory;
local checkpoint files cannot be guaranteed across a workspace reset. Token
resets cannot be detected by this code. Continue when the user gives the signal.

## Verified publication and approval boundary

The complete measured project was published and file hashes verified at commit
`c0710dbd57701d97924ba645f2b5dfd2a4162e9f` on 10 October 2026.
The profile README has **not** been updated: automatic approval review rejected
that separate-repository edit as outside the explicit authorization for this
request. Ask for permission to add the Bike Demand Forecasting link there.
The repository URL rename also remains pending because the GitHub connector
has no rename operation and browser fallback needs approval before use.
Once the user approves these two exact actions, complete them and verify both
URLs. The experiment is finished; do not retrain or repeat its publication.
