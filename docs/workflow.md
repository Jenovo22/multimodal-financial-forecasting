# Team Workflow

This document turns the current project state into three coordinated workstreams. Each workstream should move through issues, branches, tests and pull requests.

## Current State

The project has a working MVP:

- Black-Scholes-Merton baseline with Greeks.
- SPY option dataset builder and HMM regime table support.
- FINN training with train/validation/test split.
- BSM-anchored residual FINN mode for safer training on a small dataset.
- Scoring pipeline, FastAPI service and local prediction dashboard.
- Collaboration structure: docs, PR template, issue templates, CI and local checks.

Current model status:

- Split strategy: `expiration`, because only one option snapshot date is available.
- Train: earlier expirations.
- Validation: middle expiration.
- Test: later expiration.
- Final test MAE: about `0.2355`.
- BSM test MAE: about `1.4930`.

Main limitation:

- The model has not yet been validated on future quote dates. The next robust step is to collect multiple daily snapshots and move to `--split-strategy timestamp`.

Repository status:

- Several professionalization and model/dashboard changes are ready but still uncommitted.
- Do not start new feature work on top of this without first committing or branching cleanly.

## Workstream 1: Data And Evaluation Foundation

Goal: make the dataset and validation process strong enough for serious research.

Scope:

- Collect multiple daily SPY option snapshots.
- Keep raw snapshots separated by date.
- Build combined canonical training datasets.
- Move from expiration split to timestamp split once enough dates exist.
- Add data manifests with row counts, dates, filters and source commands.
- Add liquidity, spread and transaction-cost fields to evaluation reports.

Recommended branches:

- `feature/daily-option-snapshots`
- `feature/timestamp-split-evaluation`
- `chore/data-manifest`

Definition of done:

- Generated data is not committed.
- Reproduction commands are documented.
- `reports/finn_training_metrics.json` is regenerated locally.
- The PR states train/validation/test periods and row counts.
- `.\scripts\run_checks.ps1` passes.

## Workstream 2: Model And Research Validation

Goal: improve FINN only when it beats strong baselines under honest validation.

Scope:

- Keep BSM residual mode as the default training path.
- Compare FINN against BSM on every evaluation split.
- Add backtesting over historical option snapshots.
- Calibrate PDE loss only after a broader dataset exists.
- Track experiments through GitHub experiment issues.
- Add model version metadata to checkpoints and API responses.

Recommended branches:

- `experiment/finn-residual-calibration`
- `experiment/pde-loss-calibration`
- `feature/backtesting-engine`
- `feature/model-version-metadata`

Definition of done:

- Experiment issue includes hypothesis, command, dataset and metrics.
- FINN metrics are compared against BSM.
- Any change to signals includes risk notes.
- Tests pass.
- No checkpoints are committed unless explicitly approved for release.

## Workstream 3: Decision Product And Collaboration Layer

Goal: make the system usable, understandable and safe for collaborators.

Scope:

- Improve dashboard clarity around prediction, strike, time, confidence and recommendation.
- Add liquidity-adjusted edge and cost-aware signal thresholds.
- Keep API contracts stable and documented.
- Add dashboard/API tests when behavior changes.
- Maintain README, docs, changelog and PR discipline.

Recommended branches:

- `feature/liquidity-aware-signals`
- `feature/dashboard-risk-panels`
- `feature/api-model-metadata`
- `chore/docs-maintenance`

Definition of done:

- User-facing behavior is documented.
- Dashboard/API outputs include caveats where needed.
- Tests cover the changed behavior.
- PR includes screenshots or generated report notes if the dashboard changed.
- `.\scripts\run_checks.ps1` passes.

## Standard Delivery Flow

1. Create or select an issue.
2. Create a branch from the latest `main`.
3. Implement the smallest useful change.
4. Run local checks.
5. Regenerate metrics/reports locally if model or dashboard behavior changed.
6. Update docs and changelog.
7. Open a PR using the template.
8. Review risks, data assumptions and tests.
9. Merge only after checks pass.

## Commands

Local checks:

```powershell
.\scripts\run_checks.ps1
```

Robust current training:

```powershell
.\.venv\Scripts\python.exe scripts\train_finn_model.py `
  --option-source Data\raw\options\SPY_options_2026-05-05.csv `
  --regime-source Data\processed\spy_regime_features.csv `
  --output artifacts\finn_model.pt `
  --split-strategy expiration `
  --final-train-split train_val `
  --epochs 500 `
  --prediction-mode bsm_residual `
  --residual-scale 0.50 `
  --early-stopping-patience 75
```

Once multiple daily snapshots exist, prefer:

```powershell
.\.venv\Scripts\python.exe scripts\train_finn_model.py `
  --option-source <combined-options-source.csv> `
  --regime-source Data\processed\spy_regime_features.csv `
  --output artifacts\finn_model.pt `
  --split-strategy timestamp `
  --final-train-split train_val `
  --epochs 500 `
  --prediction-mode bsm_residual
```

## Immediate Next Action

Before opening new workstreams, commit the current professionalization work:

```powershell
git add .
git commit -m "Organize repository for collaborative development"
```

Then create issues for the three workstreams above and start with Workstream 1.

