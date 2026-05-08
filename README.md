# Multimodal Financial Forecasting

MVP for an HMM + FINN option-pricing pipeline. The current project includes:

- Black-Scholes-Merton baseline with Greeks.
- SPY market/regime dataset bridge.
- HMM regime detector and persisted regime table support.
- FINN model scaffold with PyTorch training utilities.
- FastAPI scoring API.
- Static prediction dashboard for local inspection.

This is a research project. Model outputs are analytical signals, not financial advice or automated trade instructions.

## Repository Map

```text
src/                  Core Python package
scripts/              Reproducible command-line workflows
tests/                Regression and smoke tests
docs/                 Collaboration and operational documentation
markdown/             Legacy MVP specs and schema references
notebooks/            Exploratory and legacy notebooks
Data/                 Local/legacy data files
artifacts/            Local model checkpoints, ignored by Git
reports/              Local generated reports, ignored by Git
.github/              CI, PR template and issue templates
```

Start with [docs/index.md](docs/index.md) for architecture, data policy, training, API and dashboard documentation. For a complete current-state overview, read [docs/project_context.md](docs/project_context.md).

## Setup

Recommended on Windows:

```powershell
.\scripts\setup_environment.bat -WithData -WithML
```

Without PyTorch/FINN training:

```powershell
.\scripts\setup_environment.bat -WithData
```

Manual install:

```powershell
python -m pip install -e ".[dev]"
python -m pip install -e ".[data]"
python -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

More detail: [markdown/environment_setup.md](markdown/environment_setup.md).

## Quality Checks

Run before opening a pull request:

```powershell
.\scripts\run_checks.ps1
```

This compiles Python files and runs the full test suite.

## Common commands

Build the SPY regime table:

```powershell
python scripts\build_spy_regime_table.py --output Data\processed\spy_regime_features.csv
```

Build a canonical option dataset for inspection:

```powershell
python scripts\download_option_chain.py --symbol SPY --min-dte-days 20 --max-dte-days 45 --quote-timestamp latest-market-date --build-dataset
```

Automate the same SPY option update:

```powershell
.\scripts\update_spy_options.ps1
```

Train FINN from an option source and the regime table with an honest
train/validation/test split:

```powershell
python scripts\train_finn_model.py --option-source Data\raw\options\SPY_options_2026-05-05.csv --regime-source Data\processed\spy_regime_features.csv --output artifacts\finn_model.pt --split-strategy expiration --final-train-split train_val --epochs 500 --prediction-mode bsm_residual --residual-scale 0.50
```

The current training default is a BSM-anchored residual FINN plus no-arbitrage
regularization. This is safer than learning the whole option price from scratch
when the dataset is still small. The script writes:

- `Data/processed/finn_split_dataset.csv` with train/validation/test labels.
- `Data/processed/finn_split_predictions.csv` with row-level predictions.
- `reports/finn_training_metrics.json` with MAE/RMSE against FINN and BSM.
- `artifacts/finn_model_selection.pt` for validation-selected diagnostics.
- `artifacts/finn_model.pt` for the final checkpoint.

Use `--split-strategy expiration` while there is only one option snapshot date.
Once several daily snapshots exist, prefer `--split-strategy timestamp` so the
test set represents future unseen dates. PDE loss is intentionally disabled by
default until that term is calibrated on a broader dataset.

Run the API:

```powershell
python scripts\run_api.py --finn-checkpoint artifacts\finn_model.pt
```

Build a local prediction dashboard:

```powershell
python scripts\build_prediction_dashboard.py --open
```

The dashboard overlays observed market price, FINN fair value, Black-Scholes price, and last traded price by strike and expiration.

## Collaboration

Use short-lived branches and pull requests. Do not commit directly to `main`.

Recommended branch names:

- `feature/<short-name>`
- `fix/<short-name>`
- `chore/<short-name>`
- `experiment/<short-name>`

See [CONTRIBUTING.md](CONTRIBUTING.md) for PR expectations, data policy and financial-signal caveats.

## Data And Artifacts

Generated datasets, model checkpoints and reports should usually stay local:

- `Data/raw/`
- `Data/processed/`
- `artifacts/`
- `reports/`

Some seed/legacy data is currently tracked. Do not remove or rewrite it casually; if the team decides to migrate data out of Git, do that in a dedicated data-migration PR.
