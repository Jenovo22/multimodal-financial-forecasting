# Multimodal Financial Forecasting

MVP for an HMM + FINN option-pricing pipeline. The current project includes:

- Black-Scholes-Merton baseline with Greeks.
- SPY market/regime dataset bridge.
- HMM regime detector and persisted regime table support.
- FINN model scaffold with PyTorch training utilities.
- FastAPI scoring API.

## Setup

Recommended on Windows:

```powershell
.\scripts\setup_environment.bat -WithML
```

Without PyTorch/FINN training:

```powershell
.\scripts\setup_environment.bat
```

Manual install:

```powershell
python -m pip install -e ".[dev]"
python -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

More detail: [markdown/environment_setup.md](markdown/environment_setup.md).

## Common commands

Build the SPY regime table:

```powershell
python scripts\build_spy_regime_table.py --output Data\processed\spy_regime_features.csv
```

Build a canonical option dataset for inspection:

```powershell
python scripts\build_option_training_dataset.py --option-source path\to\options.csv --output Data\processed\option_training_dataset.csv
```

Train FINN from an option source and the regime table:

```powershell
python scripts\train_finn_model.py --option-source path\to\options.csv --regime-source Data\processed\spy_regime_features.csv --output artifacts\finn_model.pt
```

Run the API:

```powershell
python scripts\run_api.py --finn-checkpoint artifacts\finn_model.pt
```
