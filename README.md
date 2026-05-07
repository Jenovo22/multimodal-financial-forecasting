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

Train FINN from an option source and the regime table:

```powershell
python scripts\train_finn_model.py --option-source Data\raw\options\SPY_options_2026-05-05.csv --regime-source Data\processed\spy_regime_features.csv --output artifacts\finn_model.pt
```

The current training default is supervised plus no-arbitrage regularization. PDE loss is intentionally disabled by default until that term is calibrated on a broader dataset.

Run the API:

```powershell
python scripts\run_api.py --finn-checkpoint artifacts\finn_model.pt
```
