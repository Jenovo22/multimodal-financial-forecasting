# Option data automation

The project can automate point 5 with `scripts/download_option_chain.py`.

## One-shot update

```powershell
.\scripts\setup_environment.bat -WithData -WithML
.\.venv\Scripts\Activate.ps1
python scripts\download_option_chain.py --symbol SPY --min-dte-days 20 --max-dte-days 45 --quote-timestamp latest-market-date --build-dataset
```

This writes:

- Raw option chain: `Data/raw/options/SPY_options_<DATE>.csv`
- Canonical option dataset: `Data/processed/option_training_dataset.csv`

`latest-market-date` aligns the option rows with the latest SPY date available in `Data/final/master_dataset_2010_present_long.csv`. In the current dataset that date is `2026-05-05`.

## Convenience script

```powershell
.\scripts\update_spy_options.ps1
```

Equivalent to downloading both calls and puts, then building the MVP canonical dataset for calls.

## Windows Task Scheduler

Run this from PowerShell after adjusting the repository path if needed:

```powershell
$repo = "C:\Users\gilse\OneDrive\Escritorio\proyecto_tam\multimodal-financial-forecasting"
schtasks /Create /TN "ProyectoTAM_UpdateSPYOptions" /SC DAILY /ST 18:30 /TR "powershell -NoProfile -ExecutionPolicy Bypass -File `"$repo\scripts\update_spy_options.ps1`"" /F
```

To test the scheduled command manually:

```powershell
schtasks /Run /TN "ProyectoTAM_UpdateSPYOptions"
```

## Notes

The downloader uses `yfinance>=0.2.54,<1.0`, which is practical for MVP automation. The upper bound avoids the current 1.x networking stack that can fail on some Windows TLS setups. For production-grade research or trading workflows, replace this source with a licensed options feed and keep the same raw CSV contract.
