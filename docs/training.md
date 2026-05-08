# Training

The current robust path trains FINN with a supervised target, no-arbitrage regularization and a Black-Scholes-Merton residual anchor.

## Preparar particiones de datos

Antes de entrenar, se puede generar una particion explicita y auditable en `Splits/` sin modificar la carpeta original `Data/`. El script preserva la estructura de `Data/` dentro de cada split y escribe manifests de auditoria.

Caso actual, con un solo snapshot de opciones:

```powershell
.\.venv\Scripts\python.exe scripts\create_data_splits.py `
  --strategy expiration `
  --clear-output
```

Caso futuro, cuando existan multiples snapshots diarios:

```powershell
.\.venv\Scripts\python.exe scripts\create_data_splits.py `
  --strategy timestamp `
  --clear-output
```

Revisar el resumen generado:

```powershell
Import-Csv Splits\expiration_split\manifests\split_summary.csv |
  Format-Table split, rows, split_column, split_mode, key_start, key_end, target_path -AutoSize
```

Opciones principales:

- `--strategy expiration`: separa por expiracion. Es la opcion recomendada mientras solo haya un snapshot de opciones.
- `--strategy timestamp`: separa por fecha o timestamp. Debe preferirse cuando ya existan varios snapshots diarios.
- `--clear-output`: limpia la particion generada previamente para evitar mezclar resultados viejos y nuevos.

El script de entrenamiento puede seguir generando sus propios splits internos si se usa `scripts\train_finn_model.py` con `--split-strategy`. La carpeta `Splits/` sirve principalmente como particion explicita y auditable de datos en disco.

## Current Recommended Training

Use this while the project has only one option snapshot date:

```powershell
.\.venv\Scripts\python.exe scripts\train_finn_model.py `
  --option-source Data\raw\options\SPY_options_2026-05-05.csv `
  --regime-source Data\processed\spy_regime_features.csv `
  --output artifacts\finn_model.pt `
  --selection-output artifacts\finn_model_selection.pt `
  --split-output Data\processed\finn_split_dataset.csv `
  --metrics-output reports\finn_training_metrics.json `
  --predictions-output Data\processed\finn_split_predictions.csv `
  --split-strategy expiration `
  --train-fraction 0.60 `
  --validation-fraction 0.20 `
  --final-train-split train_val `
  --epochs 500 `
  --batch-size 64 `
  --learning-rate 0.001 `
  --prediction-mode bsm_residual `
  --residual-scale 0.50 `
  --lambda-boundary 0.1 `
  --lambda-pde 0 `
  --lambda-arbitrage 1.0 `
  --early-stopping-patience 75
```

## Split Strategy

- `expiration`: preferred when there is one option quote date. The model trains on earlier expirations and tests on later expirations.
- `timestamp`: preferred once multiple daily option snapshots exist. The model trains on older dates and tests on future unseen dates.
- `random`: only for smoke tests or debugging. It is not a reliable financial validation strategy.

## Outputs

- `Data/processed/finn_split_dataset.csv`: rows with `train`, `validation` or `test`.
- `Data/processed/finn_split_predictions.csv`: row-level errors and predictions.
- `reports/finn_training_metrics.json`: model metrics versus BSM.
- `artifacts/finn_model_selection.pt`: validation-selected model.
- `artifacts/finn_model.pt`: final model used by API/scoring.

## Metric Interpretation

Primary metrics:

- `mae`: average dollar pricing error.
- `rmse`: penalizes large pricing misses.
- `baseline_mae`: BSM comparison on the same rows.
- `mape_pct`: relative error, useful but unstable for very cheap options.
- `delta_outside_bounds`: should be zero.
- `gamma_negative`: should be watched because negative gamma can indicate unstable learned Greeks.

A model is not operationally strong just because MAE improves. Also inspect spreads, liquidity, regime label, transaction costs and whether the test set is truly future data.

## Exhaustive Training Plan

1. Collect option snapshots daily.
2. Keep raw snapshots separated by date.
3. Build a combined canonical option dataset.
4. Train with `--split-strategy timestamp`.
5. Compare FINN against BSM on the future test window.
6. Score the latest option chain with realistic transaction cost and safety margin.
7. Review dashboard candidates by liquidity before considering any simulated trade.
