# Data And Evaluation Workflow

This document describes the data/evaluation workstream for the `experiment/data-evaluation` branch.

## Goal

Move the project from a single option snapshot toward a temporally robust evaluation process. The current model has only been evaluated with an expiration split from one quote timestamp. The next robust milestone is a multi-day option dataset that supports `--split-strategy timestamp`.

## Current Limitation

The current local option source is:

```text
Data/raw/options/SPY_options_2026-05-05.csv
```

It contains one quote timestamp:

```text
2026-05-05T00:00:00Z
```

With one timestamp, the system can validate across expirations but cannot yet prove generalization to future quote dates.

## Snapshot Combination

Use `scripts/combine_option_snapshots.py` to combine multiple raw option snapshots into one canonical option source plus a manifest.

Default command:

```powershell
.\.venv\Scripts\python.exe scripts\combine_option_snapshots.py `
  --source-dir Data\raw\options `
  --output Data\processed\combined_option_snapshots.csv `
  --manifest-output Data\processed\combined_option_snapshots_manifest.json
```

Explicit snapshot list:

```powershell
.\.venv\Scripts\python.exe scripts\combine_option_snapshots.py `
  --snapshot Data\raw\options\SPY_options_2026-05-05.csv `
  --snapshot Data\raw\options\SPY_options_2026-05-06.csv `
  --output Data\processed\combined_option_snapshots.csv `
  --manifest-output Data\processed\combined_option_snapshots_manifest.json
```

Require timestamp-split readiness:

```powershell
.\.venv\Scripts\python.exe scripts\combine_option_snapshots.py `
  --source-dir Data\raw\options `
  --require-timestamp-split-ready
```

This fails until at least three unique quote timestamps exist by default.

## Manifest

The manifest records:

- generation timestamp
- source files
- raw and normalized row counts
- rows before and after deduplication
- duplicate rows removed
- unique quote timestamps
- rows by timestamp
- option types
- expirations
- timestamp-split readiness

The key field is:

```json
"timestamp_split_ready": true
```

This should be true before switching production research training to `--split-strategy timestamp`.

## Deduplication

The combination step deduplicates by:

```text
timestamp
underlying_symbol
option_symbol
```

If a duplicate appears, the last row wins. This supports re-downloading a snapshot and keeping the latest copy without double-counting the same option contract at the same quote timestamp.

## Training With Combined Snapshots

Once the combined option source has at least three timestamps:

```powershell
.\.venv\Scripts\python.exe scripts\train_finn_model.py `
  --option-source Data\processed\combined_option_snapshots.csv `
  --regime-source Data\processed\spy_regime_features.csv `
  --split-strategy timestamp `
  --final-train-split train_val `
  --prediction-mode bsm_residual `
  --hidden-dims 32,32 `
  --residual-scale 0.50 `
  --epochs 500
```

Important: the market dataset must include matching dates for the option quote timestamps. If the option snapshots use dates that are not present in `Data/final/master_dataset_2010_present_long.csv`, dataset enrichment will fail or produce no usable rows.

## Definition Of Done For This Workstream

- Multiple raw snapshots are collected.
- The combined source has at least three quote timestamps.
- The manifest documents row counts and date coverage.
- FINN is trained with `--split-strategy timestamp`.
- Metrics compare FINN against BSM on future unseen quote dates.
- Generated CSV/JSON/checkpoint/report files are not committed unless explicitly approved.

