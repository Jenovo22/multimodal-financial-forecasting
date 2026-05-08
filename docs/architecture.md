# Architecture

The MVP is an options-pricing research system for SPY option chains. It combines an analytical baseline, market regime detection, a FINN model, an API and a local dashboard.

## Main Flow

```text
Raw market/options data
        |
        v
DatasetBuilder / option normalizer
        |
        v
Canonical option training frame
        |
        +--> Black-Scholes-Merton baseline
        |
        +--> HMM regime table
        |
        v
FINN training / scoring
        |
        v
Scored options CSV
        |
        +--> FastAPI scoring service
        |
        +--> Prediction dashboard
```

## Key Modules

- `src/contracts.py`: shared typed contracts for dataset rows, baseline inputs, FINN inputs and system outputs.
- `src/baseline/`: Black-Scholes-Merton pricing and Greeks.
- `src/data/`: source normalization, option dataset building and FINN frame assembly.
- `src/models/hmm/`: regime detectors and persisted regime table support.
- `src/models/finn/`: FINN model, PDE/no-arbitrage losses and training/evaluation utilities.
- `src/pipeline/`: orchestration for scoring rows and datasets.
- `src/api/`: FastAPI app exposing health, metadata and scoring endpoints.
- `src/visualization/`: local dashboard generation.
- `scripts/`: reproducible command-line entrypoints.
- `tests/`: regression tests for contracts, data, models, pipeline, API and dashboard.

## Current Production-Like Path

1. Build or refresh `Data/processed/spy_regime_features.csv`.
2. Build or refresh a raw option chain under `Data/raw/options/`.
3. Train FINN with a train/validation/test split.
4. Score the option dataset into `Data/processed/scored_options.csv`.
5. Run the API or generate `reports/prediction_dashboard.html`.

Generated outputs are local artifacts and should not be committed by default.

