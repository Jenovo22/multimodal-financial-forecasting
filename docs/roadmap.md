# Roadmap

## Near Term

- Keep the current FINN residual training path stable.
- Add transaction cost, spread and liquidity-aware signal thresholds.
- Accumulate multiple daily option snapshots.
- Move evaluation from expiration split to timestamp split once enough snapshots exist.
- Add dashboard panels for liquidity-adjusted edge and model-vs-BSM test history.

## Medium Term

- Add backtesting over historical option snapshots.
- Calibrate PDE loss on broader data before increasing `--lambda-pde`.
- Add model version metadata to checkpoints and API responses.
- Add reproducible dataset manifests or DVC/cloud storage.
- Add monitoring reports for prediction drift and regime transitions.

## Later

- Expand beyond SPY after the SPY workflow is validated.
- Add multimodal/text signals when the data pipeline is stable.
- Add broker/execution integration only after paper-trading validation.

