# API

The API exposes the current HMM + FINN scoring pipeline through FastAPI.

## Run Locally

```powershell
.\.venv\Scripts\python.exe scripts\run_api.py --finn-checkpoint artifacts\finn_model.pt
```

If port `8000` is already busy:

```powershell
.\.venv\Scripts\python.exe scripts\run_api.py --finn-checkpoint artifacts\finn_model.pt --port 8001
```

## Endpoints

- `GET /health`: runtime readiness, service version and startup errors.
- `GET /metadata`: loaded regime/model configuration.
- `POST /score/row`: score one canonical option row.
- `POST /score/batch`: score multiple canonical option rows.

## Score Row Contract

Required fields:

- `timestamp`
- `underlying_symbol`
- `option_symbol`
- `option_type`
- `S`
- `K`
- `T`
- `r`
- `market_price`
- `implied_volatility`

Optional fields include bid/ask, mid price, volume, open interest, realized volatility, returns, sentiment and dividend yield.

## Operational Notes

The API currently returns research signals: `BUY`, `SELL` or `HOLD`. These are not trade instructions. Production use would require authentication, request logging, model versioning, cost/slippage modeling and monitoring.

