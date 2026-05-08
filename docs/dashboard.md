# Dashboard

The dashboard is a static HTML report for inspecting scored option predictions.

## Build

```powershell
.\.venv\Scripts\python.exe scripts\build_prediction_dashboard.py `
  --scored-path Data\processed\scored_options.csv `
  --output reports\prediction_dashboard.html `
  --open
```

## Inputs

The primary input is a scored options CSV with:

- market price
- FINN fair value
- BSM price
- edge
- signal
- regime label and probabilities
- Greeks

The builder can also use the option dataset and market history to enrich the view.

## How To Read It

- Use the temporal view to understand quote date, expiration and time-to-expiration.
- Use the strike curve to compare market, FINN and BSM by strike.
- Use the decision cards to inspect signal, edge and confidence.
- Use the simulator to estimate simple return scenarios.

## Caveats

Very cheap options can show large percentage upside even when the dollar edge is small. Always inspect liquidity, spread, delta and expiration before interpreting a signal.

