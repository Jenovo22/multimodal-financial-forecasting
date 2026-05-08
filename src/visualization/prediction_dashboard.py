"""Build a local HTML dashboard for option predictions versus observed prices."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_SCORED_COLUMNS = (
    "timestamp",
    "underlying_symbol",
    "option_symbol",
    "market_price",
    "bs_price",
    "fair_value",
    "edge",
    "delta",
    "gamma",
    "regime_label",
    "signal",
)

OPTION_METADATA_COLUMNS = (
    "timestamp",
    "underlying_symbol",
    "option_symbol",
    "option_type",
    "K",
    "T",
    "S",
    "expiration_date",
    "lastPrice",
    "mid_price",
    "bid",
    "ask",
    "volume",
    "open_interest",
    "implied_volatility",
    "spread",
    "relative_spread",
)


@dataclass(slots=True, frozen=True)
class DashboardConfig:
    scored_path: Path = Path("Data/processed/scored_options.csv")
    option_dataset_path: Path = Path("Data/processed/option_training_dataset.csv")
    market_history_path: Path | None = Path("Data/final/master_dataset_2010_present_long.csv")
    output_path: Path = Path("reports/prediction_dashboard.html")
    title: str = "Proyecto TAM - Prediction Dashboard"
    history_days: int = 252


def build_dashboard_frame(
    scored_source: pd.DataFrame | str | Path,
    option_dataset_source: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """Join scored outputs with option metadata used by the chart."""

    scored = _load_frame(scored_source)
    option_dataset = _load_frame(option_dataset_source)
    _validate_columns(scored, REQUIRED_SCORED_COLUMNS, name="scored_source")

    join_keys = [
        column
        for column in ("timestamp", "underlying_symbol", "option_symbol")
        if column in scored.columns and column in option_dataset.columns
    ]
    if "option_symbol" not in join_keys:
        raise ValueError("Both sources must contain option_symbol.")

    metadata_columns = [
        column for column in OPTION_METADATA_COLUMNS if column in option_dataset.columns
    ]
    option_metadata = option_dataset[metadata_columns].drop_duplicates(subset=join_keys)

    merged = scored.merge(
        option_metadata,
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    if "K" not in merged.columns or merged["K"].isna().any():
        raise ValueError("Option metadata must provide strike column K for charting.")
    if "expiration_date" not in merged.columns or merged["expiration_date"].isna().any():
        raise ValueError("Option metadata must provide expiration_date for charting.")

    if "lastPrice" in merged.columns:
        merged = merged.rename(columns={"lastPrice": "last_price"})

    numeric_columns = [
        "market_price",
        "bs_price",
        "fair_value",
        "edge",
        "delta",
        "gamma",
        "K",
        "T",
        "S",
        "last_price",
        "mid_price",
        "bid",
        "ask",
        "volume",
        "open_interest",
        "implied_volatility",
        "spread",
        "relative_spread",
    ]
    for column in numeric_columns:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

    return add_decision_metrics(
        merged.sort_values(["expiration_date", "K", "option_symbol"]).reset_index(drop=True)
    )


def add_decision_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add intuitive decision-support columns to scored option rows."""

    enriched = frame.copy()
    safe_market = enriched["market_price"].clip(lower=0.01)
    if "relative_spread" not in enriched.columns:
        enriched["relative_spread"] = pd.NA
    if "volume" not in enriched.columns:
        enriched["volume"] = pd.NA
    if "open_interest" not in enriched.columns:
        enriched["open_interest"] = pd.NA
    enriched["predicted_price"] = enriched["fair_value"]
    enriched["expected_profit"] = enriched["edge"]
    enriched["expected_return_pct"] = (enriched["edge"] / safe_market) * 100.0
    enriched["dte_days"] = _days_to_expiration(enriched)

    edge_strength = (enriched["edge"].abs() / safe_market / 0.03).clip(upper=1.0)
    spread_score = _spread_score(enriched)
    volume_score = _log_score(enriched, "volume", scale=500.0)
    open_interest_score = _log_score(enriched, "open_interest", scale=1000.0)
    liquidity_score = (volume_score + open_interest_score) / 2.0
    agreement_score = (
        1.0
        - ((enriched["fair_value"] - enriched["bs_price"]).abs() / safe_market / 0.25).clip(
            upper=1.0
        )
    )
    regime_confidence = _regime_confidence(enriched)

    confidence = (
        0.30 * edge_strength
        + 0.20 * spread_score
        + 0.15 * liquidity_score
        + 0.20 * agreement_score
        + 0.15 * regime_confidence
    ).clip(lower=0.0, upper=1.0)
    enriched["confidence_score"] = (confidence * 100.0).round(1)
    enriched["confidence_label"] = pd.cut(
        enriched["confidence_score"],
        bins=[-0.1, 45.0, 70.0, 100.0],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    enriched["opportunity_score"] = (
        enriched["expected_return_pct"].abs() * confidence
    ).round(4)
    enriched["recommendation"] = [
        _recommendation(signal, confidence_score)
        for signal, confidence_score in zip(enriched["signal"], enriched["confidence_score"])
    ]
    enriched["decision_reason"] = [
        _decision_reason(row)
        for row in enriched[
            [
                "edge",
                "expected_return_pct",
                "confidence_label",
                "relative_spread",
                "volume",
                "open_interest",
            ]
        ].to_dict(orient="records")
    ]
    return enriched.sort_values(["expiration_date", "K", "option_symbol"]).reset_index(drop=True)


def _days_to_expiration(frame: pd.DataFrame) -> pd.Series:
    if "T" in frame.columns:
        dte = pd.to_numeric(frame["T"], errors="coerce") * 365.0
    else:
        dte = pd.Series(float("nan"), index=frame.index, dtype="float64")
    missing = dte.isna()
    if missing.any() and {"timestamp", "expiration_date"}.issubset(frame.columns):
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        expirations = pd.to_datetime(frame["expiration_date"], errors="coerce", utc=True)
        calendar_dte = (expirations - timestamps).dt.total_seconds() / 86400.0
        dte = dte.where(dte.notna(), calendar_dte)
    return dte


def _spread_score(frame: pd.DataFrame) -> pd.Series:
    if "relative_spread" in frame.columns:
        relative_spread = pd.to_numeric(frame["relative_spread"], errors="coerce")
    elif {"bid", "ask", "market_price"}.issubset(frame.columns):
        bid = pd.to_numeric(frame["bid"], errors="coerce")
        ask = pd.to_numeric(frame["ask"], errors="coerce")
        market_price = pd.to_numeric(frame["market_price"], errors="coerce").clip(lower=0.01)
        relative_spread = (ask - bid) / market_price
    else:
        return pd.Series(0.5, index=frame.index)
    return (1.0 - (relative_spread.fillna(0.25) / 0.25).clip(upper=1.0)).clip(lower=0.0)


def _log_score(frame: pd.DataFrame, column: str, *, scale: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.35, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    denominator = math.log1p(scale)
    return values.map(lambda value: min(math.log1p(float(value)) / denominator, 1.0))


def _regime_confidence(frame: pd.DataFrame) -> pd.Series:
    probability_columns = [
        column
        for column in frame.columns
        if column.startswith("regime_probability_")
    ]
    if not probability_columns:
        return pd.Series(0.5, index=frame.index)
    probabilities = frame[probability_columns].apply(pd.to_numeric, errors="coerce")
    return probabilities.max(axis=1).fillna(0.5).clip(lower=0.0, upper=1.0)


def _recommendation(signal: object, confidence_score: float) -> str:
    normalized = str(signal).upper()
    if confidence_score < 45.0:
        return "WATCH"
    if normalized == "BUY":
        return "BUY"
    if normalized == "SELL":
        return "SELL / AVOID LONG"
    return "HOLD"


def _decision_reason(row: dict[str, object]) -> str:
    edge = _float_value(row.get("edge"))
    expected_return = _float_value(row.get("expected_return_pct"))
    relative_spread = _float_value(row.get("relative_spread"))
    volume = _float_value(row.get("volume"))
    open_interest = _float_value(row.get("open_interest"))
    confidence = str(row.get("confidence_label", "LOW"))
    liquidity = "liquid" if volume >= 50 or open_interest >= 500 else "thin"
    spread = "tight spread" if relative_spread <= 0.08 else "wide spread"
    return (
        f"{confidence} confidence; edge {edge:+.4f} "
        f"({expected_return:+.2f}%); {spread}; {liquidity} book"
    )


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def build_market_history_frame(
    market_history_source: pd.DataFrame | str | Path | None,
    *,
    symbol: str = "SPY",
    history_days: int = 252,
) -> pd.DataFrame:
    """Build a compact underlying-price history used by the dashboard."""

    if market_history_source is None:
        return pd.DataFrame()
    path = Path(market_history_source) if not isinstance(market_history_source, pd.DataFrame) else None
    if path is not None and not path.exists():
        return pd.DataFrame()

    frame = _load_frame(market_history_source)
    if frame.empty:
        return pd.DataFrame()
    date_column = "date" if "date" in frame.columns else "timestamp"
    symbol_column = "asset" if "asset" in frame.columns else "underlying_symbol"
    price_column = "price_used" if "price_used" in frame.columns else "S"
    required = [date_column, symbol_column, price_column]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()

    filtered = frame.loc[frame[symbol_column].astype(str).str.upper() == symbol.upper()].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered["date"] = pd.to_datetime(filtered[date_column], errors="coerce", utc=True)
    filtered["price"] = pd.to_numeric(filtered[price_column], errors="coerce")
    optional_columns = {
        "return_1d": "return_1d",
        "realized_volatility": "volatility_20d_ann",
        "vix": "vix_fred",
        "risk_free_rate": "dgs2",
    }
    output = filtered[["date", "price"]].copy()
    for output_column, source_column in optional_columns.items():
        if source_column in filtered.columns:
            output[output_column] = pd.to_numeric(filtered[source_column], errors="coerce")
    output = output.dropna(subset=["date", "price"]).sort_values("date")
    if history_days > 0:
        output = output.tail(history_days)
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    return output.reset_index(drop=True)


def save_prediction_dashboard(config: DashboardConfig | None = None) -> Path:
    """Build and save the dashboard HTML file."""

    resolved = config or DashboardConfig()
    frame = build_dashboard_frame(
        resolved.scored_path,
        resolved.option_dataset_path,
    )
    history = build_market_history_frame(
        resolved.market_history_path,
        history_days=resolved.history_days,
    )
    html = render_prediction_dashboard_html(
        frame,
        title=resolved.title,
        history_frame=history,
    )
    resolved.output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.output_path.write_text(html, encoding="utf-8")
    return resolved.output_path


def render_prediction_dashboard_html(
    frame: pd.DataFrame,
    *,
    title: str = "Proyecto TAM - Prediction Dashboard",
    history_frame: pd.DataFrame | None = None,
) -> str:
    """Render a self-contained HTML dashboard with an SVG chart."""

    if "confidence_label" not in frame.columns:
        frame = add_decision_metrics(frame)

    records = _json_records(frame)
    payload = json.dumps(records, ensure_ascii=True)
    history_payload = json.dumps(
        _json_records(history_frame if history_frame is not None else pd.DataFrame()),
        ensure_ascii=True,
    )
    expirations = sorted(str(value) for value in frame["expiration_date"].dropna().unique())
    signal_counts = frame["signal"].value_counts().to_dict()
    buy_count = int(signal_counts.get("BUY", 0))
    sell_count = int(signal_counts.get("SELL", 0))
    hold_count = int(signal_counts.get("HOLD", 0))
    edge_mean = float(frame["edge"].mean()) if not frame.empty else 0.0
    market_mean = float(frame["market_price"].mean()) if not frame.empty else 0.0
    fair_mean = float(frame["fair_value"].mean()) if not frame.empty else 0.0
    high_confidence_count = int((frame["confidence_label"] == "HIGH").sum())
    best_row = (
        frame.sort_values("opportunity_score", ascending=False).iloc[0]
        if not frame.empty
        else None
    )
    best_label = (
        f"{best_row['recommendation']} {best_row['option_symbol']}"
        if best_row is not None
        else "N/A"
    )
    strike_values = sorted(float(value) for value in frame["K"].dropna().unique())
    reference_spot = (
        float(frame["S"].dropna().median())
        if "S" in frame.columns and not frame["S"].dropna().empty
        else (strike_values[len(strike_values) // 2] if strike_values else 0.0)
    )
    default_temporal_strike = (
        min(strike_values, key=lambda value: abs(value - reference_spot))
        if strike_values
        else 0.0
    )
    strike_options_html = "".join(
        (
            f'<option value="{strike:.6f}"'
            f'{" selected" if strike == default_temporal_strike else ""}>'
            f'{strike:g}</option>'
        )
        for strike in strike_values
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape_text(title)}</title>
  <style>
    :root {{
      --ink: #17211f;
      --muted: #697a75;
      --paper: #f6f1e8;
      --panel: #fffaf1;
      --line: #dccfbf;
      --market: #1f6f8b;
      --finn: #b84a39;
      --bs: #5b6b31;
      --last: #6b5aa6;
      --buy: #0f7a4d;
      --sell: #b23b3b;
      --watch: #a36d1f;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #fff7d8, var(--paper) 42%, #ece1d2);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(30px, 5vw, 52px);
      letter-spacing: -0.04em;
    }}
    .subtitle {{
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 16px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card, .panel {{
      background: color-mix(in srgb, var(--panel) 88%, white);
      border: 1px solid var(--line);
      box-shadow: 0 18px 45px rgba(57, 43, 24, 0.09);
      border-radius: 18px;
    }}
    .card {{
      padding: 16px;
    }}
    .card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .card strong {{
      display: block;
      margin-top: 7px;
      font-size: 25px;
    }}
    .panel {{
      padding: 18px;
      margin-top: 14px;
    }}
    .decision-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(260px, 0.8fr);
      gap: 14px;
      align-items: stretch;
    }}
    @media (max-width: 860px) {{
      .decision-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      align-items: center;
      margin-bottom: 12px;
    }}
    label {{
      color: var(--muted);
      font-size: 14px;
    }}
    select, input[type="checkbox"] {{
      accent-color: var(--finn);
    }}
    select {{
      margin-left: 6px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 7px 10px;
      background: white;
      color: var(--ink);
    }}
    input[type="number"] {{
      margin-left: 6px;
      width: 88px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 7px 10px;
      background: white;
      color: var(--ink);
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 10px 0 2px;
      color: var(--muted);
      font-size: 13px;
    }}
    .dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 5px;
    }}
    svg {{
      width: 100%;
      min-height: 460px;
      background: linear-gradient(180deg, #fffdf8, #fff7eb);
      border-radius: 14px;
      border: 1px solid var(--line);
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .BUY {{
      color: var(--buy);
      font-weight: 700;
    }}
    .SELL {{
      color: var(--sell);
      font-weight: 700;
    }}
    .WATCH, .HOLD {{
      color: var(--watch);
      font-weight: 700;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 9px;
      background: #f0e4d5;
      color: var(--ink);
      font-size: 12px;
      font-weight: 700;
    }}
    .badge.HIGH {{ background: #d7f0df; color: var(--buy); }}
    .badge.MEDIUM {{ background: #f5e8c7; color: #8a6417; }}
    .badge.LOW {{ background: #f2d9d7; color: var(--sell); }}
    .note {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
      margin-top: 10px;
    }}
  </style>
</head>
<body>
<main>
  <h1>{_escape_text(title)}</h1>
  <p class="subtitle">FINN prediction, observed market price, Black-Scholes baseline, and last traded price overlaid by strike.</p>

  <section class="cards">
    <div class="card"><span>Rows</span><strong>{len(frame):,}</strong></div>
    <div class="card"><span>BUY</span><strong>{buy_count:,}</strong></div>
    <div class="card"><span>SELL</span><strong>{sell_count:,}</strong></div>
    <div class="card"><span>HOLD</span><strong>{hold_count:,}</strong></div>
    <div class="card"><span>High confidence</span><strong>{high_confidence_count:,}</strong></div>
    <div class="card"><span>Best opportunity</span><strong>{_escape_text(best_label)}</strong></div>
    <div class="card"><span>Mean edge</span><strong>{edge_mean:.4f}</strong></div>
    <div class="card"><span>Market / FINN avg</span><strong>{market_mean:.2f} / {fair_mean:.2f}</strong></div>
  </section>

  <section class="panel">
    <h2>Historical Context</h2>
    <p class="note">Underlying SPY history gives context for the quote date used by the option predictions. This is not a backtest yet; it is a visual anchor for where the market was when the model scored the chain.</p>
    <svg id="historyChart" role="img" aria-label="SPY historical price chart"></svg>
  </section>

  <section class="panel">
    <h2>Decision View</h2>
    <p class="note">The predicted price shown here is FINN fair value: a model-implied target price for the option, not a guaranteed future market price. Recommendations are research signals ranked by edge, liquidity, spread, model agreement and regime confidence.</p>
    <div class="decision-grid">
      <svg id="decisionChart" role="img" aria-label="Decision map"></svg>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Option</th><th>Action</th><th>Conf.</th><th>Target</th><th>Mkt</th><th>Exp. %</th><th>Score</th>
            </tr>
          </thead>
          <tbody id="opportunityTable"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section class="panel">
    <h2>Option Simulator</h2>
    <p class="note">Estimate the model-implied return if price moves from current market price to FINN predicted fair value. Contract multiplier is 100. This is a research calculator, not execution advice.</p>
    <div class="controls">
      <label>Option
        <select id="simOptionSelect"></select>
      </label>
      <label>Action
        <select id="simActionSelect">
          <option value="AUTO">Use recommendation</option>
          <option value="BUY">Buy option</option>
          <option value="SELL">Sell / avoid long</option>
          <option value="WATCH">Watch only</option>
        </select>
      </label>
      <label>Contracts
        <input id="simContracts" type="number" min="1" step="1" value="1">
      </label>
      <label>Cost / contract
        <input id="simCost" type="number" min="0" step="0.01" value="0">
      </label>
    </div>
    <div class="cards" id="simulatorCards"></div>
    <p class="note" id="simulatorReason"></p>
  </section>

  <section class="panel">
    <h2>Temporal Structure</h2>
    <p class="note">Select a strike to see the same contract family across expirations. The horizontal axis is days to expiration, so this view makes the time scale explicit.</p>
    <div class="controls">
      <label>Strike
        <select id="temporalStrikeSelect">
          {strike_options_html}
        </select>
      </label>
      <span class="note" id="temporalSummary"></span>
    </div>
    <svg id="temporalChart" role="img" aria-label="Temporal prediction chart"></svg>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Expiration</th><th>DTE</th><th>Market</th><th>Predicted</th><th>BSM</th><th>Last</th><th>Exp. %</th><th>Action</th><th>Conf.</th>
          </tr>
        </thead>
        <tbody id="temporalTable"></tbody>
      </table>
    </div>
  </section>

  <section class="panel">
    <div class="controls">
      <label>Expiration
        <select id="expirationSelect">
          {"".join(f'<option value="{_escape_text(value)}">{_escape_text(value)}</option>' for value in expirations)}
        </select>
      </label>
      <label>Signal
        <select id="signalSelect">
          <option value="ALL">ALL</option>
          <option value="BUY">BUY</option>
          <option value="SELL">SELL</option>
          <option value="HOLD">HOLD</option>
        </select>
      </label>
      <label><input type="checkbox" id="showMarket" checked> Market price</label>
      <label><input type="checkbox" id="showFinn" checked> FINN fair value</label>
      <label><input type="checkbox" id="showBs" checked> BSM price</label>
      <label><input type="checkbox" id="showLast"> Last traded price</label>
    </div>
    <div class="legend">
      <span><i class="dot" style="background:var(--market)"></i>Market price</span>
      <span><i class="dot" style="background:var(--finn)"></i>FINN fair value</span>
      <span><i class="dot" style="background:var(--bs)"></i>BSM price</span>
      <span><i class="dot" style="background:var(--last)"></i>Last traded price</span>
    </div>
    <svg id="chart" role="img" aria-label="Prediction chart"></svg>
    <p class="note">Use the overlay to see whether FINN is above or below the current market curve. If FINN is above market, the model sees positive edge; if it is below market, the model sees negative edge.</p>
  </section>

  <section class="panel">
    <h2>Visible Rows</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Option</th><th>Strike</th><th>Action</th><th>Conf.</th><th>Market</th><th>Predicted</th><th>Exp. %</th><th>BSM</th><th>Edge</th><th>Delta</th><th>Gamma</th>
          </tr>
        </thead>
        <tbody id="rowsTable"></tbody>
      </table>
    </div>
  </section>
</main>

<script>
const predictionRows = {payload};
const marketHistoryRows = {history_payload};
const colors = {{
  market_price: "#1f6f8b",
  fair_value: "#b84a39",
  bs_price: "#5b6b31",
  last_price: "#6b5aa6"
}};
const labels = {{
  market_price: "Market",
  fair_value: "FINN",
  bs_price: "BSM",
  last_price: "Last"
}};

const expirationSelect = document.getElementById("expirationSelect");
const signalSelect = document.getElementById("signalSelect");
const historyChart = document.getElementById("historyChart");
const chart = document.getElementById("chart");
const decisionChart = document.getElementById("decisionChart");
const temporalChart = document.getElementById("temporalChart");
const table = document.getElementById("rowsTable");
const opportunityTable = document.getElementById("opportunityTable");
const temporalTable = document.getElementById("temporalTable");
const temporalStrikeSelect = document.getElementById("temporalStrikeSelect");
const temporalSummary = document.getElementById("temporalSummary");
const simOptionSelect = document.getElementById("simOptionSelect");
const simActionSelect = document.getElementById("simActionSelect");
const simContracts = document.getElementById("simContracts");
const simCost = document.getElementById("simCost");
const simulatorCards = document.getElementById("simulatorCards");
const simulatorReason = document.getElementById("simulatorReason");
for (const id of ["showMarket", "showFinn", "showBs", "showLast", "expirationSelect", "signalSelect", "temporalStrikeSelect", "simOptionSelect", "simActionSelect", "simContracts", "simCost"]) {{
  document.getElementById(id).addEventListener("change", render);
}}
initializeSimulatorOptions();

function selectedRows() {{
  const expiration = expirationSelect.value;
  const signal = signalSelect.value;
  return predictionRows
    .filter(row => row.expiration_date === expiration)
    .filter(row => signal === "ALL" || row.signal === signal)
    .sort((a, b) => a.K - b.K);
}}

function visibleSeries() {{
  const series = [];
  if (document.getElementById("showMarket").checked) series.push("market_price");
  if (document.getElementById("showFinn").checked) series.push("fair_value");
  if (document.getElementById("showBs").checked) series.push("bs_price");
  if (document.getElementById("showLast").checked) series.push("last_price");
  return series;
}}

function render() {{
  const rows = selectedRows();
  const series = visibleSeries();
  renderHistoryChart();
  renderDecisionChart(rows);
  renderOpportunityTable(rows);
  renderTemporalChart(series);
  renderTemporalTable();
  renderSimulator();
  renderChart(rows, series);
  renderTable(rows);
}}

function renderHistoryChart() {{
  const width = 1120;
  const height = 340;
  const margin = {{left: 64, right: 28, top: 24, bottom: 58}};
  historyChart.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  historyChart.innerHTML = "";
  const rows = marketHistoryRows.filter(row => Number.isFinite(Number(row.price)));
  if (rows.length < 2) {{
    historyChart.innerHTML = `<text x="${{width / 2}}" y="${{height / 2}}" text-anchor="middle" fill="#697a75">No historical market data available.</text>`;
    return;
  }}
  const prices = rows.map(row => Number(row.price));
  const yMinRaw = Math.min(...prices);
  const yMaxRaw = Math.max(...prices);
  const yPad = Math.max((yMaxRaw - yMinRaw) * 0.08, 1);
  const yMin = yMinRaw - yPad;
  const yMax = yMaxRaw + yPad;
  const xScale = index => margin.left + (index / Math.max(rows.length - 1, 1)) * (width - margin.left - margin.right);
  const yScale = value => height - margin.bottom - ((value - yMin) / Math.max(yMax - yMin, 1)) * (height - margin.top - margin.bottom);
  for (let i = 0; i <= 4; i++) {{
    const yValue = yMin + ((yMax - yMin) * i / 4);
    const y = yScale(yValue);
    historyChart.insertAdjacentHTML("beforeend", `<line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{y}}" y2="${{y}}" stroke="#e6d9c8"></line>`);
    historyChart.insertAdjacentHTML("beforeend", `<text x="${{margin.left - 10}}" y="${{y + 4}}" text-anchor="end" fill="#697a75" font-size="12">${{fmt(yValue)}}</text>`);
  }}
  const d = rows.map((row, index) => `${{index === 0 ? "M" : "L"}} ${{xScale(index).toFixed(2)}} ${{yScale(Number(row.price)).toFixed(2)}}`).join(" ");
  historyChart.insertAdjacentHTML("beforeend", `<path d="${{d}}" fill="none" stroke="#1f6f8b" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></path>`);
  const labels = [0, Math.floor(rows.length / 2), rows.length - 1];
  for (const index of labels) {{
    const x = xScale(index);
    historyChart.insertAdjacentHTML("beforeend", `<text x="${{x}}" y="${{height - margin.bottom + 24}}" text-anchor="middle" fill="#697a75" font-size="12">${{rows[index].date}}</text>`);
  }}
  historyChart.insertAdjacentHTML("beforeend", `<text x="${{width / 2}}" y="${{height - 14}}" text-anchor="middle" fill="#697a75" font-size="13">Historical date</text>`);
  historyChart.insertAdjacentHTML("beforeend", `<text x="18" y="${{height / 2}}" transform="rotate(-90, 18, ${{height / 2}})" text-anchor="middle" fill="#697a75" font-size="13">SPY price</text>`);
}}

function renderDecisionChart(rows) {{
  const width = 560;
  const height = 460;
  const margin = {{left: 62, right: 28, top: 24, bottom: 58}};
  decisionChart.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  decisionChart.innerHTML = "";
  if (rows.length === 0) {{
    decisionChart.innerHTML = `<text x="${{width / 2}}" y="${{height / 2}}" text-anchor="middle" fill="#697a75">No rows for this filter.</text>`;
    return;
  }}
  const xs = rows.map(row => Number(row.K)).filter(Number.isFinite);
  const ys = rows.map(row => Number(row.expected_return_pct)).filter(Number.isFinite);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yAbs = Math.max(...ys.map(value => Math.abs(value)), 1);
  const yMin = -yAbs * 1.12;
  const yMax = yAbs * 1.12;
  const xScale = value => margin.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * (width - margin.left - margin.right);
  const yScale = value => height - margin.bottom - ((value - yMin) / Math.max(yMax - yMin, 1)) * (height - margin.top - margin.bottom);
  drawDecisionGrid(width, height, margin, xMin, xMax, yMin, yMax, xScale, yScale);
  for (const row of rows) {{
    const x = xScale(Number(row.K));
    const y = yScale(Number(row.expected_return_pct));
    const confidence = Number(row.confidence_score);
    const radius = 4 + Math.max(0, Math.min(confidence, 100)) / 12;
    const fill = row.recommendation === "BUY" ? "#0f7a4d" : row.recommendation.startsWith("SELL") ? "#b23b3b" : "#a36d1f";
    decisionChart.insertAdjacentHTML("beforeend", `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="${{radius.toFixed(1)}}" fill="${{fill}}" opacity="0.78" stroke="#fffaf1" stroke-width="1.5"><title>${{row.option_symbol}} | ${{row.recommendation}} | target ${{fmt(row.predicted_price)}} | expected ${{fmt(row.expected_return_pct)}}% | confidence ${{fmt(confidence)}}%</title></circle>`);
  }}
}}

function drawDecisionGrid(width, height, margin, xMin, xMax, yMin, yMax, xScale, yScale) {{
  const gridColor = "#e6d9c8";
  const axisColor = "#9f907f";
  const zeroY = yScale(0);
  for (let i = 0; i <= 4; i++) {{
    const yValue = yMin + ((yMax - yMin) * i / 4);
    const y = yScale(yValue);
    decisionChart.insertAdjacentHTML("beforeend", `<line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{y}}" y2="${{y}}" stroke="${{gridColor}}"></line>`);
    decisionChart.insertAdjacentHTML("beforeend", `<text x="${{margin.left - 10}}" y="${{y + 4}}" text-anchor="end" fill="#697a75" font-size="12">${{fmt(yValue)}}%</text>`);
  }}
  for (let i = 0; i <= 4; i++) {{
    const xValue = xMin + ((xMax - xMin) * i / 4);
    const x = xScale(xValue);
    decisionChart.insertAdjacentHTML("beforeend", `<line x1="${{x}}" x2="${{x}}" y1="${{margin.top}}" y2="${{height - margin.bottom}}" stroke="${{gridColor}}" opacity="0.7"></line>`);
    decisionChart.insertAdjacentHTML("beforeend", `<text x="${{x}}" y="${{height - margin.bottom + 24}}" text-anchor="middle" fill="#697a75" font-size="12">${{xValue.toFixed(0)}}</text>`);
  }}
  decisionChart.insertAdjacentHTML("beforeend", `<line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{zeroY}}" y2="${{zeroY}}" stroke="${{axisColor}}" stroke-width="2"></line>`);
  decisionChart.insertAdjacentHTML("beforeend", `<text x="${{width / 2}}" y="${{height - 14}}" text-anchor="middle" fill="#697a75" font-size="13">Strike</text>`);
  decisionChart.insertAdjacentHTML("beforeend", `<text x="18" y="${{height / 2}}" transform="rotate(-90, 18, ${{height / 2}})" text-anchor="middle" fill="#697a75" font-size="13">Expected return %</text>`);
}}

function temporalRows() {{
  const selectedStrike = Number(temporalStrikeSelect.value);
  return predictionRows
    .filter(row => Math.abs(Number(row.K) - selectedStrike) < 1e-6)
    .sort((a, b) => Number(a.dte_days) - Number(b.dte_days));
}}

function renderTemporalChart(series) {{
  const rows = temporalRows();
  const width = 1120;
  const height = 430;
  const margin = {{left: 64, right: 28, top: 24, bottom: 62}};
  temporalChart.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  temporalChart.innerHTML = "";
  if (rows.length === 0 || series.length === 0) {{
    temporalChart.innerHTML = `<text x="${{width / 2}}" y="${{height / 2}}" text-anchor="middle" fill="#697a75">No temporal rows for this strike.</text>`;
    temporalSummary.textContent = "";
    return;
  }}
  temporalSummary.textContent = `${{rows.length}} expirations | DTE ${{fmt(rows[0].dte_days)}} to ${{fmt(rows[rows.length - 1].dte_days)}} days`;
  const xs = rows.map(row => Number(row.dte_days)).filter(Number.isFinite);
  const ys = [];
  for (const key of series) {{
    for (const row of rows) {{
      const value = Number(row[key]);
      if (Number.isFinite(value)) ys.push(value);
    }}
  }}
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMinRaw = Math.min(...ys);
  const yMaxRaw = Math.max(...ys);
  const yPad = Math.max((yMaxRaw - yMinRaw) * 0.08, 0.25);
  const yMin = Math.max(0, yMinRaw - yPad);
  const yMax = yMaxRaw + yPad;
  const xScale = value => margin.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * (width - margin.left - margin.right);
  const yScale = value => height - margin.bottom - ((value - yMin) / Math.max(yMax - yMin, 1)) * (height - margin.top - margin.bottom);
  drawTemporalGrid(width, height, margin, xMin, xMax, yMin, yMax, xScale, yScale);
  for (const key of series) {{
    const points = rows
      .map(row => [xScale(Number(row.dte_days)), yScale(Number(row[key])), row])
      .filter(point => Number.isFinite(point[0]) && Number.isFinite(point[1]));
    if (!points.length) continue;
    const d = points.map((point, index) => `${{index === 0 ? "M" : "L"}} ${{point[0].toFixed(2)}} ${{point[1].toFixed(2)}}`).join(" ");
    temporalChart.insertAdjacentHTML("beforeend", `<path d="${{d}}" fill="none" stroke="${{colors[key]}}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></path>`);
    for (const [x, y, row] of points) {{
      temporalChart.insertAdjacentHTML("beforeend", `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="4" fill="${{colors[key]}}"><title>${{row.expiration_date}} | DTE ${{fmt(row.dte_days)}} | ${{labels[key]}} ${{fmt(row[key])}}</title></circle>`);
    }}
  }}
}}

function drawTemporalGrid(width, height, margin, xMin, xMax, yMin, yMax, xScale, yScale) {{
  const axisColor = "#9f907f";
  const gridColor = "#e6d9c8";
  for (let i = 0; i <= 5; i++) {{
    const yValue = yMin + ((yMax - yMin) * i / 5);
    const y = yScale(yValue);
    temporalChart.insertAdjacentHTML("beforeend", `<line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{y}}" y2="${{y}}" stroke="${{gridColor}}"></line>`);
    temporalChart.insertAdjacentHTML("beforeend", `<text x="${{margin.left - 10}}" y="${{y + 4}}" text-anchor="end" fill="#697a75" font-size="12">${{fmt(yValue)}}</text>`);
  }}
  for (let i = 0; i <= 5; i++) {{
    const xValue = xMin + ((xMax - xMin) * i / 5);
    const x = xScale(xValue);
    temporalChart.insertAdjacentHTML("beforeend", `<line x1="${{x}}" x2="${{x}}" y1="${{margin.top}}" y2="${{height - margin.bottom}}" stroke="${{gridColor}}" opacity="0.7"></line>`);
    temporalChart.insertAdjacentHTML("beforeend", `<text x="${{x}}" y="${{height - margin.bottom + 24}}" text-anchor="middle" fill="#697a75" font-size="12">${{fmt(xValue)}}</text>`);
  }}
  temporalChart.insertAdjacentHTML("beforeend", `<line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{height - margin.bottom}}" y2="${{height - margin.bottom}}" stroke="${{axisColor}}"></line>`);
  temporalChart.insertAdjacentHTML("beforeend", `<line x1="${{margin.left}}" x2="${{margin.left}}" y1="${{margin.top}}" y2="${{height - margin.bottom}}" stroke="${{axisColor}}"></line>`);
  temporalChart.insertAdjacentHTML("beforeend", `<text x="${{width / 2}}" y="${{height - 14}}" text-anchor="middle" fill="#697a75" font-size="13">Days to expiration</text>`);
  temporalChart.insertAdjacentHTML("beforeend", `<text x="18" y="${{height / 2}}" transform="rotate(-90, 18, ${{height / 2}})" text-anchor="middle" fill="#697a75" font-size="13">Option price</text>`);
}}

function renderChart(rows, series) {{
  const width = 1120;
  const height = 460;
  const margin = {{left: 64, right: 28, top: 24, bottom: 58}};
  chart.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  chart.innerHTML = "";
  if (rows.length === 0 || series.length === 0) {{
    chart.innerHTML = `<text x="${{width / 2}}" y="${{height / 2}}" text-anchor="middle" fill="#697a75">No rows for this filter.</text>`;
    return;
  }}

  const xs = rows.map(row => Number(row.K)).filter(Number.isFinite);
  const ys = [];
  for (const key of series) {{
    for (const row of rows) {{
      const value = Number(row[key]);
      if (Number.isFinite(value)) ys.push(value);
    }}
  }}
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMinRaw = Math.min(...ys);
  const yMaxRaw = Math.max(...ys);
  const yPad = Math.max((yMaxRaw - yMinRaw) * 0.08, 0.5);
  const yMin = Math.max(0, yMinRaw - yPad);
  const yMax = yMaxRaw + yPad;
  const xScale = value => margin.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * (width - margin.left - margin.right);
  const yScale = value => height - margin.bottom - ((value - yMin) / Math.max(yMax - yMin, 1)) * (height - margin.top - margin.bottom);

  drawGrid(width, height, margin, xMin, xMax, yMin, yMax, xScale, yScale);

  for (const key of series) {{
    const points = rows
      .map(row => [xScale(Number(row.K)), yScale(Number(row[key])), row])
      .filter(point => Number.isFinite(point[0]) && Number.isFinite(point[1]));
    if (!points.length) continue;
    const d = points.map((point, index) => `${{index === 0 ? "M" : "L"}} ${{point[0].toFixed(2)}} ${{point[1].toFixed(2)}}`).join(" ");
    chart.insertAdjacentHTML("beforeend", `<path d="${{d}}" fill="none" stroke="${{colors[key]}}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></path>`);
    for (const [x, y, row] of points) {{
      const signalClass = row.signal === "BUY" ? "#0f7a4d" : row.signal === "SELL" ? "#b23b3b" : colors[key];
      chart.insertAdjacentHTML("beforeend", `<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="3.5" fill="${{colors[key]}}" stroke="${{signalClass}}" stroke-width="1"><title>${{row.option_symbol}} | ${{labels[key]}} ${{fmt(row[key])}}</title></circle>`);
    }}
  }}
}}

function drawGrid(width, height, margin, xMin, xMax, yMin, yMax, xScale, yScale) {{
  const axisColor = "#9f907f";
  const gridColor = "#e6d9c8";
  for (let i = 0; i <= 5; i++) {{
    const yValue = yMin + ((yMax - yMin) * i / 5);
    const y = yScale(yValue);
    chart.insertAdjacentHTML("beforeend", `<line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{y}}" y2="${{y}}" stroke="${{gridColor}}"></line>`);
    chart.insertAdjacentHTML("beforeend", `<text x="${{margin.left - 10}}" y="${{y + 4}}" text-anchor="end" fill="#697a75" font-size="12">${{fmt(yValue)}}</text>`);
  }}
  for (let i = 0; i <= 6; i++) {{
    const xValue = xMin + ((xMax - xMin) * i / 6);
    const x = xScale(xValue);
    chart.insertAdjacentHTML("beforeend", `<line x1="${{x}}" x2="${{x}}" y1="${{margin.top}}" y2="${{height - margin.bottom}}" stroke="${{gridColor}}" opacity="0.7"></line>`);
    chart.insertAdjacentHTML("beforeend", `<text x="${{x}}" y="${{height - margin.bottom + 24}}" text-anchor="middle" fill="#697a75" font-size="12">${{xValue.toFixed(0)}}</text>`);
  }}
  chart.insertAdjacentHTML("beforeend", `<line x1="${{margin.left}}" x2="${{width - margin.right}}" y1="${{height - margin.bottom}}" y2="${{height - margin.bottom}}" stroke="${{axisColor}}"></line>`);
  chart.insertAdjacentHTML("beforeend", `<line x1="${{margin.left}}" x2="${{margin.left}}" y1="${{margin.top}}" y2="${{height - margin.bottom}}" stroke="${{axisColor}}"></line>`);
  chart.insertAdjacentHTML("beforeend", `<text x="${{width / 2}}" y="${{height - 14}}" text-anchor="middle" fill="#697a75" font-size="13">Strike</text>`);
  chart.insertAdjacentHTML("beforeend", `<text x="18" y="${{height / 2}}" transform="rotate(-90, 18, ${{height / 2}})" text-anchor="middle" fill="#697a75" font-size="13">Option price</text>`);
}}

function renderTable(rows) {{
  table.innerHTML = rows.slice(0, 80).map(row => `
    <tr>
      <td>${{escapeHtml(row.option_symbol)}}</td>
      <td>${{fmt(row.K)}}</td>
      <td class="${{row.signal}}">${{escapeHtml(row.recommendation)}}</td>
      <td><span class="badge ${{row.confidence_label}}">${{escapeHtml(row.confidence_label)}} ${{fmt(row.confidence_score)}}%</span></td>
      <td>${{fmt(row.market_price)}}</td>
      <td>${{fmt(row.predicted_price)}}</td>
      <td>${{fmt(row.expected_return_pct)}}%</td>
      <td>${{fmt(row.bs_price)}}</td>
      <td>${{fmt(row.edge)}}</td>
      <td>${{fmt(row.delta)}}</td>
      <td>${{fmt(row.gamma)}}</td>
    </tr>
  `).join("");
}}

function renderOpportunityTable(rows) {{
  const ranked = [...rows].sort((a, b) => Number(b.opportunity_score) - Number(a.opportunity_score)).slice(0, 12);
  opportunityTable.innerHTML = ranked.map(row => `
    <tr>
      <td><span title="${{escapeHtml(row.decision_reason)}}">${{escapeHtml(row.option_symbol)}}</span></td>
      <td class="${{row.signal}}">${{escapeHtml(row.recommendation)}}</td>
      <td><span class="badge ${{row.confidence_label}}">${{fmt(row.confidence_score)}}%</span></td>
      <td>${{fmt(row.predicted_price)}}</td>
      <td>${{fmt(row.market_price)}}</td>
      <td>${{fmt(row.expected_return_pct)}}%</td>
      <td>${{fmt(row.opportunity_score)}}</td>
    </tr>
  `).join("");
}}

function renderTemporalTable() {{
  const rows = temporalRows();
  temporalTable.innerHTML = rows.map(row => `
    <tr>
      <td>${{escapeHtml(row.expiration_date)}}</td>
      <td>${{fmt(row.dte_days)}}</td>
      <td>${{fmt(row.market_price)}}</td>
      <td>${{fmt(row.predicted_price)}}</td>
      <td>${{fmt(row.bs_price)}}</td>
      <td>${{fmt(row.last_price)}}</td>
      <td>${{fmt(row.expected_return_pct)}}%</td>
      <td class="${{row.signal}}">${{escapeHtml(row.recommendation)}}</td>
      <td><span class="badge ${{row.confidence_label}}">${{fmt(row.confidence_score)}}%</span></td>
    </tr>
  `).join("");
}}

function initializeSimulatorOptions() {{
  const ranked = [...predictionRows].sort((a, b) => Number(b.opportunity_score) - Number(a.opportunity_score));
  simOptionSelect.innerHTML = ranked.map(row => `
    <option value="${{escapeHtml(row.option_symbol)}}">${{escapeHtml(row.option_symbol)}} | ${{escapeHtml(row.recommendation)}} | ${{fmt(row.expected_return_pct)}}%</option>
  `).join("");
}}

function selectedSimulatorRow() {{
  return predictionRows.find(row => row.option_symbol === simOptionSelect.value) || predictionRows[0];
}}

function renderSimulator() {{
  const row = selectedSimulatorRow();
  if (!row) {{
    simulatorCards.innerHTML = "";
    simulatorReason.textContent = "No option selected.";
    return;
  }}
  const contracts = Math.max(1, Number(simContracts.value) || 1);
  const costPerContract = Math.max(0, Number(simCost.value) || 0);
  const action = resolveSimulatorAction(row);
  const multiplier = 100;
  const market = Number(row.market_price);
  const target = Number(row.predicted_price);
  const edge = target - market;
  const premiumCapital = market * multiplier * contracts;
  const totalCosts = costPerContract * contracts;
  let expectedPnl = 0;
  let capitalBase = Math.max(premiumCapital + totalCosts, 0.01);
  let riskText = "Premium paid is the simplified capital at risk.";
  if (action === "BUY") {{
    expectedPnl = edge * multiplier * contracts - totalCosts;
  }} else if (action === "SELL") {{
    expectedPnl = -edge * multiplier * contracts - totalCosts;
    capitalBase = Math.max(market * multiplier * contracts, 0.01);
    riskText = "Short option risk/margin is not fully modeled; prefer risk-defined spreads.";
  }} else {{
    expectedPnl = 0;
    capitalBase = 1;
    riskText = "Watch only: no trade assumed.";
  }}
  const roi = (expectedPnl / capitalBase) * 100;
  simulatorCards.innerHTML = `
    <div class="card"><span>Recommended action</span><strong>${{escapeHtml(actionLabel(action))}}</strong></div>
    <div class="card"><span>Current market</span><strong>${{fmt(market)}}</strong></div>
    <div class="card"><span>Predicted target</span><strong>${{fmt(target)}}</strong></div>
    <div class="card"><span>Expected P/L</span><strong>${{fmt(expectedPnl)}}</strong></div>
    <div class="card"><span>ROI estimate</span><strong>${{fmt(roi)}}%</strong></div>
    <div class="card"><span>Confidence</span><strong>${{fmt(row.confidence_score)}}%</strong></div>
  `;
  simulatorReason.textContent = `${{row.decision_reason}}. ${{riskText}}`;
}}

function resolveSimulatorAction(row) {{
  const selected = simActionSelect.value;
  if (selected !== "AUTO") return selected;
  if (row.recommendation === "BUY") return "BUY";
  if (String(row.recommendation).startsWith("SELL")) return "SELL";
  return "WATCH";
}}

function actionLabel(action) {{
  if (action === "BUY") return "BUY option";
  if (action === "SELL") return "SELL / avoid long";
  return "WATCH";
}}

function fmt(value) {{
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  return number.toFixed(Math.abs(number) >= 10 ? 2 : 4);
}}

function escapeHtml(value) {{
  return String(value ?? "").replace(/[&<>"']/g, char => ({{
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }}[char]));
}}

render();
</script>
</body>
</html>
"""


def _load_frame(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(Path(source))


def _validate_columns(frame: pd.DataFrame, required: tuple[str, ...], *, name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    cleaned = frame.where(pd.notna(frame), None)
    records: list[dict[str, object]] = []
    for record in cleaned.to_dict(orient="records"):
        records.append({key: _json_safe_value(value) for key, value in record.items()})
    return records


def _json_safe_value(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    return value


def _escape_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
