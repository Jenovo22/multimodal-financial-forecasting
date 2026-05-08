from __future__ import annotations

import pandas as pd

from src.visualization import (
    DashboardConfig,
    build_dashboard_frame,
    render_prediction_dashboard_html,
    save_prediction_dashboard,
)


def test_build_dashboard_frame_joins_scored_outputs_with_option_metadata():
    scored = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05T00:00:00Z",
                "underlying_symbol": "SPY",
                "option_symbol": "SPY260529C00658000",
                "market_price": 76.44,
                "bs_price": 72.82,
                "fair_value": 76.29,
                "edge": -0.15,
                "delta": 0.53,
                "gamma": 0.03,
                "regime_label": "transition_uncertainty",
                "signal": "SELL",
            }
        ]
    )
    options = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05T00:00:00Z",
                "underlying_symbol": "SPY",
                "option_symbol": "SPY260529C00658000",
                "option_type": "call",
                "K": 658.0,
                "T": 24.0 / 365.0,
                "S": 723.77,
                "expiration_date": "2026-05-29",
                "lastPrice": 77.42,
                "mid_price": 76.44,
            }
        ]
    )

    frame = build_dashboard_frame(scored, options)

    assert frame.loc[0, "K"] == 658.0
    assert frame.loc[0, "expiration_date"] == "2026-05-29"
    assert frame.loc[0, "last_price"] == 77.42
    assert frame.loc[0, "predicted_price"] == 76.29
    assert "confidence_score" in frame.columns
    assert "recommendation" in frame.columns


def test_render_prediction_dashboard_html_embeds_rows():
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05T00:00:00Z",
                "underlying_symbol": "SPY",
                "option_symbol": "SPY260529C00658000",
                "market_price": 76.44,
                "bs_price": 72.82,
                "fair_value": 76.29,
                "edge": -0.15,
                "delta": 0.53,
                "gamma": 0.03,
                "regime_label": "transition_uncertainty",
                "signal": "SELL",
                "K": 658.0,
                "expiration_date": "2026-05-29",
                "last_price": 77.42,
            }
        ]
    )

    html = render_prediction_dashboard_html(frame)

    assert "predictionRows" in html
    assert "SPY260529C00658000" in html
    assert "FINN fair value" in html
    assert "Market price" in html
    assert "Decision View" in html
    assert "Expected return %" in html


def test_save_prediction_dashboard_writes_html(tmp_path):
    scored_path = tmp_path / "scored.csv"
    options_path = tmp_path / "options.csv"
    output_path = tmp_path / "dashboard.html"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05T00:00:00Z",
                "underlying_symbol": "SPY",
                "option_symbol": "SPY260529C00658000",
                "market_price": 76.44,
                "bs_price": 72.82,
                "fair_value": 76.29,
                "edge": -0.15,
                "delta": 0.53,
                "gamma": 0.03,
                "regime_label": "transition_uncertainty",
                "signal": "SELL",
            }
        ]
    ).to_csv(scored_path, index=False)
    pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05T00:00:00Z",
                "underlying_symbol": "SPY",
                "option_symbol": "SPY260529C00658000",
                "option_type": "call",
                "K": 658.0,
                "T": 24.0 / 365.0,
                "S": 723.77,
                "expiration_date": "2026-05-29",
                "lastPrice": 77.42,
            }
        ]
    ).to_csv(options_path, index=False)

    saved = save_prediction_dashboard(
        DashboardConfig(
            scored_path=scored_path,
            option_dataset_path=options_path,
            output_path=output_path,
        )
    )

    assert saved == output_path
    assert output_path.exists()
    assert "predictionRows" in output_path.read_text(encoding="utf-8")
