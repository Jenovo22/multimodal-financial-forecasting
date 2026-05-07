from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.baseline import price_black_scholes
from src.contracts import BaselineInput
from src.pipeline import (
    OptionScoringConfig,
    build_and_save_option_scores,
    score_option_rows,
    system_outputs_to_frame,
)


def build_market_source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-05-05",
                "asset": "SPY",
                "ticker": "SPY",
                "price_used": 510.25,
                "volume": 123456789.0,
                "return_1d": 0.012,
                "return_5d": 0.035,
                "volatility_20d_ann": 0.185,
                "vix_fred": 22.5,
                "dgs2": 4.1,
            }
        ]
    )


def build_option_source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05",
                "underlying_symbol": "SPY",
                "option_symbol": "SPY_20260605_520_C",
                "option_type": "call",
                "K": 520.0,
                "T": 30.0 / 365.0,
                "market_price": 8.35,
                "bid": 8.30,
                "ask": 8.40,
                "mid_price": 8.35,
            }
        ]
    )


def build_regime_source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05",
                "underlying_symbol": "SPY",
                "regime_label": "stress_high_volatility",
                "sigma_regime": 0.31,
                "regime_probability_stable_low_volatility": 0.10,
                "regime_probability_stress_high_volatility": 0.70,
                "regime_probability_transition_uncertainty": 0.20,
            }
        ]
    )


def test_score_option_rows_runs_end_to_end_with_regime_table():
    outputs = score_option_rows(
        option_source=build_option_source(),
        market_source=build_market_source(),
        regime_source=build_regime_source(),
        config=OptionScoringConfig(
            option_source_path=Path("option_source.csv"),
            transaction_cost_estimate=0.05,
            safety_margin=0.10,
        ),
    )

    assert len(outputs) == 1
    output = outputs[0]

    expected_baseline = price_black_scholes(
        BaselineInput(
            S=510.25,
            K=520.0,
            T=30.0 / 365.0,
            r=0.041,
            sigma=0.225,
            option_type="call",
            dividend_yield=0.0,
        )
    )
    expected_finn = price_black_scholes(
        BaselineInput(
            S=510.25,
            K=520.0,
            T=30.0 / 365.0,
            r=0.041,
            sigma=0.31,
            option_type="call",
            dividend_yield=0.0,
        )
    )

    assert output.option_symbol == "SPY_20260605_520_C"
    assert output.regime_label == "stress_high_volatility"
    assert output.regime_probabilities == (0.10, 0.70, 0.20)
    assert math.isclose(output.bs_price, expected_baseline.bs_price)
    assert math.isclose(output.fair_value, expected_finn.bs_price)
    assert math.isclose(output.edge, expected_finn.bs_price - 8.35)
    assert output.signal == "BUY"


def test_system_outputs_to_frame_expands_named_probability_columns():
    outputs = score_option_rows(
        option_source=build_option_source(),
        market_source=build_market_source(),
        regime_source=build_regime_source(),
        config=OptionScoringConfig(option_source_path=Path("option_source.csv")),
    )
    frame = system_outputs_to_frame(outputs)

    assert list(frame["regime_probabilities"])[0] == (0.10, 0.70, 0.20)
    assert math.isclose(frame.loc[0, "regime_probability_stable_low_volatility"], 0.10)
    assert math.isclose(frame.loc[0, "regime_probability_stress_high_volatility"], 0.70)
    assert math.isclose(frame.loc[0, "regime_probability_transition_uncertainty"], 0.20)


def test_build_and_save_option_scores_writes_csv(tmp_path):
    option_path = tmp_path / "options.csv"
    market_path = tmp_path / "market.csv"
    regime_path = tmp_path / "regimes.csv"
    output_path = tmp_path / "scored.csv"

    build_option_source().to_csv(option_path, index=False)
    build_market_source().to_csv(market_path, index=False)
    build_regime_source().to_csv(regime_path, index=False)

    frame = build_and_save_option_scores(
        OptionScoringConfig(
            option_source_path=option_path,
            market_source_path=market_path,
            regime_source_path=regime_path,
            output_path=output_path,
        ),
        dataset_builder=None,
        pipeline=None,
    )

    assert output_path.exists()
    assert len(frame) == 1
