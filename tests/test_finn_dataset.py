from __future__ import annotations

import math

import pandas as pd
import pytest

from src.data import (
    FINNDatasetConfig,
    build_finn_training_examples,
    build_finn_training_frame,
    frame_to_finn_inputs,
    frame_to_finn_targets,
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
                "market_price": 8.40,
                "bid": 8.30,
                "ask": 8.40,
                "mid_price": 8.35,
            }
        ]
    )


def build_regime_source(timestamp: str = "2026-05-05") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "underlying_symbol": "SPY",
                "regime_label": "stress_high_volatility",
                "sigma_regime": 0.31,
                "regime_probability_stable_low_volatility": 0.10,
                "regime_probability_stress_high_volatility": 0.70,
                "regime_probability_transition_uncertainty": 0.20,
            }
        ]
    )


def test_build_finn_training_frame_joins_options_market_and_regimes():
    frame = build_finn_training_frame(
        option_source=build_option_source(),
        market_source=build_market_source(),
        regime_source=build_regime_source(),
    )

    assert len(frame) == 1
    assert frame.loc[0, "timestamp"] == "2026-05-05T00:00:00Z"
    assert math.isclose(frame.loc[0, "S"], 510.25)
    assert math.isclose(frame.loc[0, "sigma_regime"], 0.31)
    assert frame.loc[0, "regime_label"] == "stress_high_volatility"
    assert math.isclose(frame.loc[0, "target_price"], 8.35)


def test_frame_to_finn_inputs_and_targets_use_canonical_probability_order():
    frame = build_finn_training_frame(
        option_source=build_option_source(),
        market_source=build_market_source(),
        regime_source=build_regime_source(),
    )

    payload = frame_to_finn_inputs(frame)[0]
    targets = frame_to_finn_targets(frame)

    assert math.isclose(payload.S, 510.25)
    assert math.isclose(payload.K, 520.0)
    assert math.isclose(payload.sigma_regime, 0.31)
    assert payload.option_type == "call"
    assert math.isclose(payload.dividend_yield, 0.0)
    assert payload.regime_probabilities == (0.10, 0.70, 0.20)
    assert targets == [8.35]


def test_build_finn_training_examples_preserves_row_metadata():
    frame = build_finn_training_frame(
        option_source=build_option_source(),
        market_source=build_market_source(),
        regime_source=build_regime_source(),
    )

    example = build_finn_training_examples(frame)[0]
    assert example.option_symbol == "SPY_20260605_520_C"
    assert example.underlying_symbol == "SPY"
    assert example.regime_label == "stress_high_volatility"
    assert math.isclose(example.target_price, 8.35)
    assert math.isclose(example.payload.sigma_regime, 0.31)


def test_finn_target_can_use_market_price_when_configured():
    frame = build_finn_training_frame(
        option_source=build_option_source(),
        market_source=build_market_source(),
        regime_source=build_regime_source(),
        config=FINNDatasetConfig(use_mid_price_if_available=False),
    )
    assert math.isclose(frame.loc[0, "target_price"], 8.40)


def test_build_finn_training_frame_rejects_missing_regime_match():
    with pytest.raises(ValueError, match="no regime match"):
        build_finn_training_frame(
            option_source=build_option_source(),
            market_source=build_market_source(),
            regime_source=build_regime_source(timestamp="2026-05-04"),
        )
