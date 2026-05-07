from __future__ import annotations

import math

import pandas as pd
import pytest

from src.contracts import HMMFeatures, HMMInput
from src.models.hmm import RegimeTableDetector, RegimeTableDetectorConfig


def build_regime_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05",
                "underlying_symbol": "SPY",
                "regime_label": "stable_low_volatility",
                "sigma_regime": 0.18,
                "regime_probability_stable_low_volatility": 0.80,
                "regime_probability_stress_high_volatility": 0.05,
                "regime_probability_transition_uncertainty": 0.15,
            },
            {
                "timestamp": "2026-05-06",
                "underlying_symbol": "SPY",
                "regime_label": "stress_high_volatility",
                "sigma_regime": 0.34,
                "regime_probability_stable_low_volatility": 0.10,
                "regime_probability_stress_high_volatility": 0.75,
                "regime_probability_transition_uncertainty": 0.15,
            },
        ]
    )


def build_hmm_input(timestamp: str) -> HMMInput:
    return HMMInput(timestamp=timestamp, features=HMMFeatures())


def test_regime_table_detector_returns_hmm_output_for_exact_timestamp():
    detector = RegimeTableDetector(build_regime_table())
    output = detector.predict(build_hmm_input("2026-05-06"))

    assert output.regime_label == "stress_high_volatility"
    assert output.regime_probabilities == (0.10, 0.75, 0.15)
    assert math.isclose(output.sigma_regime, 0.34)


def test_regime_table_detector_can_use_previous_available_timestamp():
    detector = RegimeTableDetector(
        build_regime_table(),
        config=RegimeTableDetectorConfig(match_mode="previous", max_staleness_days=2),
    )
    output = detector.predict(build_hmm_input("2026-05-07"))

    assert output.regime_label == "stress_high_volatility"
    assert math.isclose(output.sigma_regime, 0.34)


def test_regime_table_detector_rejects_missing_exact_timestamp():
    detector = RegimeTableDetector(build_regime_table())
    with pytest.raises(KeyError, match="No regime row"):
        detector.predict(build_hmm_input("2026-05-07"))


def test_regime_table_detector_requires_symbol_filter_for_multi_asset_tables():
    frame = pd.concat(
        [
            build_regime_table(),
            build_regime_table().assign(underlying_symbol="QQQ"),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="multiple underlyings"):
        RegimeTableDetector(frame)

    detector = RegimeTableDetector(
        frame,
        config=RegimeTableDetectorConfig(underlying_symbol="SPY"),
    )
    output = detector.predict(build_hmm_input("2026-05-05"))
    assert output.regime_label == "stable_low_volatility"
