from __future__ import annotations

import math

import numpy as np
import pytest

from src.contracts import HMMFeatures, HMMInput
from src.models.hmm import HMMRegimeDetector


def build_synthetic_hmm_inputs() -> list[HMMInput]:
    rng = np.random.default_rng(42)
    samples: list[HMMInput] = []

    regimes = [
        {
            "name": "stable",
            "length": 45,
            "return_1d_mean": 0.0008,
            "return_1d_std": 0.003,
            "return_5d_mean": 0.004,
            "return_5d_std": 0.008,
            "realized_volatility_mean": 0.11,
            "realized_volatility_std": 0.015,
            "implied_volatility_mean": 0.13,
            "implied_volatility_std": 0.015,
            "volume_mean": 9.5e7,
            "volume_std": 9e6,
        },
        {
            "name": "transition",
            "length": 40,
            "return_1d_mean": -0.0005,
            "return_1d_std": 0.008,
            "return_5d_mean": -0.002,
            "return_5d_std": 0.018,
            "realized_volatility_mean": 0.23,
            "realized_volatility_std": 0.025,
            "implied_volatility_mean": 0.27,
            "implied_volatility_std": 0.025,
            "volume_mean": 1.45e8,
            "volume_std": 1.2e7,
        },
        {
            "name": "stress",
            "length": 50,
            "return_1d_mean": -0.003,
            "return_1d_std": 0.015,
            "return_5d_mean": -0.015,
            "return_5d_std": 0.03,
            "realized_volatility_mean": 0.46,
            "realized_volatility_std": 0.04,
            "implied_volatility_mean": 0.52,
            "implied_volatility_std": 0.04,
            "volume_mean": 2.3e8,
            "volume_std": 2e7,
        },
    ]

    day_index = 0
    for regime in regimes:
        for _ in range(regime["length"]):
            timestamp = f"2020-01-{(day_index % 28) + 1:02d}T00:00:00Z"
            samples.append(
                HMMInput(
                    timestamp=timestamp,
                    features=HMMFeatures(
                        return_1d=float(rng.normal(regime["return_1d_mean"], regime["return_1d_std"])),
                        return_5d=float(rng.normal(regime["return_5d_mean"], regime["return_5d_std"])),
                        realized_volatility=float(
                            max(
                                rng.normal(
                                    regime["realized_volatility_mean"],
                                    regime["realized_volatility_std"],
                                ),
                                1e-4,
                            )
                        ),
                        implied_volatility=float(
                            max(
                                rng.normal(
                                    regime["implied_volatility_mean"],
                                    regime["implied_volatility_std"],
                                ),
                                1e-4,
                            )
                        ),
                        volume=float(
                            max(rng.normal(regime["volume_mean"], regime["volume_std"]), 1.0)
                        ),
                    ),
                )
            )
            day_index += 1

    return samples


def test_hmm_regime_detector_fits_and_predicts_probabilities():
    detector = HMMRegimeDetector()
    samples = build_synthetic_hmm_inputs()
    detector.fit(samples)

    assert detector.is_fitted
    output = detector.predict(samples[-1])
    assert output.regime_label in {
        "stable_low_volatility",
        "stress_high_volatility",
        "transition_uncertainty",
    }
    assert len(output.regime_probabilities) == 3
    assert math.isclose(sum(output.regime_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert output.sigma_regime > 0


def test_hmm_regime_detector_assigns_stable_and_stress_by_sigma_ranking():
    detector = HMMRegimeDetector()
    detector.fit(build_synthetic_hmm_inputs())

    assert detector.state_sigma_values_ is not None
    low_vol_state = int(np.argmin(detector.state_sigma_values_))
    high_vol_state = int(np.argmax(detector.state_sigma_values_))

    assert detector.label_from_state(low_vol_state) == "stable_low_volatility"
    assert detector.label_from_state(high_vol_state) == "stress_high_volatility"


def test_hmm_regime_detector_handles_missing_feature_values():
    detector = HMMRegimeDetector()
    samples = build_synthetic_hmm_inputs()
    samples[3] = HMMInput(
        timestamp=samples[3].timestamp,
        features=HMMFeatures(
            return_1d=samples[3].features.return_1d,
            return_5d=None,
            realized_volatility=samples[3].features.realized_volatility,
            implied_volatility=samples[3].features.implied_volatility,
            volume=None,
        ),
    )

    detector.fit(samples)
    output = detector.predict(samples[3])
    assert output.sigma_regime > 0
    assert math.isclose(sum(output.regime_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_hmm_regime_detector_requires_fit_before_predict():
    detector = HMMRegimeDetector()
    sample = build_synthetic_hmm_inputs()[0]
    with pytest.raises(RuntimeError):
        detector.predict(sample)


def test_hmm_regime_detector_exposes_probabilities_in_label_order():
    detector = HMMRegimeDetector()
    detector.state_to_label_ = {
        0: "stress_high_volatility",
        1: "stable_low_volatility",
        2: "transition_uncertainty",
    }

    probabilities = detector.probabilities_by_label((0.20, 0.70, 0.10))

    assert probabilities == (0.70, 0.20, 0.10)
