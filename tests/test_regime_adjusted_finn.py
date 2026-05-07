from __future__ import annotations

import math

from src.baseline import price_black_scholes
from src.contracts import BaselineInput, FINNInput
from src.models.finn import RegimeAdjustedBlackScholesFINN


def test_regime_adjusted_finn_matches_black_scholes_with_sigma_regime():
    payload = FINNInput(
        S=510.25,
        K=520.0,
        T=30.0 / 365.0,
        r=0.041,
        sigma_regime=0.31,
        option_type="call",
        dividend_yield=0.01,
        regime_probabilities=(0.10, 0.70, 0.20),
    )
    model = RegimeAdjustedBlackScholesFINN()

    output = model.predict(payload)
    expected = price_black_scholes(
        BaselineInput(
            S=payload.S,
            K=payload.K,
            T=payload.T,
            r=payload.r,
            sigma=payload.sigma_regime,
            option_type=payload.option_type,
            dividend_yield=payload.dividend_yield,
        )
    )

    assert math.isclose(output.fair_value, expected.bs_price)
    assert math.isclose(output.delta, expected.bs_delta)
    assert math.isclose(output.gamma, expected.bs_gamma)
    assert math.isclose(output.vega or 0.0, expected.bs_vega or 0.0)
    assert math.isclose(output.theta or 0.0, expected.bs_theta or 0.0)
    assert output.pde_residual == 0.0


def test_regime_adjusted_finn_fit_checks_feature_target_alignment():
    model = RegimeAdjustedBlackScholesFINN()
    try:
        model.fit([], [1.0])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected fit to reject misaligned features and targets.")
