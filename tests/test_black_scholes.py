from __future__ import annotations

from dataclasses import replace
import math

import pytest

from src.baseline import (
    BlackScholesPricer,
    GreeksEngine,
    build_baseline_input,
    compute_black_scholes_greeks,
    price_black_scholes,
    validate_baseline_input,
)
from src.contracts import BaselineInput, DatasetRow


CALL_WITH_DIVIDENDS = BaselineInput(
    S=100.0,
    K=100.0,
    T=1.0,
    r=0.05,
    sigma=0.2,
    option_type="call",
    dividend_yield=0.02,
)

PUT_WITH_DIVIDENDS = BaselineInput(
    S=100.0,
    K=100.0,
    T=1.0,
    r=0.05,
    sigma=0.2,
    option_type="put",
    dividend_yield=0.02,
)


def assert_baseline_output_matches(
    payload: BaselineInput,
    *,
    price: float,
    delta: float,
    gamma: float,
    vega: float,
    theta: float,
    rho: float,
) -> None:
    result = price_black_scholes(payload)
    assert math.isclose(result.bs_price, price, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.bs_delta, delta, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.bs_gamma, gamma, rel_tol=0.0, abs_tol=1e-12)
    assert result.bs_vega is not None
    assert result.bs_theta is not None
    assert result.bs_rho is not None
    assert math.isclose(result.bs_vega, vega, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.bs_theta, theta, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.bs_rho, rho, rel_tol=0.0, abs_tol=1e-12)


def test_price_black_scholes_matches_call_reference_values():
    assert_baseline_output_matches(
        CALL_WITH_DIVIDENDS,
        price=9.227005508154036,
        delta=0.586851146134764,
        gamma=0.018950578755008718,
        vega=37.901157510017434,
        theta=-5.0893189139983335,
        rho=49.45810910532236,
    )


def test_price_black_scholes_matches_put_reference_values():
    assert_baseline_output_matches(
        PUT_WITH_DIVIDENDS,
        price=6.330080627549918,
        delta=-0.3933475271719913,
        gamma=0.018950578755008718,
        vega=37.901157510017434,
        theta=-2.293569138108274,
        rho=-45.66483334474905,
    )


def test_put_call_parity_holds_with_dividends():
    call_result = price_black_scholes(CALL_WITH_DIVIDENDS)
    put_result = price_black_scholes(PUT_WITH_DIVIDENDS)
    lhs = call_result.bs_price - put_result.bs_price
    rhs = (
        CALL_WITH_DIVIDENDS.S * math.exp(-CALL_WITH_DIVIDENDS.dividend_yield * CALL_WITH_DIVIDENDS.T)
        - CALL_WITH_DIVIDENDS.K * math.exp(-CALL_WITH_DIVIDENDS.r * CALL_WITH_DIVIDENDS.T)
    )
    assert math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=1e-12)


def test_compute_black_scholes_greeks_matches_pricer_output():
    direct = price_black_scholes(CALL_WITH_DIVIDENDS)
    greeks = compute_black_scholes_greeks(CALL_WITH_DIVIDENDS)
    wrapper_price = BlackScholesPricer().price(CALL_WITH_DIVIDENDS)
    wrapper_greeks = GreeksEngine().compute(CALL_WITH_DIVIDENDS)
    assert greeks == direct
    assert wrapper_price == direct
    assert wrapper_greeks == direct


def test_build_baseline_input_uses_dividend_yield_when_present():
    row = DatasetRow(
        timestamp="2026-04-01T00:00:00Z",
        underlying_symbol="SPY",
        option_symbol="SPY_20260501_520_C",
        option_type="call",
        S=518.25,
        K=520.0,
        T=0.08219,
        r=0.045,
        market_price=8.40,
        implied_volatility=0.214,
        dividend_yield=0.012,
    )
    payload = build_baseline_input(row)
    assert payload.dividend_yield == 0.012


def test_build_baseline_input_defaults_dividend_yield_to_zero():
    row = DatasetRow(
        timestamp="2026-04-01T00:00:00Z",
        underlying_symbol="SPY",
        option_symbol="SPY_20260501_520_C",
        option_type="call",
        S=518.25,
        K=520.0,
        T=0.08219,
        r=0.045,
        market_price=8.40,
        implied_volatility=0.214,
    )
    payload = build_baseline_input(row)
    assert payload.dividend_yield == 0.0


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("S", 0.0),
        ("K", 0.0),
        ("T", 0.0),
        ("sigma", 0.0),
        ("dividend_yield", -0.01),
    ],
)
def test_validate_baseline_input_rejects_invalid_numeric_values(
    field_name: str,
    field_value: float,
) -> None:
    payload = BaselineInput(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type="call",
        dividend_yield=0.02,
    )
    payload = replace(payload, **{field_name: field_value})
    with pytest.raises(ValueError):
        validate_baseline_input(payload)


def test_validate_baseline_input_rejects_invalid_option_type():
    payload = BaselineInput(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type="call",
        dividend_yield=0.02,
    )
    invalid_payload = replace(payload, option_type="straddle")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_baseline_input(invalid_payload)
