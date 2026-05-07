"""Analytical Black-Scholes-Merton baseline module."""

from __future__ import annotations

import math

from src.contracts import BaselineInput, BaselineOutput, DatasetRow

VALID_OPTION_TYPES = {"call", "put"}
SQRT_TWO = math.sqrt(2.0)
SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def build_baseline_input(row: DatasetRow) -> BaselineInput:
    """Map one dataset row into the baseline model input contract."""

    return BaselineInput(
        S=row.S,
        K=row.K,
        T=row.T,
        r=row.r,
        sigma=row.implied_volatility,
        dividend_yield=row.dividend_yield or 0.0,
        option_type=row.option_type,
    )


def validate_baseline_input(payload: BaselineInput) -> None:
    """Fail fast on invalid baseline inputs."""

    if payload.option_type not in VALID_OPTION_TYPES:
        raise ValueError(f"Unsupported option_type: {payload.option_type!r}")
    if payload.S <= 0 or payload.K <= 0 or payload.T <= 0 or payload.sigma <= 0:
        raise ValueError("S, K, T and sigma must be strictly positive.")
    if payload.dividend_yield < 0:
        raise ValueError("dividend_yield must be non-negative.")


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_TWO))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_TWO_PI


def _compute_d1_d2(payload: BaselineInput) -> tuple[float, float]:
    sigma_sqrt_t = payload.sigma * math.sqrt(payload.T)
    log_moneyness = math.log(payload.S / payload.K)
    carry = payload.r - payload.dividend_yield + 0.5 * payload.sigma * payload.sigma
    d1 = (log_moneyness + carry * payload.T) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return d1, d2


def _compute_baseline_output(payload: BaselineInput) -> BaselineOutput:
    d1, d2 = _compute_d1_d2(payload)
    q = payload.dividend_yield
    discount_r = math.exp(-payload.r * payload.T)
    discount_q = math.exp(-q * payload.T)
    nd1 = _normal_cdf(d1)
    nd2 = _normal_cdf(d2)
    pdf_d1 = _normal_pdf(d1)

    if payload.option_type == "call":
        price = payload.S * discount_q * nd1 - payload.K * discount_r * nd2
        delta = discount_q * nd1
        theta = (
            -payload.S * discount_q * pdf_d1 * payload.sigma / (2.0 * math.sqrt(payload.T))
            - payload.r * payload.K * discount_r * nd2
            + q * payload.S * discount_q * nd1
        )
        rho = payload.K * payload.T * discount_r * nd2
    else:
        n_minus_d1 = _normal_cdf(-d1)
        n_minus_d2 = _normal_cdf(-d2)
        price = payload.K * discount_r * n_minus_d2 - payload.S * discount_q * n_minus_d1
        delta = discount_q * (nd1 - 1.0)
        theta = (
            -payload.S * discount_q * pdf_d1 * payload.sigma / (2.0 * math.sqrt(payload.T))
            + payload.r * payload.K * discount_r * n_minus_d2
            - q * payload.S * discount_q * n_minus_d1
        )
        rho = -payload.K * payload.T * discount_r * n_minus_d2

    gamma = discount_q * pdf_d1 / (payload.S * payload.sigma * math.sqrt(payload.T))
    vega = payload.S * discount_q * pdf_d1 * math.sqrt(payload.T)

    return BaselineOutput(
        bs_price=price,
        bs_delta=delta,
        bs_gamma=gamma,
        bs_vega=vega,
        bs_theta=theta,
        bs_rho=rho,
    )


def price_black_scholes(payload: BaselineInput) -> BaselineOutput:
    """Compute analytical Black-Scholes-Merton price and Greeks."""

    validate_baseline_input(payload)
    return _compute_baseline_output(payload)


class BlackScholesPricer:
    """Thin object wrapper used by the pipeline."""

    def price(self, payload: BaselineInput) -> BaselineOutput:
        return price_black_scholes(payload)
