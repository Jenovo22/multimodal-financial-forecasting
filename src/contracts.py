"""Typed contracts aligned with the MVP schema documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]
RegimeLabel = Literal[
    "stable_low_volatility",
    "stress_high_volatility",
    "transition_uncertainty",
    "unknown",
]
SignalType = Literal["BUY", "SELL", "HOLD"]

REGIME_LABEL_ORDER: tuple[RegimeLabel, RegimeLabel, RegimeLabel] = (
    "stable_low_volatility",
    "stress_high_volatility",
    "transition_uncertainty",
)

REGIME_PROBABILITY_COLUMNS: tuple[str, str, str] = (
    "regime_probability_stable_low_volatility",
    "regime_probability_stress_high_volatility",
    "regime_probability_transition_uncertainty",
)


@dataclass(slots=True, frozen=True)
class Metadata:
    project_name: str
    mvp_underlying: Literal["SPY", "SPX"]
    mvp_derivative: str
    frequency: Literal["daily", "intraday"]
    created_for: str


@dataclass(slots=True, frozen=True)
class DatasetRow:
    timestamp: str
    underlying_symbol: str
    option_symbol: str
    option_type: OptionType
    S: float
    K: float
    T: float
    r: float
    market_price: float
    implied_volatility: float
    bid: float | None = None
    ask: float | None = None
    mid_price: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    realized_volatility: float | None = None
    return_1d: float | None = None
    return_5d: float | None = None
    sentiment_score: float | None = None
    text_embedding: tuple[float, ...] | None = None
    event_count: int | None = None
    dividend_yield: float | None = None


@dataclass(slots=True, frozen=True)
class BaselineInput:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType
    dividend_yield: float = 0.0


@dataclass(slots=True, frozen=True)
class BaselineOutput:
    bs_price: float
    bs_delta: float
    bs_gamma: float
    bs_vega: float | None = None
    bs_theta: float | None = None
    bs_rho: float | None = None


@dataclass(slots=True, frozen=True)
class HMMFeatures:
    return_1d: float | None = None
    return_5d: float | None = None
    realized_volatility: float | None = None
    implied_volatility: float | None = None
    volume: float | None = None
    sentiment_score: float | None = None
    z_t: tuple[float, ...] | None = None


@dataclass(slots=True, frozen=True)
class HMMInput:
    timestamp: str
    features: HMMFeatures


@dataclass(slots=True, frozen=True)
class HMMOutput:
    regime_label: RegimeLabel
    regime_probabilities: tuple[float, float, float]
    sigma_regime: float


@dataclass(slots=True, frozen=True)
class FINNInput:
    S: float
    K: float
    T: float
    r: float
    sigma_regime: float
    option_type: OptionType
    dividend_yield: float = 0.0
    regime_probabilities: tuple[float, float, float] | None = None
    z_t: tuple[float, ...] | None = None


@dataclass(slots=True, frozen=True)
class FINNOutput:
    fair_value: float
    delta: float
    gamma: float
    vega: float | None = None
    theta: float | None = None
    pde_residual: float | None = None


@dataclass(slots=True, frozen=True)
class SystemOutput:
    timestamp: str
    underlying_symbol: str
    option_symbol: str
    market_price: float
    bs_price: float
    fair_value: float
    edge: float
    delta: float
    gamma: float
    regime_label: RegimeLabel
    regime_probabilities: tuple[float, float, float]
    signal: SignalType
    transaction_cost_estimate: float | None = None
    safety_margin: float | None = None
