"""FINN-compatible analytical fallback using regime-adjusted Black-Scholes-Merton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.baseline import price_black_scholes
from src.contracts import BaselineInput, FINNInput, FINNOutput


@dataclass(slots=True, frozen=True)
class RegimeAdjustedBlackScholesConfig:
    pde_residual_for_analytical_solution: float = 0.0


class RegimeAdjustedBlackScholesFINN:
    """Drop-in FINN component that prices with sigma_regime instead of implied vol.

    This is intentionally analytical. It lets the HMM + FINN pipeline run end-to-end
    while the neural FINN implementation is plugged in behind the same contract.
    """

    def __init__(self, config: RegimeAdjustedBlackScholesConfig | None = None) -> None:
        self.config = config or RegimeAdjustedBlackScholesConfig()
        self.is_trained = True

    def fit(self, features: Sequence[FINNInput], targets: Sequence[float]) -> None:
        """No-op fit for API compatibility with trainable FINN implementations."""

        if len(features) != len(targets):
            raise ValueError("features and targets must have the same length.")
        self.is_trained = True

    def predict(self, payload: FINNInput) -> FINNOutput:
        baseline_output = price_black_scholes(
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
        return FINNOutput(
            fair_value=baseline_output.bs_price,
            delta=baseline_output.bs_delta,
            gamma=baseline_output.bs_gamma,
            vega=baseline_output.bs_vega,
            theta=baseline_output.bs_theta,
            pde_residual=self.config.pde_residual_for_analytical_solution,
        )
