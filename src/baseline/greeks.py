"""Analytical Greeks engine backed by the baseline implementation."""

from __future__ import annotations

from src.baseline.black_scholes import price_black_scholes
from src.contracts import BaselineInput, BaselineOutput


def compute_black_scholes_greeks(payload: BaselineInput) -> BaselineOutput:
    """Return Black-Scholes price and sensitivities for one contract."""

    return price_black_scholes(payload)


class GreeksEngine:
    """Service wrapper for future pipeline integration."""

    def compute(self, payload: BaselineInput) -> BaselineOutput:
        return compute_black_scholes_greeks(payload)
