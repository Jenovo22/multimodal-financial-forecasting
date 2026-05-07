"""Loss helpers for a PyTorch-based Finance-Informed Neural Network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - exercised indirectly when torch is absent.
    torch = None
    Tensor = Any


@dataclass(slots=True, frozen=True)
class PDELossWeights:
    lambda_data: float = 1.0
    lambda_boundary: float = 0.1
    lambda_pde: float = 0.1
    lambda_arbitrage: float = 0.0


def total_finn_loss(
    data_loss_value: float,
    boundary_loss_value: float,
    pde_loss_value: float,
    arbitrage_loss_value: float = 0.0,
    weights: PDELossWeights | None = None,
) -> float:
    """Combine scalar or tensor loss terms with the configured lambdas."""

    weights = weights or PDELossWeights()
    return (
        weights.lambda_data * data_loss_value
        + weights.lambda_boundary * boundary_loss_value
        + weights.lambda_pde * pde_loss_value
        + weights.lambda_arbitrage * arbitrage_loss_value
    )


def black_scholes_pde_residual(
    inputs: Mapping[str, Tensor],
    predictions: Tensor,
) -> Tensor:
    """Return the mean squared Black-Scholes PDE residual on time-to-maturity T."""

    _require_torch()
    values = _flatten_predictions(predictions)

    S = inputs["S"]
    T = inputs["T"]
    r = inputs["r"]
    sigma = inputs["sigma_regime"]
    q = inputs.get("dividend_yield")
    if q is None:
        q = torch.zeros_like(r)

    dV_dS = torch.autograd.grad(
        values.sum(),
        S,
        create_graph=True,
        retain_graph=True,
    )[0]
    d2V_dS2 = torch.autograd.grad(
        dV_dS.sum(),
        S,
        create_graph=True,
        retain_graph=True,
    )[0]
    dV_dT = torch.autograd.grad(
        values.sum(),
        T,
        create_graph=True,
        retain_graph=True,
    )[0]

    rhs = 0.5 * sigma.square() * S.square() * d2V_dS2 + (r - q) * S * dV_dS - r * values
    residual = dV_dT - rhs
    return residual.square().mean()


def boundary_condition_loss(
    inputs: Mapping[str, Tensor],
    predictions: Tensor,
) -> Tensor:
    """Penalize violations of vanilla option price bounds."""

    _require_torch()
    values = _flatten_predictions(predictions)
    S = inputs["S"]
    K = inputs["K"]
    T = inputs["T"]
    r = inputs["r"]
    q = inputs.get("dividend_yield")
    if q is None:
        q = torch.zeros_like(r)

    discount_r = torch.exp(-r * T)
    discount_q = torch.exp(-q * T)
    option_sign = inputs["option_type_sign"]

    call_mask = option_sign > 0
    call_lower = torch.clamp(S * discount_q - K * discount_r, min=0.0)
    call_upper = S * discount_q
    put_lower = torch.clamp(K * discount_r - S * discount_q, min=0.0)
    put_upper = K * discount_r

    lower_bound = torch.where(call_mask, call_lower, put_lower)
    upper_bound = torch.where(call_mask, call_upper, put_upper)
    below_penalty = torch.relu(lower_bound - values)
    above_penalty = torch.relu(values - upper_bound)
    return (below_penalty.square() + above_penalty.square()).mean()


def arbitrage_regularization_loss(
    inputs: Mapping[str, Tensor],
    predictions: Tensor,
) -> Tensor:
    """Penalize Delta/Gamma violations of basic no-arbitrage constraints."""

    _require_torch()
    values = _flatten_predictions(predictions)
    S = inputs["S"]
    T = inputs["T"]
    r = inputs["r"]
    q = inputs.get("dividend_yield")
    if q is None:
        q = torch.zeros_like(r)
    option_sign = inputs["option_type_sign"]

    dV_dS = torch.autograd.grad(
        values.sum(),
        S,
        create_graph=True,
        retain_graph=True,
    )[0]
    d2V_dS2 = torch.autograd.grad(
        dV_dS.sum(),
        S,
        create_graph=True,
        retain_graph=True,
    )[0]

    discount_q = torch.exp(-q * T)
    call_mask = option_sign > 0

    call_lower_delta = torch.zeros_like(dV_dS)
    call_upper_delta = discount_q
    put_lower_delta = -discount_q
    put_upper_delta = torch.zeros_like(dV_dS)

    lower_delta = torch.where(call_mask, call_lower_delta, put_lower_delta)
    upper_delta = torch.where(call_mask, call_upper_delta, put_upper_delta)

    delta_below_penalty = torch.relu(lower_delta - dV_dS)
    delta_above_penalty = torch.relu(dV_dS - upper_delta)
    gamma_penalty = torch.relu(-d2V_dS2)

    return (
        delta_below_penalty.square()
        + delta_above_penalty.square()
        + gamma_penalty.square()
    ).mean()


def _flatten_predictions(predictions: Tensor) -> Tensor:
    if predictions.ndim == 0:
        return predictions.reshape(1)
    if predictions.ndim == 2 and predictions.shape[-1] == 1:
        return predictions.reshape(-1)
    return predictions


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for FINN PDE losses. Install the optional 'ml' dependency."
        )
