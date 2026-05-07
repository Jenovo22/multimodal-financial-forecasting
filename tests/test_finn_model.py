from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from src.baseline import price_black_scholes
from src.contracts import BaselineInput, FINNInput
from src.models.finn import (
    FINNConfig,
    FINNPricingModel,
    arbitrage_regularization_loss,
    black_scholes_pde_residual,
    boundary_condition_loss,
)


SQRT_TWO = math.sqrt(2.0)


def _normal_cdf(values):
    return 0.5 * (1.0 + torch.erf(values / SQRT_TWO))


def build_raw_torch_inputs():
    S = torch.tensor([510.25, 498.0], dtype=torch.float32, requires_grad=True)
    K = torch.tensor([520.0, 500.0], dtype=torch.float32)
    T = torch.tensor([30.0 / 365.0, 45.0 / 365.0], dtype=torch.float32, requires_grad=True)
    r = torch.tensor([0.041, 0.038], dtype=torch.float32)
    sigma = torch.tensor([0.31, 0.24], dtype=torch.float32, requires_grad=True)
    q = torch.tensor([0.01, 0.0], dtype=torch.float32)
    option_type_sign = torch.tensor([1.0, -1.0], dtype=torch.float32)
    regime_probabilities = torch.tensor(
        [
            [0.10, 0.70, 0.20],
            [0.60, 0.10, 0.30],
        ],
        dtype=torch.float32,
    )
    return {
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "sigma_regime": sigma,
        "dividend_yield": q,
        "option_type_sign": option_type_sign,
        "regime_probabilities": regime_probabilities,
    }


def analytical_black_scholes_from_raw(raw_inputs):
    S = raw_inputs["S"]
    K = raw_inputs["K"]
    T = raw_inputs["T"]
    r = raw_inputs["r"]
    sigma = raw_inputs["sigma_regime"]
    q = raw_inputs["dividend_yield"]
    option_type_sign = raw_inputs["option_type_sign"]

    sigma_sqrt_t = sigma * torch.sqrt(T)
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma.square()) * T) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    discount_r = torch.exp(-r * T)
    discount_q = torch.exp(-q * T)

    call = S * discount_q * _normal_cdf(d1) - K * discount_r * _normal_cdf(d2)
    put = K * discount_r * _normal_cdf(-d2) - S * discount_q * _normal_cdf(-d1)
    return torch.where(option_type_sign > 0, call, put)


def build_training_samples() -> tuple[list[FINNInput], list[float]]:
    features: list[FINNInput] = []
    targets: list[float] = []
    probabilities = [
        (0.70, 0.10, 0.20),
        (0.10, 0.70, 0.20),
        (0.20, 0.20, 0.60),
    ]
    for index in range(18):
        option_type = "call" if index % 2 == 0 else "put"
        payload = FINNInput(
            S=470.0 + 4.0 * index,
            K=500.0 + 5.0 * ((index % 5) - 2),
            T=(20.0 + index) / 365.0,
            r=0.03 + 0.0005 * index,
            sigma_regime=0.16 + 0.005 * (index % 6),
            option_type=option_type,
            dividend_yield=0.01 if index % 3 == 0 else 0.0,
            regime_probabilities=probabilities[index % len(probabilities)],
        )
        target = price_black_scholes(
            BaselineInput(
                S=payload.S,
                K=payload.K,
                T=payload.T,
                r=payload.r,
                sigma=payload.sigma_regime,
                option_type=payload.option_type,
                dividend_yield=payload.dividend_yield,
            )
        ).bs_price
        features.append(payload)
        targets.append(target)
    return features, targets


def test_black_scholes_pde_residual_is_small_for_analytical_solution():
    raw_inputs = build_raw_torch_inputs()
    predictions = analytical_black_scholes_from_raw(raw_inputs)

    residual = black_scholes_pde_residual(raw_inputs, predictions)
    boundary = boundary_condition_loss(raw_inputs, predictions)
    arbitrage = arbitrage_regularization_loss(raw_inputs, predictions)

    assert residual.item() < 1e-8
    assert boundary.item() < 1e-10
    assert arbitrage.item() < 1e-10


def test_finn_model_fit_predict_save_and_load(tmp_path):
    features, targets = build_training_samples()
    model = FINNPricingModel(
        FINNConfig(
            hidden_dims=(32, 32),
            epochs=40,
            batch_size=6,
            learning_rate=1e-3,
            lambda_boundary=0.0,
            lambda_pde=0.0,
            lambda_arbitrage=0.0,
        )
    )

    model.fit(features, targets)
    prediction = model.predict(features[0])

    assert model.is_trained is True
    assert model.training_history_[0]["data_loss"] > model.training_history_[-1]["data_loss"]
    assert math.isfinite(prediction.fair_value)
    assert math.isfinite(prediction.pde_residual or 0.0)
    assert abs(prediction.fair_value - targets[0]) < 6.0

    checkpoint = model.save(tmp_path / "finn_model.pt")
    loaded = FINNPricingModel.load(checkpoint)
    loaded_prediction = loaded.predict(features[0])

    assert math.isclose(
        prediction.fair_value,
        loaded_prediction.fair_value,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
