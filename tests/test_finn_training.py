from __future__ import annotations

import math

import pandas as pd
import pytest

from src.contracts import FINNInput
from src.models.finn import (
    FINNConfig,
    FINNPricingModel,
    FINNSplitConfig,
    evaluate_finn_model,
    frame_for_final_training,
    frame_for_split,
    split_finn_training_frame,
    split_summary,
)


torch = pytest.importorskip("torch")


def build_training_frame() -> pd.DataFrame:
    rows = []
    expirations = ["2026-05-29", "2026-06-05", "2026-06-12", "2026-06-18"]
    for expiration_index, expiration in enumerate(expirations):
        for strike_index, strike in enumerate([700.0, 710.0, 720.0, 730.0]):
            market_price = max(723.0 - strike, 1.0) + expiration_index * 0.4
            rows.append(
                {
                    "timestamp": "2026-05-05T00:00:00Z",
                    "underlying_symbol": "SPY",
                    "option_symbol": f"SPY_{expiration.replace('-', '')}_{int(strike)}_C",
                    "option_type": "call",
                    "S": 723.0,
                    "K": strike,
                    "T": (24.0 + 7.0 * expiration_index) / 365.0,
                    "r": 0.04,
                    "market_price": market_price,
                    "target_price": market_price,
                    "implied_volatility": 0.18 + 0.01 * strike_index,
                    "sigma_regime": 0.16,
                    "dividend_yield": 0.0,
                    "regime_label": "transition_uncertainty",
                    "regime_probability_stable_low_volatility": 0.1,
                    "regime_probability_stress_high_volatility": 0.2,
                    "regime_probability_transition_uncertainty": 0.7,
                    "expiration_date": expiration,
                }
            )
    return pd.DataFrame(rows)


def test_split_finn_training_frame_by_expiration_keeps_future_test_group():
    frame = build_training_frame()

    split = split_finn_training_frame(frame, FINNSplitConfig(strategy="expiration"))

    assert split["split"].value_counts().to_dict() == {
        "train": 8,
        "validation": 4,
        "test": 4,
    }
    assert set(frame_for_split(split, "test")["expiration_date"]) == {"2026-06-18"}
    assert frame_for_final_training(split, "train_val")["split"].isin(
        ["train", "validation"]
    ).all()
    assert split_summary(split)["test"]["rows"] == 4
    assert split_summary(split)["test"]["timestamps"] == ["2026-05-05T00:00:00Z"]


def test_split_finn_training_frame_by_timestamp_keeps_future_snapshot_for_test():
    frame = build_training_frame()
    timestamps = [
        "2026-05-05T00:00:00Z",
        "2026-05-06T00:00:00Z",
        "2026-05-07T00:00:00Z",
        "2026-05-08T00:00:00Z",
    ]
    frame["timestamp"] = [
        timestamps[index // 4]
        for index in range(len(frame))
    ]

    split = split_finn_training_frame(frame, FINNSplitConfig(strategy="timestamp"))

    assert split["split"].value_counts().to_dict() == {
        "train": 8,
        "validation": 4,
        "test": 4,
    }
    assert set(frame_for_split(split, "test")["timestamp"]) == {"2026-05-08T00:00:00Z"}
    assert split_summary(split)["test"]["timestamps"] == ["2026-05-08T00:00:00Z"]


def test_finn_fit_records_validation_loss_and_early_stopping():
    features = [
        FINNInput(
            S=723.0,
            K=700.0 + index,
            T=30.0 / 365.0,
            r=0.04,
            sigma_regime=0.16,
            option_type="call",
            regime_probabilities=(0.1, 0.2, 0.7),
        )
        for index in range(8)
    ]
    targets = [25.0 - index * 0.2 for index in range(8)]
    model = FINNPricingModel(
        FINNConfig(
            hidden_dims=(16,),
            epochs=5,
            batch_size=4,
            lambda_boundary=0.0,
            lambda_pde=0.0,
            lambda_arbitrage=0.0,
            early_stopping_patience=3,
        )
    )

    model.fit(
        features[:6],
        targets[:6],
        validation_features=features[6:],
        validation_targets=targets[6:],
    )

    assert model.best_epoch_ is not None
    assert model.best_validation_loss_ is not None
    assert "validation_data_loss" in model.training_history_[-1]


def test_evaluate_finn_model_returns_metrics_and_predictions():
    frame = build_training_frame().head(8)
    features = [
        FINNInput(
            S=float(row.S),
            K=float(row.K),
            T=float(row.T),
            r=float(row.r),
            sigma_regime=float(row.sigma_regime),
            option_type="call",
            regime_probabilities=(
                float(row.regime_probability_stable_low_volatility),
                float(row.regime_probability_stress_high_volatility),
                float(row.regime_probability_transition_uncertainty),
            ),
        )
        for row in frame.itertuples(index=False)
    ]
    targets = [float(value) for value in frame["target_price"]]
    model = FINNPricingModel(
        FINNConfig(
            hidden_dims=(16,),
            epochs=3,
            batch_size=4,
            lambda_boundary=0.0,
            lambda_pde=0.0,
            lambda_arbitrage=0.0,
        )
    )
    model.fit(features, targets)

    metrics, predictions = evaluate_finn_model(model, frame)

    assert metrics["rows"] == len(frame)
    assert math.isfinite(metrics["mae"])
    assert "predicted_fair_value" in predictions.columns
    assert "baseline_bs_price" in predictions.columns
