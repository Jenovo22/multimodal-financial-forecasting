from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import APIServerConfig, create_app
from src.baseline import BlackScholesPricer
from src.contracts import BaselineInput, FINNInput
from src.models.finn import FINNConfig, FINNPricingModel, RegimeAdjustedBlackScholesFINN
from src.models.hmm import RegimeTableDetector
from src.pipeline.hmm_finn_pipeline import HMMFINNPipeline, PipelineConfig
from src.baseline import price_black_scholes


def build_regime_source():
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05",
                "underlying_symbol": "SPY",
                "regime_label": "stress_high_volatility",
                "sigma_regime": 0.31,
                "regime_probability_stable_low_volatility": 0.10,
                "regime_probability_stress_high_volatility": 0.70,
                "regime_probability_transition_uncertainty": 0.20,
            }
        ]
    )


def build_row_payload() -> dict[str, object]:
    return {
        "timestamp": "2026-05-05T00:00:00Z",
        "underlying_symbol": "SPY",
        "option_symbol": "SPY_20260605_520_C",
        "option_type": "call",
        "S": 510.25,
        "K": 520.0,
        "T": 30.0 / 365.0,
        "r": 0.041,
        "market_price": 8.35,
        "bid": 8.30,
        "ask": 8.40,
        "mid_price": 8.35,
        "implied_volatility": 0.225,
        "realized_volatility": 0.185,
        "return_1d": 0.012,
        "return_5d": 0.035,
        "volume": 123456789.0,
        "dividend_yield": 0.01,
    }


def build_test_client() -> TestClient:
    detector = RegimeTableDetector(build_regime_source())
    pipeline = HMMFINNPipeline(
        baseline=BlackScholesPricer(),
        regime_detector=detector,
        finn_model=RegimeAdjustedBlackScholesFINN(),
        config=PipelineConfig(transaction_cost_estimate=0.05, safety_margin=0.10),
    )
    app = create_app(
        APIServerConfig(
            regime_source_path=Path("tests/fixtures/regimes.csv"),
            transaction_cost_estimate=0.05,
            safety_margin=0.10,
        ),
        pipeline=pipeline,
        regime_row_count=1,
    )
    return TestClient(app)


def test_health_and_metadata_report_ready_runtime():
    client = build_test_client()

    health = client.get("/health")
    metadata = client.get("/metadata")

    assert health.status_code == 200
    assert health.json()["ready"] is True
    assert health.json()["regime_row_count"] == 1
    assert health.json()["finn_component_name"] == "RegimeAdjustedBlackScholesFINN"

    assert metadata.status_code == 200
    assert metadata.json()["finn_component"] == "RegimeAdjustedBlackScholesFINN"
    assert metadata.json()["transaction_cost_estimate"] == 0.05


def test_score_row_returns_system_output():
    client = build_test_client()

    response = client.post("/score/row", json=build_row_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["option_symbol"] == "SPY_20260605_520_C"
    assert body["regime_label"] == "stress_high_volatility"
    assert body["regime_probabilities"] == [0.1, 0.7, 0.2]
    assert body["signal"] == "BUY"
    assert body["fair_value"] > body["market_price"]


def test_score_batch_returns_count_and_items():
    client = build_test_client()

    response = client.post(
        "/score/batch",
        json={"rows": [build_row_payload(), build_row_payload()]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["items"]) == 2


def test_score_row_rejects_invalid_contract_payload():
    client = build_test_client()
    row = build_row_payload()
    row["S"] = 0.0

    response = client.post("/score/row", json=row)

    assert response.status_code == 422
    assert "S must be > 0" in response.json()["detail"]


def test_api_reports_degraded_state_when_runtime_cannot_start(tmp_path):
    app = create_app(
        APIServerConfig(
            regime_source_path=tmp_path / "missing_regimes.csv",
        )
    )
    client = TestClient(app)

    health = client.get("/health")
    score = client.post("/score/row", json=build_row_payload())

    assert health.status_code == 200
    assert health.json()["ready"] is False
    assert health.json()["status"] == "degraded"
    assert score.status_code == 503


def test_api_can_load_trained_finn_checkpoint(tmp_path):
    pytest.importorskip("torch")
    regime_path = tmp_path / "regimes.csv"
    build_regime_source().to_csv(regime_path, index=False)

    features: list[FINNInput] = []
    targets: list[float] = []
    for index in range(10):
        payload = FINNInput(
            S=490.0 + 2.0 * index,
            K=500.0,
            T=(20.0 + index) / 365.0,
            r=0.03 + 0.001 * index,
            sigma_regime=0.18 + 0.004 * index,
            option_type="call",
            dividend_yield=0.01,
            regime_probabilities=(0.2, 0.6, 0.2),
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

    model = FINNPricingModel(
        FINNConfig(
            hidden_dims=(16, 16),
            epochs=20,
            batch_size=5,
            lambda_boundary=0.0,
            lambda_pde=0.0,
            lambda_arbitrage=0.0,
        )
    )
    model.fit(features, targets)
    checkpoint_path = model.save(tmp_path / "finn_model.pt")

    app = create_app(
        APIServerConfig(
            regime_source_path=regime_path,
            finn_checkpoint_path=checkpoint_path,
        )
    )
    client = TestClient(app)

    metadata = client.get("/metadata")
    response = client.post("/score/row", json=build_row_payload())

    assert metadata.status_code == 200
    assert metadata.json()["finn_component"] == "FINNPricingModel"
    assert response.status_code == 200
