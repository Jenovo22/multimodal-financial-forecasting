from __future__ import annotations

import math

from src.baseline import BlackScholesPricer
from src.contracts import DatasetRow, FINNInput, FINNOutput, HMMOutput
from src.pipeline import HMMFINNPipeline, PipelineConfig


class StubRegimeDetector:
    def __init__(self, output: HMMOutput) -> None:
        self.output = output
        self.received = []

    def predict(self, payload):
        self.received.append(payload)
        return self.output


class StubFINNModel:
    def __init__(self, output: FINNOutput) -> None:
        self.output = output
        self.received: list[FINNInput] = []

    def predict(self, payload: FINNInput) -> FINNOutput:
        self.received.append(payload)
        return self.output


def build_dataset_row() -> DatasetRow:
    return DatasetRow(
        timestamp="2026-05-05T00:00:00Z",
        underlying_symbol="SPY",
        option_symbol="SPY_20260605_520_C",
        option_type="call",
        S=510.25,
        K=520.0,
        T=30.0 / 365.0,
        r=0.041,
        market_price=8.35,
        bid=8.30,
        ask=8.40,
        mid_price=8.35,
        implied_volatility=0.225,
        realized_volatility=0.185,
        return_1d=0.012,
        return_5d=0.035,
        volume=123456789.0,
        dividend_yield=0.01,
    )


def test_score_row_runs_baseline_hmm_and_finn_together():
    hmm_output = HMMOutput(
        regime_label="stress_high_volatility",
        regime_probabilities=(0.1, 0.2, 0.7),
        sigma_regime=0.31,
    )
    finn_output = FINNOutput(
        fair_value=9.10,
        delta=0.61,
        gamma=0.027,
        vega=18.5,
        theta=-4.2,
    )
    pipeline = HMMFINNPipeline(
        baseline=BlackScholesPricer(),
        regime_detector=StubRegimeDetector(hmm_output),
        finn_model=StubFINNModel(finn_output),
        config=PipelineConfig(transaction_cost_estimate=0.05, safety_margin=0.10),
    )

    scored = pipeline.score_row(build_dataset_row())
    assert scored.timestamp == "2026-05-05T00:00:00Z"
    assert scored.underlying_symbol == "SPY"
    assert scored.option_symbol == "SPY_20260605_520_C"
    assert math.isclose(scored.market_price, 8.35)
    assert math.isclose(scored.fair_value, 9.10)
    assert math.isclose(scored.edge, 0.75)
    assert scored.signal == "BUY"
    assert scored.regime_label == "stress_high_volatility"
    assert scored.regime_probabilities == (0.1, 0.2, 0.7)
    assert scored.delta == 0.61
    assert scored.gamma == 0.027
    assert scored.bs_price > 0


def test_build_finn_input_uses_sigma_regime_and_probabilities():
    hmm_output = HMMOutput(
        regime_label="transition_uncertainty",
        regime_probabilities=(0.25, 0.50, 0.25),
        sigma_regime=0.28,
    )
    finn_model = StubFINNModel(
        FINNOutput(
            fair_value=8.35,
            delta=0.58,
            gamma=0.024,
        )
    )
    pipeline = HMMFINNPipeline(
        baseline=BlackScholesPricer(),
        regime_detector=StubRegimeDetector(hmm_output),
        finn_model=finn_model,
    )

    pipeline.score_row(build_dataset_row())
    payload = finn_model.received[0]
    assert math.isclose(payload.sigma_regime, 0.28)
    assert payload.regime_probabilities == (0.25, 0.50, 0.25)
    assert payload.option_type == "call"
    assert math.isclose(payload.dividend_yield, 0.01)


def test_market_reference_price_can_fall_back_to_market_price():
    hmm_output = HMMOutput(
        regime_label="stable_low_volatility",
        regime_probabilities=(0.8, 0.15, 0.05),
        sigma_regime=0.14,
    )
    pipeline = HMMFINNPipeline(
        baseline=BlackScholesPricer(),
        regime_detector=StubRegimeDetector(hmm_output),
        finn_model=StubFINNModel(
            FINNOutput(
                fair_value=8.30,
                delta=0.52,
                gamma=0.02,
            )
        ),
        config=PipelineConfig(use_mid_price_if_available=False),
    )

    row = build_dataset_row()
    scored = pipeline.score_row(row)
    assert math.isclose(scored.market_price, row.market_price)


def test_score_rows_returns_batch_outputs():
    hmm_output = HMMOutput(
        regime_label="transition_uncertainty",
        regime_probabilities=(0.2, 0.6, 0.2),
        sigma_regime=0.24,
    )
    pipeline = HMMFINNPipeline(
        baseline=BlackScholesPricer(),
        regime_detector=StubRegimeDetector(hmm_output),
        finn_model=StubFINNModel(
            FINNOutput(
                fair_value=8.50,
                delta=0.57,
                gamma=0.023,
            )
        ),
    )

    rows = [build_dataset_row(), build_dataset_row()]
    outputs = pipeline.score_rows(rows)
    assert len(outputs) == 2
    assert all(output.option_symbol == "SPY_20260605_520_C" for output in outputs)
