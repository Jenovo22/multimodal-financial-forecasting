"""End-to-end orchestration for baseline pricing, HMM regime detection and FINN scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from src.baseline import build_baseline_input
from src.contracts import (
    BaselineInput,
    BaselineOutput,
    DatasetRow,
    FINNInput,
    FINNOutput,
    HMMFeatures,
    HMMInput,
    HMMOutput,
    SignalType,
    SystemOutput,
)


class BaselineComponent(Protocol):
    def price(self, payload: BaselineInput) -> BaselineOutput: ...


class HMMComponent(Protocol):
    def predict(self, payload: HMMInput) -> HMMOutput: ...


class FINNComponent(Protocol):
    def predict(self, payload: FINNInput) -> FINNOutput: ...


@dataclass(slots=True, frozen=True)
class PipelineConfig:
    transaction_cost_estimate: float = 0.0
    safety_margin: float = 0.0
    use_mid_price_if_available: bool = True


def signal_from_edge(
    edge: float,
    transaction_cost_estimate: float = 0.0,
    safety_margin: float = 0.0,
) -> SignalType:
    """Translate fair-value edge into a discrete trading signal."""

    threshold = transaction_cost_estimate + safety_margin
    if edge > threshold:
        return "BUY"
    if edge < -threshold:
        return "SELL"
    return "HOLD"


class HMMFINNPipeline:
    """Orchestrates baseline pricing, regime detection and FINN scoring."""

    def __init__(
        self,
        baseline: BaselineComponent,
        regime_detector: HMMComponent,
        finn_model: FINNComponent,
        config: PipelineConfig | None = None,
    ) -> None:
        self.baseline = baseline
        self.regime_detector = regime_detector
        self.finn_model = finn_model
        self.config = config or PipelineConfig()

    def build_hmm_input(self, row: DatasetRow) -> HMMInput:
        return HMMInput(
            timestamp=row.timestamp,
            features=HMMFeatures(
                return_1d=row.return_1d,
                return_5d=row.return_5d,
                realized_volatility=row.realized_volatility,
                implied_volatility=row.implied_volatility,
                volume=row.volume,
                sentiment_score=row.sentiment_score,
                z_t=row.text_embedding,
            ),
        )

    def build_finn_input(self, row: DatasetRow, hmm_output: HMMOutput) -> FINNInput:
        return FINNInput(
            S=row.S,
            K=row.K,
            T=row.T,
            r=row.r,
            sigma_regime=hmm_output.sigma_regime,
            option_type=row.option_type,
            dividend_yield=row.dividend_yield or 0.0,
            regime_probabilities=hmm_output.regime_probabilities,
            z_t=row.text_embedding,
        )

    def market_reference_price(self, row: DatasetRow) -> float:
        if self.config.use_mid_price_if_available and row.mid_price is not None:
            return row.mid_price
        return row.market_price

    def score_row(self, row: DatasetRow) -> SystemOutput:
        baseline_output = self.baseline.price(build_baseline_input(row))
        hmm_output = self.regime_detector.predict(self.build_hmm_input(row))
        finn_output = self.finn_model.predict(self.build_finn_input(row, hmm_output))

        market_price = self.market_reference_price(row)
        edge = finn_output.fair_value - market_price
        signal = signal_from_edge(
            edge=edge,
            transaction_cost_estimate=self.config.transaction_cost_estimate,
            safety_margin=self.config.safety_margin,
        )

        return SystemOutput(
            timestamp=row.timestamp,
            underlying_symbol=row.underlying_symbol,
            option_symbol=row.option_symbol,
            market_price=market_price,
            bs_price=baseline_output.bs_price,
            fair_value=finn_output.fair_value,
            edge=edge,
            delta=finn_output.delta,
            gamma=finn_output.gamma,
            regime_label=hmm_output.regime_label,
            regime_probabilities=hmm_output.regime_probabilities,
            signal=signal,
            transaction_cost_estimate=self.config.transaction_cost_estimate,
            safety_margin=self.config.safety_margin,
        )

    def score_rows(self, rows: Sequence[DatasetRow]) -> list[SystemOutput]:
        return [self.score_row(row) for row in rows]
