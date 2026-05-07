"""FastAPI application exposing the current HMM + FINN MVP scoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.baseline import BlackScholesPricer
from src.contracts import DatasetRow, OptionType, RegimeLabel, SystemOutput
from src.data import validate_dataset_row
from src.models.finn import FINNPricingModel, RegimeAdjustedBlackScholesFINN
from src.models.hmm import RegimeTableDetector, RegimeTableDetectorConfig
from src.pipeline.hmm_finn_pipeline import HMMFINNPipeline, PipelineConfig


@dataclass(slots=True, frozen=True)
class APIServerConfig:
    service_name: str = "proyecto-tam-api"
    service_version: str = "0.1.0"
    regime_source_path: Path = Path("Data/processed/spy_regime_features.csv")
    finn_checkpoint_path: Path | None = None
    transaction_cost_estimate: float = 0.0
    safety_margin: float = 0.0
    use_mid_price_if_available: bool = True
    regime_match_mode: Literal["exact", "previous"] = "exact"
    regime_max_staleness_days: int | None = None
    regime_underlying_symbol: str | None = "SPY"


@dataclass(slots=True)
class APIRuntime:
    config: APIServerConfig
    pipeline: HMMFINNPipeline | None
    regime_row_count: int | None
    finn_component_name: str
    startup_error: str | None = None


class DatasetRowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

    def to_contract(self) -> DatasetRow:
        row = DatasetRow(**self.model_dump())
        issues = validate_dataset_row(row)
        if issues:
            raise ValueError("; ".join(issues))
        return row


class ScoreBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[DatasetRowRequest] = Field(default_factory=list)


class SystemOutputResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    regime_probabilities: list[float]
    signal: Literal["BUY", "SELL", "HOLD"]
    transaction_cost_estimate: float | None = None
    safety_margin: float | None = None

    @classmethod
    def from_contract(cls, output: SystemOutput) -> "SystemOutputResponse":
        return cls(
            timestamp=output.timestamp,
            underlying_symbol=output.underlying_symbol,
            option_symbol=output.option_symbol,
            market_price=output.market_price,
            bs_price=output.bs_price,
            fair_value=output.fair_value,
            edge=output.edge,
            delta=output.delta,
            gamma=output.gamma,
            regime_label=output.regime_label,
            regime_probabilities=[float(value) for value in output.regime_probabilities],
            signal=output.signal,
            transaction_cost_estimate=output.transaction_cost_estimate,
            safety_margin=output.safety_margin,
        )


class ScoreBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    items: list[SystemOutputResponse]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    ready: bool
    service_name: str
    service_version: str
    regime_source_path: str
    regime_row_count: int | None = None
    finn_component_name: str
    startup_error: str | None = None


class MetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_name: str
    service_version: str
    regime_source_path: str
    regime_row_count: int | None = None
    regime_match_mode: Literal["exact", "previous"]
    regime_max_staleness_days: int | None = None
    regime_underlying_symbol: str | None = None
    transaction_cost_estimate: float
    safety_margin: float
    use_mid_price_if_available: bool
    baseline_component: str
    regime_component: str
    finn_component: str


def create_app(
    config: APIServerConfig | None = None,
    *,
    pipeline: HMMFINNPipeline | None = None,
    regime_row_count: int | None = None,
) -> FastAPI:
    """Create a FastAPI app backed by the current scoring pipeline."""

    resolved = config or APIServerConfig()
    runtime = _build_runtime(
        resolved,
        pipeline=pipeline,
        regime_row_count=regime_row_count,
    )

    app = FastAPI(
        title="Proyecto TAM API",
        version=resolved.service_version,
        description="Scoring API for the current HMM + FINN MVP pipeline.",
    )
    app.state.runtime = runtime

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current = _runtime(app)
        return HealthResponse(
            status="ok" if current.pipeline is not None else "degraded",
            ready=current.pipeline is not None,
            service_name=current.config.service_name,
            service_version=current.config.service_version,
            regime_source_path=str(current.config.regime_source_path),
            regime_row_count=current.regime_row_count,
            finn_component_name=current.finn_component_name,
            startup_error=current.startup_error,
        )

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata() -> MetadataResponse:
        current = _runtime(app)
        return MetadataResponse(
            service_name=current.config.service_name,
            service_version=current.config.service_version,
            regime_source_path=str(current.config.regime_source_path),
            regime_row_count=current.regime_row_count,
            regime_match_mode=current.config.regime_match_mode,
            regime_max_staleness_days=current.config.regime_max_staleness_days,
            regime_underlying_symbol=current.config.regime_underlying_symbol,
            transaction_cost_estimate=current.config.transaction_cost_estimate,
            safety_margin=current.config.safety_margin,
            use_mid_price_if_available=current.config.use_mid_price_if_available,
            baseline_component="BlackScholesPricer",
            regime_component="RegimeTableDetector",
            finn_component=current.finn_component_name,
        )

    @app.post("/score/row", response_model=SystemOutputResponse)
    def score_row(payload: DatasetRowRequest) -> SystemOutputResponse:
        scorer = _pipeline_or_503(app)
        row = _validated_row(payload)
        return SystemOutputResponse.from_contract(_score_row(scorer, row))

    @app.post("/score/batch", response_model=ScoreBatchResponse)
    def score_batch(payload: ScoreBatchRequest) -> ScoreBatchResponse:
        scorer = _pipeline_or_503(app)
        rows = [_validated_row(row) for row in payload.rows]
        outputs = _score_rows(scorer, rows)
        return ScoreBatchResponse(
            count=len(outputs),
            items=[SystemOutputResponse.from_contract(output) for output in outputs],
        )

    return app


def _runtime(app: FastAPI) -> APIRuntime:
    return app.state.runtime


def _pipeline_or_503(app: FastAPI) -> HMMFINNPipeline:
    current = _runtime(app)
    if current.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=current.startup_error or "Scoring pipeline is not available.",
        )
    return current.pipeline


def _validated_row(payload: DatasetRowRequest) -> DatasetRow:
    try:
        return payload.to_contract()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _score_row(pipeline: HMMFINNPipeline, row: DatasetRow) -> SystemOutput:
    try:
        return pipeline.score_row(row)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _score_rows(
    pipeline: HMMFINNPipeline,
    rows: list[DatasetRow],
) -> list[SystemOutput]:
    outputs: list[SystemOutput] = []
    for row in rows:
        outputs.append(_score_row(pipeline, row))
    return outputs


def _build_runtime(
    config: APIServerConfig,
    *,
    pipeline: HMMFINNPipeline | None = None,
    regime_row_count: int | None = None,
) -> APIRuntime:
    if pipeline is not None:
        return APIRuntime(
            config=config,
            pipeline=pipeline,
            regime_row_count=regime_row_count,
            finn_component_name=type(pipeline.finn_model).__name__,
        )

    try:
        if config.finn_checkpoint_path is not None:
            finn_model = FINNPricingModel.load(config.finn_checkpoint_path)
        else:
            finn_model = RegimeAdjustedBlackScholesFINN()

        detector = RegimeTableDetector(
            config.regime_source_path,
            config=RegimeTableDetectorConfig(
                underlying_symbol=config.regime_underlying_symbol,
                match_mode=config.regime_match_mode,
                max_staleness_days=config.regime_max_staleness_days,
            ),
        )
        return APIRuntime(
            config=config,
            pipeline=HMMFINNPipeline(
                baseline=BlackScholesPricer(),
                regime_detector=detector,
                finn_model=finn_model,
                config=PipelineConfig(
                    transaction_cost_estimate=config.transaction_cost_estimate,
                    safety_margin=config.safety_margin,
                    use_mid_price_if_available=config.use_mid_price_if_available,
                ),
            ),
            regime_row_count=len(detector.frame),
            finn_component_name=type(finn_model).__name__,
        )
    except Exception as exc:
        return APIRuntime(
            config=config,
            pipeline=None,
            regime_row_count=None,
            finn_component_name="unavailable",
            startup_error=str(exc),
        )
