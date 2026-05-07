"""Batch utilities for scoring option rows with baseline, regimes and FINN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

from src.baseline import BlackScholesPricer
from src.contracts import (
    REGIME_LABEL_ORDER,
    REGIME_PROBABILITY_COLUMNS,
    SystemOutput,
)
from src.data import DatasetBuilder
from src.models.finn import RegimeAdjustedBlackScholesFINN
from src.models.hmm import RegimeTableDetector, RegimeTableDetectorConfig
from src.pipeline.hmm_finn_pipeline import HMMFINNPipeline, PipelineConfig


@dataclass(slots=True, frozen=True)
class OptionScoringConfig:
    option_source_path: Path
    market_source_path: Path = Path("Data/final/master_dataset_2010_present_long.csv")
    regime_source_path: Path = Path("Data/processed/spy_regime_features.csv")
    output_path: Path = Path("Data/processed/scored_option_dataset.csv")
    transaction_cost_estimate: float = 0.0
    safety_margin: float = 0.0
    use_mid_price_if_available: bool = True
    regime_match_mode: Literal["exact", "previous"] = "exact"
    regime_max_staleness_days: int | None = None
    regime_underlying_symbol: str | None = "SPY"


def build_option_scoring_pipeline(
    regime_source: pd.DataFrame | str | Path,
    *,
    config: OptionScoringConfig | None = None,
    baseline: BlackScholesPricer | None = None,
    regime_detector: RegimeTableDetector | None = None,
    finn_model: RegimeAdjustedBlackScholesFINN | None = None,
) -> HMMFINNPipeline:
    """Build the default scoring pipeline used for batch inference."""

    resolved = config or OptionScoringConfig(option_source_path=Path("option_source.csv"))
    detector = regime_detector or RegimeTableDetector(
        regime_source,
        config=RegimeTableDetectorConfig(
            underlying_symbol=resolved.regime_underlying_symbol,
            match_mode=resolved.regime_match_mode,
            max_staleness_days=resolved.regime_max_staleness_days,
        ),
    )
    return HMMFINNPipeline(
        baseline=baseline or BlackScholesPricer(),
        regime_detector=detector,
        finn_model=finn_model or RegimeAdjustedBlackScholesFINN(),
        config=PipelineConfig(
            transaction_cost_estimate=resolved.transaction_cost_estimate,
            safety_margin=resolved.safety_margin,
            use_mid_price_if_available=resolved.use_mid_price_if_available,
        ),
    )


def score_option_rows(
    option_source: pd.DataFrame | str | Path,
    market_source: pd.DataFrame | str | Path,
    regime_source: pd.DataFrame | str | Path,
    *,
    dataset_builder: DatasetBuilder | None = None,
    pipeline: HMMFINNPipeline | None = None,
    config: OptionScoringConfig | None = None,
) -> list[SystemOutput]:
    """Score option rows end-to-end from raw option and market sources."""

    builder = dataset_builder or DatasetBuilder()
    option_frame = builder.build_option_training_dataset(option_source, market_source)
    rows = builder.frame_to_dataset_rows(option_frame)
    scorer = pipeline or build_option_scoring_pipeline(
        regime_source,
        config=config,
    )
    return scorer.score_rows(rows)


def system_outputs_to_frame(outputs: Sequence[SystemOutput]) -> pd.DataFrame:
    """Convert scored outputs into a dataframe with named regime probability columns."""

    rows: list[dict[str, object]] = []
    for output in outputs:
        row = {
            "timestamp": output.timestamp,
            "underlying_symbol": output.underlying_symbol,
            "option_symbol": output.option_symbol,
            "market_price": output.market_price,
            "bs_price": output.bs_price,
            "fair_value": output.fair_value,
            "edge": output.edge,
            "delta": output.delta,
            "gamma": output.gamma,
            "regime_label": output.regime_label,
            "regime_probabilities": output.regime_probabilities,
            "signal": output.signal,
            "transaction_cost_estimate": output.transaction_cost_estimate,
            "safety_margin": output.safety_margin,
        }
        for index, label in enumerate(REGIME_LABEL_ORDER):
            probability = output.regime_probabilities[index]
            row[f"regime_probability_{label}"] = probability
        rows.append(row)

    if not rows:
        columns = [
            "timestamp",
            "underlying_symbol",
            "option_symbol",
            "market_price",
            "bs_price",
            "fair_value",
            "edge",
            "delta",
            "gamma",
            "regime_label",
            "regime_probabilities",
            "signal",
            "transaction_cost_estimate",
            "safety_margin",
            *REGIME_PROBABILITY_COLUMNS,
        ]
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows)


def save_system_output_frame(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist system outputs as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_and_save_option_scores(
    config: OptionScoringConfig,
    *,
    dataset_builder: DatasetBuilder | None = None,
    pipeline: HMMFINNPipeline | None = None,
) -> pd.DataFrame:
    """Build scored option outputs from configured sources and save them."""

    outputs = score_option_rows(
        option_source=config.option_source_path,
        market_source=config.market_source_path,
        regime_source=config.regime_source_path,
        dataset_builder=dataset_builder,
        pipeline=pipeline,
        config=config,
    )
    frame = system_outputs_to_frame(outputs)
    save_system_output_frame(frame, config.output_path)
    return frame
