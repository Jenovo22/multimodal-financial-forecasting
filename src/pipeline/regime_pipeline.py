"""Utilities for fitting HMM regimes and exporting a date-level regime table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.contracts import HMMInput, HMMOutput
from src.data import DatasetBuilder
from src.models.hmm import HMMRegimeDetector


@dataclass(slots=True, frozen=True)
class RegimeTableConfig:
    market_source_path: Path = Path("Data/final/master_dataset_2010_present_long.csv")
    output_path: Path = Path("Data/processed/spy_regime_features.csv")
    fit_end_timestamp: str | None = None
    include_market_context: bool = True


def build_regime_table(
    market_source: pd.DataFrame | str | Path,
    *,
    dataset_builder: DatasetBuilder | None = None,
    detector: HMMRegimeDetector | None = None,
    fit_end_timestamp: str | None = None,
    include_market_context: bool = True,
) -> pd.DataFrame:
    """Fit the HMM and return one regime row per SPY market date."""

    builder = dataset_builder or DatasetBuilder()
    regime_detector = detector or HMMRegimeDetector()
    context = builder.build_market_context_frame(market_source)
    samples = builder.build_hmm_inputs(market_source)
    fit_samples = _select_fit_samples(samples, fit_end_timestamp)

    regime_detector.fit(fit_samples)
    outputs = regime_detector.predict_many(samples)
    regime_frame = _outputs_to_frame(
        samples=samples,
        outputs=outputs,
        detector=regime_detector,
    )

    if include_market_context:
        market_columns = [
            "timestamp",
            "underlying_symbol",
            "S",
            "return_1d",
            "return_5d",
            "realized_volatility",
            "implied_volatility_proxy",
            "r",
        ]
        available_market_columns = [name for name in market_columns if name in context.columns]
        regime_frame = context[available_market_columns].merge(
            regime_frame,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )

    return regime_frame.sort_values("timestamp").reset_index(drop=True)


def save_regime_table(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist a regime table as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_and_save_regime_table(config: RegimeTableConfig | None = None) -> pd.DataFrame:
    """Build the default SPY regime table and write it to disk."""

    resolved = config or RegimeTableConfig()
    frame = build_regime_table(
        resolved.market_source_path,
        fit_end_timestamp=resolved.fit_end_timestamp,
        include_market_context=resolved.include_market_context,
    )
    save_regime_table(frame, resolved.output_path)
    return frame


def _select_fit_samples(
    samples: Sequence[HMMInput],
    fit_end_timestamp: str | None,
) -> list[HMMInput]:
    if fit_end_timestamp is None:
        return list(samples)

    fit_end = pd.to_datetime(fit_end_timestamp, errors="raise", utc=True)
    selected = [
        sample
        for sample in samples
        if pd.to_datetime(sample.timestamp, errors="coerce", utc=True) <= fit_end
    ]
    if not selected:
        raise ValueError("No HMM samples are available at or before fit_end_timestamp.")
    return selected


def _outputs_to_frame(
    *,
    samples: Sequence[HMMInput],
    outputs: Sequence[HMMOutput],
    detector: HMMRegimeDetector,
) -> pd.DataFrame:
    if len(samples) != len(outputs):
        raise ValueError("samples and outputs must have the same length.")

    rows: list[dict[str, object]] = []
    for sample, output in zip(samples, outputs):
        row: dict[str, object] = {
            "timestamp": sample.timestamp,
            "regime_label": output.regime_label,
            "sigma_regime": output.sigma_regime,
        }
        row.update(_probabilities_by_label(output.regime_probabilities, detector))
        rows.append(row)

    return pd.DataFrame(rows)


def _probabilities_by_label(
    probabilities: tuple[float, ...],
    detector: HMMRegimeDetector,
) -> dict[str, float]:
    labels = list(detector.config.labels)
    return {
        f"regime_probability_{label}": float(probabilities[index])
        if index < len(probabilities)
        else 0.0
        for index, label in enumerate(labels)
    }
