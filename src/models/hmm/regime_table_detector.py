"""HMM-compatible detector backed by a persisted regime table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src.contracts import (
    HMMInput,
    HMMOutput,
    REGIME_PROBABILITY_COLUMNS,
    RegimeLabel,
)


@dataclass(slots=True, frozen=True)
class RegimeTableDetectorConfig:
    timestamp_column: str = "timestamp"
    underlying_symbol_column: str = "underlying_symbol"
    underlying_symbol: str | None = None
    regime_label_column: str = "regime_label"
    sigma_regime_column: str = "sigma_regime"
    probability_columns: tuple[str, str, str] = REGIME_PROBABILITY_COLUMNS
    match_mode: Literal["exact", "previous"] = "exact"
    max_staleness_days: int | None = None


class RegimeTableDetector:
    """Serve HMMOutput objects from a precomputed date-level regime table."""

    def __init__(
        self,
        source: pd.DataFrame | str | Path,
        config: RegimeTableDetectorConfig | None = None,
    ) -> None:
        self.config = config or RegimeTableDetectorConfig()
        self.frame = self._prepare_frame(self._load_frame(source))

    def predict(self, payload: HMMInput) -> HMMOutput:
        row = self._match_row(payload.timestamp)
        return HMMOutput(
            regime_label=self._regime_label(row[self.config.regime_label_column]),
            regime_probabilities=tuple(
                float(row[column]) for column in self.config.probability_columns
            ),
            sigma_regime=float(row[self.config.sigma_regime_column]),
        )

    def predict_many(self, payloads: list[HMMInput]) -> list[HMMOutput]:
        return [self.predict(payload) for payload in payloads]

    def _load_frame(self, source: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        return pd.read_csv(Path(source))

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = self._missing_required_columns(frame)
        if missing:
            raise ValueError(f"Regime table is missing required columns: {missing}")

        working = frame.copy()
        if self.config.underlying_symbol_column in working.columns:
            working[self.config.underlying_symbol_column] = (
                working[self.config.underlying_symbol_column].astype(str).str.upper()
            )
            if self.config.underlying_symbol is not None:
                symbol = self.config.underlying_symbol.upper()
                working = working.loc[
                    working[self.config.underlying_symbol_column] == symbol
                ].copy()
                if working.empty:
                    raise ValueError(f"No regime rows found for underlying_symbol={symbol!r}.")
            elif working[self.config.underlying_symbol_column].nunique() > 1:
                raise ValueError(
                    "Regime table has multiple underlyings; configure underlying_symbol."
                )

        working[self.config.timestamp_column] = self._normalize_timestamp_series(
            working[self.config.timestamp_column]
        )
        if working[self.config.timestamp_column].isna().any():
            raise ValueError("Regime table contains timestamps that cannot be parsed.")

        duplicate_mask = working.duplicated(subset=[self.config.timestamp_column], keep=False)
        if duplicate_mask.any():
            raise ValueError("Regime table contains duplicate timestamps after filtering.")

        probability_sums = working[list(self.config.probability_columns)].sum(axis=1)
        invalid_probabilities = (
            working[list(self.config.probability_columns)].lt(0).any(axis=1)
            | working[list(self.config.probability_columns)].gt(1).any(axis=1)
            | (probability_sums - 1.0).abs().gt(1e-6)
        )
        if invalid_probabilities.any():
            raise ValueError("Regime table probabilities must be in [0, 1] and sum to 1.")

        if working[self.config.sigma_regime_column].le(0).any():
            raise ValueError("Regime table sigma_regime values must be positive.")

        return working.sort_values(self.config.timestamp_column).reset_index(drop=True)

    def _missing_required_columns(self, frame: pd.DataFrame) -> list[str]:
        required = [
            self.config.timestamp_column,
            self.config.regime_label_column,
            self.config.sigma_regime_column,
            *self.config.probability_columns,
        ]
        return [column for column in required if column not in frame.columns]

    def _match_row(self, timestamp: str) -> pd.Series:
        normalized = self._normalize_timestamp(timestamp)
        timestamp_column = self.config.timestamp_column

        if self.config.match_mode == "exact":
            matches = self.frame.loc[self.frame[timestamp_column] == normalized]
            if matches.empty:
                raise KeyError(f"No regime row found for timestamp={normalized!r}.")
            return matches.iloc[0]

        target = pd.to_datetime(normalized, utc=True)
        timestamps = pd.to_datetime(self.frame[timestamp_column], utc=True)
        candidates = self.frame.loc[timestamps <= target]
        if candidates.empty:
            raise KeyError(f"No previous regime row found for timestamp={normalized!r}.")

        row = candidates.iloc[-1]
        if self.config.max_staleness_days is not None:
            row_timestamp = pd.to_datetime(row[timestamp_column], utc=True)
            staleness_days = (target - row_timestamp).days
            if staleness_days > self.config.max_staleness_days:
                raise KeyError(
                    "Previous regime row is older than max_staleness_days "
                    f"for timestamp={normalized!r}."
                )
        return row

    def _regime_label(self, value: object) -> RegimeLabel:
        label = str(value)
        if label in {
            "stable_low_volatility",
            "stress_high_volatility",
            "transition_uncertainty",
            "unknown",
        }:
            return label  # type: ignore[return-value]
        raise ValueError(f"Invalid regime_label={label!r}.")

    def _normalize_timestamp(self, value: object) -> str:
        parsed = pd.to_datetime(value, errors="raise", utc=True)
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _normalize_timestamp_series(self, values: pd.Series) -> pd.Series:
        return pd.to_datetime(values, errors="coerce", utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
