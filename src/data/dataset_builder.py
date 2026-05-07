"""Dataset assembly utilities that bridge market context and option rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.contracts import DatasetRow, HMMFeatures, HMMInput
from src.data.option_source import OptionSourceConfig, OptionSourceNormalizer

CURRENT_MARKET_REQUIRED_COLUMNS = (
    "date",
    "asset",
    "ticker",
    "price_used",
    "volume",
    "return_1d",
    "return_5d",
    "volatility_20d_ann",
    "vix_fred",
    "dgs2",
)

OPTION_SOURCE_BASE_COLUMNS = (
    "timestamp",
    "underlying_symbol",
    "option_symbol",
    "option_type",
    "K",
    "T",
)

MVP_REQUIRED_COLUMNS = (
    "timestamp",
    "underlying_symbol",
    "option_symbol",
    "option_type",
    "S",
    "K",
    "T",
    "r",
    "market_price",
    "implied_volatility",
)

MVP_RECOMMENDED_COLUMNS = (
    "bid",
    "ask",
    "mid_price",
    "volume",
    "open_interest",
    "realized_volatility",
    "return_1d",
    "return_5d",
    "sentiment_score",
    "text_embedding",
    "event_count",
    "dividend_yield",
)


@dataclass(slots=True, frozen=True)
class DatasetBuilderConfig:
    underlying_symbol: str = "SPY"
    option_type: str = "call"
    min_time_to_maturity_years: float = 20.0 / 365.0
    max_time_to_maturity_years: float = 45.0 / 365.0
    min_moneyness: float = 0.90
    max_moneyness: float = 1.10
    min_volume: float = 10.0
    max_relative_spread: float = 0.25
    required_columns: tuple[str, ...] = MVP_REQUIRED_COLUMNS
    market_required_columns: tuple[str, ...] = CURRENT_MARKET_REQUIRED_COLUMNS
    option_source_base_columns: tuple[str, ...] = OPTION_SOURCE_BASE_COLUMNS
    market_date_column: str = "date"
    market_asset_column: str = "asset"
    market_ticker_column: str = "ticker"
    market_price_column: str = "price_used"
    market_volume_column: str = "volume"
    market_return_1d_column: str = "return_1d"
    market_return_5d_column: str = "return_5d"
    market_realized_volatility_column: str = "volatility_20d_ann"
    market_implied_volatility_proxy_column: str = "vix_fred"
    market_implied_volatility_scale: float = 0.01
    market_risk_free_rate_column: str = "dgs2"
    market_risk_free_rate_scale: float = 0.01
    market_sentiment_column: str | None = None
    market_dividend_yield: float = 0.0
    normalize_raw_option_source: bool = True
    option_quote_timestamp_override: str | None = None
    option_implied_volatility_percent_threshold: float = 3.0


def validate_dataset_row(row: DatasetRow) -> list[str]:
    """Return a list of contract violations for one dataset row."""

    issues: list[str] = []
    if not row.timestamp:
        issues.append("timestamp is required")
    if not row.option_symbol:
        issues.append("option_symbol is required")
    if row.S <= 0:
        issues.append("S must be > 0")
    if row.K <= 0:
        issues.append("K must be > 0")
    if row.T <= 0:
        issues.append("T must be > 0")
    if row.market_price <= 0:
        issues.append("market_price must be > 0")
    if row.implied_volatility <= 0:
        issues.append("implied_volatility must be > 0")
    if row.option_type not in {"call", "put"}:
        issues.append("option_type must be 'call' or 'put'")
    if row.bid is not None and row.ask is not None and row.bid > row.ask:
        issues.append("bid must be <= ask")
    if row.mid_price is not None and row.mid_price <= 0:
        issues.append("mid_price must be > 0 when present")
    return issues


class DatasetBuilder:
    """Bridge the current multimodal market dataset with the option pricing scaffold."""

    def __init__(self, config: DatasetBuilderConfig | None = None) -> None:
        self.config = config or DatasetBuilderConfig()

    def missing_required_columns(self, columns: Iterable[str]) -> list[str]:
        available = set(columns)
        return [name for name in self.config.required_columns if name not in available]

    def missing_market_source_columns(self, columns: Iterable[str]) -> list[str]:
        available = set(columns)
        return [name for name in self.config.market_required_columns if name not in available]

    def missing_option_source_base_columns(self, columns: Iterable[str]) -> list[str]:
        available = set(columns)
        return [name for name in self.config.option_source_base_columns if name not in available]

    def load_frame(self, source: pd.DataFrame | str | Path) -> pd.DataFrame:
        """Load a dataframe either from memory or from a CSV file path."""

        if isinstance(source, pd.DataFrame):
            return source.copy()
        path = Path(source)
        return pd.read_csv(path)

    def build_market_context_frame(self, source: pd.DataFrame | str | Path) -> pd.DataFrame:
        """Build a normalized SPY market context frame from the current multimodal dataset."""

        raw = self.load_frame(source)
        missing = self.missing_market_source_columns(raw.columns)
        if missing:
            raise ValueError(f"Market source is missing required columns: {missing}")

        filtered = raw.loc[
            raw[self.config.market_asset_column] == self.config.underlying_symbol
        ].copy()
        if filtered.empty:
            raise ValueError(
                f"No rows found for underlying_symbol={self.config.underlying_symbol!r} "
                "in the market source."
            )

        filtered["timestamp"] = self._normalize_timestamp_series(
            filtered[self.config.market_date_column]
        )
        context = pd.DataFrame(
            {
                "timestamp": filtered["timestamp"],
                "underlying_symbol": filtered[self.config.market_asset_column],
                "ticker": filtered[self.config.market_ticker_column],
                "S": filtered[self.config.market_price_column],
                "market_volume": filtered[self.config.market_volume_column],
                "return_1d": filtered[self.config.market_return_1d_column],
                "return_5d": filtered[self.config.market_return_5d_column],
                "realized_volatility": filtered[self.config.market_realized_volatility_column],
                "implied_volatility_proxy": (
                    filtered[self.config.market_implied_volatility_proxy_column]
                    * self.config.market_implied_volatility_scale
                ),
                "r": (
                    filtered[self.config.market_risk_free_rate_column]
                    * self.config.market_risk_free_rate_scale
                ),
                "dividend_yield": self.config.market_dividend_yield,
            }
        )
        if self.config.market_sentiment_column and self.config.market_sentiment_column in filtered.columns:
            context["sentiment_score"] = filtered[self.config.market_sentiment_column]
        else:
            context["sentiment_score"] = pd.NA

        context = context.sort_values("timestamp").reset_index(drop=True)
        return context

    def build_hmm_inputs(self, source: pd.DataFrame | str | Path) -> list[HMMInput]:
        """Convert the multimodal market dataset into HMM input objects."""

        context = self.build_market_context_frame(source)
        samples: list[HMMInput] = []
        for row in context.itertuples(index=False):
            samples.append(
                HMMInput(
                    timestamp=row.timestamp,
                    features=HMMFeatures(
                        return_1d=self._optional_float(row.return_1d),
                        return_5d=self._optional_float(row.return_5d),
                        realized_volatility=self._optional_float(row.realized_volatility),
                        implied_volatility=self._optional_float(row.implied_volatility_proxy),
                        volume=self._optional_float(row.market_volume),
                        sentiment_score=self._optional_float(row.sentiment_score),
                    ),
                )
            )
        return samples

    def build_option_training_dataset(
        self,
        option_source: pd.DataFrame | str | Path,
        market_source: pd.DataFrame | str | Path,
    ) -> pd.DataFrame:
        """Enrich option rows with SPY market context from the current multimodal dataset."""

        option_df = self._load_option_source_frame(option_source)
        missing = self.missing_option_source_base_columns(option_df.columns)
        if missing:
            raise ValueError(f"Option source is missing required columns: {missing}")

        if "market_price" not in option_df.columns and "mid_price" in option_df.columns:
            option_df["market_price"] = option_df["mid_price"]

        context = self.build_market_context_frame(market_source)
        merged = option_df.merge(
            context,
            on=["timestamp", "underlying_symbol"],
            how="left",
            validate="many_to_one",
        )

        self._fill_column_from_context(merged, "S", "S")
        if merged["S"].isna().all():
            raise ValueError("Unable to enrich option rows with underlying spot price S.")

        self._fill_column_from_context(merged, "return_1d", "return_1d")
        self._fill_column_from_context(merged, "return_5d", "return_5d")
        self._fill_column_from_context(merged, "realized_volatility", "realized_volatility")
        self._fill_column_from_context(merged, "r", "r")
        self._fill_column_from_context(merged, "dividend_yield", "dividend_yield")
        self._fill_column_from_context(merged, "sentiment_score", "sentiment_score")
        self._fill_column_from_context(merged, "implied_volatility", "implied_volatility_proxy")

        merged["option_type"] = merged["option_type"].astype(str).str.lower()
        if "mid_price" in merged.columns and "market_price" in merged.columns:
            merged["market_price"] = merged["market_price"].fillna(merged["mid_price"])
        self._derive_option_metrics(merged)
        merged = self.filter_option_training_dataset(merged)
        if merged.empty:
            raise ValueError("No option rows remain after applying the MVP filters.")

        missing_final = self.missing_required_columns(merged.columns)
        if missing_final:
            raise ValueError(
                "Final option training dataset is missing required columns after enrichment: "
                f"{missing_final}"
            )

        validation_issues = self.validate_frame(merged)
        if validation_issues:
            raise ValueError(
                "Option training dataset contains invalid rows after enrichment: "
                f"{validation_issues[:5]}"
            )

        drop_columns = ["ticker", "market_volume", "implied_volatility_proxy"]
        return merged.drop(
            columns=[name for name in drop_columns if name in merged.columns]
        ).reset_index(drop=True)

    def filter_option_training_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the MVP option-universe filters on an enriched option dataset."""

        filtered = frame.copy()
        filtered = filtered.loc[
            filtered["underlying_symbol"].astype(str).str.upper() == self.config.underlying_symbol
        ]
        filtered = filtered.loc[
            filtered["option_type"].astype(str).str.lower() == self.config.option_type
        ]
        filtered = filtered.loc[
            filtered["T"].between(
                self.config.min_time_to_maturity_years,
                self.config.max_time_to_maturity_years,
                inclusive="both",
            )
        ]

        if "moneyness" in filtered.columns:
            filtered = filtered.loc[
                filtered["moneyness"].between(
                    self.config.min_moneyness,
                    self.config.max_moneyness,
                    inclusive="both",
                )
            ]

        if "volume" in filtered.columns:
            filtered = filtered.loc[
                filtered["volume"].isna() | (filtered["volume"] >= self.config.min_volume)
            ]

        if "relative_spread" in filtered.columns:
            filtered = filtered.loc[
                filtered["relative_spread"].isna()
                | (filtered["relative_spread"] <= self.config.max_relative_spread)
            ]

        return filtered.reset_index(drop=True)

    def validate_frame(self, frame: pd.DataFrame) -> list[tuple[int, list[str]]]:
        """Validate a dataframe against the DatasetRow contract."""

        issues: list[tuple[int, list[str]]] = []
        for index, row in frame.iterrows():
            payload = self._dataset_row_from_mapping(row)
            row_issues = validate_dataset_row(payload)
            if row_issues:
                issues.append((index, row_issues))
        return issues

    def frame_to_dataset_rows(self, frame: pd.DataFrame) -> list[DatasetRow]:
        """Convert a validated dataframe into DatasetRow objects."""

        missing = self.missing_required_columns(frame.columns)
        if missing:
            raise ValueError(f"Dataset frame is missing required columns: {missing}")

        validation_issues = self.validate_frame(frame)
        if validation_issues:
            raise ValueError(f"Dataset frame contains invalid rows: {validation_issues[:5]}")

        return [
            self._dataset_row_from_mapping(row)
            for _, row in frame.reset_index(drop=True).iterrows()
        ]

    def build(self, source: object) -> pd.DataFrame:
        """Build the bridge dataset from market data only or market + option sources.

        Supported forms:
        - market_source
        - {"market_source": market_source}
        - {"market_source": market_source, "option_source": option_source}
        """

        if isinstance(source, Mapping):
            if "market_source" not in source:
                raise ValueError("Mapping source must contain a 'market_source' entry.")
            market_source = source["market_source"]
            option_source = source.get("option_source")
            if option_source is None:
                return self.build_market_context_frame(market_source)
            return self.build_option_training_dataset(option_source, market_source)

        return self.build_market_context_frame(source)

    def _normalize_option_source(self, option_df: pd.DataFrame) -> pd.DataFrame:
        working = option_df.copy()
        if "timestamp" not in working.columns and "date" in working.columns:
            working = working.rename(columns={"date": "timestamp"})
        if "underlying_symbol" not in working.columns and "asset" in working.columns:
            working = working.rename(columns={"asset": "underlying_symbol"})
        if "timestamp" in working.columns:
            working["timestamp"] = self._normalize_timestamp_series(working["timestamp"])
        if "underlying_symbol" in working.columns:
            working["underlying_symbol"] = working["underlying_symbol"].astype(str).str.upper()
        return working

    def _load_option_source_frame(
        self,
        source: pd.DataFrame | str | Path,
    ) -> pd.DataFrame:
        if self.config.normalize_raw_option_source:
            return OptionSourceNormalizer(
                OptionSourceConfig(
                    underlying_symbol=self.config.underlying_symbol,
                    quote_timestamp_override=self.config.option_quote_timestamp_override,
                    implied_volatility_percent_threshold=(
                        self.config.option_implied_volatility_percent_threshold
                    ),
                )
            ).normalize(source)

        option_df = self.load_frame(source)
        return self._normalize_option_source(option_df)

    def _fill_column_from_context(
        self,
        frame: pd.DataFrame,
        target_column: str,
        context_column: str,
    ) -> None:
        if context_column not in frame.columns:
            return
        if target_column not in frame.columns:
            frame[target_column] = frame[context_column]
            return
        frame[target_column] = frame[target_column].where(
            frame[target_column].notna(),
            frame[context_column],
        )

    def _derive_option_metrics(self, frame: pd.DataFrame) -> None:
        if "mid_price" not in frame.columns and {"bid", "ask"}.issubset(frame.columns):
            frame["mid_price"] = (frame["bid"] + frame["ask"]) / 2.0
        elif {"bid", "ask"}.issubset(frame.columns):
            computed_mid = (frame["bid"] + frame["ask"]) / 2.0
            frame["mid_price"] = frame["mid_price"].where(frame["mid_price"].notna(), computed_mid)

        if {"bid", "ask"}.issubset(frame.columns):
            frame["spread"] = frame["ask"] - frame["bid"]
        if {"spread", "mid_price"}.issubset(frame.columns):
            denominator = frame["mid_price"].where(frame["mid_price"] > 0)
            frame["relative_spread"] = frame["spread"] / denominator
        if {"S", "K"}.issubset(frame.columns):
            frame["moneyness"] = frame["S"] / frame["K"]
            frame["log_moneyness"] = np.log(frame["moneyness"].where(frame["moneyness"] > 0))

    def _dataset_row_from_mapping(self, row: Mapping[str, object]) -> DatasetRow:
        return DatasetRow(
            timestamp=str(row["timestamp"]),
            underlying_symbol=str(row["underlying_symbol"]),
            option_symbol=str(row["option_symbol"]),
            option_type=str(row["option_type"]).lower(),  # type: ignore[arg-type]
            S=float(row["S"]),
            K=float(row["K"]),
            T=float(row["T"]),
            r=float(row["r"]),
            market_price=float(row["market_price"]),
            implied_volatility=float(row["implied_volatility"]),
            bid=self._maybe_float(row.get("bid")),
            ask=self._maybe_float(row.get("ask")),
            mid_price=self._maybe_float(row.get("mid_price")),
            volume=self._maybe_float(row.get("volume")),
            open_interest=self._maybe_float(row.get("open_interest")),
            realized_volatility=self._maybe_float(row.get("realized_volatility")),
            return_1d=self._maybe_float(row.get("return_1d")),
            return_5d=self._maybe_float(row.get("return_5d")),
            sentiment_score=self._maybe_float(row.get("sentiment_score")),
            event_count=self._maybe_int(row.get("event_count")),
            dividend_yield=self._maybe_float(row.get("dividend_yield")),
        )

    def _normalize_timestamp_series(self, values: pd.Series) -> pd.Series:
        return pd.to_datetime(values, errors="coerce", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _optional_float(self, value: object) -> float | None:
        if pd.isna(value):
            return None
        return float(value)

    def _maybe_float(self, value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _maybe_int(self, value: object) -> int | None:
        if value is None or pd.isna(value):
            return None
        return int(value)
