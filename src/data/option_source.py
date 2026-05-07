"""Normalization utilities for raw option-chain sources."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

DEFAULT_OPTION_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "quote_timestamp", "quote_datetime", "quoteDate", "lastTradeDate"),
    "underlying_symbol": ("underlying_symbol", "underlying", "underlyingSymbol", "asset"),
    "option_symbol": ("option_symbol", "contractSymbol", "optionSymbol"),
    "option_type": ("option_type", "type", "right", "putCall", "cp_flag", "optionType"),
    "K": ("K", "strike", "strikePrice"),
    "T": ("T", "time_to_maturity", "time_to_maturity_years"),
    "expiration_date": ("expiration_date", "expiration", "expiry", "expiryDate", "expirationDate"),
    "market_price": ("market_price", "lastPrice", "mark", "markPrice", "optionPrice", "close"),
    "implied_volatility": ("implied_volatility", "impliedVolatility", "iv"),
    "bid": ("bid",),
    "ask": ("ask",),
    "mid_price": ("mid_price", "mid", "midPrice"),
    "volume": ("volume",),
    "open_interest": ("open_interest", "openInterest"),
    "dividend_yield": ("dividend_yield",),
}

OCC_OPTION_SYMBOL_RE = re.compile(
    r"^(?P<underlying>[A-Z]{1,6})(?P<expiration>\d{6})(?P<option_type>[CP])(?P<strike>\d{8})$"
)
CANONICAL_OPTION_SYMBOL_RE = re.compile(
    r"^(?P<underlying>[A-Z0-9._-]+)_(?P<expiration>\d{8})_(?P<strike>\d+(?:\.\d+)?)_(?P<option_type>C|P|CALL|PUT)$"
)


@dataclass(slots=True, frozen=True)
class OptionSourceConfig:
    underlying_symbol: str = "SPY"
    quote_timestamp_override: str | None = None
    day_count_basis: float = 365.0
    implied_volatility_percent_threshold: float = 3.0
    source_aliases: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: DEFAULT_OPTION_SOURCE_ALIASES.copy()
    )


class OptionSourceNormalizer:
    """Convert raw option-chain data into the project's canonical schema."""

    def __init__(self, config: OptionSourceConfig | None = None) -> None:
        self.config = config or OptionSourceConfig()

    def load_frame(self, source: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        return pd.read_csv(Path(source))

    def normalize(self, source: pd.DataFrame | str | Path) -> pd.DataFrame:
        frame = self._rename_aliases(self.load_frame(source))
        working = frame.copy()

        if self.config.quote_timestamp_override is not None:
            working["timestamp"] = self.config.quote_timestamp_override
        elif "timestamp" not in working.columns:
            raise ValueError(
                "Option source must provide a timestamp column or quote_timestamp_override."
            )

        symbol_details = self._parse_option_symbols(working.get("option_symbol"))

        if "underlying_symbol" not in working.columns:
            working["underlying_symbol"] = symbol_details["underlying_symbol"]
        else:
            working["underlying_symbol"] = working["underlying_symbol"].where(
                working["underlying_symbol"].notna(),
                symbol_details["underlying_symbol"],
            )
        working["underlying_symbol"] = working["underlying_symbol"].fillna(
            self.config.underlying_symbol
        )
        working["underlying_symbol"] = working["underlying_symbol"].astype(str).str.upper()

        if "option_type" not in working.columns:
            working["option_type"] = symbol_details["option_type"]
        else:
            working["option_type"] = working["option_type"].where(
                working["option_type"].notna(),
                symbol_details["option_type"],
            )
        working["option_type"] = working["option_type"].map(self._normalize_option_type)

        if "K" not in working.columns:
            working["K"] = symbol_details["K"]
        else:
            working["K"] = working["K"].where(working["K"].notna(), symbol_details["K"])

        if "expiration_date" not in working.columns:
            working["expiration_date"] = symbol_details["expiration_date"]
        else:
            working["expiration_date"] = working["expiration_date"].where(
                working["expiration_date"].notna(),
                symbol_details["expiration_date"],
            )

        working["timestamp"] = self._normalize_timestamp_series(working["timestamp"])
        if "expiration_date" in working.columns:
            working["expiration_date"] = self._normalize_date_series(working["expiration_date"])

        if "T" not in working.columns or working["T"].isna().any():
            if "expiration_date" not in working.columns:
                raise ValueError(
                    "Option source must provide T or expiration_date / parseable option_symbol."
                )
            working["T"] = self._time_to_maturity_from_dates(
                working["timestamp"],
                working["expiration_date"],
            )

        if "option_symbol" not in working.columns:
            working["option_symbol"] = self._build_canonical_option_symbol(working)
        else:
            missing_symbols = working["option_symbol"].isna()
            if missing_symbols.any():
                working.loc[missing_symbols, "option_symbol"] = self._build_canonical_option_symbol(
                    working.loc[missing_symbols]
                )

        numeric_columns = (
            "K",
            "T",
            "market_price",
            "implied_volatility",
            "bid",
            "ask",
            "mid_price",
            "volume",
            "open_interest",
            "dividend_yield",
        )
        for column in numeric_columns:
            if column in working.columns:
                working[column] = pd.to_numeric(working[column], errors="coerce")

        if "mid_price" not in working.columns and {"bid", "ask"}.issubset(working.columns):
            working["mid_price"] = (working["bid"] + working["ask"]) / 2.0
        elif {"bid", "ask"}.issubset(working.columns):
            computed_mid = (working["bid"] + working["ask"]) / 2.0
            working["mid_price"] = working["mid_price"].where(
                working["mid_price"].notna(),
                computed_mid,
            )

        if "market_price" not in working.columns and "mid_price" in working.columns:
            working["market_price"] = working["mid_price"]
        elif "market_price" in working.columns and "mid_price" in working.columns:
            working["market_price"] = working["market_price"].where(
                working["market_price"].notna(),
                working["mid_price"],
            )

        if "implied_volatility" in working.columns:
            working["implied_volatility"] = self._normalize_implied_volatility_series(
                working["implied_volatility"]
            )

        if "dividend_yield" in working.columns:
            working["dividend_yield"] = working["dividend_yield"].fillna(0.0)

        canonical_order = [
            "timestamp",
            "underlying_symbol",
            "option_symbol",
            "option_type",
            "K",
            "T",
            "market_price",
            "implied_volatility",
            "bid",
            "ask",
            "mid_price",
            "volume",
            "open_interest",
            "dividend_yield",
            "expiration_date",
        ]
        existing = [column for column in canonical_order if column in working.columns]
        others = [column for column in working.columns if column not in existing]
        return working[existing + others].reset_index(drop=True)

    def _rename_aliases(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        for canonical_name, aliases in self.config.source_aliases.items():
            if canonical_name in working.columns:
                continue
            for alias in aliases:
                if alias in working.columns:
                    working = working.rename(columns={alias: canonical_name})
                    break
        return working

    def _parse_option_symbols(self, symbols: pd.Series | None) -> pd.DataFrame:
        if symbols is None:
            return pd.DataFrame(
                {
                    "underlying_symbol": pd.Series(dtype="object"),
                    "option_type": pd.Series(dtype="object"),
                    "K": pd.Series(dtype="float64"),
                    "expiration_date": pd.Series(dtype="object"),
                }
            )

        details = []
        for symbol in symbols.fillna("").astype(str):
            parsed = self._parse_option_symbol(symbol)
            details.append(
                {
                    "underlying_symbol": parsed.get("underlying_symbol"),
                    "option_type": parsed.get("option_type"),
                    "K": parsed.get("K"),
                    "expiration_date": parsed.get("expiration_date"),
                }
            )
        return pd.DataFrame(details, index=symbols.index)

    def _parse_option_symbol(self, symbol: str) -> dict[str, object]:
        cleaned = symbol.strip().upper()
        if not cleaned:
            return {}

        occ_match = OCC_OPTION_SYMBOL_RE.match(cleaned)
        if occ_match:
            expiration_raw = occ_match.group("expiration")
            expiration_date = pd.to_datetime(expiration_raw, format="%y%m%d", utc=True).strftime(
                "%Y-%m-%d"
            )
            strike = int(occ_match.group("strike")) / 1000.0
            option_type = "call" if occ_match.group("option_type") == "C" else "put"
            return {
                "underlying_symbol": occ_match.group("underlying"),
                "option_type": option_type,
                "K": strike,
                "expiration_date": expiration_date,
            }

        canonical_match = CANONICAL_OPTION_SYMBOL_RE.match(cleaned)
        if canonical_match:
            expiration_date = pd.to_datetime(
                canonical_match.group("expiration"),
                format="%Y%m%d",
                utc=True,
            ).strftime("%Y-%m-%d")
            option_type = canonical_match.group("option_type")
            return {
                "underlying_symbol": canonical_match.group("underlying"),
                "option_type": self._normalize_option_type(option_type),
                "K": float(canonical_match.group("strike")),
                "expiration_date": expiration_date,
            }

        return {}

    def _build_canonical_option_symbol(self, frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype="object")

        missing = [
            column
            for column in ("underlying_symbol", "expiration_date", "K", "option_type")
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                "Cannot build canonical option_symbol because these columns are missing: "
                f"{missing}"
            )

        symbols = []
        for row in frame.itertuples(index=False):
            expiration = pd.to_datetime(row.expiration_date, utc=True).strftime("%Y%m%d")
            strike = self._format_strike(float(row.K))
            option_type = "C" if str(row.option_type).lower() == "call" else "P"
            symbols.append(f"{str(row.underlying_symbol).upper()}_{expiration}_{strike}_{option_type}")
        return pd.Series(symbols, index=frame.index, dtype="object")

    def _time_to_maturity_from_dates(
        self,
        timestamps: pd.Series,
        expiration_dates: pd.Series,
    ) -> pd.Series:
        quote = pd.to_datetime(timestamps, errors="coerce", utc=True)
        expiry = pd.to_datetime(expiration_dates, errors="coerce", utc=True)
        delta_days = (expiry - quote).dt.total_seconds() / (24.0 * 60.0 * 60.0)
        return delta_days / self.config.day_count_basis

    def _normalize_implied_volatility_series(self, values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        threshold = self.config.implied_volatility_percent_threshold
        mask = numeric.abs() > threshold
        numeric.loc[mask] = numeric.loc[mask] / 100.0
        return numeric

    def _normalize_option_type(self, value: object) -> str | pd.NA:
        if value is None or pd.isna(value):
            return pd.NA
        normalized = str(value).strip().lower()
        if normalized in {"call", "c"}:
            return "call"
        if normalized in {"put", "p"}:
            return "put"
        raise ValueError(f"Unsupported option_type value: {value!r}")

    def _normalize_timestamp_series(self, values: pd.Series | object) -> pd.Series:
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        parsed = pd.to_datetime(values, errors="coerce", utc=True).dt.normalize()
        return parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _normalize_date_series(self, values: pd.Series) -> pd.Series:
        return pd.to_datetime(values, errors="coerce", utc=True).dt.strftime("%Y-%m-%d")

    def _format_strike(self, strike: float) -> str:
        if math.isclose(strike, round(strike), rel_tol=0.0, abs_tol=1e-9):
            return str(int(round(strike)))
        return np.format_float_positional(strike, trim="-")


def normalize_option_source(
    source: pd.DataFrame | str | Path,
    config: OptionSourceConfig | None = None,
) -> pd.DataFrame:
    """Normalize a raw option-chain source into the canonical project schema."""

    return OptionSourceNormalizer(config=config).normalize(source)
