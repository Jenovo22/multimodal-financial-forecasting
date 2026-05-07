"""Download option chains into the raw schema consumed by the dataset builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_MARKET_SOURCE = Path("Data/final/master_dataset_2010_present_long.csv")
DEFAULT_RAW_OPTIONS_DIR = Path("Data/raw/options")
LATEST_MARKET_DATE_SENTINEL = "latest-market-date"


@dataclass(slots=True, frozen=True)
class OptionChainDownloadConfig:
    symbol: str = "SPY"
    expirations: tuple[str, ...] = ()
    max_expirations: int | None = 8
    min_dte_days: float | None = 20.0
    max_dte_days: float | None = 45.0
    option_types: tuple[str, ...] = ("call", "put")
    quote_timestamp: str | None = LATEST_MARKET_DATE_SENTINEL
    market_source: pd.DataFrame | str | Path | None = DEFAULT_MARKET_SOURCE


def download_yfinance_option_chain(config: OptionChainDownloadConfig) -> pd.DataFrame:
    """Download an option chain through yfinance and return a raw option-source frame."""

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised by integration usage.
        raise RuntimeError(
            "yfinance is required to download option chains. Install it with "
            '`python -m pip install -e ".[data]"` or run setup with -WithData.'
        ) from exc

    symbol = config.symbol.upper()
    ticker = yf.Ticker(symbol)
    available_expirations = tuple(str(value) for value in ticker.options)
    if not available_expirations:
        raise ValueError(f"No option expirations were returned for symbol={symbol!r}.")

    quote_timestamp = resolve_quote_timestamp(
        config.quote_timestamp,
        market_source=config.market_source,
        symbol=symbol,
    )
    expirations = select_expirations(
        available_expirations=available_expirations,
        requested_expirations=config.expirations,
        max_expirations=config.max_expirations,
        quote_timestamp=quote_timestamp,
        min_dte_days=config.min_dte_days,
        max_dte_days=config.max_dte_days,
    )
    downloaded_at = utc_now_iso()

    frames: list[pd.DataFrame] = []
    for expiration in expirations:
        chain = ticker.option_chain(expiration)
        frames.append(
            build_raw_option_chain_frame(
                symbol=symbol,
                expiration=expiration,
                calls=chain.calls,
                puts=chain.puts,
                option_types=config.option_types,
                quote_timestamp=quote_timestamp,
                downloaded_at=downloaded_at,
            )
        )

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        raise ValueError(f"No option rows were downloaded for symbol={symbol!r}.")
    return raw


def build_raw_option_chain_frame(
    *,
    symbol: str,
    expiration: str,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    option_types: tuple[str, ...] = ("call", "put"),
    quote_timestamp: str,
    downloaded_at: str,
) -> pd.DataFrame:
    """Attach project metadata to yfinance calls/puts frames."""

    requested = normalize_option_types(option_types)
    frames: list[pd.DataFrame] = []
    for option_type, source in (("call", calls), ("put", puts)):
        if option_type not in requested or source.empty:
            continue
        frame = source.copy()
        frame["underlying_symbol"] = symbol.upper()
        frame["option_type"] = option_type
        frame["expiration_date"] = expiration
        frame["quote_timestamp"] = quote_timestamp
        frame["downloaded_at"] = downloaded_at
        frame["source"] = "yfinance"
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_expirations(
    *,
    available_expirations: tuple[str, ...],
    requested_expirations: tuple[str, ...] = (),
    max_expirations: int | None = 8,
    quote_timestamp: str | None = None,
    min_dte_days: float | None = None,
    max_dte_days: float | None = None,
) -> tuple[str, ...]:
    """Select expirations while failing early for mistyped requested dates."""

    if requested_expirations:
        missing = sorted(set(requested_expirations) - set(available_expirations))
        if missing:
            raise ValueError(
                "Requested expirations are not available for this symbol: "
                f"{missing}. Available examples: {list(available_expirations[:5])}"
            )
        return requested_expirations

    if max_expirations is None:
        return available_expirations
        if max_expirations <= 0:
            raise ValueError("max_expirations must be positive when provided.")

    selected = available_expirations
    if min_dte_days is not None or max_dte_days is not None:
        if quote_timestamp is None:
            raise ValueError("quote_timestamp is required when filtering expirations by DTE.")
        selected = filter_expirations_by_dte(
            available_expirations=available_expirations,
            quote_timestamp=quote_timestamp,
            min_dte_days=min_dte_days,
            max_dte_days=max_dte_days,
        )

    return selected if max_expirations is None else selected[:max_expirations]


def filter_expirations_by_dte(
    *,
    available_expirations: tuple[str, ...],
    quote_timestamp: str,
    min_dte_days: float | None = None,
    max_dte_days: float | None = None,
) -> tuple[str, ...]:
    """Keep expirations inside a days-to-expiration window."""

    quote_date = pd.to_datetime(quote_timestamp, utc=True).date()
    selected: list[str] = []
    for expiration in available_expirations:
        expiration_date = pd.to_datetime(expiration, utc=True).date()
        dte_days = (expiration_date - quote_date).days
        if min_dte_days is not None and dte_days < min_dte_days:
            continue
        if max_dte_days is not None and dte_days > max_dte_days:
            continue
        selected.append(expiration)

    if not selected:
        raise ValueError(
            "No expirations remain after applying the DTE window "
            f"min_dte_days={min_dte_days}, max_dte_days={max_dte_days}, "
            f"quote_timestamp={quote_timestamp!r}."
        )
    return tuple(selected)


def resolve_quote_timestamp(
    quote_timestamp: str | None,
    *,
    market_source: pd.DataFrame | str | Path | None,
    symbol: str,
) -> str:
    """Resolve the date used to join downloaded options to market context."""

    if quote_timestamp in (None, "", "today"):
        return datetime.now(timezone.utc).date().isoformat()
    if str(quote_timestamp).lower() == LATEST_MARKET_DATE_SENTINEL:
        if market_source is None:
            raise ValueError("market_source is required when using latest-market-date.")
        return latest_market_date(market_source, symbol=symbol)
    return pd.to_datetime(quote_timestamp, utc=True).date().isoformat()


def latest_market_date(
    market_source: pd.DataFrame | str | Path,
    *,
    symbol: str,
) -> str:
    """Return the latest date available for one underlying in the market dataset."""

    frame = market_source.copy() if isinstance(market_source, pd.DataFrame) else pd.read_csv(market_source)
    date_column = "date" if "date" in frame.columns else "timestamp"
    symbol_column = "asset" if "asset" in frame.columns else "underlying_symbol"
    if date_column not in frame.columns or symbol_column not in frame.columns:
        raise ValueError(
            "Market source must contain date/timestamp and asset/underlying_symbol columns."
        )

    filtered = frame.loc[frame[symbol_column].astype(str).str.upper() == symbol.upper()]
    if filtered.empty:
        raise ValueError(f"No market rows found for symbol={symbol!r}.")
    dates = pd.to_datetime(filtered[date_column], errors="coerce", utc=True)
    latest = dates.max()
    if pd.isna(latest):
        raise ValueError(f"No parseable market dates found for symbol={symbol!r}.")
    return latest.date().isoformat()


def normalize_option_types(option_types: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(str(value).lower() for value in option_types)
    invalid = sorted(set(normalized) - {"call", "put"})
    if invalid:
        raise ValueError(f"option_types must contain only 'call' and/or 'put': {invalid}")
    return normalized


def default_raw_options_path(
    *,
    output_dir: str | Path = DEFAULT_RAW_OPTIONS_DIR,
    symbol: str,
    quote_timestamp: str,
) -> Path:
    date_part = pd.to_datetime(quote_timestamp, utc=True).date().isoformat()
    return Path(output_dir) / f"{symbol.upper()}_options_{date_part}.csv"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
