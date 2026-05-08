from __future__ import annotations

import pytest
import pandas as pd

from src.data.options_downloader import (
    build_raw_option_chain_frame,
    default_raw_options_path,
    filter_expirations_by_dte,
    latest_market_date,
    resolve_quote_timestamp,
    select_expirations,
)


def test_latest_market_date_filters_by_symbol():
    market = pd.DataFrame(
        [
            {"date": "2026-05-01", "asset": "SPY"},
            {"date": "2026-05-05", "asset": "SPY"},
            {"date": "2026-05-06", "asset": "QQQ"},
        ]
    )

    assert latest_market_date(market, symbol="SPY") == "2026-05-05"


def test_resolve_quote_timestamp_supports_latest_market_date():
    market = pd.DataFrame(
        [
            {"date": "2026-05-01", "asset": "SPY"},
            {"date": "2026-05-05", "asset": "SPY"},
        ]
    )

    assert (
        resolve_quote_timestamp(
            "latest-market-date",
            market_source=market,
            symbol="SPY",
        )
        == "2026-05-05"
    )


def test_build_raw_option_chain_frame_adds_project_metadata():
    calls = pd.DataFrame(
        [
            {
                "contractSymbol": "SPY260605C00520000",
                "strike": 520.0,
                "lastPrice": 8.4,
                "bid": 8.3,
                "ask": 8.5,
                "impliedVolatility": 0.214,
            }
        ]
    )
    puts = pd.DataFrame(
        [
            {
                "contractSymbol": "SPY260605P00520000",
                "strike": 520.0,
                "lastPrice": 7.9,
                "bid": 7.8,
                "ask": 8.0,
                "impliedVolatility": 0.221,
            }
        ]
    )

    frame = build_raw_option_chain_frame(
        symbol="spy",
        expiration="2026-06-05",
        calls=calls,
        puts=puts,
        option_types=("call",),
        quote_timestamp="2026-05-05",
        downloaded_at="2026-05-07T12:00:00+00:00",
    )

    assert len(frame) == 1
    assert frame.loc[0, "underlying_symbol"] == "SPY"
    assert frame.loc[0, "option_type"] == "call"
    assert frame.loc[0, "expiration_date"] == "2026-06-05"
    assert frame.loc[0, "quote_timestamp"] == "2026-05-05"
    assert frame.loc[0, "downloaded_at"] == "2026-05-07T12:00:00+00:00"
    assert frame.loc[0, "source"] == "yfinance"


def test_select_expirations_uses_nearest_dates_and_validates_requests():
    available = ("2026-05-15", "2026-05-22", "2026-05-29")

    assert select_expirations(
        available_expirations=available,
        max_expirations=2,
    ) == ("2026-05-15", "2026-05-22")

    with pytest.raises(ValueError, match="Requested expirations"):
        select_expirations(
            available_expirations=available,
            requested_expirations=("2026-06-05",),
        )


def test_select_expirations_can_filter_by_dte_window():
    available = ("2026-05-08", "2026-05-15", "2026-05-29", "2026-06-05", "2026-06-26")

    selected = select_expirations(
        available_expirations=available,
        quote_timestamp="2026-05-05",
        min_dte_days=20.0,
        max_dte_days=45.0,
        max_expirations=8,
    )

    assert selected == ("2026-05-29", "2026-06-05")


def test_filter_expirations_by_dte_raises_when_window_is_empty():
    with pytest.raises(ValueError, match="No expirations remain"):
        filter_expirations_by_dte(
            available_expirations=("2026-05-08",),
            quote_timestamp="2026-05-05",
            min_dte_days=20.0,
            max_dte_days=45.0,
        )


def test_default_raw_options_path_is_stable():
    path = default_raw_options_path(
        output_dir="Data/raw/options",
        symbol="spy",
        quote_timestamp="2026-05-05T00:00:00Z",
    )

    assert path.parts[-4:-1] == ("Data", "raw", "options")
    assert path.name == "SPY_options_2026-05-05.csv"
