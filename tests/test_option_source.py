from __future__ import annotations

import math

import pandas as pd

from src.data import OptionSourceConfig, OptionSourceNormalizer, normalize_option_source


def test_normalize_option_source_parses_occ_symbol_and_daily_timestamp():
    frame = normalize_option_source(
        pd.DataFrame(
            [
                {
                    "contractSymbol": "SPY260605C00520000",
                    "lastTradeDate": "2026-05-05T19:30:00Z",
                    "lastPrice": 8.4,
                    "bid": 8.3,
                    "ask": 8.5,
                    "impliedVolatility": 0.214,
                }
            ]
        )
    )

    assert frame.loc[0, "timestamp"] == "2026-05-05T00:00:00Z"
    assert frame.loc[0, "underlying_symbol"] == "SPY"
    assert frame.loc[0, "option_symbol"] == "SPY260605C00520000"
    assert frame.loc[0, "option_type"] == "call"
    assert math.isclose(frame.loc[0, "K"], 520.0)
    assert math.isclose(frame.loc[0, "T"], 31.0 / 365.0)
    assert frame.loc[0, "expiration_date"] == "2026-06-05"


def test_normalize_option_source_scales_percent_implied_volatility():
    frame = normalize_option_source(
        pd.DataFrame(
            [
                {
                    "contractSymbol": "SPY260605C00520000",
                    "lastTradeDate": "2026-05-05T19:30:00Z",
                    "lastPrice": 8.4,
                    "impliedVolatility": 21.4,
                }
            ]
        )
    )

    assert math.isclose(frame.loc[0, "implied_volatility"], 0.214)


def test_normalize_option_source_can_build_canonical_symbol_when_missing():
    normalizer = OptionSourceNormalizer(
        OptionSourceConfig(
            underlying_symbol="SPY",
            quote_timestamp_override="2026-05-05",
        )
    )
    frame = normalizer.normalize(
        pd.DataFrame(
            [
                {
                    "expiration_date": "2026-06-05",
                    "strike": 520.0,
                    "option_type": "call",
                    "lastPrice": 8.4,
                    "impliedVolatility": 0.214,
                }
            ]
        )
    )

    assert frame.loc[0, "option_symbol"] == "SPY_20260605_520_C"
    assert frame.loc[0, "underlying_symbol"] == "SPY"
    assert frame.loc[0, "timestamp"] == "2026-05-05T00:00:00Z"
