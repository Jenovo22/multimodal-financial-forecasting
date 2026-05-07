from __future__ import annotations

import math

import pandas as pd

from src.data.dataset_builder import DatasetBuilder


def build_market_source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-05-05",
                "asset": "SPY",
                "ticker": "SPY",
                "price_used": 510.25,
                "volume": 123456789.0,
                "return_1d": 0.012,
                "return_5d": 0.035,
                "volatility_20d_ann": 0.185,
                "vix_fred": 22.5,
                "dgs2": 4.1,
            },
            {
                "date": "2026-05-05",
                "asset": "GLD",
                "ticker": "GLD",
                "price_used": 230.5,
                "volume": 2500000.0,
                "return_1d": -0.002,
                "return_5d": 0.01,
                "volatility_20d_ann": 0.11,
                "vix_fred": 22.5,
                "dgs2": 4.1,
            },
        ]
    )


def test_build_market_context_frame_filters_spy_and_normalizes_fields():
    builder = DatasetBuilder()
    context = builder.build_market_context_frame(build_market_source())
    assert list(context["underlying_symbol"]) == ["SPY"]
    assert context.loc[0, "timestamp"] == "2026-05-05T00:00:00Z"
    assert context.loc[0, "S"] == 510.25
    assert math.isclose(context.loc[0, "implied_volatility_proxy"], 0.225)
    assert math.isclose(context.loc[0, "r"], 0.041)
    assert context.loc[0, "realized_volatility"] == 0.185


def test_build_hmm_inputs_uses_market_context_proxy_fields():
    builder = DatasetBuilder()
    samples = builder.build_hmm_inputs(build_market_source())
    assert len(samples) == 1
    sample = samples[0]
    assert sample.timestamp == "2026-05-05T00:00:00Z"
    assert math.isclose(sample.features.implied_volatility or 0.0, 0.225)
    assert math.isclose(sample.features.realized_volatility or 0.0, 0.185)
    assert sample.features.volume == 123456789.0


def test_build_option_training_dataset_enriches_option_rows_from_market_context():
    builder = DatasetBuilder()
    option_source = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05",
                "underlying_symbol": "SPY",
                "option_symbol": "SPY_20260605_520_C",
                "option_type": "call",
                "K": 520.0,
                "T": 30.0 / 365.0,
                "market_price": 8.4,
            }
        ]
    )
    enriched = builder.build_option_training_dataset(option_source, build_market_source())
    assert enriched.loc[0, "timestamp"] == "2026-05-05T00:00:00Z"
    assert enriched.loc[0, "S"] == 510.25
    assert math.isclose(enriched.loc[0, "r"], 0.041)
    assert math.isclose(enriched.loc[0, "implied_volatility"], 0.225)
    assert enriched.loc[0, "realized_volatility"] == 0.185
    assert enriched.loc[0, "return_5d"] == 0.035
    assert math.isclose(enriched.loc[0, "moneyness"], 510.25 / 520.0)
    assert "volume" not in enriched.columns
    assert enriched.loc[0, "dividend_yield"] == 0.0


def test_build_option_training_dataset_normalizes_yfinance_style_option_rows():
    builder = DatasetBuilder()
    option_source = pd.DataFrame(
        [
            {
                "contractSymbol": "SPY260605C00520000",
                "lastTradeDate": "2026-05-05T19:30:00Z",
                "strike": 520.0,
                "lastPrice": 8.4,
                "bid": 8.3,
                "ask": 8.5,
                "volume": 120.0,
                "openInterest": 450.0,
                "impliedVolatility": 0.214,
            }
        ]
    )
    enriched = builder.build_option_training_dataset(option_source, build_market_source())

    assert len(enriched) == 1
    assert enriched.loc[0, "timestamp"] == "2026-05-05T00:00:00Z"
    assert enriched.loc[0, "underlying_symbol"] == "SPY"
    assert enriched.loc[0, "option_symbol"] == "SPY260605C00520000"
    assert enriched.loc[0, "option_type"] == "call"
    assert math.isclose(enriched.loc[0, "K"], 520.0)
    assert math.isclose(enriched.loc[0, "T"], 31.0 / 365.0)
    assert math.isclose(enriched.loc[0, "mid_price"], 8.4)
    assert math.isclose(enriched.loc[0, "relative_spread"], (8.5 - 8.3) / 8.4)


def test_build_option_training_dataset_applies_mvp_filters():
    builder = DatasetBuilder()
    option_source = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05",
                "underlying_symbol": "SPY",
                "option_symbol": "SPY_20260605_520_C",
                "option_type": "call",
                "K": 520.0,
                "T": 30.0 / 365.0,
                "market_price": 8.4,
                "bid": 8.3,
                "ask": 8.5,
                "volume": 50.0,
            },
            {
                "timestamp": "2026-05-05",
                "underlying_symbol": "SPY",
                "option_symbol": "SPY_20260605_650_C",
                "option_type": "call",
                "K": 650.0,
                "T": 30.0 / 365.0,
                "market_price": 0.5,
                "bid": 0.1,
                "ask": 1.0,
                "volume": 50.0,
            },
        ]
    )
    enriched = builder.build_option_training_dataset(option_source, build_market_source())

    assert len(enriched) == 1
    assert enriched.loc[0, "option_symbol"] == "SPY_20260605_520_C"


def test_build_dispatch_supports_mapping_sources():
    builder = DatasetBuilder()
    option_source = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-05",
                "underlying_symbol": "SPY",
                "option_symbol": "SPY_20260605_520_C",
                "option_type": "call",
                "K": 520.0,
                "T": 30.0 / 365.0,
                "market_price": 8.4,
            }
        ]
    )
    result = builder.build(
        {
            "market_source": build_market_source(),
            "option_source": option_source,
        }
    )
    assert list(result["underlying_symbol"]) == ["SPY"]
    assert "S" in result.columns


def test_missing_market_source_columns_detects_bridge_requirements():
    builder = DatasetBuilder()
    missing = builder.missing_market_source_columns(["date", "asset"])
    assert "price_used" in missing
    assert "dgs2" in missing
