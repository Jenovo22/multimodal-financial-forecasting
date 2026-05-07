from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.pipeline import build_regime_table, save_regime_table


def build_market_source() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    regimes = [
        (0.001, 0.003, 0.11, 13.0, 95_000_000.0),
        (-0.001, 0.008, 0.24, 26.0, 145_000_000.0),
        (-0.003, 0.015, 0.45, 50.0, 230_000_000.0),
    ]
    day = 0
    for return_mean, return_std, vol, vix, volume in regimes:
        for _ in range(30):
            rows.append(
                {
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                    "asset": "SPY",
                    "ticker": "SPY",
                    "price_used": 100.0 + day,
                    "volume": max(float(rng.normal(volume, 5_000_000.0)), 1.0),
                    "return_1d": float(rng.normal(return_mean, return_std)),
                    "return_5d": float(rng.normal(return_mean * 5.0, return_std * 2.0)),
                    "volatility_20d_ann": float(max(rng.normal(vol, 0.01), 1e-4)),
                    "vix_fred": float(max(rng.normal(vix, 1.0), 1e-4)),
                    "dgs2": 4.0,
                }
            )
            day += 1
    return pd.DataFrame(rows)


def test_build_regime_table_outputs_named_probabilities_and_context():
    frame = build_regime_table(build_market_source())
    probability_columns = [
        "regime_probability_stable_low_volatility",
        "regime_probability_stress_high_volatility",
        "regime_probability_transition_uncertainty",
    ]
    assert len(frame) == 90
    assert "S" in frame.columns
    assert "regime_label" in frame.columns
    assert "sigma_regime" in frame.columns
    assert all(column in frame.columns for column in probability_columns)

    probability_sums = frame[probability_columns].sum(axis=1)
    assert all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-6) for value in probability_sums)


def test_build_regime_table_can_fit_with_cutoff_and_predict_full_frame():
    frame = build_regime_table(
        build_market_source(),
        fit_end_timestamp="2020-02-29T00:00:00Z",
    )
    assert len(frame) == 90
    assert frame["timestamp"].iloc[-1] == "2020-03-30T00:00:00Z"


def test_save_regime_table_writes_csv(tmp_path):
    frame = build_regime_table(build_market_source())
    output_path = save_regime_table(frame, tmp_path / "regimes.csv")
    loaded = pd.read_csv(output_path)
    assert output_path.exists()
    assert len(loaded) == len(frame)
