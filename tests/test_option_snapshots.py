from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data import (
    OptionSnapshotCombineConfig,
    combine_option_snapshots,
    discover_option_snapshot_paths,
    write_combined_option_snapshots,
    write_option_snapshot_manifest,
)


def raw_snapshot(quote_timestamp: str, *, duplicate_first: bool = False) -> pd.DataFrame:
    rows = [
        {
            "contractSymbol": "SPY260529C00700000",
            "lastPrice": 25.0,
            "bid": 24.9,
            "ask": 25.1,
            "volume": 100,
            "openInterest": 1000,
            "impliedVolatility": 0.20,
            "underlying_symbol": "SPY",
            "option_type": "call",
            "expiration_date": "2026-05-29",
            "quote_timestamp": quote_timestamp,
            "downloaded_at": quote_timestamp,
            "source": "test",
        },
        {
            "contractSymbol": "SPY260529P00700000",
            "lastPrice": 1.5,
            "bid": 1.4,
            "ask": 1.6,
            "volume": 80,
            "openInterest": 900,
            "impliedVolatility": 0.21,
            "underlying_symbol": "SPY",
            "option_type": "put",
            "expiration_date": "2026-05-29",
            "quote_timestamp": quote_timestamp,
            "downloaded_at": quote_timestamp,
            "source": "test",
        },
    ]
    if duplicate_first:
        rows.append({**rows[0], "lastPrice": 26.0})
    return pd.DataFrame(rows)


def test_combine_option_snapshots_normalizes_deduplicates_and_builds_manifest():
    first = raw_snapshot("2026-05-05", duplicate_first=True)
    second = raw_snapshot("2026-05-06")

    combined, manifest = combine_option_snapshots(
        [first, second],
        config=OptionSnapshotCombineConfig(timestamp_split_min_groups=3),
    )

    assert len(combined) == 4
    assert manifest["source_count"] == 2
    assert manifest["rows_before_dedup"] == 5
    assert manifest["rows_after_dedup"] == 4
    assert manifest["duplicate_rows_removed"] == 1
    assert manifest["timestamps"] == [
        "2026-05-05T00:00:00Z",
        "2026-05-06T00:00:00Z",
    ]
    assert manifest["timestamp_split_ready"] is False
    assert manifest["row_count_by_option_type"] == {"call": 2, "put": 2}
    assert combined["timestamp"].is_monotonic_increasing
    latest_duplicate = combined.loc[
        (combined["timestamp"] == "2026-05-05T00:00:00Z")
        & (combined["option_symbol"] == "SPY260529C00700000"),
        "market_price",
    ].iloc[0]
    assert latest_duplicate == pytest.approx(26.0)


def test_combine_option_snapshots_marks_timestamp_split_ready_with_three_groups():
    snapshots = [
        raw_snapshot("2026-05-05"),
        raw_snapshot("2026-05-06"),
        raw_snapshot("2026-05-07"),
    ]

    _, manifest = combine_option_snapshots(snapshots)

    assert manifest["timestamp_count"] == 3
    assert manifest["timestamp_split_ready"] is True


def test_discover_option_snapshot_paths_and_writers(tmp_path):
    first_path = tmp_path / "SPY_options_2026-05-05.csv"
    second_path = tmp_path / "SPY_options_2026-05-06.csv"
    raw_snapshot("2026-05-05").to_csv(first_path, index=False)
    raw_snapshot("2026-05-06").to_csv(second_path, index=False)

    paths = discover_option_snapshot_paths(tmp_path, pattern="SPY_options_*.csv")
    combined, manifest = combine_option_snapshots(paths)
    combined_path = write_combined_option_snapshots(
        combined,
        tmp_path / "combined.csv",
    )
    manifest_path = write_option_snapshot_manifest(
        manifest,
        tmp_path / "manifest.json",
    )

    assert paths == [first_path, second_path]
    assert combined_path.exists()
    assert manifest_path.exists()
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest["rows_after_dedup"] == 4


def test_discover_option_snapshot_paths_raises_when_empty(tmp_path):
    with pytest.raises(ValueError, match="No option snapshot files matched"):
        discover_option_snapshot_paths(tmp_path)

