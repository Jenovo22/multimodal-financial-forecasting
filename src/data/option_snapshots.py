"""Utilities for combining multiple option-chain snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data.option_source import OptionSourceConfig, OptionSourceNormalizer


DEFAULT_SNAPSHOT_DEDUP_COLUMNS = ("timestamp", "underlying_symbol", "option_symbol")
DEFAULT_SNAPSHOT_SORT_COLUMNS = (
    "timestamp",
    "expiration_date",
    "option_type",
    "K",
    "option_symbol",
)


@dataclass(slots=True, frozen=True)
class OptionSnapshotCombineConfig:
    underlying_symbol: str = "SPY"
    deduplicate: bool = True
    dedup_columns: tuple[str, ...] = DEFAULT_SNAPSHOT_DEDUP_COLUMNS
    sort_columns: tuple[str, ...] = DEFAULT_SNAPSHOT_SORT_COLUMNS
    timestamp_split_min_groups: int = 3


def discover_option_snapshot_paths(
    source_dir: str | Path,
    *,
    pattern: str = "*_options_*.csv",
) -> list[Path]:
    """Return sorted option snapshot CSV paths from a directory."""

    directory = Path(source_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Snapshot directory does not exist: {directory}")
    paths = sorted(path for path in directory.glob(pattern) if path.is_file())
    if not paths:
        raise ValueError(f"No option snapshot files matched {directory / pattern}.")
    return paths


def combine_option_snapshots(
    sources: Sequence[str | Path | pd.DataFrame],
    *,
    config: OptionSnapshotCombineConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Normalize, combine and deduplicate option snapshots.

    Returns the combined canonical option-source frame plus a manifest describing
    source files, row counts and timestamp-split readiness.
    """

    resolved = config or OptionSnapshotCombineConfig()
    if not sources:
        raise ValueError("At least one option snapshot source is required.")

    normalizer = OptionSourceNormalizer(
        OptionSourceConfig(underlying_symbol=resolved.underlying_symbol)
    )
    frames: list[pd.DataFrame] = []
    source_entries: list[dict[str, object]] = []

    for source in sources:
        raw = _load_frame(source)
        normalized = normalizer.normalize(raw)
        frames.append(normalized)
        source_entries.append(
            _source_manifest_entry(
                source=source,
                raw_rows=len(raw),
                normalized=normalized,
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    rows_before_dedup = len(combined)
    dedup_columns = tuple(
        column for column in resolved.dedup_columns if column in combined.columns
    )
    if resolved.deduplicate:
        if len(dedup_columns) != len(resolved.dedup_columns):
            missing = sorted(set(resolved.dedup_columns) - set(dedup_columns))
            raise ValueError(f"Cannot deduplicate snapshots; missing columns: {missing}")
        combined = combined.drop_duplicates(subset=list(dedup_columns), keep="last")

    sort_columns = [column for column in resolved.sort_columns if column in combined.columns]
    if sort_columns:
        combined = combined.sort_values(sort_columns)
    combined = combined.reset_index(drop=True)

    manifest = _combined_manifest(
        combined=combined,
        source_entries=source_entries,
        rows_before_dedup=rows_before_dedup,
        dedup_columns=dedup_columns,
        config=resolved,
    )
    return combined, manifest


def write_combined_option_snapshots(
    frame: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Persist a combined option snapshot frame."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_option_snapshot_manifest(
    manifest: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Persist a manifest as JSON."""

    import json

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _load_frame(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(Path(source))


def _source_manifest_entry(
    *,
    source: str | Path | pd.DataFrame,
    raw_rows: int,
    normalized: pd.DataFrame,
) -> dict[str, object]:
    return {
        "source": "<dataframe>" if isinstance(source, pd.DataFrame) else str(Path(source)),
        "raw_rows": int(raw_rows),
        "normalized_rows": int(len(normalized)),
        "timestamps": _unique_strings(normalized, "timestamp"),
        "underlying_symbols": _unique_strings(normalized, "underlying_symbol"),
        "option_types": _unique_strings(normalized, "option_type"),
        "expirations": _unique_strings(normalized, "expiration_date"),
    }


def _combined_manifest(
    *,
    combined: pd.DataFrame,
    source_entries: list[dict[str, object]],
    rows_before_dedup: int,
    dedup_columns: tuple[str, ...],
    config: OptionSnapshotCombineConfig,
) -> dict[str, object]:
    timestamp_counts = (
        combined["timestamp"].value_counts().sort_index().astype(int).to_dict()
        if "timestamp" in combined.columns
        else {}
    )
    timestamp_count = len(timestamp_counts)
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "underlying_symbol": config.underlying_symbol.upper(),
        "source_count": len(source_entries),
        "sources": source_entries,
        "rows_before_dedup": int(rows_before_dedup),
        "rows_after_dedup": int(len(combined)),
        "duplicate_rows_removed": int(rows_before_dedup - len(combined)),
        "deduplicate": config.deduplicate,
        "dedup_columns": list(dedup_columns),
        "timestamps": _unique_strings(combined, "timestamp"),
        "timestamp_count": timestamp_count,
        "rows_by_timestamp": {str(key): int(value) for key, value in timestamp_counts.items()},
        "timestamp_split_min_groups": config.timestamp_split_min_groups,
        "timestamp_split_ready": timestamp_count >= config.timestamp_split_min_groups,
        "underlying_symbols": _unique_strings(combined, "underlying_symbol"),
        "option_types": _unique_strings(combined, "option_type"),
        "expirations": _unique_strings(combined, "expiration_date"),
        "row_count_by_option_type": _value_counts(combined, "option_type"),
        "row_count_by_expiration": _value_counts(combined, "expiration_date"),
    }


def _unique_strings(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    values = frame[column].dropna().astype(str).unique()
    return sorted(values.tolist())


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = frame[column].dropna().astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}

