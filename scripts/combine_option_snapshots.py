"""Combine multiple raw option snapshots into one canonical option source."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import (
    OptionSnapshotCombineConfig,
    combine_option_snapshots,
    discover_option_snapshot_paths,
    write_combined_option_snapshots,
    write_option_snapshot_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=Path("Data/raw/options"),
        type=Path,
        help="Directory containing option snapshot CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_options_*.csv",
        help="Glob pattern used with --source-dir when --snapshot is omitted.",
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        type=Path,
        help="Explicit snapshot CSV path. Repeat for multiple files.",
    )
    parser.add_argument(
        "--symbol",
        default="SPY",
        help="Underlying symbol used by the normalizer when source rows are incomplete.",
    )
    parser.add_argument(
        "--output",
        default=Path("Data/processed/combined_option_snapshots.csv"),
        type=Path,
        help="Output CSV path for the combined canonical option source.",
    )
    parser.add_argument(
        "--manifest-output",
        default=Path("Data/processed/combined_option_snapshots_manifest.json"),
        type=Path,
        help="Output JSON path for the combination manifest.",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Keep duplicate rows instead of deduplicating by timestamp/symbol/option.",
    )
    parser.add_argument(
        "--timestamp-split-min-groups",
        default=3,
        type=int,
        help="Minimum unique timestamps required to be timestamp-split ready.",
    )
    parser.add_argument(
        "--require-timestamp-split-ready",
        action="store_true",
        help="Fail when the combined manifest has fewer timestamp groups than required.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = args.snapshot or discover_option_snapshot_paths(
        args.source_dir,
        pattern=args.pattern,
    )
    combined, manifest = combine_option_snapshots(
        sources,
        config=OptionSnapshotCombineConfig(
            underlying_symbol=args.symbol,
            deduplicate=not args.no_deduplicate,
            timestamp_split_min_groups=args.timestamp_split_min_groups,
        ),
    )
    if args.require_timestamp_split_ready and not manifest["timestamp_split_ready"]:
        raise SystemExit(
            "Combined snapshots are not timestamp-split ready: "
            f"{manifest['timestamp_count']} unique timestamps found, "
            f"{manifest['timestamp_split_min_groups']} required."
        )

    output_path = write_combined_option_snapshots(combined, args.output)
    manifest_path = write_option_snapshot_manifest(manifest, args.manifest_output)

    print(f"Combined {manifest['source_count']} snapshot source(s).")
    print(
        "Rows: "
        f"{manifest['rows_after_dedup']} after dedup "
        f"({manifest['duplicate_rows_removed']} removed)."
    )
    print(
        "Timestamp split ready: "
        f"{manifest['timestamp_split_ready']} "
        f"({manifest['timestamp_count']} timestamp group(s))."
    )
    print(f"Saved combined option source to {output_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()

