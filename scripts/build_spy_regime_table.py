"""Build the default SPY regime table from the repo market dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import RegimeTableConfig, build_and_save_regime_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-source",
        type=Path,
        default=Path("Data/final/master_dataset_2010_present_long.csv"),
        help="Input market dataset CSV in long format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/processed/spy_regime_features.csv"),
        help="Output CSV path for the generated regime table.",
    )
    parser.add_argument(
        "--fit-end-timestamp",
        default=None,
        help="Optional timestamp cutoff for fitting the HMM before predicting all rows.",
    )
    parser.add_argument(
        "--no-market-context",
        action="store_true",
        help="Write only regime columns and timestamp.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_and_save_regime_table(
        RegimeTableConfig(
            market_source_path=args.market_source,
            output_path=args.output,
            fit_end_timestamp=args.fit_end_timestamp,
            include_market_context=not args.no_market_context,
        )
    )
    print(f"Saved {len(frame):,} regime rows to {args.output}")


if __name__ == "__main__":
    main()
