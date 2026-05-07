"""Build a FINN-ready option training dataset from option, market and regime CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import build_finn_training_frame, save_finn_training_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--option-source",
        required=True,
        type=Path,
        help="CSV with option rows to enrich.",
    )
    parser.add_argument(
        "--market-source",
        default=Path("Data/final/master_dataset_2010_present_long.csv"),
        type=Path,
        help="Market/macro source CSV used by DatasetBuilder.",
    )
    parser.add_argument(
        "--regime-source",
        default=Path("Data/processed/spy_regime_features.csv"),
        type=Path,
        help="Precomputed regime table CSV.",
    )
    parser.add_argument(
        "--output",
        default=Path("Data/processed/finn_training_dataset.csv"),
        type=Path,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_finn_training_frame(
        option_source=args.option_source,
        market_source=args.market_source,
        regime_source=args.regime_source,
    )
    save_finn_training_frame(frame, args.output)
    print(f"Saved {len(frame):,} FINN training rows to {args.output}")


if __name__ == "__main__":
    main()
