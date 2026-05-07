"""Build a canonical option training dataset from raw option chains and market context."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import DatasetBuilder, DatasetBuilderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--option-source",
        required=True,
        type=Path,
        help="Raw option-chain CSV to normalize and enrich.",
    )
    parser.add_argument(
        "--market-source",
        default=Path("Data/final/master_dataset_2010_present_long.csv"),
        type=Path,
        help="Market/macro source CSV used to enrich option rows.",
    )
    parser.add_argument(
        "--output",
        default=Path("Data/processed/option_training_dataset.csv"),
        type=Path,
        help="Output CSV path for the canonical option dataset.",
    )
    parser.add_argument(
        "--underlying-symbol",
        default="SPY",
        help="Underlying symbol used by the MVP filters.",
    )
    parser.add_argument(
        "--option-type",
        default="call",
        choices=("call", "put"),
        help="Option type retained by the MVP filters.",
    )
    parser.add_argument(
        "--quote-timestamp",
        default=None,
        help="Override timestamp for raw sources that represent one quote snapshot.",
    )
    parser.add_argument(
        "--min-volume",
        default=10.0,
        type=float,
        help="Minimum option volume for the MVP filter when volume is present.",
    )
    parser.add_argument(
        "--max-relative-spread",
        default=0.25,
        type=float,
        help="Maximum relative spread for the MVP filter when bid/ask are present.",
    )
    parser.add_argument(
        "--min-moneyness",
        default=0.90,
        type=float,
        help="Minimum S/K ratio retained by the MVP filter.",
    )
    parser.add_argument(
        "--max-moneyness",
        default=1.10,
        type=float,
        help="Maximum S/K ratio retained by the MVP filter.",
    )
    parser.add_argument(
        "--min-ttm-years",
        default=20.0 / 365.0,
        type=float,
        help="Minimum time to maturity in years.",
    )
    parser.add_argument(
        "--max-ttm-years",
        default=45.0 / 365.0,
        type=float,
        help="Maximum time to maturity in years.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = DatasetBuilder(
        DatasetBuilderConfig(
            underlying_symbol=args.underlying_symbol.upper(),
            option_type=args.option_type,
            min_time_to_maturity_years=args.min_ttm_years,
            max_time_to_maturity_years=args.max_ttm_years,
            min_moneyness=args.min_moneyness,
            max_moneyness=args.max_moneyness,
            min_volume=args.min_volume,
            max_relative_spread=args.max_relative_spread,
            option_quote_timestamp_override=args.quote_timestamp,
        )
    )
    frame = builder.build_option_training_dataset(
        option_source=args.option_source,
        market_source=args.market_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame):,} canonical option rows to {args.output}")


if __name__ == "__main__":
    main()
