"""Download option-chain data and optionally build the canonical training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import DatasetBuilder, DatasetBuilderConfig
from src.data.options_downloader import (
    DEFAULT_MARKET_SOURCE,
    DEFAULT_RAW_OPTIONS_DIR,
    LATEST_MARKET_DATE_SENTINEL,
    OptionChainDownloadConfig,
    default_raw_options_path,
    download_yfinance_option_chain,
    resolve_quote_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol to download.")
    parser.add_argument(
        "--expiration",
        action="append",
        dest="expirations",
        default=[],
        help="Expiration date to download. Repeat for multiple dates. Defaults to nearest dates.",
    )
    parser.add_argument(
        "--max-expirations",
        default=8,
        type=int,
        help="Maximum number of nearest expirations to download when --expiration is omitted.",
    )
    parser.add_argument(
        "--min-dte-days",
        default=20.0,
        type=float,
        help="Minimum days to expiration when --expiration is omitted.",
    )
    parser.add_argument(
        "--max-dte-days",
        default=45.0,
        type=float,
        help="Maximum days to expiration when --expiration is omitted.",
    )
    parser.add_argument(
        "--option-type",
        default="both",
        choices=("call", "put", "both"),
        help="Raw option type universe to download.",
    )
    parser.add_argument(
        "--quote-timestamp",
        default=LATEST_MARKET_DATE_SENTINEL,
        help="Quote date used for market-data joins. Use 'today' or 'latest-market-date'.",
    )
    parser.add_argument(
        "--market-source",
        default=DEFAULT_MARKET_SOURCE,
        type=Path,
        help="Market source used when resolving latest-market-date and building datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_RAW_OPTIONS_DIR,
        type=Path,
        help="Directory for raw option-chain CSV files.",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
        type=Path,
        help="Explicit raw output CSV path. Defaults to Data/raw/options/<SYMBOL>_options_<DATE>.csv.",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Also build Data/processed/option_training_dataset.csv from the downloaded raw CSV.",
    )
    parser.add_argument(
        "--dataset-output",
        default=Path("Data/processed/option_training_dataset.csv"),
        type=Path,
        help="Canonical option dataset output when --build-dataset is used.",
    )
    parser.add_argument(
        "--dataset-option-type",
        default="call",
        choices=("call", "put"),
        help="Option type retained by the MVP dataset filters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    option_types = ("call", "put") if args.option_type == "both" else (args.option_type,)
    quote_timestamp = resolve_quote_timestamp(
        args.quote_timestamp,
        market_source=args.market_source,
        symbol=args.symbol,
    )

    frame = download_yfinance_option_chain(
        OptionChainDownloadConfig(
            symbol=args.symbol,
            expirations=tuple(args.expirations),
            max_expirations=args.max_expirations,
            min_dte_days=args.min_dte_days,
            max_dte_days=args.max_dte_days,
            option_types=option_types,
            quote_timestamp=quote_timestamp,
            market_source=args.market_source,
        )
    )

    raw_output = args.raw_output or default_raw_options_path(
        output_dir=args.output_dir,
        symbol=args.symbol,
        quote_timestamp=quote_timestamp,
    )
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_output, index=False)
    print(f"Saved {len(frame):,} raw option rows to {raw_output}")

    if args.build_dataset:
        builder = DatasetBuilder(
            DatasetBuilderConfig(
                underlying_symbol=args.symbol.upper(),
                option_type=args.dataset_option_type,
            )
        )
        dataset = builder.build_option_training_dataset(
            option_source=raw_output,
            market_source=args.market_source,
        )
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(args.dataset_output, index=False)
        print(f"Saved {len(dataset):,} canonical option rows to {args.dataset_output}")


if __name__ == "__main__":
    main()
