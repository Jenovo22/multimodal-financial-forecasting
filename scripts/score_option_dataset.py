"""Score option rows with baseline + regime table + FINN-compatible fallback."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.baseline import BlackScholesPricer
from src.models.finn import FINNPricingModel
from src.models.hmm import RegimeTableDetector, RegimeTableDetectorConfig
from src.pipeline import (
    HMMFINNPipeline,
    OptionScoringConfig,
    PipelineConfig,
    build_and_save_option_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--option-source",
        required=True,
        type=Path,
        help="CSV with option rows to score.",
    )
    parser.add_argument(
        "--market-source",
        default=Path("Data/final/master_dataset_2010_present_long.csv"),
        type=Path,
        help="Market/macro source CSV used to enrich option rows.",
    )
    parser.add_argument(
        "--regime-source",
        default=Path("Data/processed/spy_regime_features.csv"),
        type=Path,
        help="Precomputed regime table CSV.",
    )
    parser.add_argument(
        "--output",
        default=Path("Data/processed/scored_option_dataset.csv"),
        type=Path,
        help="Output CSV path for scored rows.",
    )
    parser.add_argument(
        "--finn-checkpoint",
        default=None,
        type=Path,
        help="Optional trained FINN checkpoint path.",
    )
    parser.add_argument(
        "--transaction-cost-estimate",
        default=0.0,
        type=float,
        help="Transaction cost estimate used by the signal threshold.",
    )
    parser.add_argument(
        "--safety-margin",
        default=0.0,
        type=float,
        help="Extra safety margin used by the signal threshold.",
    )
    parser.add_argument(
        "--regime-match-mode",
        choices=("exact", "previous"),
        default="exact",
        help="How to match regime rows by timestamp.",
    )
    parser.add_argument(
        "--regime-max-staleness-days",
        default=None,
        type=int,
        help="Maximum age for previous-match regime rows.",
    )
    parser.add_argument(
        "--use-market-price",
        action="store_true",
        help="Use market_price directly instead of mid_price when both are present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = None
    if args.finn_checkpoint is not None:
        pipeline = HMMFINNPipeline(
            baseline=BlackScholesPricer(),
            regime_detector=RegimeTableDetector(
                args.regime_source,
                config=RegimeTableDetectorConfig(
                    underlying_symbol="SPY",
                    match_mode=args.regime_match_mode,
                    max_staleness_days=args.regime_max_staleness_days,
                ),
            ),
            finn_model=FINNPricingModel.load(args.finn_checkpoint),
            config=PipelineConfig(
                transaction_cost_estimate=args.transaction_cost_estimate,
                safety_margin=args.safety_margin,
                use_mid_price_if_available=not args.use_market_price,
            ),
        )

    frame = build_and_save_option_scores(
        OptionScoringConfig(
            option_source_path=args.option_source,
            market_source_path=args.market_source,
            regime_source_path=args.regime_source,
            output_path=args.output,
            transaction_cost_estimate=args.transaction_cost_estimate,
            safety_margin=args.safety_margin,
            use_mid_price_if_available=not args.use_market_price,
            regime_match_mode=args.regime_match_mode,
            regime_max_staleness_days=args.regime_max_staleness_days,
        ),
        pipeline=pipeline,
    )
    print(f"Saved {len(frame):,} scored option rows to {args.output}")


if __name__ == "__main__":
    main()
