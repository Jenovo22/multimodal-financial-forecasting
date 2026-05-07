"""Run the Proyecto TAM FastAPI application locally with uvicorn."""

from __future__ import annotations

import argparse

import uvicorn

from src.api import APIServerConfig, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to expose.")
    parser.add_argument(
        "--regime-source",
        default="Data/processed/spy_regime_features.csv",
        help="Regime CSV used by the API runtime.",
    )
    parser.add_argument(
        "--finn-checkpoint",
        default=None,
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
        help="How to match incoming timestamps against regime rows.",
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
    app = create_app(
        APIServerConfig(
            regime_source_path=args.regime_source,
            finn_checkpoint_path=args.finn_checkpoint,
            transaction_cost_estimate=args.transaction_cost_estimate,
            safety_margin=args.safety_margin,
            use_mid_price_if_available=not args.use_market_price,
            regime_match_mode=args.regime_match_mode,
            regime_max_staleness_days=args.regime_max_staleness_days,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
