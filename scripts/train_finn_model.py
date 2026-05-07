"""Train a PyTorch FINN model from option, market and regime CSV sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import build_finn_training_frame, frame_to_finn_inputs, frame_to_finn_targets
from src.models.finn import FINNConfig, FINNPricingModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--option-source",
        required=True,
        type=Path,
        help="CSV with option rows to train on.",
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
        default=Path("artifacts/finn_model.pt"),
        type=Path,
        help="Output checkpoint path.",
    )
    parser.add_argument("--epochs", default=200, type=int, help="Training epochs.")
    parser.add_argument("--batch-size", default=64, type=int, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--lambda-boundary",
        default=0.1,
        type=float,
        help="Boundary-condition loss weight.",
    )
    parser.add_argument(
        "--lambda-pde",
        default=0.1,
        type=float,
        help="PDE residual loss weight.",
    )
    parser.add_argument(
        "--lambda-arbitrage",
        default=0.0,
        type=float,
        help="No-arbitrage regularization weight.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Torch device preference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_finn_training_frame(
        option_source=args.option_source,
        market_source=args.market_source,
        regime_source=args.regime_source,
    )
    features = frame_to_finn_inputs(frame)
    targets = frame_to_finn_targets(frame)

    model = FINNPricingModel(
        FINNConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lambda_boundary=args.lambda_boundary,
            lambda_pde=args.lambda_pde,
            lambda_arbitrage=args.lambda_arbitrage,
            device=args.device,
        )
    )
    model.fit(features, targets)
    checkpoint_path = model.save(args.output)
    final_metrics = model.training_history_[-1] if model.training_history_ else {}
    print(f"Saved FINN checkpoint to {checkpoint_path}")
    if final_metrics:
        print(
            "Final losses: "
            f"total={final_metrics['total_loss']:.6f}, "
            f"data={final_metrics['data_loss']:.6f}, "
            f"boundary={final_metrics['boundary_loss']:.6f}, "
            f"pde={final_metrics['pde_loss']:.6f}, "
            f"arbitrage={final_metrics['arbitrage_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
