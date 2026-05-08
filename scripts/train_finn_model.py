"""Train a PyTorch FINN model from option, market and regime CSV sources."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.data import build_finn_training_frame, frame_to_finn_inputs, frame_to_finn_targets
from src.models.finn import (
    FINNConfig,
    FINNPricingModel,
    FINNSplitConfig,
    evaluate_finn_model,
    frame_for_final_training,
    frame_for_split,
    split_finn_training_frame,
    split_summary,
)


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
    parser.add_argument(
        "--selection-output",
        default=Path("artifacts/finn_model_selection.pt"),
        type=Path,
        help="Checkpoint trained only on the train split and selected with validation.",
    )
    parser.add_argument(
        "--split-output",
        default=Path("Data/processed/finn_split_dataset.csv"),
        type=Path,
        help="Output CSV path with train/validation/test split labels.",
    )
    parser.add_argument(
        "--metrics-output",
        default=Path("reports/finn_training_metrics.json"),
        type=Path,
        help="Output JSON path for split and evaluation metrics.",
    )
    parser.add_argument(
        "--predictions-output",
        default=Path("Data/processed/finn_split_predictions.csv"),
        type=Path,
        help="Output CSV path for row-level predictions on each split.",
    )
    parser.add_argument(
        "--split-strategy",
        default="expiration",
        choices=("expiration", "timestamp", "random"),
        help=(
            "How to split the current data. Use expiration for one quote date; "
            "use timestamp once multiple daily snapshots exist."
        ),
    )
    parser.add_argument(
        "--train-fraction",
        default=0.60,
        type=float,
        help="Fraction of groups/rows assigned to train.",
    )
    parser.add_argument(
        "--validation-fraction",
        default=0.20,
        type=float,
        help="Fraction of groups/rows assigned to validation.",
    )
    parser.add_argument(
        "--random-state",
        default=42,
        type=int,
        help="Random seed for random split and model initialization.",
    )
    parser.add_argument(
        "--final-train-split",
        default="train_val",
        choices=("train", "train_val", "all"),
        help=(
            "Rows used for the final checkpoint. Use train_val to keep test untouched; "
            "all maximizes fit but invalidates unbiased test metrics."
        ),
    )
    parser.add_argument(
        "--final-epochs",
        default=None,
        type=int,
        help="Final checkpoint epochs. Defaults to best validation epoch.",
    )
    parser.add_argument("--epochs", default=300, type=int, help="Training epochs.")
    parser.add_argument("--batch-size", default=64, type=int, help="Mini-batch size.")
    parser.add_argument(
        "--hidden-dims",
        default=(64, 64),
        type=_parse_hidden_dims,
        help="Comma-separated hidden layer widths, for example '64,64' or '128,64'.",
    )
    parser.add_argument(
        "--activation",
        default="silu",
        choices=("silu", "relu", "tanh", "gelu"),
        help="Hidden-layer activation function.",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--prediction-mode",
        default="bsm_residual",
        choices=("bsm_residual", "direct"),
        help=(
            "FINN output strategy. bsm_residual anchors predictions to analytical "
            "BSM and learns a bounded residual; direct learns price from scratch."
        ),
    )
    parser.add_argument(
        "--residual-scale",
        default=0.50,
        type=float,
        help="Residual bound as a fraction of max(abs(BSM anchor), residual floor).",
    )
    parser.add_argument(
        "--residual-anchor-floor",
        default=1.0,
        type=float,
        help="Minimum dollar scale used by the residual anchor.",
    )
    parser.add_argument(
        "--lambda-boundary",
        default=0.1,
        type=float,
        help="Boundary-condition loss weight.",
    )
    parser.add_argument(
        "--lambda-pde",
        default=0.0,
        type=float,
        help="PDE residual loss weight.",
    )
    parser.add_argument(
        "--lambda-arbitrage",
        default=1.0,
        type=float,
        help="No-arbitrage regularization weight.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Torch device preference.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=50,
        type=int,
        help="Epochs without validation improvement before stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        default=1e-5,
        type=float,
        help="Minimum validation-loss improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-6,
        type=float,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        default=5.0,
        type=float,
        help="Gradient clipping norm. Use a negative value to disable clipping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_finn_training_frame(
        option_source=args.option_source,
        market_source=args.market_source,
        regime_source=args.regime_source,
    )
    split_frame = split_finn_training_frame(
        frame,
        FINNSplitConfig(
            strategy=args.split_strategy,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            random_state=args.random_state,
        ),
    )
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(args.split_output, index=False)

    train_frame = frame_for_split(split_frame, "train")
    validation_frame = frame_for_split(split_frame, "validation")
    test_frame = frame_for_split(split_frame, "test")

    train_features = frame_to_finn_inputs(train_frame)
    train_targets = frame_to_finn_targets(train_frame)
    validation_features = frame_to_finn_inputs(validation_frame)
    validation_targets = frame_to_finn_targets(validation_frame)

    selection_config = _build_config(args, epochs=args.epochs)
    selection_model = FINNPricingModel(selection_config)
    selection_model.fit(
        train_features,
        train_targets,
        validation_features=validation_features,
        validation_targets=validation_targets,
    )
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection_checkpoint_path = selection_model.save(args.selection_output)

    selection_metrics, prediction_frames = _evaluate_all_splits(
        selection_model,
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
    )

    final_epochs = args.final_epochs or selection_model.best_epoch_ or len(
        selection_model.training_history_
    )
    final_frame = frame_for_final_training(split_frame, args.final_train_split)
    final_features = frame_to_finn_inputs(final_frame)
    final_targets = frame_to_finn_targets(final_frame)
    final_model = FINNPricingModel(_build_config(args, epochs=final_epochs))
    final_model.fit(final_features, final_targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = final_model.save(args.output)
    final_test_metrics, final_test_predictions = evaluate_finn_model(final_model, test_frame)
    final_test_predictions["split"] = "test"
    final_test_predictions["model_role"] = "final"

    all_predictions = prediction_frames + [final_test_predictions]
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    if all_predictions:
        import pandas as pd

        pd.concat(all_predictions, ignore_index=True).to_csv(
            args.predictions_output,
            index=False,
        )

    metrics_payload = {
        "split_config": {
            "strategy": args.split_strategy,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "random_state": args.random_state,
            "final_train_split": args.final_train_split,
        },
        "model_config": asdict(selection_config),
        "split_summary": split_summary(split_frame),
        "selection_model": {
            "checkpoint": str(selection_checkpoint_path),
            "best_epoch": selection_model.best_epoch_,
            "best_validation_loss": selection_model.best_validation_loss_,
            "history_length": len(selection_model.training_history_),
            "metrics": selection_metrics,
        },
        "final_model": {
            "checkpoint": str(final_checkpoint_path),
            "epochs": final_epochs,
            "train_rows": int(len(final_frame)),
            "test_metrics": final_test_metrics,
        },
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    print(f"Saved split dataset to {args.split_output}")
    print(f"Saved selection checkpoint to {selection_checkpoint_path}")
    print(f"Saved final checkpoint to {final_checkpoint_path}")
    print(f"Saved metrics to {args.metrics_output}")
    print(f"Saved split predictions to {args.predictions_output}")
    final_metrics = selection_model.training_history_[-1] if selection_model.training_history_ else {}
    if final_metrics:
        print(
            "Selection losses: "
            f"total={final_metrics['total_loss']:.6f}, "
            f"data={final_metrics['data_loss']:.6f}, "
            f"boundary={final_metrics['boundary_loss']:.6f}, "
            f"pde={final_metrics['pde_loss']:.6f}, "
            f"arbitrage={final_metrics['arbitrage_loss']:.6f}"
        )
    print(
        "Selection test metrics: "
        f"MAE={selection_metrics['test']['mae']:.6f}, "
        f"RMSE={selection_metrics['test']['rmse']:.6f}, "
        f"baseline_MAE={selection_metrics['test']['baseline_mae']:.6f}"
    )
    print(
        "Final model test metrics: "
        f"MAE={final_test_metrics['mae']:.6f}, "
        f"RMSE={final_test_metrics['rmse']:.6f}, "
        f"baseline_MAE={final_test_metrics['baseline_mae']:.6f}"
    )


def _build_config(args: argparse.Namespace, *, epochs: int) -> FINNConfig:
    gradient_clip_norm = (
        None if args.gradient_clip_norm < 0 else args.gradient_clip_norm
    )
    return FINNConfig(
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        prediction_mode=args.prediction_mode,
        residual_scale=args.residual_scale,
        residual_anchor_floor=args.residual_anchor_floor,
        lambda_boundary=args.lambda_boundary,
        lambda_pde=args.lambda_pde,
        lambda_arbitrage=args.lambda_arbitrage,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        gradient_clip_norm=gradient_clip_norm,
        random_state=args.random_state,
        device=args.device,
    )


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError("hidden-dims must contain positive integers.")
    try:
        dims = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hidden-dims must contain positive integers."
        ) from exc
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("hidden-dims must be positive.")
    return dims


def _evaluate_all_splits(
    model: FINNPricingModel,
    *,
    train_frame,
    validation_frame,
    test_frame,
) -> tuple[dict[str, dict[str, float | int | None]], list]:
    metrics = {}
    prediction_frames = []
    for split_name, frame in [
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
    ]:
        split_metrics, predictions = evaluate_finn_model(model, frame)
        metrics[split_name] = split_metrics
        predictions["split"] = split_name
        predictions["model_role"] = "selection"
        prediction_frames.append(predictions)
    return metrics, prediction_frames


if __name__ == "__main__":
    main()
