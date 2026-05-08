"""Run a controlled FINN architecture validation matrix.

The script calls scripts/train_finn_model.py repeatedly and aggregates the
metrics needed to compare FINN variants against BSM on the same split.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class ArchitectureExperiment:
    name: str
    hidden_dims: str
    activation: str = "silu"
    residual_scale: float = 0.50
    lambda_boundary: float = 0.1
    lambda_pde: float = 0.0
    lambda_arbitrage: float = 1.0


DEFAULT_EXPERIMENTS: tuple[ArchitectureExperiment, ...] = (
    ArchitectureExperiment(
        name="baseline_64x64_silu_scale050",
        hidden_dims="64,64",
        activation="silu",
        residual_scale=0.50,
    ),
    ArchitectureExperiment(
        name="compact_32x32_silu_scale050",
        hidden_dims="32,32",
        activation="silu",
        residual_scale=0.50,
    ),
    ArchitectureExperiment(
        name="wide_128x128_silu_scale050",
        hidden_dims="128,128",
        activation="silu",
        residual_scale=0.50,
    ),
    ArchitectureExperiment(
        name="baseline_64x64_gelu_scale050",
        hidden_dims="64,64",
        activation="gelu",
        residual_scale=0.50,
    ),
    ArchitectureExperiment(
        name="conservative_64x64_silu_scale025",
        hidden_dims="64,64",
        activation="silu",
        residual_scale=0.25,
    ),
    ArchitectureExperiment(
        name="flexible_64x64_silu_scale075",
        hidden_dims="64,64",
        activation="silu",
        residual_scale=0.75,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--option-source",
        default=Path("Data/raw/options/SPY_options_2026-05-05.csv"),
        type=Path,
        help="Raw option source consumed by train_finn_model.py.",
    )
    parser.add_argument(
        "--market-source",
        default=Path("Data/final/master_dataset_2010_present_long.csv"),
        type=Path,
        help="Market source consumed by train_finn_model.py.",
    )
    parser.add_argument(
        "--regime-source",
        default=Path("Data/processed/spy_regime_features.csv"),
        type=Path,
        help="Regime source consumed by train_finn_model.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("reports/finn_architecture_validation"),
        type=Path,
        help="Directory for per-experiment outputs and aggregate summaries.",
    )
    parser.add_argument(
        "--split-strategy",
        default="expiration",
        choices=("expiration", "timestamp", "random"),
        help="Split strategy forwarded to train_finn_model.py.",
    )
    parser.add_argument("--epochs", default=300, type=int, help="Max training epochs.")
    parser.add_argument("--batch-size", default=64, type=int, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=75,
        type=int,
        help="Early-stopping patience forwarded to train_finn_model.py.",
    )
    parser.add_argument(
        "--random-state",
        default=42,
        type=int,
        help="Random seed forwarded to train_finn_model.py.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Experiment name to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-experiments",
        default=None,
        type=int,
        help="Optional cap for quick smoke validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = _select_experiments(args.include, args.max_experiments)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment in experiments:
        print(f"\n=== Running {experiment.name} ===", flush=True)
        experiment_dir = args.output_dir / experiment.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = experiment_dir / "metrics.json"
        _run_training(args, experiment, experiment_dir, metrics_path)
        rows.append(_summary_row(experiment, metrics_path))

    summary_json = args.output_dir / "summary.json"
    summary_csv = args.output_dir / "summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_summary_csv(summary_csv, rows)
    _print_summary(rows)
    print(f"\nSaved architecture summary to {summary_json}")
    print(f"Saved architecture table to {summary_csv}")


def _select_experiments(
    include: list[str] | None,
    max_experiments: int | None,
) -> list[ArchitectureExperiment]:
    experiments = list(DEFAULT_EXPERIMENTS)
    if include:
        names = set(include)
        experiments = [experiment for experiment in experiments if experiment.name in names]
        missing = sorted(names - {experiment.name for experiment in experiments})
        if missing:
            raise SystemExit(f"Unknown experiments requested: {missing}")
    if max_experiments is not None:
        if max_experiments <= 0:
            raise SystemExit("--max-experiments must be positive.")
        experiments = experiments[:max_experiments]
    if not experiments:
        raise SystemExit("No architecture experiments selected.")
    return experiments


def _run_training(
    args: argparse.Namespace,
    experiment: ArchitectureExperiment,
    experiment_dir: Path,
    metrics_path: Path,
) -> None:
    command = [
        sys.executable,
        "scripts/train_finn_model.py",
        "--option-source",
        str(args.option_source),
        "--market-source",
        str(args.market_source),
        "--regime-source",
        str(args.regime_source),
        "--output",
        str(experiment_dir / "finn_model.pt"),
        "--selection-output",
        str(experiment_dir / "finn_model_selection.pt"),
        "--split-output",
        str(experiment_dir / "finn_split_dataset.csv"),
        "--metrics-output",
        str(metrics_path),
        "--predictions-output",
        str(experiment_dir / "finn_split_predictions.csv"),
        "--split-strategy",
        args.split_strategy,
        "--final-train-split",
        "train_val",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--hidden-dims",
        experiment.hidden_dims,
        "--activation",
        experiment.activation,
        "--prediction-mode",
        "bsm_residual",
        "--residual-scale",
        str(experiment.residual_scale),
        "--lambda-boundary",
        str(experiment.lambda_boundary),
        "--lambda-pde",
        str(experiment.lambda_pde),
        "--lambda-arbitrage",
        str(experiment.lambda_arbitrage),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--random-state",
        str(args.random_state),
    ]
    subprocess.run(command, check=True)


def _summary_row(
    experiment: ArchitectureExperiment,
    metrics_path: Path,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selection_test = metrics["selection_model"]["metrics"]["test"]
    final_test = metrics["final_model"]["test_metrics"]
    return {
        "name": experiment.name,
        "hidden_dims": experiment.hidden_dims,
        "activation": experiment.activation,
        "residual_scale": experiment.residual_scale,
        "selection_best_epoch": metrics["selection_model"]["best_epoch"],
        "selection_test_mae": selection_test["mae"],
        "selection_test_rmse": selection_test["rmse"],
        "final_epochs": metrics["final_model"]["epochs"],
        "final_test_mae": final_test["mae"],
        "final_test_rmse": final_test["rmse"],
        "baseline_test_mae": final_test["baseline_mae"],
        "delta_outside_bounds": final_test["delta_outside_bounds"],
        "gamma_negative": final_test["gamma_negative"],
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: row["final_test_mae"])
    print("\nArchitecture validation summary:")
    print("name final_MAE final_RMSE BSM_MAE gamma_negative")
    for row in ordered:
        print(
            f"{row['name']} "
            f"{row['final_test_mae']:.6f} "
            f"{row['final_test_rmse']:.6f} "
            f"{row['baseline_test_mae']:.6f} "
            f"{row['gamma_negative']}"
        )


if __name__ == "__main__":
    main()

