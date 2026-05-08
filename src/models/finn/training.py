"""Training split and evaluation utilities for FINN models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.baseline import price_black_scholes
from src.contracts import BaselineInput
from src.data import FINNDatasetConfig, frame_to_finn_inputs, frame_to_finn_targets
from src.models.finn.finn_model import FINNPricingModel

SplitStrategy = Literal["expiration", "timestamp", "random"]
FinalTrainSplit = Literal["train", "train_val", "all"]


@dataclass(slots=True, frozen=True)
class FINNSplitConfig:
    strategy: SplitStrategy = "expiration"
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    random_state: int = 42
    timestamp_column: str = "timestamp"
    expiration_column: str = "expiration_date"
    split_column: str = "split"


def split_finn_training_frame(
    frame: pd.DataFrame,
    config: FINNSplitConfig | None = None,
) -> pd.DataFrame:
    """Add a train/validation/test split column to a FINN training frame."""

    resolved = config or FINNSplitConfig()
    _validate_split_fractions(resolved)
    if frame.empty:
        raise ValueError("Cannot split an empty training frame.")
    if resolved.strategy == "expiration":
        return _split_by_ordered_group(
            frame,
            resolved,
            group_column=resolved.expiration_column,
            group_label="expiration",
        )
    if resolved.strategy == "timestamp":
        return _split_by_ordered_group(
            frame,
            resolved,
            group_column=resolved.timestamp_column,
            group_label="timestamp",
        )
    if resolved.strategy == "random":
        return _split_random(frame, resolved)
    raise ValueError(f"Unsupported split strategy: {resolved.strategy!r}")


def frame_for_split(
    frame: pd.DataFrame,
    split_name: str,
    *,
    split_column: str = "split",
) -> pd.DataFrame:
    """Return a copy of the requested split."""

    if split_column not in frame.columns:
        raise ValueError(f"Frame does not contain split column {split_column!r}.")
    return frame.loc[frame[split_column] == split_name].reset_index(drop=True).copy()


def frame_for_final_training(
    split_frame: pd.DataFrame,
    final_train_split: FinalTrainSplit,
    *,
    split_column: str = "split",
) -> pd.DataFrame:
    """Resolve which rows should be used for the final checkpoint."""

    if final_train_split == "train":
        selected = ["train"]
    elif final_train_split == "train_val":
        selected = ["train", "validation"]
    elif final_train_split == "all":
        selected = ["train", "validation", "test"]
    else:
        raise ValueError(f"Unsupported final_train_split={final_train_split!r}.")
    return split_frame.loc[split_frame[split_column].isin(selected)].reset_index(drop=True).copy()


def split_summary(
    split_frame: pd.DataFrame,
    *,
    split_column: str = "split",
) -> dict[str, dict[str, object]]:
    """Return row counts and expiration coverage per split."""

    if split_column not in split_frame.columns:
        raise ValueError(f"Frame does not contain split column {split_column!r}.")
    summary: dict[str, dict[str, object]] = {}
    for split_name, group in split_frame.groupby(split_column):
        expirations = (
            sorted(group["expiration_date"].astype(str).unique())
            if "expiration_date" in group.columns
            else []
        )
        timestamps = (
            sorted(group["timestamp"].astype(str).unique())
            if "timestamp" in group.columns
            else []
        )
        summary[str(split_name)] = {
            "rows": int(len(group)),
            "expirations": expirations,
            "timestamps": timestamps,
            "strike_min": _optional_float(group["K"].min()) if "K" in group.columns else None,
            "strike_max": _optional_float(group["K"].max()) if "K" in group.columns else None,
        }
    return summary


def evaluate_finn_model(
    model: FINNPricingModel,
    frame: pd.DataFrame,
    *,
    config: FINNDatasetConfig | None = None,
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    """Evaluate a trained FINN model and return metrics plus row-level predictions."""

    if frame.empty:
        return _empty_metrics(), pd.DataFrame()
    resolved = config or FINNDatasetConfig()
    payloads = frame_to_finn_inputs(frame, config=resolved)
    targets = frame_to_finn_targets(frame, config=resolved)

    predictions = []
    for payload in payloads:
        output = model.predict(payload)
        predictions.append(
            {
                "fair_value": output.fair_value,
                "delta": output.delta,
                "gamma": output.gamma,
                "vega": output.vega,
                "theta": output.theta,
                "pde_residual": output.pde_residual,
            }
        )

    prediction_frame = frame.reset_index(drop=True).copy()
    for column in ["fair_value", "delta", "gamma", "vega", "theta", "pde_residual"]:
        prediction_frame[f"predicted_{column}"] = [
            item[column] for item in predictions
        ]
    prediction_frame["target_price"] = targets
    prediction_frame["prediction_error"] = (
        prediction_frame["predicted_fair_value"] - prediction_frame["target_price"]
    )
    prediction_frame["absolute_error"] = prediction_frame["prediction_error"].abs()
    prediction_frame["squared_error"] = prediction_frame["prediction_error"].pow(2)

    baseline_prices = _baseline_prices(prediction_frame)
    if baseline_prices is not None:
        prediction_frame["baseline_bs_price"] = baseline_prices
        baseline_error = prediction_frame["baseline_bs_price"] - prediction_frame["target_price"]
    else:
        baseline_error = None

    metrics = _prediction_metrics(
        prediction_frame,
        baseline_error=baseline_error,
    )
    return metrics, prediction_frame


def _split_by_ordered_group(
    frame: pd.DataFrame,
    config: FINNSplitConfig,
    *,
    group_column: str,
    group_label: str,
) -> pd.DataFrame:
    if group_column not in frame.columns:
        raise ValueError(f"{group_label.title()} split requires column {group_column!r}.")
    groups = sorted(frame[group_column].astype(str).unique())
    if len(groups) < 3:
        raise ValueError(f"{group_label.title()} split requires at least three groups.")
    train_count, validation_count = _split_group_counts(
        len(groups),
        config.train_fraction,
        config.validation_fraction,
    )
    train_groups = set(groups[:train_count])
    validation_groups = set(groups[train_count : train_count + validation_count])

    split_frame = frame.copy()
    split_frame[config.split_column] = "test"
    split_frame.loc[
        split_frame[group_column].astype(str).isin(train_groups),
        config.split_column,
    ] = "train"
    split_frame.loc[
        split_frame[group_column].astype(str).isin(validation_groups),
        config.split_column,
    ] = "validation"
    return split_frame.reset_index(drop=True)


def _split_random(frame: pd.DataFrame, config: FINNSplitConfig) -> pd.DataFrame:
    shuffled = frame.sample(frac=1.0, random_state=config.random_state).reset_index(drop=True)
    train_count = max(1, int(len(shuffled) * config.train_fraction))
    validation_count = max(1, int(len(shuffled) * config.validation_fraction))
    if train_count + validation_count >= len(shuffled):
        validation_count = max(1, len(shuffled) - train_count - 1)
    split_frame = shuffled.copy()
    split_frame[config.split_column] = "test"
    split_frame.loc[: train_count - 1, config.split_column] = "train"
    split_frame.loc[
        train_count : train_count + validation_count - 1,
        config.split_column,
    ] = "validation"
    return split_frame


def _split_group_counts(
    group_count: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[int, int]:
    train_count = max(1, int(math.floor(group_count * train_fraction)))
    validation_count = max(1, int(round(group_count * validation_fraction)))
    if train_count + validation_count >= group_count:
        train_count = max(1, group_count - 2)
        validation_count = 1
    return train_count, validation_count


def _validate_split_fractions(config: FINNSplitConfig) -> None:
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    if config.train_fraction + config.validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1.")


def _prediction_metrics(
    prediction_frame: pd.DataFrame,
    *,
    baseline_error: pd.Series | None,
) -> dict[str, float | int | None]:
    target = prediction_frame["target_price"].clip(lower=0.01)
    error = prediction_frame["prediction_error"]
    abs_error = error.abs()
    metrics: dict[str, float | int | None] = {
        "rows": int(len(prediction_frame)),
        "mae": float(abs_error.mean()),
        "rmse": float(prediction_frame["squared_error"].mean() ** 0.5),
        "mean_error": float(error.mean()),
        "mape_pct": float(((abs_error / target) * 100.0).mean()),
        "target_mean": float(prediction_frame["target_price"].mean()),
        "prediction_mean": float(prediction_frame["predicted_fair_value"].mean()),
        "delta_outside_bounds": int(_delta_outside_bounds(prediction_frame).sum()),
        "gamma_negative": int((prediction_frame["predicted_gamma"] < -1e-6).sum()),
    }
    if baseline_error is not None:
        metrics["baseline_mae"] = float(baseline_error.abs().mean())
        metrics["baseline_rmse"] = float((baseline_error.pow(2).mean()) ** 0.5)
    else:
        metrics["baseline_mae"] = None
        metrics["baseline_rmse"] = None
    return metrics


def _baseline_prices(frame: pd.DataFrame) -> pd.Series | None:
    required = ["S", "K", "T", "r", "implied_volatility", "option_type"]
    if any(column not in frame.columns for column in required):
        return None
    prices = []
    for row in frame.itertuples(index=False):
        row_map = row._asdict()
        try:
            prices.append(
                price_black_scholes(
                    BaselineInput(
                        S=float(row_map["S"]),
                        K=float(row_map["K"]),
                        T=float(row_map["T"]),
                        r=float(row_map["r"]),
                        sigma=float(row_map["implied_volatility"]),
                        option_type=str(row_map["option_type"]).lower(),  # type: ignore[arg-type]
                        dividend_yield=_optional_float(row_map.get("dividend_yield")) or 0.0,
                    )
                ).bs_price
            )
        except (ValueError, TypeError):
            prices.append(float("nan"))
    return pd.Series(prices, index=frame.index)


def _delta_outside_bounds(frame: pd.DataFrame) -> pd.Series:
    option_type = frame["option_type"].astype(str).str.lower()
    delta = pd.to_numeric(frame["predicted_delta"], errors="coerce")
    call_invalid = option_type.eq("call") & ((delta < -1e-6) | (delta > 1.0 + 1e-6))
    put_invalid = option_type.eq("put") & ((delta < -1.0 - 1e-6) | (delta > 1e-6))
    return call_invalid | put_invalid


def _empty_metrics() -> dict[str, float | int | None]:
    return {
        "rows": 0,
        "mae": None,
        "rmse": None,
        "mean_error": None,
        "mape_pct": None,
        "target_mean": None,
        "prediction_mean": None,
        "delta_outside_bounds": 0,
        "gamma_negative": 0,
        "baseline_mae": None,
        "baseline_rmse": None,
    }


def _optional_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
