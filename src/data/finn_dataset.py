"""Build FINN-ready training features from options, market context and regimes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.contracts import (
    FINNInput,
    REGIME_PROBABILITY_COLUMNS,
    OptionType,
    RegimeLabel,
)
from src.data.dataset_builder import DatasetBuilder


@dataclass(slots=True, frozen=True)
class FINNDatasetConfig:
    timestamp_column: str = "timestamp"
    underlying_symbol_column: str = "underlying_symbol"
    option_symbol_column: str = "option_symbol"
    option_type_column: str = "option_type"
    market_price_column: str = "market_price"
    mid_price_column: str = "mid_price"
    target_price_column: str = "target_price"
    regime_label_column: str = "regime_label"
    sigma_regime_column: str = "sigma_regime"
    probability_columns: tuple[str, str, str] = REGIME_PROBABILITY_COLUMNS
    use_mid_price_if_available: bool = True
    require_complete_regime: bool = True


@dataclass(slots=True, frozen=True)
class FINNTrainingExample:
    timestamp: str
    underlying_symbol: str
    option_symbol: str
    payload: FINNInput
    target_price: float
    market_price: float
    regime_label: RegimeLabel | None = None


def build_finn_training_frame(
    option_source: pd.DataFrame | str | Path,
    market_source: pd.DataFrame | str | Path,
    regime_source: pd.DataFrame | str | Path,
    *,
    dataset_builder: DatasetBuilder | None = None,
    config: FINNDatasetConfig | None = None,
) -> pd.DataFrame:
    """Enrich option rows with market context and attach precomputed HMM regimes."""

    builder = dataset_builder or DatasetBuilder()
    option_frame = builder.build_option_training_dataset(option_source, market_source)
    return attach_regime_features(option_frame, regime_source, config=config)


def attach_regime_features(
    option_frame: pd.DataFrame,
    regime_source: pd.DataFrame | str | Path,
    *,
    config: FINNDatasetConfig | None = None,
) -> pd.DataFrame:
    """Join a validated option frame with a date-level regime table."""

    resolved = config or FINNDatasetConfig()
    options = option_frame.copy()
    regimes = _load_frame(regime_source)
    _validate_regime_source(regimes, resolved)

    options[resolved.timestamp_column] = _normalize_timestamp_series(
        options[resolved.timestamp_column]
    )
    regimes[resolved.timestamp_column] = _normalize_timestamp_series(
        regimes[resolved.timestamp_column]
    )

    regime_columns = [
        resolved.timestamp_column,
        resolved.regime_label_column,
        resolved.sigma_regime_column,
        *resolved.probability_columns,
    ]
    join_keys = [resolved.timestamp_column]
    if resolved.underlying_symbol_column in regimes.columns:
        regimes[resolved.underlying_symbol_column] = (
            regimes[resolved.underlying_symbol_column].astype(str).str.upper()
        )
        options[resolved.underlying_symbol_column] = (
            options[resolved.underlying_symbol_column].astype(str).str.upper()
        )
        regime_columns.insert(1, resolved.underlying_symbol_column)
        join_keys.append(resolved.underlying_symbol_column)

    duplicate_mask = regimes.duplicated(subset=join_keys, keep=False)
    if duplicate_mask.any():
        raise ValueError(f"Regime source contains duplicate rows for join keys: {join_keys}")

    stale_regime_columns = [
        column
        for column in [
            resolved.regime_label_column,
            resolved.sigma_regime_column,
            *resolved.probability_columns,
        ]
        if column in options.columns
    ]
    if stale_regime_columns:
        options = options.drop(columns=stale_regime_columns)

    joined = options.merge(
        regimes[regime_columns],
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    joined[resolved.target_price_column] = _target_price_series(joined, resolved)
    _validate_finn_training_frame(joined, resolved)
    return joined.reset_index(drop=True)


def frame_to_finn_inputs(
    frame: pd.DataFrame,
    *,
    config: FINNDatasetConfig | None = None,
) -> list[FINNInput]:
    """Convert a FINN training frame into contract objects consumed by the model."""

    resolved = config or FINNDatasetConfig()
    working = frame.copy()
    if resolved.target_price_column not in working.columns:
        working[resolved.target_price_column] = _target_price_series(working, resolved)
    _validate_finn_training_frame(working, resolved)

    payloads: list[FINNInput] = []
    for row in working.itertuples(index=False):
        row_map = row._asdict()
        payloads.append(
            FINNInput(
                S=float(row_map["S"]),
                K=float(row_map["K"]),
                T=float(row_map["T"]),
                r=float(row_map["r"]),
                sigma_regime=float(row_map[resolved.sigma_regime_column]),
                option_type=_option_type(row_map[resolved.option_type_column]),
                dividend_yield=_float_or_default_zero(row_map.get("dividend_yield")),
                regime_probabilities=tuple(
                    float(row_map[column]) for column in resolved.probability_columns
                ),
                z_t=_optional_vector(row_map.get("text_embedding")),
            )
        )
    return payloads


def frame_to_finn_targets(
    frame: pd.DataFrame,
    *,
    config: FINNDatasetConfig | None = None,
) -> list[float]:
    """Return the supervised pricing target aligned with frame_to_finn_inputs."""

    resolved = config or FINNDatasetConfig()
    if resolved.target_price_column in frame.columns:
        target = pd.to_numeric(frame[resolved.target_price_column], errors="coerce")
    else:
        target = _target_price_series(frame, resolved)
    if target.isna().any() or target.le(0).any():
        raise ValueError(f"{resolved.target_price_column} must be positive and non-missing.")
    return [float(value) for value in target]


def build_finn_training_examples(
    frame: pd.DataFrame,
    *,
    config: FINNDatasetConfig | None = None,
) -> list[FINNTrainingExample]:
    """Build metadata-rich examples for model training or API smoke tests."""

    resolved = config or FINNDatasetConfig()
    payloads = frame_to_finn_inputs(frame, config=resolved)
    targets = frame_to_finn_targets(frame, config=resolved)
    examples: list[FINNTrainingExample] = []
    for index, row in frame.reset_index(drop=True).iterrows():
        examples.append(
            FINNTrainingExample(
                timestamp=str(row[resolved.timestamp_column]),
                underlying_symbol=str(row[resolved.underlying_symbol_column]),
                option_symbol=str(row[resolved.option_symbol_column]),
                payload=payloads[index],
                target_price=targets[index],
                market_price=float(row[resolved.market_price_column]),
                regime_label=_maybe_regime_label(row.get(resolved.regime_label_column)),
            )
        )
    return examples


def save_finn_training_frame(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist a FINN training frame as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _load_frame(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(Path(source))


def _validate_regime_source(frame: pd.DataFrame, config: FINNDatasetConfig) -> None:
    required = [
        config.timestamp_column,
        config.regime_label_column,
        config.sigma_regime_column,
        *config.probability_columns,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Regime source is missing required columns: {missing}")


def _validate_finn_training_frame(frame: pd.DataFrame, config: FINNDatasetConfig) -> None:
    required = [
        config.timestamp_column,
        config.underlying_symbol_column,
        config.option_symbol_column,
        config.option_type_column,
        "S",
        "K",
        "T",
        "r",
        config.market_price_column,
        config.target_price_column,
        config.sigma_regime_column,
        *config.probability_columns,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"FINN training frame is missing required columns: {missing}")

    if config.require_complete_regime:
        regime_columns = [
            config.regime_label_column,
            config.sigma_regime_column,
            *config.probability_columns,
        ]
        missing_regime_count = frame[regime_columns].isna().any(axis=1).sum()
        if missing_regime_count:
            raise ValueError(f"{missing_regime_count} option rows have no regime match.")

    for column in ["S", "K", "T", config.market_price_column, config.target_price_column]:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or numeric.le(0).any():
            raise ValueError(f"{column} must be positive and non-missing.")

    r = pd.to_numeric(frame["r"], errors="coerce")
    if r.isna().any():
        raise ValueError("r must be numeric and non-missing.")

    sigma = pd.to_numeric(frame[config.sigma_regime_column], errors="coerce")
    if sigma.isna().any() or sigma.le(0).any():
        raise ValueError(f"{config.sigma_regime_column} must be positive and non-missing.")

    if "dividend_yield" in frame.columns:
        dividend_yield = pd.to_numeric(frame["dividend_yield"], errors="coerce").fillna(0.0)
        if dividend_yield.lt(0).any():
            raise ValueError("dividend_yield must be non-negative when present.")

    probabilities = frame[list(config.probability_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    probability_sums = probabilities.sum(axis=1)
    invalid_probabilities = (
        probabilities.isna().any(axis=1)
        | probabilities.lt(0).any(axis=1)
        | probabilities.gt(1).any(axis=1)
        | (probability_sums - 1.0).abs().gt(1e-6)
    )
    if invalid_probabilities.any():
        raise ValueError("Regime probabilities must be in [0, 1] and sum to 1.")

    option_types = frame[config.option_type_column].astype(str).str.lower()
    invalid_option_types = ~option_types.isin(["call", "put"])
    if invalid_option_types.any():
        raise ValueError("option_type must be 'call' or 'put'.")


def _target_price_series(frame: pd.DataFrame, config: FINNDatasetConfig) -> pd.Series:
    market_price = pd.to_numeric(frame[config.market_price_column], errors="coerce")
    if config.use_mid_price_if_available and config.mid_price_column in frame.columns:
        mid_price = pd.to_numeric(frame[config.mid_price_column], errors="coerce")
        return mid_price.where(mid_price.notna(), market_price)
    return market_price


def _option_type(value: object) -> OptionType:
    option_type = str(value).lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    return option_type  # type: ignore[return-value]


def _maybe_regime_label(value: object) -> RegimeLabel | None:
    if value is None or pd.isna(value):
        return None
    label = str(value)
    if label in {
        "stable_low_volatility",
        "stress_high_volatility",
        "transition_uncertainty",
        "unknown",
    }:
        return label  # type: ignore[return-value]
    raise ValueError(f"Invalid regime_label={label!r}.")


def _optional_vector(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(float(item) for item in value)
    return None


def _normalize_timestamp_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _float_or_default_zero(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)
