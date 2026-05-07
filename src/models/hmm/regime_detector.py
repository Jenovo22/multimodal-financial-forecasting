"""Gaussian HMM market regime detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from hmmlearn.hmm import GaussianHMM

from src.contracts import HMMInput, HMMOutput, REGIME_LABEL_ORDER, RegimeLabel

DEFAULT_REGIME_LABELS = REGIME_LABEL_ORDER

DEFAULT_FEATURE_ORDER = (
    "return_1d",
    "return_5d",
    "realized_volatility",
    "implied_volatility",
    "volume",
)

DEFAULT_VOLATILITY_FEATURES = ("realized_volatility", "implied_volatility")


@dataclass(slots=True, frozen=True)
class HMMRegimeConfig:
    n_states: int = 3
    labels: tuple[RegimeLabel, RegimeLabel, RegimeLabel] = DEFAULT_REGIME_LABELS
    sigma_floor: float = 1e-6
    feature_order: tuple[str, ...] = DEFAULT_FEATURE_ORDER
    volatility_feature_names: tuple[str, ...] = DEFAULT_VOLATILITY_FEATURES
    covariance_type: Literal["diag", "full", "spherical", "tied"] = "diag"
    volume_transform: Literal["log1p", "none"] = "log1p"
    n_iter: int = 200
    tol: float = 1e-3
    min_covar: float = 1e-6
    random_state: int = 42


class HMMRegimeDetector:
    """Fit and apply a Gaussian HMM on market context features."""

    def __init__(self, config: HMMRegimeConfig | None = None) -> None:
        self.config = config or HMMRegimeConfig()
        if self.config.n_states != len(self.config.labels):
            raise ValueError("n_states and labels length must match.")

        self.is_fitted = False
        self.model: GaussianHMM | None = None
        self.feature_medians_: np.ndarray | None = None
        self.feature_means_: np.ndarray | None = None
        self.feature_stds_: np.ndarray | None = None
        self.state_sigma_values_: np.ndarray | None = None
        self.state_to_label_: dict[int, RegimeLabel] = {}

    def fit(self, samples: Sequence[HMMInput]) -> None:
        raw_matrix = self._extract_feature_matrix(samples)
        standardized = self._fit_feature_pipeline(raw_matrix)

        self.model = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            min_covar=self.config.min_covar,
            random_state=self.config.random_state,
        )
        self.model.fit(standardized)

        posterior = self.model.predict_proba(standardized)
        self.state_sigma_values_ = self._estimate_state_sigma_values(raw_matrix, posterior)
        self.state_to_label_ = self._build_state_label_mapping(self.state_sigma_values_)
        self.is_fitted = True

    def predict(self, sample: HMMInput) -> HMMOutput:
        self._ensure_fitted()
        outputs = self.predict_many([sample])
        return outputs[0]

    def predict_many(self, samples: Sequence[HMMInput]) -> list[HMMOutput]:
        self._ensure_fitted()
        raw_matrix = self._extract_feature_matrix(samples)
        standardized = self._transform_with_fitted_pipeline(raw_matrix)
        posterior = self.model.predict_proba(standardized)

        assert self.state_sigma_values_ is not None
        outputs: list[HMMOutput] = []
        for row in posterior:
            label_probabilities = self.probabilities_by_label(row)
            label_index = int(np.argmax(label_probabilities))
            sigma_regime = max(float(np.dot(row, self.state_sigma_values_)), self.config.sigma_floor)
            outputs.append(
                HMMOutput(
                    regime_label=self.config.labels[label_index],
                    regime_probabilities=label_probabilities,
                    sigma_regime=sigma_regime,
                )
            )
        return outputs

    def probabilities_by_label(self, state_probabilities: Sequence[float]) -> tuple[float, ...]:
        """Aggregate raw HMM state probabilities into the configured label order."""

        values: dict[RegimeLabel, float] = {label: 0.0 for label in self.config.labels}
        for state_index, probability in enumerate(state_probabilities):
            label = self.label_from_state(state_index)
            if label != "unknown":
                values[label] = values.get(label, 0.0) + float(probability)

        return tuple(values.get(label, 0.0) for label in self.config.labels)

    def label_from_state(self, state_index: int) -> RegimeLabel:
        if self.state_to_label_:
            return self.state_to_label_.get(state_index, "unknown")
        if 0 <= state_index < len(self.config.labels):
            return self.config.labels[state_index]
        return "unknown"

    def _fit_feature_pipeline(self, raw_matrix: np.ndarray) -> np.ndarray:
        if raw_matrix.shape[0] < self.config.n_states:
            raise ValueError("Need at least as many samples as HMM states to fit the detector.")

        transformed = self._apply_raw_transforms(raw_matrix)
        medians = np.nanmedian(transformed, axis=0)
        if np.isnan(medians).any():
            missing_columns = [
                self.config.feature_order[index]
                for index, value in enumerate(medians)
                if np.isnan(value)
            ]
            raise ValueError(
                "Cannot fit HMM because some features are entirely missing: "
                f"{missing_columns}"
            )

        imputed = np.where(np.isnan(transformed), medians, transformed)
        means = imputed.mean(axis=0)
        stds = np.maximum(imputed.std(axis=0), self.config.sigma_floor)

        self.feature_medians_ = medians
        self.feature_means_ = means
        self.feature_stds_ = stds
        return (imputed - means) / stds

    def _transform_with_fitted_pipeline(self, raw_matrix: np.ndarray) -> np.ndarray:
        transformed = self._apply_raw_transforms(raw_matrix)

        assert self.feature_medians_ is not None
        assert self.feature_means_ is not None
        assert self.feature_stds_ is not None

        imputed = np.where(np.isnan(transformed), self.feature_medians_, transformed)
        return (imputed - self.feature_means_) / self.feature_stds_

    def _extract_feature_matrix(self, samples: Sequence[HMMInput]) -> np.ndarray:
        if not samples:
            raise ValueError("At least one HMMInput sample is required.")

        rows: list[list[float]] = []
        for sample in samples:
            features = sample.features
            rows.append(
                [
                    self._to_float(getattr(features, feature_name))
                    for feature_name in self.config.feature_order
                ]
            )
        return np.asarray(rows, dtype=float)

    def _apply_raw_transforms(self, raw_matrix: np.ndarray) -> np.ndarray:
        transformed = raw_matrix.copy()
        if self.config.volume_transform == "log1p" and "volume" in self.config.feature_order:
            volume_index = self.config.feature_order.index("volume")
            column = transformed[:, volume_index]
            valid_mask = ~np.isnan(column)
            column[valid_mask] = np.log1p(np.clip(column[valid_mask], a_min=0.0, a_max=None))
            transformed[:, volume_index] = column
        return transformed

    def _estimate_state_sigma_values(
        self,
        raw_matrix: np.ndarray,
        posterior: np.ndarray,
    ) -> np.ndarray:
        volatility_indices = [
            self.config.feature_order.index(feature_name)
            for feature_name in self.config.volatility_feature_names
            if feature_name in self.config.feature_order
        ]
        if not volatility_indices:
            raise ValueError("At least one volatility feature is required to label HMM states.")

        volatility_block = raw_matrix[:, volatility_indices]
        row_volatility = np.nanmean(volatility_block, axis=1)

        sigma_values = np.full(self.config.n_states, self.config.sigma_floor, dtype=float)
        valid_rows = ~np.isnan(row_volatility)
        if not np.any(valid_rows):
            return sigma_values

        for state_index in range(self.config.n_states):
            state_weights = posterior[valid_rows, state_index]
            weight_sum = float(state_weights.sum())
            if weight_sum <= 0:
                continue
            sigma_values[state_index] = max(
                float(np.dot(state_weights, row_volatility[valid_rows]) / weight_sum),
                self.config.sigma_floor,
            )
        return sigma_values

    def _build_state_label_mapping(self, sigma_values: np.ndarray) -> dict[int, RegimeLabel]:
        if len(sigma_values) != self.config.n_states:
            raise ValueError("State sigma vector length does not match n_states.")

        ranked_states = np.argsort(sigma_values)
        mapping: dict[int, RegimeLabel] = {}
        if self.config.n_states == 3:
            mapping[int(ranked_states[0])] = "stable_low_volatility"
            mapping[int(ranked_states[-1])] = "stress_high_volatility"
            middle_states = [int(index) for index in ranked_states[1:-1]]
            for state_index in middle_states:
                mapping[state_index] = "transition_uncertainty"
        else:
            for position, state_index in enumerate(ranked_states):
                label_index = min(position, len(self.config.labels) - 1)
                mapping[int(state_index)] = self.config.labels[label_index]

        for state_index in range(self.config.n_states):
            mapping.setdefault(state_index, "unknown")
        return mapping

    def _ensure_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("The HMMRegimeDetector must be fitted before prediction.")

    def _to_float(self, value: object) -> float:
        if value is None:
            return np.nan
        return float(value)
