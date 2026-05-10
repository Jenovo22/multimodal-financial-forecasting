"""PyTorch implementation of a trainable Finance-Informed Neural Network."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - exercised indirectly when torch is absent.
    torch = None
    Tensor = Any
    nn = Any
    F = None

from src.contracts import FINNInput, FINNOutput
from src.models.finn.pde_loss import (
    PDELossWeights,
    arbitrage_regularization_loss,
    black_scholes_pde_residual,
    boundary_condition_loss,
    total_finn_loss,
)


@dataclass(slots=True, frozen=True)
class FINNConfig:
    input_dim: int = 12
    hidden_dims: tuple[int, ...] = (32, 32)
    activation: str = "silu"
    prediction_mode: str = "bsm_residual"
    residual_scale: float = 0.50
    residual_anchor_floor: float = 1.0
    residual_experts: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 64
    epochs: int = 300
    lambda_data: float = 1.0
    lambda_boundary: float = 0.1
    lambda_pde: float = 0.0
    lambda_arbitrage: float = 1.0
    feature_std_floor: float = 1e-6
    gradient_clip_norm: float | None = 5.0
    early_stopping_patience: int | None = 50
    early_stopping_min_delta: float = 1e-5
    random_state: int = 42
    device: str = "cpu"


class FINNPricingModel:
    """Trainable pricing model with PDE and no-arbitrage regularization."""

    def __init__(self, config: FINNConfig | None = None) -> None:
        self.config = config or FINNConfig()
        self.is_trained = False
        self.model_: nn.Module | None = None
        self.feature_mean_: Tensor | None = None
        self.feature_std_: Tensor | None = None
        self.training_history_: list[dict[str, float]] = []
        self.best_epoch_: int | None = None
        self.best_validation_loss_: float | None = None

    def fit(
        self,
        features: Sequence[FINNInput],
        targets: Sequence[float],
        *,
        validation_features: Sequence[FINNInput] | None = None,
        validation_targets: Sequence[float] | None = None,
    ) -> None:
        _require_torch()
        if not features:
            raise ValueError("At least one FINNInput sample is required.")
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same length.")
        if (validation_features is None) != (validation_targets is None):
            raise ValueError("validation_features and validation_targets must be provided together.")
        if validation_features is not None and len(validation_features) != len(validation_targets):
            raise ValueError("validation_features and validation_targets must have the same length.")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.config.epochs <= 0:
            raise ValueError("epochs must be positive.")
        valid_prediction_modes = {"direct", "bsm_residual", "bsm_residual_mixture"}
        if self.config.prediction_mode not in valid_prediction_modes:
            raise ValueError(
                "prediction_mode must be 'direct', 'bsm_residual' or "
                "'bsm_residual_mixture'."
            )
        if self.config.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative.")
        if self.config.residual_anchor_floor <= 0:
            raise ValueError("residual_anchor_floor must be positive.")
        if (
            self.config.prediction_mode == "bsm_residual_mixture"
            and self.config.residual_experts < 2
        ):
            raise ValueError(
                "residual_experts must be at least 2 for bsm_residual_mixture."
            )

        torch.manual_seed(self.config.random_state)
        device = self._resolve_device()

        dataset = _build_raw_dataset(features, device=device)
        targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device).reshape(-1)

        feature_matrix = self._encode_feature_matrix(dataset)
        self.feature_mean_ = feature_matrix.mean(dim=0)
        self.feature_std_ = _stable_feature_std(
            feature_matrix,
            floor=self.config.feature_std_floor,
        )

        self.model_ = _FINNNetwork(
            input_dim=feature_matrix.shape[1],
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
            output_dim=_network_output_dim(self.config),
            zero_initialize_output=_uses_bsm_anchor(self.config.prediction_mode),
        ).to(device)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        weights = PDELossWeights(
            lambda_data=self.config.lambda_data,
            lambda_boundary=self.config.lambda_boundary,
            lambda_pde=self.config.lambda_pde,
            lambda_arbitrage=self.config.lambda_arbitrage,
        )

        self.training_history_ = []
        self.best_epoch_ = None
        self.best_validation_loss_ = None
        best_state_dict: dict[str, Tensor] | None = None
        epochs_without_improvement = 0
        batch_size = min(self.config.batch_size, len(features))
        for epoch in range(self.config.epochs):
            self.model_.train()
            permutation = torch.randperm(len(features), device=device)
            batch_metrics: list[dict[str, float]] = []

            for start in range(0, len(features), batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_inputs = _slice_raw_dataset(dataset, batch_indices, requires_grad=True)
                batch_targets = targets_tensor[batch_indices]
                batch_features = self._encode_feature_matrix(batch_inputs)
                standardized = self._standardize_features(batch_features)
                raw_predictions = self.model_(standardized)
                predictions = self._price_from_network(batch_inputs, raw_predictions)

                data_loss = F.mse_loss(predictions, batch_targets)
                boundary_loss = boundary_condition_loss(batch_inputs, predictions)
                pde_loss = black_scholes_pde_residual(batch_inputs, predictions)
                arbitrage_loss = arbitrage_regularization_loss(batch_inputs, predictions)
                loss = total_finn_loss(
                    data_loss,
                    boundary_loss,
                    pde_loss,
                    arbitrage_loss_value=arbitrage_loss,
                    weights=weights,
                )

                optimizer.zero_grad()
                loss.backward()
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_.parameters(),
                        self.config.gradient_clip_norm,
                    )
                optimizer.step()

                batch_metrics.append(
                    {
                        "total_loss": float(loss.detach().cpu()),
                        "data_loss": float(data_loss.detach().cpu()),
                        "boundary_loss": float(boundary_loss.detach().cpu()),
                        "pde_loss": float(pde_loss.detach().cpu()),
                        "arbitrage_loss": float(arbitrage_loss.detach().cpu()),
                    }
                )

            epoch_metrics = _mean_metric_dict(batch_metrics)
            if validation_features is not None and validation_targets is not None:
                validation_data_loss = self._validation_data_loss(
                    validation_features,
                    validation_targets,
                )
                epoch_metrics["validation_data_loss"] = validation_data_loss
                improved = (
                    self.best_validation_loss_ is None
                    or validation_data_loss
                    < self.best_validation_loss_ - self.config.early_stopping_min_delta
                )
                if improved:
                    self.best_validation_loss_ = validation_data_loss
                    self.best_epoch_ = epoch + 1
                    best_state_dict = {
                        name: value.detach().cpu().clone()
                        for name, value in self.model_.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            self.training_history_.append(epoch_metrics)

            if (
                validation_features is not None
                and self.config.early_stopping_patience is not None
                and epochs_without_improvement >= self.config.early_stopping_patience
            ):
                break

        if best_state_dict is not None:
            self.model_.load_state_dict(
                {name: value.to(device) for name, value in best_state_dict.items()}
            )
        elif self.best_epoch_ is None:
            self.best_epoch_ = len(self.training_history_)

        self.is_trained = True

    def predict(self, payload: FINNInput) -> FINNOutput:
        _require_torch()
        self._ensure_ready()

        device = self._resolve_device()
        raw_inputs = _build_raw_dataset([payload], device=device, requires_grad=True)
        features = self._encode_feature_matrix(raw_inputs)
        standardized = self._standardize_features(features)
        raw_predictions = self.model_(standardized)
        predictions = self._price_from_network(raw_inputs, raw_predictions)

        fair_value = predictions[0]
        dV_dS = torch.autograd.grad(
            fair_value,
            raw_inputs["S"],
            create_graph=True,
            retain_graph=True,
        )[0][0]
        d2V_dS2 = torch.autograd.grad(
            dV_dS,
            raw_inputs["S"],
            create_graph=True,
            retain_graph=True,
        )[0][0]
        dV_dSigma = torch.autograd.grad(
            fair_value,
            raw_inputs["sigma_regime"],
            create_graph=True,
            retain_graph=True,
        )[0][0]
        dV_dT = torch.autograd.grad(
            fair_value,
            raw_inputs["T"],
            create_graph=True,
            retain_graph=True,
        )[0][0]
        pde_residual = black_scholes_pde_residual(raw_inputs, predictions)

        return FINNOutput(
            fair_value=float(fair_value.detach().cpu()),
            delta=float(dV_dS.detach().cpu()),
            gamma=float(d2V_dS2.detach().cpu()),
            vega=float(dV_dSigma.detach().cpu()),
            theta=float((-dV_dT).detach().cpu()),
            pde_residual=float(pde_residual.detach().cpu()),
        )

    def save(self, output_path: str | Path) -> Path:
        _require_torch()
        self._ensure_ready()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.model_.state_dict(),
                "feature_mean": self.feature_mean_.detach().cpu(),
                "feature_std": self.feature_std_.detach().cpu(),
                "training_history": self.training_history_,
                "best_epoch": self.best_epoch_,
                "best_validation_loss": self.best_validation_loss_,
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls,
        source_path: str | Path,
        config: FINNConfig | None = None,
    ) -> "FINNPricingModel":
        _require_torch()

        checkpoint = torch.load(Path(source_path), map_location="cpu")
        saved_config_payload = dict(checkpoint["config"])
        saved_config_payload.setdefault("residual_experts", 3)
        saved_config = FINNConfig(**saved_config_payload)
        resolved_config = config or saved_config
        model = cls(resolved_config)
        model.model_ = _FINNNetwork(
            input_dim=saved_config.input_dim,
            hidden_dims=saved_config.hidden_dims,
            activation=saved_config.activation,
            output_dim=_network_output_dim(saved_config),
            zero_initialize_output=False,
        ).to(model._resolve_device())
        model.model_.load_state_dict(checkpoint["state_dict"])
        model.model_.eval()
        model.feature_mean_ = checkpoint["feature_mean"].to(model._resolve_device())
        model.feature_std_ = checkpoint["feature_std"].to(model._resolve_device())
        model.training_history_ = list(checkpoint.get("training_history", []))
        model.best_epoch_ = checkpoint.get("best_epoch")
        model.best_validation_loss_ = checkpoint.get("best_validation_loss")
        model.is_trained = True
        return model

    def _validation_data_loss(
        self,
        features: Sequence[FINNInput],
        targets: Sequence[float],
    ) -> float:
        assert torch is not None
        assert F is not None
        assert self.model_ is not None
        device = self._resolve_device()
        dataset = _build_raw_dataset(features, device=device)
        targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device).reshape(-1)
        with torch.no_grad():
            feature_matrix = self._encode_feature_matrix(dataset)
            standardized = self._standardize_features(feature_matrix)
            raw_predictions = self.model_(standardized)
            predictions = self._price_from_network(dataset, raw_predictions)
            loss = F.mse_loss(predictions, targets_tensor)
        return float(loss.detach().cpu())

    def _encode_feature_matrix(self, raw_inputs: dict[str, Tensor]) -> Tensor:
        S = raw_inputs["S"]
        K = raw_inputs["K"]
        T = raw_inputs["T"]
        r = raw_inputs["r"]
        sigma = raw_inputs["sigma_regime"]
        option_sign = raw_inputs["option_type_sign"]
        dividend_yield = raw_inputs["dividend_yield"]
        probabilities = raw_inputs["regime_probabilities"]

        log_moneyness = torch.log(S / K)
        sqrt_T = torch.sqrt(T)
        feature_columns = [
            S,
            K,
            log_moneyness,
            T,
            sqrt_T,
            r,
            sigma,
            option_sign,
            dividend_yield,
            probabilities[:, 0],
            probabilities[:, 1],
            probabilities[:, 2],
        ]
        return torch.stack(feature_columns, dim=1)

    def _standardize_features(self, features: Tensor) -> Tensor:
        assert self.feature_mean_ is not None
        assert self.feature_std_ is not None
        return (features - self.feature_mean_) / self.feature_std_

    def _price_from_network(
        self,
        raw_inputs: dict[str, Tensor],
        raw_output: Tensor,
    ) -> Tensor:
        assert torch is not None
        assert F is not None
        if self.config.prediction_mode == "direct":
            return F.softplus(raw_output).reshape(-1)
        if self.config.prediction_mode == "bsm_residual":
            anchor = _torch_black_scholes_price(raw_inputs)
            raw_residual = raw_output.reshape(-1)
            residual_scale = (
                torch.clamp(
                    anchor.detach().abs(),
                    min=self.config.residual_anchor_floor,
                )
                * self.config.residual_scale
            )
            return torch.clamp(anchor + residual_scale * torch.tanh(raw_residual), min=0.0)
        if self.config.prediction_mode == "bsm_residual_mixture":
            anchor = _torch_black_scholes_price(raw_inputs)
            mixture_outputs = raw_output.reshape(-1, self.config.residual_experts * 2)
            raw_residuals = mixture_outputs[:, : self.config.residual_experts]
            gate_logits = mixture_outputs[:, self.config.residual_experts :]
            gate_weights = torch.softmax(gate_logits, dim=1)
            blended_residual = torch.sum(gate_weights * torch.tanh(raw_residuals), dim=1)
            residual_scale = (
                torch.clamp(
                    anchor.detach().abs(),
                    min=self.config.residual_anchor_floor,
                )
                * self.config.residual_scale
            )
            return torch.clamp(anchor + residual_scale * blended_residual, min=0.0)
        raise ValueError(
            "prediction_mode must be 'direct', 'bsm_residual' or "
            "'bsm_residual_mixture'."
        )

    def _resolve_device(self) -> "torch.device":
        assert torch is not None
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _ensure_ready(self) -> None:
        if not self.is_trained or self.model_ is None:
            raise RuntimeError("The FINNPricingModel must be trained or loaded before prediction.")
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("Feature normalization statistics are missing.")


def _uses_bsm_anchor(prediction_mode: str) -> bool:
    return prediction_mode in {"bsm_residual", "bsm_residual_mixture"}


def _network_output_dim(config: FINNConfig) -> int:
    if config.prediction_mode == "bsm_residual_mixture":
        return config.residual_experts * 2
    return 1


class _FINNNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        activation: str,
        *,
        output_dim: int,
        zero_initialize_output: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        activation_module = _activation_module(activation)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_module())
            current_dim = hidden_dim
        output_layer = nn.Linear(current_dim, output_dim)
        if zero_initialize_output:
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


SQRT_TWO = math.sqrt(2.0)


def _torch_black_scholes_price(raw_inputs: dict[str, Tensor]) -> Tensor:
    assert torch is not None

    S = raw_inputs["S"]
    K = raw_inputs["K"]
    T = raw_inputs["T"]
    r = raw_inputs["r"]
    sigma = raw_inputs["sigma_regime"]
    q = raw_inputs["dividend_yield"]
    option_type_sign = raw_inputs["option_type_sign"]

    sigma_sqrt_t = sigma * torch.sqrt(T)
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma.square()) * T) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    discount_r = torch.exp(-r * T)
    discount_q = torch.exp(-q * T)

    call = S * discount_q * _torch_normal_cdf(d1) - K * discount_r * _torch_normal_cdf(d2)
    put = K * discount_r * _torch_normal_cdf(-d2) - S * discount_q * _torch_normal_cdf(-d1)
    return torch.where(option_type_sign > 0, call, put)


def _torch_normal_cdf(values: Tensor) -> Tensor:
    assert torch is not None
    return 0.5 * (1.0 + torch.erf(values / SQRT_TWO))


def _build_raw_dataset(
    features: Sequence[FINNInput],
    *,
    device: "torch.device",
    requires_grad: bool = False,
) -> dict[str, Tensor]:
    assert torch is not None

    def build(values: list[float], grad: bool = False) -> Tensor:
        tensor = torch.tensor(values, dtype=torch.float32, device=device)
        if grad:
            tensor = tensor.clone().detach().requires_grad_(True)
        return tensor

    regime_probabilities = [
        _probabilities_or_uniform(sample.regime_probabilities)
        for sample in features
    ]
    return {
        "S": build([sample.S for sample in features], grad=requires_grad),
        "K": build([sample.K for sample in features]),
        "T": build([sample.T for sample in features], grad=requires_grad),
        "r": build([sample.r for sample in features]),
        "sigma_regime": build(
            [sample.sigma_regime for sample in features],
            grad=requires_grad,
        ),
        "dividend_yield": build([sample.dividend_yield for sample in features]),
        "option_type_sign": build(
            [1.0 if sample.option_type == "call" else -1.0 for sample in features]
        ),
        "regime_probabilities": torch.tensor(
            regime_probabilities,
            dtype=torch.float32,
            device=device,
        ),
    }


def _slice_raw_dataset(
    dataset: dict[str, Tensor],
    indices: Tensor,
    *,
    requires_grad: bool,
) -> dict[str, Tensor]:
    assert torch is not None

    sliced: dict[str, Tensor] = {}
    for name, tensor in dataset.items():
        values = tensor[indices]
        if requires_grad and name in {"S", "T", "sigma_regime"}:
            values = values.clone().detach().requires_grad_(True)
        else:
            values = values.clone()
        sliced[name] = values
    return sliced


def _probabilities_or_uniform(
    probabilities: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    if probabilities is None:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return tuple(float(value) for value in probabilities)  # type: ignore[return-value]


def _activation_module(name: str) -> type[nn.Module]:
    assert nn is not Any
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU
    if normalized == "tanh":
        return nn.Tanh
    if normalized == "gelu":
        return nn.GELU
    return nn.SiLU


def _stable_feature_std(features: Tensor, *, floor: float) -> Tensor:
    """Avoid exploding input gradients for columns that are constant in one chain."""

    assert torch is not None
    raw_std = features.std(dim=0, unbiased=False)
    return torch.where(raw_std < floor, torch.ones_like(raw_std), raw_std)


def _mean_metric_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {
            "total_loss": 0.0,
            "data_loss": 0.0,
            "boundary_loss": 0.0,
            "pde_loss": 0.0,
            "arbitrage_loss": 0.0,
        }
    return {
        key: sum(item[key] for item in items) / len(items)
        for key in items[0]
    }


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for FINNPricingModel. Install the optional 'ml' dependency."
        )
