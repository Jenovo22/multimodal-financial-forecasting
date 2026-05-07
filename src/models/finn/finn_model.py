"""PyTorch implementation of a trainable Finance-Informed Neural Network."""

from __future__ import annotations

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
    hidden_dims: tuple[int, ...] = (64, 64)
    activation: str = "silu"
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

    def fit(self, features: Sequence[FINNInput], targets: Sequence[float]) -> None:
        _require_torch()
        if not features:
            raise ValueError("At least one FINNInput sample is required.")
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same length.")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.config.epochs <= 0:
            raise ValueError("epochs must be positive.")

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
        batch_size = min(self.config.batch_size, len(features))
        for _ in range(self.config.epochs):
            permutation = torch.randperm(len(features), device=device)
            batch_metrics: list[dict[str, float]] = []

            for start in range(0, len(features), batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_inputs = _slice_raw_dataset(dataset, batch_indices, requires_grad=True)
                batch_targets = targets_tensor[batch_indices]
                batch_features = self._encode_feature_matrix(batch_inputs)
                standardized = self._standardize_features(batch_features)
                predictions = self.model_(standardized).reshape(-1)

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

            self.training_history_.append(_mean_metric_dict(batch_metrics))

        self.is_trained = True

    def predict(self, payload: FINNInput) -> FINNOutput:
        _require_torch()
        self._ensure_ready()

        device = self._resolve_device()
        raw_inputs = _build_raw_dataset([payload], device=device, requires_grad=True)
        features = self._encode_feature_matrix(raw_inputs)
        standardized = self._standardize_features(features)
        predictions = self.model_(standardized).reshape(-1)

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
        saved_config = FINNConfig(**checkpoint["config"])
        resolved_config = config or saved_config
        model = cls(resolved_config)
        model.model_ = _FINNNetwork(
            input_dim=saved_config.input_dim,
            hidden_dims=saved_config.hidden_dims,
            activation=saved_config.activation,
        ).to(model._resolve_device())
        model.model_.load_state_dict(checkpoint["state_dict"])
        model.model_.eval()
        model.feature_mean_ = checkpoint["feature_mean"].to(model._resolve_device())
        model.feature_std_ = checkpoint["feature_std"].to(model._resolve_device())
        model.training_history_ = list(checkpoint.get("training_history", []))
        model.is_trained = True
        return model

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


class _FINNNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], activation: str) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        activation_module = _activation_module(activation)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_module())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        raw_output = self.network(features)
        return F.softplus(raw_output).reshape(-1)


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
