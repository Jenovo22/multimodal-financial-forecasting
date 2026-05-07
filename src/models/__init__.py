"""Model packages for HMM and FINN components."""

from src.models.finn.finn_model import FINNConfig, FINNPricingModel
from src.models.finn.pde_loss import (
    PDELossWeights,
    arbitrage_regularization_loss,
    black_scholes_pde_residual,
    boundary_condition_loss,
)
from src.models.finn.regime_adjusted import (
    RegimeAdjustedBlackScholesConfig,
    RegimeAdjustedBlackScholesFINN,
)
from src.models.hmm.regime_detector import HMMRegimeConfig, HMMRegimeDetector
from src.models.hmm.regime_table_detector import (
    RegimeTableDetector,
    RegimeTableDetectorConfig,
)

__all__ = [
    "FINNConfig",
    "FINNPricingModel",
    "PDELossWeights",
    "arbitrage_regularization_loss",
    "black_scholes_pde_residual",
    "boundary_condition_loss",
    "HMMRegimeConfig",
    "HMMRegimeDetector",
    "RegimeTableDetector",
    "RegimeTableDetectorConfig",
    "RegimeAdjustedBlackScholesConfig",
    "RegimeAdjustedBlackScholesFINN",
]
