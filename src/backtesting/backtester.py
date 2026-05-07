"""Skeleton for the backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.contracts import SystemOutput


@dataclass(slots=True, frozen=True)
class BacktestConfig:
    commission_per_contract: float = 0.0
    slippage_bps: float = 0.0
    min_liquidity: float = 0.0
    max_gross_exposure: float = 1.0


@dataclass(slots=True, frozen=True)
class BacktestMetrics:
    cumulative_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    turnover: float = 0.0
    average_exposure: float = 0.0


class Backtester:
    """Placeholder for walk-forward execution and portfolio accounting."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, scored_rows: Sequence[SystemOutput]) -> BacktestMetrics:
        raise NotImplementedError("Implement the backtesting loop here.")
