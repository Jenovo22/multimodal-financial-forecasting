from src.contracts import DatasetRow
from src.data.dataset_builder import DatasetBuilder, validate_dataset_row
from src.models.finn.pde_loss import PDELossWeights, total_finn_loss
from src.pipeline.hmm_finn_pipeline import signal_from_edge


def test_dataset_builder_reports_missing_columns():
    builder = DatasetBuilder()
    missing = builder.missing_required_columns(["timestamp", "S"])
    assert "underlying_symbol" in missing
    assert "implied_volatility" in missing


def test_validate_dataset_row_flags_invalid_values():
    row = DatasetRow(
        timestamp="2026-04-01T00:00:00Z",
        underlying_symbol="SPY",
        option_symbol="",
        option_type="call",
        S=0.0,
        K=520.0,
        T=0.1,
        r=0.04,
        market_price=8.4,
        implied_volatility=0.2,
    )
    issues = validate_dataset_row(row)
    assert "option_symbol is required" in issues
    assert "S must be > 0" in issues


def test_signal_from_edge_uses_cost_threshold():
    assert signal_from_edge(0.50, transaction_cost_estimate=0.10, safety_margin=0.05) == "BUY"
    assert signal_from_edge(-0.50, transaction_cost_estimate=0.10, safety_margin=0.05) == "SELL"
    assert signal_from_edge(0.10, transaction_cost_estimate=0.05, safety_margin=0.05) == "HOLD"


def test_total_finn_loss_combines_all_terms():
    weights = PDELossWeights(
        lambda_data=1.0,
        lambda_boundary=2.0,
        lambda_pde=3.0,
        lambda_arbitrage=4.0,
    )
    total = total_finn_loss(1.0, 2.0, 3.0, arbitrage_loss_value=4.0, weights=weights)
    assert total == 30.0
