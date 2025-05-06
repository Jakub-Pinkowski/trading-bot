import numpy as np
import pandas as pd
import pytest

from app.analysis.dataset_metrics import calculate_dataset_metrics


@pytest.fixture
def sample_trades_with_metrics():
    """Sample trades data with metrics for testing."""
    return pd.DataFrame([
        {
            "symbol": "AAPL",
            "entry_time": pd.Timestamp("2023-12-11 18:00:49"),
            "entry_side": "B",
            "entry_price": 192.26,
            "entry_net_amount": 961.3,
            "exit_time": pd.Timestamp("2023-12-11 19:15:30"),
            "exit_side": "S",
            "exit_price": 195.50,
            "exit_net_amount": 586.5,
            "size": 3,
            "total_commission": 1.76,
            "pnl": 15.98,
            "pnl_pct": 0.0166,
            "trade_duration": 74.68
        },
        {
            "symbol": "MSFT",
            "entry_time": pd.Timestamp("2023-12-12 10:30:15"),
            "entry_side": "B",
            "entry_price": 350.75,
            "entry_net_amount": 701.5,
            "exit_time": pd.Timestamp("2023-12-12 11:45:20"),
            "exit_side": "S",
            "exit_price": 348.25,
            "exit_net_amount": 696.5,
            "size": 2,
            "total_commission": 1.25,
            "pnl": -6.25,
            "pnl_pct": -0.0089,
            "trade_duration": 75.08
        },
        {
            "symbol": "GOOGL",
            "entry_time": pd.Timestamp("2023-12-13 09:45:10"),
            "entry_side": "S",
            "entry_price": 135.50,
            "entry_net_amount": 1355.0,
            "exit_time": pd.Timestamp("2023-12-13 14:30:45"),
            "exit_side": "B",
            "exit_price": 133.25,
            "exit_net_amount": 1332.5,
            "size": 10,
            "total_commission": 2.50,
            "pnl": 20.0,
            "pnl_pct": 0.0148,
            "trade_duration": 285.58
        }
    ])


def test_calculate_dataset_metrics(sample_trades_with_metrics):
    """Test calculating dataset metrics with sample data."""

    metrics = calculate_dataset_metrics(sample_trades_with_metrics)

    # Verify the metrics are calculated correctly
    assert isinstance(metrics, dict)

    # Win Rate (2 out of 3 trades are profitable)
    assert metrics["win_rate"] == pytest.approx(66.6667, 0.01)

    # Average Win (average of the two profitable trades)
    assert metrics["average_win"] == pytest.approx(17.99, 0.01)

    # Average Loss (the one losing trade)
    assert metrics["average_loss"] == pytest.approx(-6.25, 0.01)

    # Profit Factor (Total Profit / Total Loss)
    assert metrics["profit_factor"] == pytest.approx(5.7568, 0.01)

    # Sharpe Ratio (Mean PnL / Std Dev of PnL)
    pnl_mean = sample_trades_with_metrics["pnl"].mean()
    pnl_std = sample_trades_with_metrics["pnl"].std()
    expected_sharpe = pnl_mean / pnl_std
    assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe, 0.01)

    # Sortino Ratio (Mean PnL / Std Dev of Negative PnL)
    losses = sample_trades_with_metrics[sample_trades_with_metrics["pnl"] <= 0]
    downside_std = losses["pnl"].std() if not losses.empty else 0
    expected_sortino = pnl_mean / downside_std if downside_std > 0 else np.nan
    if np.isnan(expected_sortino):
        assert np.isnan(metrics["sortino_ratio"])
    else:
        assert metrics["sortino_ratio"] == pytest.approx(expected_sortino, 0.01)

    # Cumulative PnL
    assert metrics["cumulative_pnl"] == pytest.approx(29.73, 0.01)

    # Maximum Drawdown
    assert metrics["max_drawdown"] == pytest.approx(6.25, 0.01)

    # Total Commission
    assert metrics["total_commission"] == pytest.approx(5.51, 0.01)


def test_calculate_dataset_metrics_all_wins():
    """Test calculating dataset metrics with all winning trades."""

    trades_df = pd.DataFrame([
        {"pnl": 10.0, "total_commission": 1.0},
        {"pnl": 20.0, "total_commission": 1.5},
        {"pnl": 15.0, "total_commission": 1.2}
    ])

    metrics = calculate_dataset_metrics(trades_df)

    # Verify the metrics
    assert metrics["win_rate"] == 100.0
    assert metrics["average_win"] == pytest.approx(15.0, 0.01)
    assert metrics["average_loss"] == 0.0
    assert np.isnan(metrics["profit_factor"])  # No losses, so profit factor is undefined
    assert metrics["cumulative_pnl"] == 45.0
    assert metrics["total_commission"] == 3.7


def test_calculate_dataset_metrics_all_losses():
    """Test calculating dataset metrics with all losing trades."""

    trades_df = pd.DataFrame([
        {"pnl": -10.0, "total_commission": 1.0},
        {"pnl": -20.0, "total_commission": 1.5},
        {"pnl": -15.0, "total_commission": 1.2}
    ])

    metrics = calculate_dataset_metrics(trades_df)

    # Verify the metrics
    assert metrics["win_rate"] == 0.0
    assert metrics["average_win"] == 0.0
    assert metrics["average_loss"] == pytest.approx(-15.0, 0.01)
    assert metrics["profit_factor"] == 0.0
    assert metrics["cumulative_pnl"] == -45.0
    assert metrics["total_commission"] == 3.7


def test_calculate_dataset_metrics_empty():
    """Test calculating dataset metrics with empty data."""

    trades_df = pd.DataFrame(columns=["pnl", "total_commission"])

    metrics = calculate_dataset_metrics(trades_df)

    # Verify the metrics
    assert metrics["win_rate"] == 0.0
    assert metrics["average_win"] == 0.0
    assert metrics["average_loss"] == 0.0
    assert np.isnan(metrics["profit_factor"])
    assert np.isnan(metrics["sharpe_ratio"])
    assert np.isnan(metrics["sortino_ratio"])
    assert metrics["cumulative_pnl"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["total_commission"] == 0.0


def test_calculate_dataset_metrics_single_trade():
    """Test calculating dataset metrics with a single trade."""

    trades_df = pd.DataFrame([
        {"pnl": 15.0, "total_commission": 1.2}
    ])

    metrics = calculate_dataset_metrics(trades_df)

    # Verify the metrics
    assert metrics["win_rate"] == 100.0
    assert metrics["average_win"] == 15.0
    assert metrics["average_loss"] == 0.0
    assert np.isnan(metrics["profit_factor"])
    assert np.isnan(metrics["sharpe_ratio"])  # Standard deviation is 0, so Sharpe is undefined
    assert np.isnan(metrics["sortino_ratio"])
    assert metrics["cumulative_pnl"] == 15.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["total_commission"] == 1.2
