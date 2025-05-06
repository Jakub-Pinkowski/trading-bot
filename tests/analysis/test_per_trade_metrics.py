import pandas as pd
import pytest

from app.analysis.per_trade_metrics import (
    calculate_pnl,
    calculate_trade_duration,
    calculate_absolute_return,
    calculate_commission_pct,
    calculate_price_move_pct,
    add_per_trade_metrics
)


@pytest.fixture
def sample_matched_trades():
    """Sample matched trades data for testing."""
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
            "total_commission": 1.76
        },
        {
            "symbol": "MSFT",
            "entry_time": pd.Timestamp("2023-12-12 10:30:15"),
            "entry_side": "S",
            "entry_price": 350.75,
            "entry_net_amount": 701.5,
            "exit_time": pd.Timestamp("2023-12-12 11:45:20"),
            "exit_side": "B",
            "exit_price": 348.25,
            "exit_net_amount": 696.5,
            "size": 2,
            "total_commission": 1.25
        }
    ])


def test_calculate_pnl_long_trade():
    """Test calculating PnL for a long trade."""

    entry_side = "B"
    exit_side = "S"
    entry_net_amount = 961.3
    exit_net_amount = 586.5
    size = 3
    total_commission = 1.76

    pnl, pnl_pct = calculate_pnl(entry_side, exit_side, entry_net_amount, exit_net_amount, size, total_commission)

    # Expected PnL: (exit_net_amount - entry_net_amount) * size - total_commission
    expected_pnl = (586.5 - 961.3) * 3 - 1.76
    assert pnl == pytest.approx(expected_pnl, 0.01)

    # Expected PnL%: pnl / entry_net_amount
    expected_pnl_pct = expected_pnl / entry_net_amount
    assert pnl_pct == pytest.approx(expected_pnl_pct, 0.0001)


def test_calculate_pnl_short_trade():
    """Test calculating PnL for a short trade."""

    entry_side = "S"
    exit_side = "B"
    entry_net_amount = 701.5
    exit_net_amount = 696.5
    size = 2
    total_commission = 1.25

    pnl, pnl_pct = calculate_pnl(entry_side, exit_side, entry_net_amount, exit_net_amount, size, total_commission)

    # Expected PnL: (entry_net_amount - exit_net_amount) * size - total_commission
    expected_pnl = (701.5 - 696.5) * 2 - 1.25
    assert pnl == pytest.approx(expected_pnl, 0.01)

    # Expected PnL%: pnl / entry_net_amount
    expected_pnl_pct = expected_pnl / entry_net_amount
    assert pnl_pct == pytest.approx(expected_pnl_pct, 0.0001)


def test_calculate_pnl_invalid_sides():
    """Test calculating PnL with invalid side combinations."""

    # Both sides are buy
    pnl, pnl_pct = calculate_pnl("B", "B", 100, 110, 1, 1)
    assert pnl == 0
    assert pnl_pct == 0

    # Both sides are sell
    pnl, pnl_pct = calculate_pnl("S", "S", 100, 90, 1, 1)
    assert pnl == 0
    assert pnl_pct == 0


def test_calculate_pnl_zero_entry_amount():
    """Test calculating PnL with zero entry amount."""

    pnl, pnl_pct = calculate_pnl("B", "S", 0, 100, 1, 1)
    assert pnl == pytest.approx(99, 0.01)  # (100 - 0) * 1 - 1
    assert pnl_pct == 0  # Avoid division by zero


def test_calculate_trade_duration():
    """Test calculating trade duration."""

    start_time = pd.Timestamp("2023-12-11 18:00:49")
    end_time = pd.Timestamp("2023-12-11 19:15:30")

    duration = calculate_trade_duration(start_time, end_time)

    # Expected duration in minutes
    expected_duration = (end_time - start_time).total_seconds() / 60.0
    assert duration == pytest.approx(expected_duration, 0.01)


def test_calculate_absolute_return_long():
    """Test calculating absolute return for a long trade."""

    entry_net_amount = 961.3
    exit_net_amount = 586.5 * 3  # Multiply by size to match the calculation
    entry_side = "B"

    abs_return = calculate_absolute_return(entry_net_amount, exit_net_amount, entry_side)

    # Expected absolute return: exit_net_amount - entry_net_amount
    expected_abs_return = exit_net_amount - entry_net_amount
    assert abs_return == pytest.approx(expected_abs_return, 0.01)


def test_calculate_absolute_return_short():
    """Test calculating absolute return for a short trade."""

    entry_net_amount = 701.5
    exit_net_amount = 696.5 * 2  # Multiply by size to match the calculation
    entry_side = "S"

    abs_return = calculate_absolute_return(entry_net_amount, exit_net_amount, entry_side)

    # Expected absolute return: -(exit_net_amount - entry_net_amount)
    expected_abs_return = -(exit_net_amount - entry_net_amount)
    assert abs_return == pytest.approx(expected_abs_return, 0.01)


def test_calculate_commission_pct():
    """Test calculating commission percentage."""

    total_commission = 1.76
    entry_net_amount = 961.3

    commission_pct = calculate_commission_pct(total_commission, entry_net_amount)

    # Expected commission percentage: (total_commission / entry_net_amount) * 100
    expected_commission_pct = (total_commission / entry_net_amount) * 100
    assert commission_pct == pytest.approx(expected_commission_pct, 0.0001)


def test_calculate_commission_pct_zero_entry():
    """Test calculating commission percentage with zero entry amount."""

    total_commission = 1.76
    entry_net_amount = 0

    commission_pct = calculate_commission_pct(total_commission, entry_net_amount)

    # Expected commission percentage: 0 (avoid division by zero)
    assert commission_pct == 0


def test_calculate_price_move_pct():
    """Test calculating price move percentage."""

    entry_price = 192.26
    exit_price = 195.50

    price_move_pct = calculate_price_move_pct(entry_price, exit_price)

    # Expected price move percentage: ((exit_price - entry_price) / entry_price) * 100
    expected_price_move_pct = ((exit_price - entry_price) / entry_price) * 100
    assert price_move_pct == pytest.approx(expected_price_move_pct, 0.0001)


def test_calculate_price_move_pct_zero_entry():
    """Test calculating price move percentage with zero entry price."""

    entry_price = 0
    exit_price = 195.50

    price_move_pct = calculate_price_move_pct(entry_price, exit_price)

    # Expected price move percentage: 0 (avoid division by zero)
    assert price_move_pct == 0


def test_add_per_trade_metrics(sample_matched_trades):
    """Test adding per-trade metrics to matched trades."""

    result = add_per_trade_metrics(sample_matched_trades)

    # Verify the result is a DataFrame with the expected structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_matched_trades)

    # Verify the metrics were added
    assert "pnl" in result.columns
    assert "pnl_pct" in result.columns
    assert "trade_duration" in result.columns

    # Verify the metrics for the first trade (long trade)
    first_trade = result.iloc[0]
    expected_pnl, expected_pnl_pct = calculate_pnl(
        first_trade["entry_side"],
        first_trade["exit_side"],
        first_trade["entry_net_amount"],
        first_trade["exit_net_amount"],
        first_trade["size"],
        first_trade["total_commission"]
    )
    assert first_trade["pnl"] == pytest.approx(expected_pnl, 0.01)
    assert first_trade["pnl_pct"] == pytest.approx(expected_pnl_pct, 0.0001)

    # Verify the metrics for the second trade (short trade)
    second_trade = result.iloc[1]
    expected_pnl, expected_pnl_pct = calculate_pnl(
        second_trade["entry_side"],
        second_trade["exit_side"],
        second_trade["entry_net_amount"],
        second_trade["exit_net_amount"],
        second_trade["size"],
        second_trade["total_commission"]
    )
    assert second_trade["pnl"] == pytest.approx(expected_pnl, 0.01)
    assert second_trade["pnl_pct"] == pytest.approx(expected_pnl_pct, 0.0001)


def test_add_per_trade_metrics_empty():
    """Test adding per-trade metrics to empty matched trades."""

    empty_df = pd.DataFrame(columns=[
        "symbol", "entry_time", "entry_side", "entry_price", "entry_net_amount",
        "exit_time", "exit_side", "exit_price", "exit_net_amount", "size", "total_commission"
    ])

    result = add_per_trade_metrics(empty_df)

    # Verify the result is an empty DataFrame with the expected columns
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert "pnl" in result.columns
    assert "pnl_pct" in result.columns
    assert "trade_duration" in result.columns
