"""
Tests for Summary Metrics Module.

Tests cover:
- Summary metrics calculation for trade sequences
- Win rate, profit factor, and return metrics
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Drawdown calculations (max drawdown, Ulcer index)
- Risk metrics (VaR, Expected Shortfall)
- Edge cases (no trades, all wins, all losses, single trade)
- Real trade sequences from strategy backtests

All tests use realistic trade data with proper metric calculations.
"""
from datetime import datetime

import pytest

from app.backtesting.metrics.per_trade_metrics import calculate_trade_metrics
from app.backtesting.metrics.summary_metrics import (
    SummaryMetrics,
    MIN_RETURNS_FOR_SHARPE,
    MIN_RETURNS_FOR_VAR,
    RISK_FREE_RATE,
    CONFIDENCE_LEVEL,
    INFINITY_REPLACEMENT
)


# ==================== Fixtures ====================

@pytest.fixture
def winning_trade_zs():
    """Create a winning trade for ZS."""
    trade = {
        'entry_time': datetime(2024, 1, 15, 10, 0),
        'exit_time': datetime(2024, 1, 15, 14, 0),
        'entry_price': 1200.0,
        'exit_price': 1210.0,
        'side': 'long'
    }
    return calculate_trade_metrics(trade, 'ZS')


@pytest.fixture
def losing_trade_zs():
    """Create a losing trade for ZS."""
    trade = {
        'entry_time': datetime(2024, 1, 16, 10, 0),
        'exit_time': datetime(2024, 1, 16, 14, 0),
        'entry_price': 1200.0,
        'exit_price': 1195.0,
        'side': 'long'
    }
    return calculate_trade_metrics(trade, 'ZS')


@pytest.fixture
def breakeven_trade_zs():
    """Create a breakeven trade for ZS."""
    trade = {
        'entry_time': datetime(2024, 1, 17, 10, 0),
        'exit_time': datetime(2024, 1, 17, 14, 0),
        'entry_price': 1200.0,
        'exit_price': 1200.0,
        'side': 'long'
    }
    return calculate_trade_metrics(trade, 'ZS')


@pytest.fixture
def mixed_trades(winning_trade_zs, losing_trade_zs):
    """Create a list of mixed winning and losing trades."""
    return [winning_trade_zs, losing_trade_zs]


@pytest.fixture
def all_winning_trades():
    """Create a list of all winning trades."""
    trades = []
    for i in range(5):
        trade = {
            'entry_time': datetime(2024, 1, 10 + i, 10, 0),
            'exit_time': datetime(2024, 1, 10 + i, 14, 0),
            'entry_price': 1200.0,
            'exit_price': 1200.0 + (i + 1) * 5,  # Increasing profits
            'side': 'long'
        }
        trades.append(calculate_trade_metrics(trade, 'ZS'))
    return trades


@pytest.fixture
def all_losing_trades():
    """Create a list of all losing trades."""
    trades = []
    for i in range(5):
        trade = {
            'entry_time': datetime(2024, 1, 10 + i, 10, 0),
            'exit_time': datetime(2024, 1, 10 + i, 14, 0),
            'entry_price': 1200.0,
            'exit_price': 1200.0 - (i + 1) * 5,  # Increasing losses
            'side': 'long'
        }
        trades.append(calculate_trade_metrics(trade, 'ZS'))
    return trades


# ==================== Test Classes ====================

class TestSummaryMetricsInitialization:
    """Test SummaryMetrics class initialization."""

    def test_initialization_with_trades(self, mixed_trades):
        """Test SummaryMetrics initializes correctly with trades."""
        metrics = SummaryMetrics(mixed_trades)

        assert metrics.trades == mixed_trades
        assert metrics.total_trades == 2
        assert len(metrics.winning_trades) == 1
        assert len(metrics.losing_trades) == 1

    def test_initialization_with_empty_list(self):
        """Test SummaryMetrics handles empty trade list."""
        metrics = SummaryMetrics([])

        assert metrics.total_trades == 0
        assert metrics.winning_trades == []
        assert metrics.losing_trades == []
        assert metrics.win_count == 0
        assert metrics.loss_count == 0

    def test_initialization_with_none(self):
        """Test SummaryMetrics handles None trades."""
        metrics = SummaryMetrics(None)

        assert metrics.total_trades == 0
        assert not metrics._has_trades()


class TestBasicTradeStatistics:
    """Test basic trade statistics calculations."""

    def test_win_rate_calculation(self, mixed_trades):
        """Test win rate is calculated correctly."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # 1 win out of 2 trades = 50%
        assert result['win_rate'] == 50.0
        assert result['winning_trades'] == 1
        assert result['losing_trades'] == 1

    def test_all_winning_trades_win_rate(self, all_winning_trades):
        """Test win rate with all winning trades."""
        metrics = SummaryMetrics(all_winning_trades)
        result = metrics.calculate_all_metrics()

        assert result['win_rate'] == 100.0
        assert result['winning_trades'] == 5
        assert result['losing_trades'] == 0

    def test_all_losing_trades_win_rate(self, all_losing_trades):
        """Test win rate with all losing trades."""
        metrics = SummaryMetrics(all_losing_trades)
        result = metrics.calculate_all_metrics()

        assert result['win_rate'] == 0.0
        assert result['winning_trades'] == 0
        assert result['losing_trades'] == 5

    def test_total_trades_count(self, all_winning_trades):
        """Test total trades count is correct."""
        metrics = SummaryMetrics(all_winning_trades)
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 5

    def test_average_trade_duration(self, mixed_trades):
        """Test average trade duration calculation."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Both trades are 4 hours
        assert result['average_trade_duration_hours'] == 4.0


class TestReturnMetrics:
    """Test return-based metrics calculations."""

    def test_total_return_percentage(self, mixed_trades):
        """Test total return percentage calculation."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Sum of return_percentage_of_contract from both trades
        expected = sum(t['return_percentage_of_contract'] for t in mixed_trades)
        assert result['total_return_percentage_of_contract'] == round(expected, 2)

    def test_average_trade_return(self, mixed_trades):
        """Test average trade return calculation."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Average of all trade returns
        total_return = sum(t['return_percentage_of_contract'] for t in mixed_trades)
        expected_avg = total_return / len(mixed_trades)
        assert result['average_trade_return_percentage_of_contract'] == round(expected_avg, 2)

    def test_average_win_percentage(self, all_winning_trades):
        """Test average win percentage calculation."""
        metrics = SummaryMetrics(all_winning_trades)
        result = metrics.calculate_all_metrics()

        winning_returns = [t['return_percentage_of_contract'] for t in all_winning_trades]
        expected_avg = sum(winning_returns) / len(winning_returns)
        assert result['average_win_percentage_of_contract'] == round(expected_avg, 2)

    def test_average_loss_percentage(self, all_losing_trades):
        """Test average loss percentage calculation."""
        metrics = SummaryMetrics(all_losing_trades)
        result = metrics.calculate_all_metrics()

        losing_returns = [t['return_percentage_of_contract'] for t in all_losing_trades]
        expected_avg = sum(losing_returns) / len(losing_returns)
        assert result['average_loss_percentage_of_contract'] == round(expected_avg, 2)

    def test_total_wins_percentage(self, mixed_trades):
        """Test total wins percentage calculation."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        winning_trades = [t for t in mixed_trades if t['return_percentage_of_contract'] > 0]
        expected = sum(t['return_percentage_of_contract'] for t in winning_trades)
        assert result['total_wins_percentage_of_contract'] == round(expected, 2)

    def test_total_losses_percentage(self, mixed_trades):
        """Test total losses percentage calculation."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        losing_trades = [t for t in mixed_trades if t['return_percentage_of_contract'] <= 0]
        expected = sum(t['return_percentage_of_contract'] for t in losing_trades)
        assert result['total_losses_percentage_of_contract'] == round(expected, 2)


class TestProfitFactor:
    """Test profit factor calculations."""

    def test_profit_factor_with_mixed_trades(self, mixed_trades):
        """Test profit factor with both wins and losses."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Profit factor = Total Wins / |Total Losses|
        assert result['profit_factor'] > 0
        assert isinstance(result['profit_factor'], float)

    def test_profit_factor_all_wins(self, all_winning_trades):
        """Test profit factor with only winning trades."""
        metrics = SummaryMetrics(all_winning_trades)
        result = metrics.calculate_all_metrics()

        # No losses = infinity replacement
        assert result['profit_factor'] == INFINITY_REPLACEMENT

    def test_profit_factor_all_losses(self, all_losing_trades):
        """Test profit factor with only losing trades."""
        metrics = SummaryMetrics(all_losing_trades)
        result = metrics.calculate_all_metrics()

        # No wins = 0
        assert result['profit_factor'] == 0.0


class TestDrawdownCalculations:
    """Test drawdown-related metrics."""

    def test_maximum_drawdown_calculation(self, mixed_trades):
        """Test maximum drawdown is calculated."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Should have a drawdown value
        assert 'maximum_drawdown_percentage' in result
        assert isinstance(result['maximum_drawdown_percentage'], float)

    def test_maximum_drawdown_all_wins(self, all_winning_trades):
        """Test maximum drawdown with all winning trades."""
        metrics = SummaryMetrics(all_winning_trades)
        result = metrics.calculate_all_metrics()

        # All wins = minimal or zero drawdown
        assert result['maximum_drawdown_percentage'] >= 0

    def test_maximum_drawdown_all_losses(self, all_losing_trades):
        """Test maximum drawdown with all losing trades."""
        metrics = SummaryMetrics(all_losing_trades)
        result = metrics.calculate_all_metrics()

        # All losses = significant drawdown
        assert result['maximum_drawdown_percentage'] > 0

    def test_ulcer_index_calculation(self, mixed_trades):
        """Test Ulcer Index is calculated."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        assert 'ulcer_index' in result
        assert isinstance(result['ulcer_index'], float)
        assert result['ulcer_index'] >= 0


class TestSharpeRatio:
    """Test Sharpe ratio calculations."""

    def test_sharpe_ratio_with_sufficient_trades(self, all_winning_trades):
        """Test Sharpe ratio with enough trades."""
        metrics = SummaryMetrics(all_winning_trades)
        result = metrics.calculate_all_metrics()

        # With 5 trades, should calculate Sharpe
        assert 'sharpe_ratio' in result
        assert isinstance(result['sharpe_ratio'], float)

    def test_sharpe_ratio_insufficient_trades(self, winning_trade_zs):
        """Test Sharpe ratio with insufficient trades."""
        metrics = SummaryMetrics([winning_trade_zs])
        result = metrics.calculate_all_metrics()

        # Less than MIN_RETURNS_FOR_SHARPE = 0
        assert result['sharpe_ratio'] == 0.0


class TestSortinoRatio:
    """Test Sortino ratio calculations."""

    def test_sortino_ratio_with_losses(self, mixed_trades):
        """Test Sortino ratio with both wins and losses."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        assert 'sortino_ratio' in result
        assert isinstance(result['sortino_ratio'], float)

    def test_sortino_ratio_no_negative_returns(self, all_winning_trades):
        """Test Sortino ratio with no negative returns."""
        metrics = SummaryMetrics(all_winning_trades)
        result = metrics.calculate_all_metrics()

        # No downside deviation = infinity replacement
        assert result['sortino_ratio'] == INFINITY_REPLACEMENT


class TestCalmarRatio:
    """Test Calmar ratio calculations."""

    def test_calmar_ratio_calculation(self, mixed_trades):
        """Test Calmar ratio is calculated."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        assert 'calmar_ratio' in result
        assert isinstance(result['calmar_ratio'], float)

    def test_calmar_ratio_zero_drawdown(self, all_winning_trades):
        """Test Calmar ratio with zero or minimal drawdown."""
        metrics = SummaryMetrics(all_winning_trades)
        result = metrics.calculate_all_metrics()

        # Zero/minimal drawdown might give very high value
        assert result['calmar_ratio'] >= 0


class TestValueAtRisk:
    """Test Value at Risk (VaR) calculations."""

    def test_var_with_sufficient_trades(self, all_losing_trades):
        """Test VaR with enough trades."""
        metrics = SummaryMetrics(all_losing_trades)
        result = metrics.calculate_all_metrics()

        # Should calculate VaR
        assert 'value_at_risk' in result
        assert result['value_at_risk'] >= 0

    def test_var_insufficient_trades(self, winning_trade_zs):
        """Test VaR with insufficient trades."""
        # Less than MIN_RETURNS_FOR_VAR
        trades = [winning_trade_zs]
        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Not enough returns = 0
        assert result['value_at_risk'] == 0.0


class TestExpectedShortfall:
    """Test Expected Shortfall (CVaR) calculations."""

    def test_expected_shortfall_with_sufficient_trades(self, all_losing_trades):
        """Test Expected Shortfall with enough trades."""
        metrics = SummaryMetrics(all_losing_trades)
        result = metrics.calculate_all_metrics()

        assert 'expected_shortfall' in result
        assert result['expected_shortfall'] >= 0

    def test_expected_shortfall_insufficient_trades(self, winning_trade_zs):
        """Test Expected Shortfall with insufficient trades."""
        trades = [winning_trade_zs]
        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        assert result['expected_shortfall'] == 0.0

    def test_expected_shortfall_greater_than_var(self, all_losing_trades):
        """Test Expected Shortfall is typically >= VaR."""
        metrics = SummaryMetrics(all_losing_trades)
        result = metrics.calculate_all_metrics()

        # ES should be >= VaR (average of worst cases vs threshold)
        assert result['expected_shortfall'] >= result['value_at_risk']


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trades_list(self):
        """Test metrics with empty trades list."""
        metrics = SummaryMetrics([])
        result = metrics.calculate_all_metrics()

        # Should return empty dict
        assert result == {}

    def test_none_trades(self):
        """Test metrics with None trades."""
        metrics = SummaryMetrics(None)
        result = metrics.calculate_all_metrics()

        assert result == {}

    def test_single_winning_trade(self, winning_trade_zs):
        """Test metrics with single winning trade."""
        metrics = SummaryMetrics([winning_trade_zs])
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 1
        assert result['win_rate'] == 100.0
        assert result['winning_trades'] == 1
        assert result['losing_trades'] == 0

    def test_single_losing_trade(self, losing_trade_zs):
        """Test metrics with single losing trade."""
        metrics = SummaryMetrics([losing_trade_zs])
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 1
        assert result['win_rate'] == 0.0
        assert result['winning_trades'] == 0
        assert result['losing_trades'] == 1

    def test_breakeven_trade_counted_as_loss(self, breakeven_trade_zs):
        """Test breakeven trade is counted as losing trade."""
        metrics = SummaryMetrics([breakeven_trade_zs])
        result = metrics.calculate_all_metrics()

        # Breakeven (including commission) should be <= 0
        assert result['losing_trades'] == 1
        assert result['winning_trades'] == 0


class TestMetricsRounding:
    """Test that all metrics are properly rounded."""

    def test_all_metrics_rounded_to_two_decimals(self, mixed_trades):
        """Test all metrics are rounded to 2 decimal places."""
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        for key, value in result.items():
            if isinstance(value, float):
                # Check that it has at most 2 decimal places
                str_value = str(value)
                if '.' in str_value:
                    decimals = len(str_value.split('.')[1])
                    assert decimals <= 2, f"{key} has more than 2 decimal places: {value}"


class TestPrivateMethods:
    """Test private helper methods."""

    def test_has_trades_method(self, mixed_trades):
        """Test _has_trades helper method."""
        metrics = SummaryMetrics(mixed_trades)
        assert metrics._has_trades() is True

        empty_metrics = SummaryMetrics([])
        assert empty_metrics._has_trades() is False

    def test_has_winning_trades_method(self, mixed_trades):
        """Test _has_winning_trades helper method."""
        metrics = SummaryMetrics(mixed_trades)
        assert metrics._has_winning_trades() is True

        losing_metrics = SummaryMetrics([mixed_trades[1]])  # Only losing trade
        assert losing_metrics._has_winning_trades() is False

    def test_has_losing_trades_method(self, mixed_trades):
        """Test _has_losing_trades helper method."""
        metrics = SummaryMetrics(mixed_trades)
        assert metrics._has_losing_trades() is True

        winning_metrics = SummaryMetrics([mixed_trades[0]])  # Only winning trade
        assert winning_metrics._has_losing_trades() is False


class TestConstants:
    """Test module constants are defined correctly."""

    def test_min_returns_for_sharpe(self):
        """Test MIN_RETURNS_FOR_SHARPE constant."""
        assert MIN_RETURNS_FOR_SHARPE == 2

    def test_min_returns_for_var(self):
        """Test MIN_RETURNS_FOR_VAR constant."""
        assert MIN_RETURNS_FOR_VAR == 5

    def test_risk_free_rate(self):
        """Test RISK_FREE_RATE constant."""
        assert RISK_FREE_RATE == 0.0

    def test_confidence_level(self):
        """Test CONFIDENCE_LEVEL constant."""
        assert CONFIDENCE_LEVEL == 0.95

    def test_infinity_replacement(self):
        """Test INFINITY_REPLACEMENT constant."""
        assert INFINITY_REPLACEMENT == 9999.99


class TestRealWorldScenarios:
    """Test with realistic trade sequences."""

    def test_profitable_strategy(self):
        """Test metrics for overall profitable strategy."""
        # 7 wins, 3 losses
        trades = []
        for i in range(10):
            if i < 7:  # Winning trades
                trade = {
                    'entry_time': datetime(2024, 1, 10 + i, 10, 0),
                    'exit_time': datetime(2024, 1, 10 + i, 14, 0),
                    'entry_price': 1200.0,
                    'exit_price': 1205.0 + i,
                    'side': 'long'
                }
            else:  # Losing trades
                trade = {
                    'entry_time': datetime(2024, 1, 10 + i, 10, 0),
                    'exit_time': datetime(2024, 1, 10 + i, 14, 0),
                    'entry_price': 1200.0,
                    'exit_price': 1198.0,
                    'side': 'long'
                }
            trades.append(calculate_trade_metrics(trade, 'ZS'))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 10
        assert result['win_rate'] == 70.0
        assert result['total_return_percentage_of_contract'] > 0
        assert result['profit_factor'] > 1.0

    def test_losing_strategy(self):
        """Test metrics for overall losing strategy."""
        # 3 wins, 7 losses
        trades = []
        for i in range(10):
            if i < 3:  # Winning trades
                trade = {
                    'entry_time': datetime(2024, 1, 10 + i, 10, 0),
                    'exit_time': datetime(2024, 1, 10 + i, 14, 0),
                    'entry_price': 1200.0,
                    'exit_price': 1202.0,
                    'side': 'long'
                }
            else:  # Losing trades
                trade = {
                    'entry_time': datetime(2024, 1, 10 + i, 10, 0),
                    'exit_time': datetime(2024, 1, 10 + i, 14, 0),
                    'entry_price': 1200.0,
                    'exit_price': 1195.0 - i,
                    'side': 'long'
                }
            trades.append(calculate_trade_metrics(trade, 'ZS'))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 10
        assert result['win_rate'] == 30.0
        assert result['total_return_percentage_of_contract'] < 0
        assert result['profit_factor'] < 1.0
        assert result['maximum_drawdown_percentage'] > 0
