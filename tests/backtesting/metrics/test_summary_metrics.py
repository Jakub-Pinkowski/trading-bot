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

This file uses shared fixtures from conftest.py:
- trade_factory: For creating individual trades
- trades_factory: For creating trade sequences
"""
import random
from datetime import datetime, timedelta

import pytest

from app.backtesting.metrics.summary_metrics import (
    SummaryMetrics,
    MIN_RETURNS_FOR_SHARPE,
    MIN_RETURNS_FOR_VAR,
    RISK_FREE_RATE,
    CONFIDENCE_LEVEL,
    INFINITY_REPLACEMENT
)


# ==================== Test Classes ====================

class TestSummaryMetricsInitialization:
    """Test SummaryMetrics class initialization."""

    def test_initialization_with_trades(self, trades_factory):
        """Test SummaryMetrics initializes correctly with trades."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
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

    def test_win_rate_calculation(self, trades_factory):
        """Test win rate is calculated correctly."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # 1 win out of 2 trades = 50%
        assert result['win_rate'] == 50.0
        assert result['winning_trades'] == 1
        assert result['losing_trades'] == 1

    def test_all_winning_trades_win_rate(self, trades_factory):
        """Test win rate with all winning trades."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        assert result['win_rate'] == 100.0
        assert result['winning_trades'] == 5
        assert result['losing_trades'] == 0

    def test_all_losing_trades_win_rate(self, trades_factory):
        """Test win rate with all losing trades."""
        all_losing = trades_factory.all_losing(count=5)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        assert result['win_rate'] == 0.0
        assert result['winning_trades'] == 0
        assert result['losing_trades'] == 5

    def test_total_trades_count(self, trades_factory):
        """Test total trades count is correct."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 5

    def test_average_trade_duration_bars(self, trade_factory):
        """Test average_trade_duration_bars uses the duration_bars field when present."""
        trade1 = trade_factory('ZS', 1200.0, 1210.0, duration_hours=4.0)
        trade1['duration_bars'] = 4.0  # 4h trade on a 1h interval = 4 bars
        trade2 = trade_factory('ZS', 1200.0, 1190.0, duration_hours=4.0)
        trade2['duration_bars'] = 8.0  # same wall-clock hours on a 30m interval = 8 bars

        result = SummaryMetrics([trade1, trade2]).calculate_all_metrics()

        assert result['average_trade_duration_bars'] == 6.0

    def test_average_trade_duration_bars_defaults_to_zero_when_missing(self, trade_factory):
        """Test average_trade_duration_bars is 0 when duration_bars is not set on trades."""
        trade = trade_factory('ZS', 1200.0, 1210.0)
        # duration_bars not set — simulates trades that bypassed runner.py

        result = SummaryMetrics([trade]).calculate_all_metrics()

        assert result['average_trade_duration_bars'] == 0.0


class TestReturnMetrics:
    """Test return-based metrics calculations."""

    def test_total_return_percentage(self, trades_factory):
        """Test total return percentage calculation."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Sum of return_percentage_of_contract from both trades
        expected = sum(t['return_percentage_of_contract'] for t in mixed_trades)
        assert result['total_return_percentage_of_contract'] == round(expected, 2)

    def test_average_trade_return(self, trades_factory):
        """Test average trade return calculation."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Average of all trade returns
        total_return = sum(t['return_percentage_of_contract'] for t in mixed_trades)
        expected_avg = total_return / len(mixed_trades)
        assert result['average_trade_return_percentage_of_contract'] == round(expected_avg, 2)

    def test_average_win_percentage(self, trades_factory):
        """Test average win percentage calculation."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        winning_returns = [t['return_percentage_of_contract'] for t in all_winning]
        expected_avg = sum(winning_returns) / len(winning_returns)
        assert result['average_win_percentage_of_contract'] == round(expected_avg, 2)

    def test_average_loss_percentage(self, trades_factory):
        """Test average loss percentage calculation."""
        all_losing = trades_factory.all_losing(count=5)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        losing_returns = [t['return_percentage_of_contract'] for t in all_losing]
        expected_avg = sum(losing_returns) / len(losing_returns)
        assert result['average_loss_percentage_of_contract'] == round(expected_avg, 2)

    def test_total_wins_percentage(self, trades_factory):
        """Test total wins percentage calculation."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        winning_trades = [t for t in mixed_trades if t['return_percentage_of_contract'] > 0]
        expected = sum(t['return_percentage_of_contract'] for t in winning_trades)
        assert result['total_wins_percentage_of_contract'] == round(expected, 2)

    def test_total_losses_percentage(self, trades_factory):
        """Test total losses percentage calculation."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        losing_trades = [t for t in mixed_trades if t['return_percentage_of_contract'] <= 0]
        expected = sum(t['return_percentage_of_contract'] for t in losing_trades)
        assert result['total_losses_percentage_of_contract'] == round(expected, 2)


class TestProfitFactor:
    """Test profit factor calculations."""

    def test_profit_factor_with_mixed_trades(self, trades_factory):
        """Test profit factor with both wins and losses."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        result = metrics.calculate_all_metrics()

        # Profit factor = Total Wins / |Total Losses|
        assert result['profit_factor'] > 0
        assert isinstance(result['profit_factor'], float)

    def test_profit_factor_all_wins(self, trades_factory):
        """Test profit factor with only winning trades."""
        all_winning = trades_factory.all_winning(count=5)
        metrics = SummaryMetrics(all_winning)
        result = metrics.calculate_all_metrics()

        # No losses = infinity replacement
        assert result['profit_factor'] == INFINITY_REPLACEMENT

    def test_profit_factor_all_losses(self, trades_factory):
        """Test profit factor with only losing trades."""
        all_losing = trades_factory.all_losing(count=5)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        # No wins = 0
        assert result['profit_factor'] == 0.0


class TestDrawdownCalculations:
    """Consolidated drawdown and Ulcer Index tests."""

    def test_drawdown_basic_scenarios(self, trades_factory):
        """Test drawdown with basic win/loss patterns."""
        # Test 1: All wins (minimal drawdown)
        all_wins = trades_factory.all_winning(count=5)
        metrics_wins = SummaryMetrics(all_wins)
        assert metrics_wins.calculate_all_metrics()['maximum_drawdown_percentage'] >= 0

        # Test 2: All losses (significant drawdown)
        all_losses = trades_factory.all_losing(count=5)
        metrics_losses = SummaryMetrics(all_losses)
        assert metrics_losses.calculate_all_metrics()['maximum_drawdown_percentage'] > 0

        # Test 3: Mixed trades
        mixed = trades_factory.mixed(win_count=3, loss_count=2)
        metrics_mixed = SummaryMetrics(mixed)
        result = metrics_mixed.calculate_all_metrics()
        assert 'maximum_drawdown_percentage' in result
        assert isinstance(result['maximum_drawdown_percentage'], float)

    @pytest.mark.parametrize("scenario,expected_has_drawdown", [
        ("peak_trough_recovery", True),
        ("sustained_decline", True),
        ("multiple_drawdowns", True),
    ])
    def test_drawdown_patterns(self, trades_factory, scenario, expected_has_drawdown):
        """Test various drawdown patterns using parametrization."""
        if scenario == "peak_trough_recovery":
            specs = [(1200, 1250), (1250, 1220), (1220, 1245)]
        elif scenario == "sustained_decline":
            specs = [(1200, 1250), (1250, 1240), (1240, 1230), (1230, 1220)]
        else:  # multiple_drawdowns
            specs = [(1200, 1230), (1230, 1220), (1220, 1240), (1240, 1210)]

        trades = trades_factory.create_sequence(specs, symbol='ZS')
        result = SummaryMetrics(trades).calculate_all_metrics()

        if expected_has_drawdown:
            assert result['maximum_drawdown_percentage'] > 0

    def test_ulcer_index_scenarios(self, trades_factory):
        """Test Ulcer Index with key scenarios."""
        # No drawdown
        no_dd = trades_factory.all_winning(count=5)
        assert SummaryMetrics(no_dd).calculate_all_metrics()['ulcer_index'] < 1.0

        # Single drawdown
        single_dd = trades_factory.create_sequence(
            [(1200, 1250), (1250, 1220), (1220, 1245)], symbol='ZS'
        )
        assert SummaryMetrics(single_dd).calculate_all_metrics()['ulcer_index'] > 0

        # Multiple drawdowns
        multi_dd = trades_factory.create_sequence(
            [(1200, 1230), (1230, 1220), (1220, 1240), (1240, 1210), (1210, 1235)],
            symbol='ZS'
        )
        assert SummaryMetrics(multi_dd).calculate_all_metrics()['ulcer_index'] > 0

    def test_drawdown_edge_cases(self, trade_factory):
        """Test drawdown edge cases."""
        # Single losing trade from zero
        single_loss = [trade_factory('ZS', 1200, 1190)]
        result = SummaryMetrics(single_loss).calculate_all_metrics()
        assert result['maximum_drawdown_percentage'] > 0

        # Zero duration trade
        zero_duration = [trade_factory('ZS', 1200, 1205, duration_hours=0.0)]
        result_zero = SummaryMetrics(zero_duration).calculate_all_metrics()
        assert 'maximum_drawdown_percentage' in result_zero

    def test_drawdown_vs_ulcer_relationship(self, trades_factory):
        """Test relationship between max drawdown and Ulcer Index."""
        specs = [(1200, 1300), (1300, 1250), (1250, 1295)]
        trades = trades_factory.create_sequence(specs, symbol='ZS')
        result = SummaryMetrics(trades).calculate_all_metrics()

        # Both should capture the drawdown
        assert result['ulcer_index'] > 0
        assert result['maximum_drawdown_percentage'] > 0
        # Ulcer considers duration, Max DD only depth


class TestRiskAdjustedRatios:
    """Consolidated tests for Sharpe, Sortino, and Calmar ratios."""

    @pytest.mark.parametrize("ratio_name,min_trades", [
        ('sharpe_ratio', MIN_RETURNS_FOR_SHARPE),
        ('sortino_ratio', MIN_RETURNS_FOR_SHARPE),
        ('calmar_ratio', 2),
    ])
    def test_ratio_with_sufficient_trades(self, trades_factory, ratio_name, min_trades):
        """Test all ratios with sufficient trades."""
        trades = trades_factory.all_winning(count=max(5, min_trades))
        result = SummaryMetrics(trades).calculate_all_metrics()

        assert ratio_name in result
        assert isinstance(result[ratio_name], (int, float))
        if ratio_name != 'calmar_ratio':
            assert result[ratio_name] > 0  # Should be positive for all wins

    @pytest.mark.parametrize("ratio_name", ['sharpe_ratio', 'sortino_ratio'])
    def test_ratio_insufficient_trades(self, trade_factory, ratio_name):
        """Test ratios with insufficient trades."""
        single_trade = [trade_factory('ZS', 1200, 1210)]
        result = SummaryMetrics(single_trade).calculate_all_metrics()

        # Sharpe should be 0 with insufficient trades
        # Sortino with a single winning trade returns INFINITY_REPLACEMENT (no downside)
        if ratio_name == 'sharpe_ratio':
            assert result[ratio_name] == 0.0
        else:  # sortino_ratio
            assert result[ratio_name] == INFINITY_REPLACEMENT

    def test_ratios_with_high_volatility(self, trades_factory):
        """Test all ratios with high volatility returns."""
        specs = [
            (1200, 1250), (1250, 1210), (1210, 1260),
            (1260, 1215), (1215, 1265)
        ]
        trades = trades_factory.create_sequence(specs, symbol='ZS')
        result = SummaryMetrics(trades).calculate_all_metrics()

        # All ratios should be calculated
        for ratio in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
            assert ratio in result
            assert isinstance(result[ratio], (int, float))

    def test_ratios_with_negative_returns(self, trades_factory):
        """Test all ratios with predominantly negative returns."""
        losing_trades = trades_factory.all_losing(count=5)
        result = SummaryMetrics(losing_trades).calculate_all_metrics()

        # Sharpe and Sortino should be negative
        assert result['sharpe_ratio'] < 0
        assert result['sortino_ratio'] < 0
        # Calmar should be negative
        assert result['calmar_ratio'] < 0

    @pytest.mark.parametrize("trade_pattern,expected_behavior", [
        ("all_wins", "sortino_infinity"),
        ("mixed", "both_calculated"),
        ("all_losses", "both_negative"),
    ])
    def test_sortino_vs_sharpe_behavior(self, trades_factory, trade_pattern, expected_behavior):
        """Test Sortino vs Sharpe ratio behavior patterns."""
        if trade_pattern == "all_wins":
            trades = trades_factory.all_winning(count=5)
            result = SummaryMetrics(trades).calculate_all_metrics()
            assert result['sortino_ratio'] == INFINITY_REPLACEMENT
            assert result['sharpe_ratio'] > 0
        elif trade_pattern == "mixed":
            trades = trades_factory.mixed(win_count=3, loss_count=2)
            result = SummaryMetrics(trades).calculate_all_metrics()
            # Both should be calculated (can be positive or negative)
            assert isinstance(result['sortino_ratio'], (int, float))
            assert isinstance(result['sharpe_ratio'], (int, float))
        else:  # all_losses
            trades = trades_factory.all_losing(count=5)
            result = SummaryMetrics(trades).calculate_all_metrics()
            assert result['sortino_ratio'] < 0
            assert result['sharpe_ratio'] < 0

    def test_calmar_with_varying_drawdowns(self, trades_factory):
        """Test Calmar ratio with different drawdown scenarios."""
        # Small drawdown
        small_dd_specs = [(1200, 1250), (1250, 1245), (1245, 1280)]
        small_dd_trades = trades_factory.create_sequence(small_dd_specs, symbol='ZS')
        small_result = SummaryMetrics(small_dd_trades).calculate_all_metrics()
        assert small_result['calmar_ratio'] > 0

        # Large drawdown
        large_dd_specs = [(1200, 1250), (1250, 1180), (1180, 1210)]
        large_dd_trades = trades_factory.create_sequence(large_dd_specs, symbol='ZS')
        large_result = SummaryMetrics(large_dd_trades).calculate_all_metrics()
        assert large_result['calmar_ratio'] > 0

        # Negative returns
        neg_specs = [(1200, 1190), (1190, 1180), (1180, 1170)]
        neg_trades = trades_factory.create_sequence(neg_specs, symbol='ZS')
        neg_result = SummaryMetrics(neg_trades).calculate_all_metrics()
        assert neg_result['calmar_ratio'] < 0


class TestValueAtRisk:
    """Test Value at Risk (VaR) calculations."""

    def test_var_with_sufficient_trades(self, trades_factory):
        """Test VaR with enough trades (MIN_RETURNS_FOR_VAR = 30)."""
        all_losing = trades_factory.all_losing(count=30)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        # Should calculate VaR with sufficient trades
        assert 'value_at_risk' in result
        assert result['value_at_risk'] > 0

    def test_var_insufficient_trades(self, trade_factory):
        """Test VaR with insufficient trades."""
        # Less than MIN_RETURNS_FOR_VAR
        winning_trade = trade_factory('ZS', 1200.0, 1210.0)
        trades = [winning_trade]
        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # Not enough returns = 0
        assert result['value_at_risk'] == 0.0

    def test_var_with_skewed_distribution(self, trades_factory):
        """Test VaR with negatively skewed return distribution (many small wins, few big losses)."""
        # Create skewed distribution: 25 small wins, 5 big losses (total >= 30 for MIN_RETURNS_FOR_VAR)
        trades = []

        # Small wins
        for i in range(25):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1202.0,
                                                       entry_time=datetime(2024, 1, 1, 10, 0) + timedelta(days=i),
                                                       exit_time=datetime(2024, 1, 1, 14, 0) + timedelta(days=i)))

        # Big losses
        for i in range(5):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1170.0 - i * 5,
                                                       entry_time=datetime(2024, 3, 1, 10, 0) + timedelta(days=i),
                                                       exit_time=datetime(2024, 3, 1, 14, 0) + timedelta(days=i)))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # VaR should capture the tail risk from big losses
        assert result['value_at_risk'] > 0

    def test_var_with_fat_tails(self, trades_factory):
        """Test VaR with fat-tailed distribution (more extreme events than normal distribution)."""
        # Create distribution with extreme outliers — need >= 30 trades for MIN_RETURNS_FOR_VAR
        price_specs = [(1200.0, 1205.0)] * 29 + [(1200.0, 1150.0)]  # 29 small wins + 1 extreme loss

        trades = trades_factory.create_sequence(price_specs, symbol='ZS')

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # VaR should be significant due to extreme loss
        assert result['value_at_risk'] > 0

    def test_var_with_extreme_losses(self, trades_factory):
        """Test VaR calculation with several extreme losses (>= 30 trades for MIN_RETURNS_FOR_VAR)."""
        # Create scenario with multiple extreme losses: 25 normal trades + 5 extreme losses
        trades = []
        for i in range(25):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1205.0,
                                                       entry_time=datetime(2024, 1, 1, 10, 0) + timedelta(days=i),
                                                       exit_time=datetime(2024, 1, 1, 14, 0) + timedelta(days=i)))
        for i in range(5):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1150.0 - i * 10,
                                                       entry_time=datetime(2024, 3, 1, 10, 0) + timedelta(days=i),
                                                       exit_time=datetime(2024, 3, 1, 14, 0) + timedelta(days=i)))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # VaR should be high due to extreme losses
        assert result['value_at_risk'] > 0

    def test_var_with_normal_distribution(self, trades_factory):
        """Test VaR with approximately normal return distribution."""
        # Create returns clustered around mean
        random.seed(42)  # For reproducibility

        trades = []
        for i in range(20):
            # Generate returns around 1% with small variance
            price_change = 1200.0 * (1 + random.gauss(0.01, 0.005))
            trades.append(trades_factory.trade_factory('ZS', 1200.0, price_change,
                                                       entry_time=datetime(2024, 1, 10, 10, 0) + timedelta(days=i),
                                                       exit_time=datetime(2024, 1, 10, 14, 0) + timedelta(days=i)))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        # VaR should be reasonable for normal distribution
        assert result['value_at_risk'] >= 0

    # Expected Shortfall (CVaR) tests

    def test_expected_shortfall_with_sufficient_trades(self, trades_factory):
        """Test Expected Shortfall with enough trades (MIN_RETURNS_FOR_VAR = 30)."""
        all_losing = trades_factory.all_losing(count=30)
        metrics = SummaryMetrics(all_losing)
        result = metrics.calculate_all_metrics()

        assert 'expected_shortfall' in result
        assert result['expected_shortfall'] > 0

    def test_expected_shortfall_insufficient_trades(self, trade_factory):
        """Test Expected Shortfall with insufficient trades."""
        winning_trade = trade_factory('ZS', 1200.0, 1210.0)
        trades = [winning_trade]
        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        assert result['expected_shortfall'] == 0.0

    def test_expected_shortfall_greater_than_var(self, trades_factory):
        """Test Expected Shortfall is typically >= VaR."""
        all_losing = trades_factory.all_losing(count=30)
        metrics = SummaryMetrics(all_losing)
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

        # Specifically hit defensive checks in private methods for coverage
        assert metrics._calculate_max_drawdown() == (0, 0)
        assert metrics._calculate_profit_factor() == 0
        assert metrics._calculate_sortino_ratio() == 0
        assert metrics._calculate_calmar_ratio() == 0
        assert metrics._calculate_ulcer_index() == 0
        assert metrics._calculate_value_at_risk() == 0
        assert metrics._calculate_expected_shortfall() == 0
        assert metrics._calculate_sharpe_ratio() == 0

    def test_std_dev_zero_hits_defensive_checks(self, trade_factory):
        """Test scenarios where standard deviation is zero."""
        # Sharpe ratio with zero standard deviation (all identical returns)
        trade = trade_factory('ZS', 1200.0, 1210.0)
        metrics = SummaryMetrics([trade, trade])
        assert metrics._calculate_sharpe_ratio() == 0

    def test_none_trades(self):
        """Test metrics with None trades."""
        metrics = SummaryMetrics(None)
        result = metrics.calculate_all_metrics()

        assert result == {}

    def test_single_winning_trade(self, trade_factory):
        """Test metrics with single winning trade."""
        winning_trade = trade_factory('ZS', 1200.0, 1210.0)
        metrics = SummaryMetrics([winning_trade])
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 1
        assert result['win_rate'] == 100.0
        assert result['winning_trades'] == 1
        assert result['losing_trades'] == 0

    def test_single_losing_trade(self, trade_factory):
        """Test metrics with single losing trade."""
        losing_trade = trade_factory('ZS', 1200.0, 1195.0)
        metrics = SummaryMetrics([losing_trade])
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 1
        assert result['win_rate'] == 0.0
        assert result['winning_trades'] == 0
        assert result['losing_trades'] == 1

    def test_breakeven_trade_counted_as_loss(self, trade_factory):
        """Test breakeven trade is counted as losing trade."""
        breakeven_trade = trade_factory('ZS', 1200.0, 1200.0)
        metrics = SummaryMetrics([breakeven_trade])
        result = metrics.calculate_all_metrics()

        # Breakeven (including commission) should be <= 0
        assert result['losing_trades'] == 1
        assert result['winning_trades'] == 0


class TestMetricsRounding:
    """Test that all metrics are properly rounded."""

    def test_all_metrics_rounded_to_two_decimals(self, trades_factory):
        """Test all metrics are rounded to 2 decimal places."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
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

    def test_calculate_consecutive_streaks_on_empty_metrics(self):
        """Test _calculate_consecutive_streaks returns (0, 0) when called directly on empty metrics."""
        metrics = SummaryMetrics([])
        assert metrics._calculate_consecutive_streaks() == (0, 0)

    def test_has_trades_method(self, trades_factory):
        """Test _has_trades helper method."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        assert metrics._has_trades() is True

        empty_metrics = SummaryMetrics([])
        assert empty_metrics._has_trades() is False

    def test_has_winning_trades_method(self, trades_factory):
        """Test _has_winning_trades helper method."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        assert metrics._has_winning_trades() is True

        losing_only = trades_factory.all_losing(count=1)
        losing_metrics = SummaryMetrics(losing_only)
        assert losing_metrics._has_winning_trades() is False

    def test_has_losing_trades_method(self, trades_factory):
        """Test _has_losing_trades helper method."""
        mixed_trades = trades_factory.mixed(win_count=1, loss_count=1)
        metrics = SummaryMetrics(mixed_trades)
        assert metrics._has_losing_trades() is True

        winning_only = trades_factory.all_winning(count=1)
        winning_metrics = SummaryMetrics(winning_only)
        assert winning_metrics._has_losing_trades() is False


class TestConstants:
    """Test module constants are defined correctly."""

    def test_min_returns_for_sharpe(self):
        """Test MIN_RETURNS_FOR_SHARPE constant."""
        assert MIN_RETURNS_FOR_SHARPE == 2

    def test_min_returns_for_var(self):
        """Test MIN_RETURNS_FOR_VAR constant."""
        assert MIN_RETURNS_FOR_VAR == 30

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

    def test_profitable_strategy(self, trades_factory):
        """Test metrics for overall profitable strategy."""
        # 7 wins, 3 losses
        trades = []

        # 7 winning trades
        for i in range(7):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1205.0 + i,
                                                       entry_time=datetime(2024, 1, 10 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 10 + i, 14, 0)))

        # 3 losing trades
        for i in range(3):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1198.0,
                                                       entry_time=datetime(2024, 1, 17 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 17 + i, 14, 0)))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 10
        assert result['win_rate'] == 70.0
        assert result['total_return_percentage_of_contract'] > 0
        assert result['profit_factor'] > 1.0

    def test_losing_strategy(self, trades_factory):
        """Test metrics for overall losing strategy."""
        # 3 wins, 7 losses
        trades = []

        # 3 winning trades
        for i in range(3):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1202.0,
                                                       entry_time=datetime(2024, 1, 10 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 10 + i, 14, 0)))

        # 7 losing trades
        for i in range(7):
            trades.append(trades_factory.trade_factory('ZS', 1200.0, 1195.0 - i,
                                                       entry_time=datetime(2024, 1, 13 + i, 10, 0),
                                                       exit_time=datetime(2024, 1, 13 + i, 14, 0)))

        metrics = SummaryMetrics(trades)
        result = metrics.calculate_all_metrics()

        assert result['total_trades'] == 10
        assert result['win_rate'] == 30.0
        assert result['total_return_percentage_of_contract'] < 0
        assert result['profit_factor'] < 1.0
        assert result['maximum_drawdown_percentage'] > 0


class TestWinLossRatio:
    """Test win/loss ratio calculation."""

    def test_win_loss_ratio_mixed_trades(self, trades_factory):
        """Test win/loss ratio is average_win / abs(average_loss)."""
        trades = trades_factory.create_sequence(
            [(1200, 1220), (1200, 1185)], symbol='ZS'
        )
        result = SummaryMetrics(trades).calculate_all_metrics()

        assert 'win_loss_ratio' in result
        assert result['win_loss_ratio'] > 0

    def test_win_loss_ratio_all_wins_returns_infinity(self, trades_factory):
        """Test win/loss ratio is INFINITY_REPLACEMENT when no losses."""
        all_winning = trades_factory.all_winning(count=5)
        result = SummaryMetrics(all_winning).calculate_all_metrics()

        assert result['win_loss_ratio'] == INFINITY_REPLACEMENT

    def test_win_loss_ratio_higher_for_bigger_wins(self, trades_factory):
        """Test win/loss ratio increases when wins are larger."""
        small_win_trades = trades_factory.create_sequence(
            [(1200, 1205), (1200, 1195)], symbol='ZS'  # 5 pt win, 5 pt loss
        )
        big_win_trades = trades_factory.create_sequence(
            [(1200, 1220), (1200, 1195)], symbol='ZS'  # 20 pt win, 5 pt loss
        )

        small_ratio = SummaryMetrics(small_win_trades).calculate_all_metrics()['win_loss_ratio']
        big_ratio = SummaryMetrics(big_win_trades).calculate_all_metrics()['win_loss_ratio']

        assert big_ratio > small_ratio

    def test_win_loss_ratio_rounded_to_two_decimals(self, trades_factory):
        """Test win/loss ratio is rounded to 2 decimal places."""
        trades = trades_factory.mixed(win_count=2, loss_count=2)
        result = SummaryMetrics(trades).calculate_all_metrics()

        value = result['win_loss_ratio']
        str_val = str(value)
        if '.' in str_val:
            assert len(str_val.split('.')[1]) <= 2


class TestConsecutiveStreaks:
    """Test max consecutive wins/losses calculation."""

    def test_max_consecutive_wins_basic(self, trades_factory):
        """Test max consecutive wins is counted correctly."""
        # 3 wins then 1 loss then 1 win
        trades = trades_factory.create_sequence(
            [(1200, 1210), (1200, 1210), (1200, 1210), (1200, 1190), (1200, 1210)],
            symbol='ZS'
        )
        result = SummaryMetrics(trades).calculate_all_metrics()

        assert result['max_consecutive_wins'] == 3

    def test_max_consecutive_losses_basic(self, trades_factory):
        """Test max consecutive losses is counted correctly."""
        # 1 win then 4 losses
        trades = trades_factory.create_sequence(
            [(1200, 1210), (1200, 1190), (1200, 1190), (1200, 1190), (1200, 1190)],
            symbol='ZS'
        )
        result = SummaryMetrics(trades).calculate_all_metrics()

        assert result['max_consecutive_losses'] == 4

    def test_consecutive_streaks_all_wins(self, trades_factory):
        """Test streaks when all trades are winning."""
        trades = trades_factory.all_winning(count=5)
        result = SummaryMetrics(trades).calculate_all_metrics()

        assert result['max_consecutive_wins'] == 5
        assert result['max_consecutive_losses'] == 0

    def test_consecutive_streaks_all_losses(self, trades_factory):
        """Test streaks when all trades are losing."""
        trades = trades_factory.all_losing(count=5)
        result = SummaryMetrics(trades).calculate_all_metrics()

        assert result['max_consecutive_wins'] == 0
        assert result['max_consecutive_losses'] == 5

    def test_consecutive_streaks_alternating(self, trades_factory):
        """Test streaks with alternating win/loss pattern."""
        trades = trades_factory.create_sequence(
            [(1200, 1210), (1200, 1190), (1200, 1210), (1200, 1190)],
            symbol='ZS'
        )
        result = SummaryMetrics(trades).calculate_all_metrics()

        assert result['max_consecutive_wins'] == 1
        assert result['max_consecutive_losses'] == 1


class TestExpectancyPerBar:
    """Test expectancy_per_bar calculation."""

    def test_expectancy_per_bar_positive_for_winning_strategy(self, trade_factory):
        """Test expectancy_per_bar is positive for profitable trades."""
        trade = trade_factory('ZS', 1200.0, 1220.0, duration_hours=4.0)
        trade['duration_bars'] = 4.0

        result = SummaryMetrics([trade]).calculate_all_metrics()

        assert result['expectancy_per_bar'] > 0

    def test_expectancy_per_bar_is_zero_when_no_duration(self, trade_factory):
        """Test expectancy_per_bar is 0 when duration_bars is missing."""
        trade = trade_factory('ZS', 1200.0, 1220.0)
        # No duration_bars set — defaults to 0

        result = SummaryMetrics([trade]).calculate_all_metrics()

        assert result['expectancy_per_bar'] == 0.0

    def test_expectancy_per_bar_longer_trade_lower_per_bar(self, trade_factory):
        """Test that the same return over more bars gives lower expectancy_per_bar."""
        short_trade = trade_factory('ZS', 1200.0, 1220.0)
        short_trade['duration_bars'] = 2.0

        long_trade = trade_factory('ZS', 1200.0, 1220.0)
        long_trade['duration_bars'] = 8.0

        short_result = SummaryMetrics([short_trade]).calculate_all_metrics()
        long_result = SummaryMetrics([long_trade]).calculate_all_metrics()

        assert short_result['expectancy_per_bar'] > long_result['expectancy_per_bar']

    def test_expectancy_per_bar_rounded_to_four_decimals(self, trade_factory):
        """Test expectancy_per_bar is rounded to 4 decimal places."""
        trade = trade_factory('ZS', 1200.0, 1210.0)
        trade['duration_bars'] = 3.0

        result = SummaryMetrics([trade]).calculate_all_metrics()

        value = result['expectancy_per_bar']
        str_val = str(value)
        if '.' in str_val:
            assert len(str_val.split('.')[1]) <= 4


class TestTimeInMarket:
    """Test time_in_market_percentage calculation."""

    def test_time_in_market_requires_dataset_total_hours(self, trade_factory):
        """Test time_in_market_percentage is 0 when dataset_total_hours not provided."""
        trade = trade_factory('ZS', 1200.0, 1210.0, duration_hours=4.0)
        result = SummaryMetrics([trade]).calculate_all_metrics()

        assert result['time_in_market_percentage'] == 0.0

    def test_time_in_market_calculated_with_dataset_hours(self, trade_factory):
        """Test time_in_market_percentage is computed when dataset_total_hours is provided."""
        trade = trade_factory('ZS', 1200.0, 1210.0, duration_hours=4.0)
        # Trade duration is 4 hours, dataset is 8 hours → 50%
        result = SummaryMetrics([trade], dataset_total_hours=8.0).calculate_all_metrics()

        assert result['time_in_market_percentage'] == 50.0

    def test_time_in_market_multiple_trades(self, trade_factory):
        """Test time_in_market_percentage sums all trade durations."""
        trade1 = trade_factory('ZS', 1200.0, 1210.0, duration_hours=2.0)
        trade2 = trade_factory('ZS', 1200.0, 1195.0, duration_hours=2.0)
        # Total trade time = 4h, dataset = 8h → 50%
        result = SummaryMetrics([trade1, trade2], dataset_total_hours=8.0).calculate_all_metrics()

        assert result['time_in_market_percentage'] == 50.0

    def test_time_in_market_zero_when_dataset_hours_is_zero(self, trade_factory):
        """Test time_in_market_percentage is 0 when dataset_total_hours is 0."""
        trade = trade_factory('ZS', 1200.0, 1210.0, duration_hours=4.0)
        result = SummaryMetrics([trade], dataset_total_hours=0).calculate_all_metrics()

        assert result['time_in_market_percentage'] == 0.0


class TestCalmarAnnualisation:
    """Test annualised Calmar ratio via dataset_total_hours."""

    def test_calmar_without_dataset_hours_uses_raw_return(self, trades_factory):
        """Test Calmar falls back to raw total return when dataset_total_hours is None."""
        trades = trades_factory.create_sequence(
            [(1200, 1250), (1250, 1230), (1230, 1260)], symbol='ZS'
        )
        result = SummaryMetrics(trades).calculate_all_metrics()

        # Should still produce a valid number (not crash)
        assert isinstance(result['calmar_ratio'], float)

    def test_calmar_with_dataset_hours_annualises_return(self, trades_factory):
        """Test Calmar uses annualised return when dataset_total_hours is provided."""
        specs = [(1200, 1250), (1250, 1230), (1230, 1260)]
        trades = trades_factory.create_sequence(specs, symbol='ZS')

        # Use 1 year of data → annualisation_factor = 1, same as raw
        result_1y = SummaryMetrics(trades, dataset_total_hours=8760).calculate_all_metrics()
        # Use 6 months → annualisation_factor = 2, Calmar should be ~2x
        result_6m = SummaryMetrics(trades, dataset_total_hours=4380).calculate_all_metrics()

        # Shorter dataset → higher Calmar (same return annualised over shorter period)
        assert result_6m['calmar_ratio'] > result_1y['calmar_ratio']

    def test_calmar_no_drawdown_returns_infinity(self, trades_factory):
        """Test Calmar returns INFINITY_REPLACEMENT when max drawdown is 0."""
        # Strictly increasing cumulative returns → no drawdown
        trades = trades_factory.all_winning(count=3)
        result = SummaryMetrics(trades, dataset_total_hours=8760).calculate_all_metrics()

        assert result['calmar_ratio'] == INFINITY_REPLACEMENT
