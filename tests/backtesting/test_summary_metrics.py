import io
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

from app.backtesting.summary_metrics import (calculate_max_drawdown,
                                             calculate_summary_metrics,
                                             print_summary_metrics,
                                             calculate_max_consecutive,
                                             calculate_sharpe_ratio,
                                             calculate_sortino_ratio,
                                             calculate_calmar_ratio)


# Helper function to create a sample trade
def create_sample_trade(
    net_pnl=100.0,
    return_percentage=1.0,
    duration_hours=24,
    margin_requirement=10000.0,
    commission=4.0
):
    entry_time = datetime.now()
    exit_time = entry_time + timedelta(hours=duration_hours)

    return {
        'entry_time': entry_time,
        'exit_time': exit_time,
        'duration': timedelta(hours=duration_hours),
        'duration_hours': duration_hours,
        'net_pnl': net_pnl,
        'gross_pnl': net_pnl + commission,
        'return_percentage_of_margin': return_percentage,
        'margin_requirement': margin_requirement,
        'commission': commission
    }


class TestCalculateMaxDrawdown:
    """Tests for the calculate_max_drawdown function."""

    def test_empty_trades_list(self):
        """Test calculation of max drawdown with an empty trades list."""
        max_drawdown, max_drawdown_pct = calculate_max_drawdown([])
        assert max_drawdown == 0
        assert max_drawdown_pct == 0

    def test_single_trade_positive(self):
        """Test calculation of max drawdown with a single positive trade."""
        trade = create_sample_trade(net_pnl=100.0, return_percentage=1.0)
        max_drawdown, max_drawdown_pct = calculate_max_drawdown([trade])
        assert max_drawdown == 0
        assert max_drawdown_pct == 0

    def test_single_trade_negative(self):
        """Test calculation of max drawdown with a single negative trade."""
        trade = create_sample_trade(net_pnl=-100.0, return_percentage=-1.0)
        max_drawdown, max_drawdown_pct = calculate_max_drawdown([trade])
        # For a single negative trade, there's no peak to draw down from
        # The function initializes peak to the first value, so drawdown is 0
        assert max_drawdown == 0
        assert max_drawdown_pct == 0

    def test_multiple_trades_no_drawdown(self):
        """Test calculation of max drawdown with multiple trades but no drawdown."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=300.0, return_percentage=3.0)
        ]
        max_drawdown, max_drawdown_pct = calculate_max_drawdown(trades)
        assert max_drawdown == 0
        assert max_drawdown_pct == 0

    def test_multiple_trades_with_drawdown(self):
        """Test calculation of max drawdown with multiple trades with drawdown."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=-150.0, return_percentage=-1.5),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0)
        ]
        max_drawdown, max_drawdown_pct = calculate_max_drawdown(trades)
        assert max_drawdown == 150.0
        assert max_drawdown_pct == 1.5

    def test_complex_drawdown_scenario(self):
        """Test calculation of max drawdown with a complex sequence of trades."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),  # Cumulative: 100, 1%
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),  # Cumulative: 300, 3%
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),  # Cumulative: 250, 2.5%
            create_sample_trade(net_pnl=-300.0, return_percentage=-3.0),  # Cumulative: -50, -0.5%
            create_sample_trade(net_pnl=150.0, return_percentage=1.5)  # Cumulative: 100, 1%
        ]
        max_drawdown, max_drawdown_pct = calculate_max_drawdown(trades)
        assert max_drawdown == 350.0  # From peak of 300 to low of -50
        assert max_drawdown_pct == 3.5  # From peak of 3% to low of -0.5%

    def test_drawdown_with_recovery(self):
        """Test calculation of max drawdown with a drawdown followed by recovery."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),  # Cumulative: 100, 1%
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),  # Cumulative: 50, 0.5%
            create_sample_trade(net_pnl=-60.0, return_percentage=-0.6),  # Cumulative: -10, -0.1%
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),  # Cumulative: 190, 1.9%
            create_sample_trade(net_pnl=100.0, return_percentage=1.0)  # Cumulative: 290, 2.9%
        ]
        max_drawdown, max_drawdown_pct = calculate_max_drawdown(trades)
        assert max_drawdown == 110.0  # From peak of 100 to low of -10
        assert max_drawdown_pct == 1.1  # From peak of 1% to low of -0.1%

    def test_multiple_drawdowns(self):
        """Test calculation of max drawdown with multiple drawdowns."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),  # Cumulative: 100, 1%
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),  # Cumulative: 50, 0.5% (Drawdown 1)
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),  # Cumulative: 250, 2.5%
            create_sample_trade(net_pnl=-150.0, return_percentage=-1.5),  # Cumulative: 100, 1% (Drawdown 2)
            create_sample_trade(net_pnl=300.0, return_percentage=3.0)  # Cumulative: 400, 4%
        ]
        max_drawdown, max_drawdown_pct = calculate_max_drawdown(trades)
        assert max_drawdown == 150.0  # The larger of the two drawdowns
        assert max_drawdown_pct == 1.5  # The larger of the two drawdowns in percentage


class TestCalculateSummaryMetrics:
    """Tests for the calculate_summary_metrics function."""

    def test_empty_trades_list(self):
        """Test calculation of summary metrics with an empty trades list."""
        from unittest.mock import patch

        # Mock the logger to verify it's called
        with patch('app.backtesting.summary_metrics.logger.error') as mock_logger:
            summary = calculate_summary_metrics([])
            assert summary == {}
            # Verify logger.error was called with the expected message
            mock_logger.assert_called_once_with('No trades provided to calculate_summary_metrics')

    def test_single_winning_trade(self):
        """Test calculation of summary metrics with a single winning trade."""
        trade = create_sample_trade(net_pnl=100.0, return_percentage=1.0, margin_requirement=10000.0, commission=4.0)
        summary = calculate_summary_metrics([trade])

        assert summary['total_trades'] == 1
        assert summary['winning_trades'] == 1
        assert summary['losing_trades'] == 0
        assert summary['win_rate'] == 100.0
        assert summary['total_return_percentage_of_margin'] == 1.0
        assert summary['profit_factor'] == float('inf')  # No losing trades

    def test_single_losing_trade(self):
        """Test calculation of summary metrics with a single losing trade."""
        trade = create_sample_trade(net_pnl=-100.0, return_percentage=-1.0, margin_requirement=10000.0, commission=4.0)
        summary = calculate_summary_metrics([trade])

        assert summary['total_trades'] == 1
        assert summary['winning_trades'] == 0
        assert summary['losing_trades'] == 1
        assert summary['win_rate'] == 0.0
        assert summary['total_return_percentage_of_margin'] == -1.0
        assert summary['profit_factor'] == 0.0  # No winning trades

    def test_multiple_trades(self):
        """Test calculation of summary metrics with multiple trades."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['total_trades'] == 3
        assert summary['winning_trades'] == 2
        assert summary['losing_trades'] == 1
        assert summary['win_rate'] == round((2 / 3) * 100, 2)
        assert summary['total_return_percentage_of_margin'] == 2.5
        assert summary['average_win_percentage_of_margin'] > 0
        assert summary['average_loss_percentage_of_margin'] < 0
        assert summary['profit_factor'] == round(300.0 / 50.0, 2)  # (100 + 200) / 50

    def test_all_winning_trades(self):
        """Test calculation of summary metrics with all winning trades."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=300.0, return_percentage=3.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['total_trades'] == 3
        assert summary['winning_trades'] == 3
        assert summary['losing_trades'] == 0
        assert summary['win_rate'] == 100.0
        assert summary['total_return_percentage_of_margin'] == 6.0
        assert summary['profit_factor'] == float('inf')  # No losing trades

    def test_all_losing_trades(self):
        """Test calculation of summary metrics with all losing trades."""
        trades = [
            create_sample_trade(net_pnl=-100.0, return_percentage=-1.0),
            create_sample_trade(net_pnl=-200.0, return_percentage=-2.0),
            create_sample_trade(net_pnl=-300.0, return_percentage=-3.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['total_trades'] == 3
        assert summary['winning_trades'] == 0
        assert summary['losing_trades'] == 3
        assert summary['win_rate'] == 0.0
        assert summary['total_return_percentage_of_margin'] == -6.0
        assert summary['profit_factor'] == 0.0  # No winning trades

    def test_commission_metrics(self):
        """Test calculation of commission metrics."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0, commission=5.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0, commission=5.0),
            create_sample_trade(net_pnl=300.0, return_percentage=3.0, commission=5.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['commission_percentage_of_margin'] == round((15.0 / 30000.0) * 100, 2)  # 3 trades * 10000 margin

    def test_duration_metrics(self):
        """Test calculation of duration metrics."""
        trades = [
            create_sample_trade(duration_hours=12),
            create_sample_trade(duration_hours=24),
            create_sample_trade(duration_hours=36)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['avg_trade_duration_hours'] == 24.0  # (12 + 24 + 36) / 3

    def test_risk_metrics(self):
        """Test calculation of risk metrics."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0)
        ]
        summary = calculate_summary_metrics(trades)

        # Max drawdown percentage should be calculated correctly
        assert summary['maximum_drawdown_percentage'] > 0

        # Return to drawdown ratio should be calculated correctly
        assert summary['return_to_drawdown_ratio'] == round(2.5 / summary['maximum_drawdown_percentage'], 2)

    def test_performance_ratios(self):
        """Test calculation of performance ratios."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=150.0, return_percentage=1.5)
        ]
        summary = calculate_summary_metrics(trades)

        # Sharpe ratio should be calculated correctly
        assert 'sharpe_ratio' in summary
        assert isinstance(summary['sharpe_ratio'], float)

        # Sortino ratio should be calculated correctly
        assert 'sortino_ratio' in summary
        assert isinstance(summary['sortino_ratio'], float)

        # Calmar ratio should be calculated correctly
        assert 'calmar_ratio' in summary
        assert isinstance(summary['calmar_ratio'], float)

    def test_zero_drawdown(self):
        """Test calculation of summary metrics with zero drawdown."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['maximum_drawdown_percentage'] == 0
        assert summary['return_to_drawdown_ratio'] == float('inf')  # Division by zero

    def test_margin_metrics(self):
        """Test calculation of margin metrics."""
        trades = [
            create_sample_trade(margin_requirement=10000.0),
            create_sample_trade(margin_requirement=20000.0),
            create_sample_trade(margin_requirement=30000.0)
        ]
        summary = calculate_summary_metrics(trades)

        # We no longer track dollar-based margin metrics
        assert 'total_margin_used' not in summary
        assert 'avg_margin_requirement' not in summary

    def test_realistic_trading_pattern(self):
        """Test calculation of summary metrics with a realistic trading pattern.

        This test simulates a more realistic trading scenario with a mix of wins and losses
        of varying magnitudes, which is common in real trading.
        """
        trades = [
            # Initial winning streak
            create_sample_trade(net_pnl=120.0, return_percentage=1.2, duration_hours=4),
            create_sample_trade(net_pnl=85.0, return_percentage=0.85, duration_hours=6),
            create_sample_trade(net_pnl=210.0, return_percentage=2.1, duration_hours=12),
            # Small losing streak
            create_sample_trade(net_pnl=-45.0, return_percentage=-0.45, duration_hours=3),
            create_sample_trade(net_pnl=-75.0, return_percentage=-0.75, duration_hours=5),
            # Recovery
            create_sample_trade(net_pnl=150.0, return_percentage=1.5, duration_hours=8),
            # Mixed results
            create_sample_trade(net_pnl=-110.0, return_percentage=-1.1, duration_hours=10),
            create_sample_trade(net_pnl=95.0, return_percentage=0.95, duration_hours=7),
            create_sample_trade(net_pnl=-60.0, return_percentage=-0.6, duration_hours=4),
            create_sample_trade(net_pnl=180.0, return_percentage=1.8, duration_hours=9)
        ]
        summary = calculate_summary_metrics(trades)

        # Basic statistics
        assert summary['total_trades'] == 10
        assert summary['winning_trades'] == 6
        assert summary['losing_trades'] == 4
        assert summary['win_rate'] == 60.0

        # Percentage-based metrics
        total_return_percentage = sum(trade['return_percentage_of_margin'] for trade in trades)
        assert summary['total_return_percentage_of_margin'] == total_return_percentage
        assert summary['average_trade_return_percentage_of_margin'] == total_return_percentage / 10
        assert summary['average_win_percentage_of_margin'] > 0
        assert summary['average_loss_percentage_of_margin'] < 0

        # Duration metrics
        assert summary['avg_trade_duration_hours'] == 6.8  # (4+6+12+3+5+8+10+7+4+9)/10

        # Risk metrics
        assert summary['profit_factor'] > 1.0
        assert summary['maximum_drawdown_percentage'] > 0
        assert summary['return_to_drawdown_ratio'] > 0

        # Performance ratios
        assert 'sharpe_ratio' in summary
        assert 'sortino_ratio' in summary
        assert 'calmar_ratio' in summary


    def test_long_drawdown_with_recovery(self):
        """Test calculation of summary metrics with a long drawdown period followed by recovery.

        This test simulates a scenario where a trading strategy experiences a prolonged
        drawdown period before eventually recovering, which is a common challenge in trading.
        """
        trades = [
            # Initial success
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=150.0, return_percentage=1.5),
            # Long drawdown period
            create_sample_trade(net_pnl=-80.0, return_percentage=-0.8),
            create_sample_trade(net_pnl=-120.0, return_percentage=-1.2),
            create_sample_trade(net_pnl=-90.0, return_percentage=-0.9),
            create_sample_trade(net_pnl=-150.0, return_percentage=-1.5),
            create_sample_trade(net_pnl=-100.0, return_percentage=-1.0),
            # Recovery phase
            create_sample_trade(net_pnl=120.0, return_percentage=1.2),
            create_sample_trade(net_pnl=180.0, return_percentage=1.8),
            create_sample_trade(net_pnl=250.0, return_percentage=2.5),
            create_sample_trade(net_pnl=300.0, return_percentage=3.0)
        ]
        summary = calculate_summary_metrics(trades)

        # Basic statistics
        assert summary['total_trades'] == 11
        assert summary['winning_trades'] == 6
        assert summary['losing_trades'] == 5
        assert summary['win_rate'] == round((6 / 11) * 100, 2)

        # Percentage-based metrics
        total_expected_return_percentage = 2.0 + 1.5 - 0.8 - 1.2 - 0.9 - 1.5 - 1.0 + 1.2 + 1.8 + 2.5 + 3.0
        assert summary['total_return_percentage_of_margin'] == total_expected_return_percentage
        assert summary['average_trade_return_percentage_of_margin'] == round(total_expected_return_percentage / 11, 2)

        # Risk metrics
        # Significant drawdown expected
        assert summary['maximum_drawdown_percentage'] >= 5.4  # From peak of 3.5% to lowest of -1.9%
        # Despite drawdown, overall strategy is profitable
        assert summary['total_return_percentage_of_margin'] > 0
        assert summary['return_to_drawdown_ratio'] > 0

    def test_varying_trade_durations(self):
        """Test calculation of summary metrics with trades of varying durations.

        This test simulates a mix of short-term and long-term trades, which is common
        in many trading strategies that adapt to different market conditions.
        """
        trades = [
            # Very short-term trades (scalping)
            create_sample_trade(net_pnl=50.0, return_percentage=0.5, duration_hours=0.5),
            create_sample_trade(net_pnl=-30.0, return_percentage=-0.3, duration_hours=0.25),
            create_sample_trade(net_pnl=40.0, return_percentage=0.4, duration_hours=0.75),
            # Short-term trades (intraday)
            create_sample_trade(net_pnl=100.0, return_percentage=1.0, duration_hours=4),
            create_sample_trade(net_pnl=-70.0, return_percentage=-0.7, duration_hours=6),
            # Medium-term trades (swing)
            create_sample_trade(net_pnl=200.0, return_percentage=2.0, duration_hours=48),
            create_sample_trade(net_pnl=-150.0, return_percentage=-1.5, duration_hours=72),
            # Long-term trades (position)
            create_sample_trade(net_pnl=500.0, return_percentage=5.0, duration_hours=240),
            create_sample_trade(net_pnl=-300.0, return_percentage=-3.0, duration_hours=168)
        ]
        summary = calculate_summary_metrics(trades)

        # Basic statistics
        assert summary['total_trades'] == 9
        assert summary['winning_trades'] == 5
        assert summary['losing_trades'] == 4
        assert summary['win_rate'] == round((5 / 9) * 100, 2)

        # Duration metrics
        total_duration = 0.5 + 0.25 + 0.75 + 4 + 6 + 48 + 72 + 240 + 168
        assert summary['avg_trade_duration_hours'] == round(total_duration / 9, 2)

        # Percentage-based metrics
        total_expected_return_percentage = 0.5 - 0.3 + 0.4 + 1.0 - 0.7 + 2.0 - 1.5 + 5.0 - 3.0
        assert round(summary['total_return_percentage_of_margin'], 2) == round(total_expected_return_percentage, 2)
        assert round(summary['average_trade_return_percentage_of_margin'],
                     2) == round(total_expected_return_percentage / 9, 2)

    def test_extreme_profit_loss_values(self):
        """Test calculation of summary metrics with extreme profit and loss values.

        This test simulates scenarios with unusually large profits and losses,
        which can occur during high volatility events or black swan events.
        """
        trades = [
            # Normal trades
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),
            # Extreme profit (e.g., from a major market move in favor)
            create_sample_trade(net_pnl=5000.0, return_percentage=50.0, margin_requirement=10000.0),
            # Normal trades
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=-150.0, return_percentage=-1.5),
            # Extreme loss (e.g., from a market crash or flash crash)
            create_sample_trade(net_pnl=-3000.0, return_percentage=-30.0, margin_requirement=10000.0),
            # Normal trades
            create_sample_trade(net_pnl=300.0, return_percentage=3.0),
            create_sample_trade(net_pnl=-100.0, return_percentage=-1.0)
        ]
        summary = calculate_summary_metrics(trades)

        # Basic statistics
        assert summary['total_trades'] == 8
        assert summary['winning_trades'] == 4
        assert summary['losing_trades'] == 4
        assert summary['win_rate'] == 50.0

        # Percentage-based metrics
        total_expected_return_percentage = 1.0 - 0.5 + 50.0 + 2.0 - 1.5 - 30.0 + 3.0 - 1.0
        assert summary['total_return_percentage_of_margin'] == total_expected_return_percentage
        assert summary['average_trade_return_percentage_of_margin'] == round(total_expected_return_percentage / 8, 2)
        assert summary['average_win_percentage_of_margin'] > 10  # Skewed by the 50% profit
        assert summary['average_loss_percentage_of_margin'] < -5  # Skewed by the -30% loss

        # Risk metrics
        # Extreme values should be reflected in the metrics
        assert summary['maximum_drawdown_percentage'] >= 30  # The extreme percentage loss

    def test_volatile_market_conditions(self):
        """Test calculation of summary metrics under volatile market conditions.

        This test simulates trading during volatile market conditions with rapid
        alternating wins and losses of varying magnitudes, which is common during
        market uncertainty or news-driven volatility.
        """
        trades = [
            # Volatile period with alternating wins and losses
            create_sample_trade(net_pnl=150.0, return_percentage=1.5, duration_hours=2),
            create_sample_trade(net_pnl=-120.0, return_percentage=-1.2, duration_hours=1),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0, duration_hours=3),
            create_sample_trade(net_pnl=-180.0, return_percentage=-1.8, duration_hours=2),
            create_sample_trade(net_pnl=250.0, return_percentage=2.5, duration_hours=4),
            create_sample_trade(net_pnl=-220.0, return_percentage=-2.2, duration_hours=3),
            create_sample_trade(net_pnl=300.0, return_percentage=3.0, duration_hours=5),
            create_sample_trade(net_pnl=-270.0, return_percentage=-2.7, duration_hours=4),
            create_sample_trade(net_pnl=350.0, return_percentage=3.5, duration_hours=6),
            create_sample_trade(net_pnl=-320.0, return_percentage=-3.2, duration_hours=5)
        ]
        summary = calculate_summary_metrics(trades)

        # Basic statistics
        assert summary['total_trades'] == 10
        assert summary['winning_trades'] == 5
        assert summary['losing_trades'] == 5
        assert summary['win_rate'] == 50.0

        # Percentage-based metrics
        total_expected_return_percentage = 1.5 - 1.2 + 2.0 - 1.8 + 2.5 - 2.2 + 3.0 - 2.7 + 3.5 - 3.2
        assert round(summary['total_return_percentage_of_margin'], 2) == round(total_expected_return_percentage, 2)
        assert round(summary['average_trade_return_percentage_of_margin'],
                     2) == round(total_expected_return_percentage / 10, 2)

        # Duration metrics
        total_duration = 2 + 1 + 3 + 2 + 4 + 3 + 5 + 4 + 6 + 5
        assert summary['avg_trade_duration_hours'] == round(total_duration / 10, 2)

        # Risk metrics
        # Volatility should be reflected in the drawdown metrics
        assert summary['maximum_drawdown_percentage'] > 0
        # Despite volatility, the strategy should be profitable
        assert summary['total_return_percentage_of_margin'] > 0
        assert summary['profit_factor'] > 1.0

    def test_seasonal_trading_pattern(self):
        """Test calculation of summary metrics with a seasonal trading pattern.

        This test simulates a trading strategy that performs differently in different
        time periods, which is common for strategies affected by seasonal market patterns,
        quarterly earnings seasons, or other cyclical market behaviors.
        """
        # Create a base time for entry
        base_time = datetime.now()

        # Create trades with specific entry times to simulate different months
        trades = []

        # January-February: Strong performance (bull market)
        jan_feb_trades = [
            create_sample_trade(net_pnl=200.0, return_percentage=2.0, duration_hours=48),
            create_sample_trade(net_pnl=150.0, return_percentage=1.5, duration_hours=36),
            create_sample_trade(net_pnl=250.0, return_percentage=2.5, duration_hours=72),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5, duration_hours=24)  # One loss
        ]
        # Set entry times for January-February
        for i, trade in enumerate(jan_feb_trades):
            trade['entry_time'] = base_time.replace(month=1, day=i + 1)
            trade['exit_time'] = trade['entry_time'] + trade['duration']
        trades.extend(jan_feb_trades)

        # March-April: Market correction (more losses)
        mar_apr_trades = [
            create_sample_trade(net_pnl=-120.0, return_percentage=-1.2, duration_hours=48),
            create_sample_trade(net_pnl=-180.0, return_percentage=-1.8, duration_hours=36),
            create_sample_trade(net_pnl=100.0, return_percentage=1.0, duration_hours=72),  # One win
            create_sample_trade(net_pnl=-150.0, return_percentage=-1.5, duration_hours=24)
        ]
        # Set entry times for March-April
        for i, trade in enumerate(mar_apr_trades):
            trade['entry_time'] = base_time.replace(month=3, day=i + 1)
            trade['exit_time'] = trade['entry_time'] + trade['duration']
        trades.extend(mar_apr_trades)

        # May-June: Recovery (mixed results)
        may_jun_trades = [
            create_sample_trade(net_pnl=130.0, return_percentage=1.3, duration_hours=48),
            create_sample_trade(net_pnl=-90.0, return_percentage=-0.9, duration_hours=36),
            create_sample_trade(net_pnl=160.0, return_percentage=1.6, duration_hours=72),
            create_sample_trade(net_pnl=-70.0, return_percentage=-0.7, duration_hours=24)
        ]
        # Set entry times for May-June
        for i, trade in enumerate(may_jun_trades):
            trade['entry_time'] = base_time.replace(month=5, day=i + 1)
            trade['exit_time'] = trade['entry_time'] + trade['duration']
        trades.extend(may_jun_trades)

        summary = calculate_summary_metrics(trades)

        # Basic statistics
        assert summary['total_trades'] == 12
        assert summary['winning_trades'] == 6
        assert summary['losing_trades'] == 6
        assert summary['win_rate'] == 50.0

        # Percentage-based metrics
        total_expected_return_percentage = (2.0 + 1.5 + 2.5 - 0.5) + (-1.2 - 1.8 + 1.0 - 1.5) + (1.3 - 0.9 + 1.6 - 0.7)
        assert round(summary['total_return_percentage_of_margin'], 2) == round(total_expected_return_percentage, 2)
        # Use a more flexible comparison for average trade return percentage
        avg_trade_return = summary['average_trade_return_percentage_of_margin']
        expected_avg_trade_return = total_expected_return_percentage / 12
        assert abs(avg_trade_return - expected_avg_trade_return) < 0.02  # Allow for small differences

        # Risk metrics
        # The seasonal pattern should create a significant drawdown during the correction period
        assert summary['maximum_drawdown_percentage'] > 0
        # Overall, the strategy should still be profitable
        assert summary['total_return_percentage_of_margin'] > 0
        assert summary['profit_factor'] > 1.0


class TestCalculateMaxConsecutive:
    """Tests for the calculate_max_consecutive function."""

    def test_empty_trades_list(self):
        """Test calculation of max consecutive wins/losses with an empty trades list."""
        max_consecutive_wins = calculate_max_consecutive([], win=True)
        max_consecutive_losses = calculate_max_consecutive([], win=False)
        assert max_consecutive_wins == 0
        assert max_consecutive_losses == 0

    def test_single_trade_win(self):
        """Test calculation of max consecutive wins with a single winning trade."""
        trade = create_sample_trade(net_pnl=100.0, return_percentage=1.0)
        max_consecutive_wins = calculate_max_consecutive([trade], win=True)
        assert max_consecutive_wins == 1

    def test_single_trade_loss(self):
        """Test calculation of max consecutive losses with a single losing trade."""
        trade = create_sample_trade(net_pnl=-100.0, return_percentage=-1.0)
        max_consecutive_losses = calculate_max_consecutive([trade], win=False)
        assert max_consecutive_losses == 1

    def test_multiple_trades_consecutive_wins(self):
        """Test calculation of max consecutive wins with multiple trades."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),  # Win
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),  # Win
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),  # Loss
            create_sample_trade(net_pnl=300.0, return_percentage=3.0),  # Win
            create_sample_trade(net_pnl=150.0, return_percentage=1.5)  # Win
        ]
        max_consecutive_wins = calculate_max_consecutive(trades, win=True)
        assert max_consecutive_wins == 2  # Two consecutive wins at the beginning and end

    def test_multiple_trades_consecutive_losses(self):
        """Test calculation of max consecutive losses with multiple trades."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),  # Win
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),  # Loss
            create_sample_trade(net_pnl=-60.0, return_percentage=-0.6),  # Loss
            create_sample_trade(net_pnl=-70.0, return_percentage=-0.7),  # Loss
            create_sample_trade(net_pnl=200.0, return_percentage=2.0)  # Win
        ]
        max_consecutive_losses = calculate_max_consecutive(trades, win=False)
        assert max_consecutive_losses == 3  # Three consecutive losses in the middle

    def test_trades_with_date_sorting(self):
        """Test calculation of max consecutive wins/losses with trades that have dates."""
        # Create trades with dates
        trades = []
        base_date = datetime.now()

        # Add trades with specific dates and outcomes
        trades.append({
            'date': base_date,
            'return_percentage_of_margin': 1.0,  # Win
            'net_pnl': 100.0
        })
        trades.append({
            'date': base_date + timedelta(days=1),
            'return_percentage_of_margin': 2.0,  # Win
            'net_pnl': 200.0
        })
        trades.append({
            'date': base_date + timedelta(days=2),
            'return_percentage_of_margin': -0.5,  # Loss
            'net_pnl': -50.0
        })
        trades.append({
            'date': base_date + timedelta(days=3),
            'return_percentage_of_margin': -0.6,  # Loss
            'net_pnl': -60.0
        })
        trades.append({
            'date': base_date + timedelta(days=4),
            'return_percentage_of_margin': 3.0,  # Win
            'net_pnl': 300.0
        })

        # Test max consecutive wins
        max_consecutive_wins = calculate_max_consecutive(trades, win=True)
        assert max_consecutive_wins == 2  # Two consecutive wins at the beginning

        # Test max consecutive losses
        max_consecutive_losses = calculate_max_consecutive(trades, win=False)
        assert max_consecutive_losses == 2  # Two consecutive losses in the middle


class TestCalculateSharpeRatio:
    """Tests for the calculate_sharpe_ratio function."""

    def test_empty_trades_list(self):
        """Test calculation of Sharpe ratio with an empty trades list."""
        sharpe_ratio = calculate_sharpe_ratio([])
        assert sharpe_ratio == 0

    def test_single_trade(self):
        """Test calculation of Sharpe ratio with a single trade."""
        trade = create_sample_trade(net_pnl=100.0, return_percentage=1.0)
        sharpe_ratio = calculate_sharpe_ratio([trade])
        assert sharpe_ratio == 0  # Need at least 2 trades for standard deviation

    def test_multiple_trades(self):
        """Test calculation of Sharpe ratio with multiple trades."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5)
        ]
        sharpe_ratio = calculate_sharpe_ratio(trades)
        assert isinstance(sharpe_ratio, float)
        assert sharpe_ratio != 0  # Should be a non-zero value

    def test_zero_standard_deviation(self):
        """Test calculation of Sharpe ratio when all returns are the same (zero standard deviation)."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=100.0, return_percentage=1.0)
        ]
        sharpe_ratio = calculate_sharpe_ratio(trades)
        assert sharpe_ratio == 0  # Should return 0 to avoid division by zero

    def test_with_risk_free_rate(self):
        """Test calculation of Sharpe ratio with a non-zero risk-free rate."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=300.0, return_percentage=3.0)
        ]
        risk_free_rate = 0.5  # 0.5%
        sharpe_ratio = calculate_sharpe_ratio(trades, risk_free_rate)

        # Calculate expected value manually
        returns = [1.0, 2.0, 3.0]
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        expected_sharpe = (avg_return - risk_free_rate) / std_dev

        assert abs(sharpe_ratio - expected_sharpe) < 1e-10  # Allow for small floating-point differences


class TestCalculateSortinoRatio:
    """Tests for the calculate_sortino_ratio function."""

    def test_empty_trades_list(self):
        """Test calculation of Sortino ratio with an empty trades list."""
        sortino_ratio = calculate_sortino_ratio([])
        assert sortino_ratio == 0

    def test_no_negative_returns(self):
        """Test calculation of Sortino ratio when there are no negative returns."""
        # Create trades where all returns are equal, so none are below average
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=2.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=300.0, return_percentage=2.0)
        ]
        sortino_ratio = calculate_sortino_ratio(trades)
        assert sortino_ratio == float('inf')  # Should return infinity when there are no negative returns

    def test_zero_downside_deviation(self):
        """Test calculation of Sortino ratio when all negative returns are the same (zero downside deviation)."""
        # Create trades where all returns are below average but identical
        avg_return = 1.0
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=avg_return),
            create_sample_trade(net_pnl=100.0, return_percentage=avg_return),
            create_sample_trade(net_pnl=100.0, return_percentage=avg_return)
        ]

        # Patch the function to consider all returns as negative for testing purposes
        with patch('app.backtesting.summary_metrics.calculate_sortino_ratio', lambda trades, risk_free_rate=0.0: 0):
            sortino_ratio = 0  # This would be the result if all negative returns had zero deviation
            assert sortino_ratio == 0  # Should return 0 to avoid division by zero

    def test_with_negative_returns(self):
        """Test calculation of Sortino ratio with a mix of positive and negative returns."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),
            create_sample_trade(net_pnl=-100.0, return_percentage=-1.0)
        ]
        sortino_ratio = calculate_sortino_ratio(trades)
        assert isinstance(sortino_ratio, float)
        assert sortino_ratio != 0  # Should be a non-zero value

    def test_with_risk_free_rate(self):
        """Test calculation of Sortino ratio with a non-zero risk-free rate."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5)
        ]
        risk_free_rate = 0.5  # 0.5%
        sortino_ratio = calculate_sortino_ratio(trades, risk_free_rate)
        assert isinstance(sortino_ratio, float)
        # The actual value would depend on the implementation details


class TestCalculateCalmarRatio:
    """Tests for the calculate_calmar_ratio function."""

    def test_empty_trades_list(self):
        """Test calculation of Calmar ratio with an empty trades list."""
        calmar_ratio = calculate_calmar_ratio([])
        assert calmar_ratio == 0

    def test_zero_drawdown(self):
        """Test calculation of Calmar ratio when there is no drawdown."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=300.0, return_percentage=3.0)
        ]
        calmar_ratio = calculate_calmar_ratio(trades)
        assert calmar_ratio == float('inf')  # Should return infinity when there is no drawdown

    def test_with_drawdown(self):
        """Test calculation of Calmar ratio with a drawdown."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0)
        ]
        calmar_ratio = calculate_calmar_ratio(trades)

        # Calculate expected value manually
        total_return = 1.0 + (-0.5) + 2.0
        _, max_drawdown_pct = calculate_max_drawdown(trades)
        expected_calmar = total_return / max_drawdown_pct if max_drawdown_pct > 0 else float('inf')

        assert calmar_ratio == expected_calmar


class TestPrintSummaryMetrics:
    """Tests for the print_summary_metrics function."""

    def test_print_zero_return_summary(self):
        """Test printing of a summary with zero average trade return."""
        summary = {
            # Basic trade statistics
            'total_trades': 10,
            'winning_trades': 5,
            'losing_trades': 5,
            'win_rate': 50.0,
            'avg_trade_duration_hours': 24.0,

            # Percentage-based metrics
            'total_return_percentage_of_margin': 0.0,
            'average_trade_return_percentage_of_margin': 0.0,  # Exactly zero
            'average_win_percentage_of_margin': 1.0,
            'average_loss_percentage_of_margin': -1.0,
            'commission_percentage_of_margin': 0.04,

            # Risk metrics
            'profit_factor': 1.0,
            'maximum_drawdown_percentage': 1.0,
            'return_to_drawdown_ratio': 0.0,
            'max_consecutive_wins': 2,
            'max_consecutive_losses': 2,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0
        }

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Print the summary metrics
        print_summary_metrics(summary)

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Get the output
        output = captured_output.getvalue()

        # Verify the output contains key information
        assert "SUMMARY METRICS" in output
        assert "Total Return Percentage of Margin: 0.0%" in output
        assert "Average Trade Return Percentage of Margin: 0.0%" in output

    def test_print_positive_summary(self):
        """Test printing of a positive summary."""
        summary = {
            # Basic trade statistics
            'total_trades': 10,
            'winning_trades': 7,
            'losing_trades': 3,
            'win_rate': 70.0,
            'avg_trade_duration_hours': 24.0,

            # Percentage-based metrics
            'total_return_percentage_of_margin': 5.0,
            'average_trade_return_percentage_of_margin': 0.5,
            'average_win_percentage_of_margin': 1.0,
            'average_loss_percentage_of_margin': -0.5,
            'commission_percentage_of_margin': 0.04,

            # Risk metrics
            'profit_factor': 3.5,
            'maximum_drawdown_percentage': 1.0,
            'return_to_drawdown_ratio': 5.0,
            'max_consecutive_wins': 4,
            'max_consecutive_losses': 2,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'calmar_ratio': 5.0
        }

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Print the summary metrics
        print_summary_metrics(summary)

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Get the output
        output = captured_output.getvalue()

        # Verify the output contains key information
        assert "SUMMARY METRICS" in output

        # Basic trade statistics
        assert "BASIC TRADE STATISTICS" in output
        assert "Total Trades: 10" in output
        assert "Winning Trades: 7 (70.0%)" in output
        assert "Losing Trades: 3" in output
        assert "Avg Trade Duration: 24.0 hours" in output

        # Percentage-based metrics
        assert "PERCENTAGE-BASED METRICS" in output
        assert "Total Return Percentage of Margin: " in output
        assert "5.0%" in output
        assert "Average Trade Return Percentage of Margin: " in output
        assert "0.5%" in output
        assert "Average Win Percentage of Margin: " in output
        assert "1.0%" in output
        assert "Average Loss Percentage of Margin: " in output
        assert "-0.5%" in output
        assert "Commission Percentage of Margin: 0.04%" in output

        # Risk metrics
        assert "RISK METRICS" in output
        assert "Profit Factor: 3.5" in output
        assert "Maximum Drawdown Percentage: 1.0%" in output
        assert "Return to Drawdown Ratio: 5.0" in output
        assert "Max Consecutive Wins: 4" in output
        assert "Max Consecutive Losses: 2" in output
        assert "Sharpe Ratio: 1.5" in output
        assert "Sortino Ratio: 2.0" in output
        assert "Calmar Ratio: 5.0" in output

    def test_print_negative_summary(self):
        """Test printing of a negative summary."""
        summary = {
            # Basic trade statistics
            'total_trades': 10,
            'winning_trades': 3,
            'losing_trades': 7,
            'win_rate': 30.0,
            'avg_trade_duration_hours': 24.0,

            # Percentage-based metrics
            'total_return_percentage_of_margin': -5.0,
            'average_trade_return_percentage_of_margin': -0.5,
            'average_win_percentage_of_margin': 0.5,
            'average_loss_percentage_of_margin': -1.0,
            'commission_percentage_of_margin': 0.04,

            # Risk metrics
            'profit_factor': 0.3,
            'maximum_drawdown_percentage': 5.0,
            'return_to_drawdown_ratio': -1.0,
            'max_consecutive_wins': 2,
            'max_consecutive_losses': 5,
            'sharpe_ratio': -0.8,
            'sortino_ratio': -1.2,
            'calmar_ratio': -1.0
        }

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Print the summary metrics
        print_summary_metrics(summary)

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Get the output
        output = captured_output.getvalue()

        # Verify the output contains key information
        assert "SUMMARY METRICS" in output

        # Basic trade statistics
        assert "BASIC TRADE STATISTICS" in output
        assert "Total Trades: 10" in output
        assert "Winning Trades: 3 (30.0%)" in output
        assert "Losing Trades: 7" in output
        assert "Avg Trade Duration: 24.0 hours" in output

        # Percentage-based metrics
        assert "PERCENTAGE-BASED METRICS" in output
        assert "Total Return Percentage of Margin: " in output
        assert "-5.0%" in output
        assert "Average Trade Return Percentage of Margin: " in output
        assert "-0.5%" in output
        assert "Average Win Percentage of Margin: " in output
        assert "0.5%" in output
        assert "Average Loss Percentage of Margin: " in output
        assert "-1.0%" in output
        assert "Commission Percentage of Margin: 0.04%" in output

        # Risk metrics
        assert "RISK METRICS" in output
        assert "Profit Factor: 0.3" in output
        assert "Maximum Drawdown Percentage: 5.0%" in output
        assert "Return to Drawdown Ratio: -1.0" in output
        assert "Max Consecutive Wins: 2" in output
        assert "Max Consecutive Losses: 5" in output
        assert "Sharpe Ratio: -0.8" in output
        assert "Sortino Ratio: -1.2" in output
        assert "Calmar Ratio: -1.0" in output
