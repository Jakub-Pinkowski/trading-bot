import io
import sys
from datetime import datetime, timedelta

from app.backtesting.summary_metrics import calculate_max_drawdown, calculate_summary_metrics, print_summary_metrics


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
        summary = calculate_summary_metrics([])
        assert summary == {}

    def test_single_winning_trade(self):
        """Test calculation of summary metrics with a single winning trade."""
        trade = create_sample_trade(net_pnl=100.0, return_percentage=1.0, margin_requirement=10000.0, commission=4.0)
        summary = calculate_summary_metrics([trade])

        assert summary['total_trades'] == 1
        assert summary['winning_trades'] == 1
        assert summary['losing_trades'] == 0
        assert summary['win_rate'] == 100.0
        assert summary['total_net_pnl'] == 100.0
        assert summary['avg_trade_net_pnl'] == 100.0
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
        assert summary['total_net_pnl'] == -100.0
        assert summary['avg_trade_net_pnl'] == -100.0
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
        assert summary['total_net_pnl'] == 250.0
        assert summary['avg_trade_net_pnl'] == round(250.0 / 3, 2)
        assert summary['total_return_percentage_of_margin'] == 2.5
        assert summary['avg_win_net'] == 150.0  # (100 + 200) / 2
        assert summary['avg_loss_net'] == -50.0
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
        assert summary['total_net_pnl'] == 600.0
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
        assert summary['total_net_pnl'] == -600.0
        assert summary['profit_factor'] == 0.0  # No winning trades

    def test_commission_metrics(self):
        """Test calculation of commission metrics."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0, commission=5.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0, commission=5.0),
            create_sample_trade(net_pnl=300.0, return_percentage=3.0, commission=5.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['total_commission_paid'] == 15.0
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

        # Max drawdown should be calculated correctly
        assert summary['max_drawdown'] > 0
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

    def test_consecutive_wins_losses(self):
        """Test calculation of maximum consecutive wins and losses."""
        # Test consecutive wins
        win_trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=150.0, return_percentage=1.5),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0),
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),
            create_sample_trade(net_pnl=120.0, return_percentage=1.2),
            create_sample_trade(net_pnl=180.0, return_percentage=1.8)
        ]
        summary = calculate_summary_metrics(win_trades)
        assert summary['max_consecutive_wins'] == 3
        assert summary['max_consecutive_losses'] == 1

        # Test consecutive losses
        loss_trades = [
            create_sample_trade(net_pnl=-50.0, return_percentage=-0.5),
            create_sample_trade(net_pnl=-70.0, return_percentage=-0.7),
            create_sample_trade(net_pnl=-90.0, return_percentage=-0.9),
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=-60.0, return_percentage=-0.6),
            create_sample_trade(net_pnl=-80.0, return_percentage=-0.8)
        ]
        summary = calculate_summary_metrics(loss_trades)
        assert summary['max_consecutive_wins'] == 1
        assert summary['max_consecutive_losses'] == 3

    def test_zero_drawdown(self):
        """Test calculation of summary metrics with zero drawdown."""
        trades = [
            create_sample_trade(net_pnl=100.0, return_percentage=1.0),
            create_sample_trade(net_pnl=200.0, return_percentage=2.0)
        ]
        summary = calculate_summary_metrics(trades)

        assert summary['max_drawdown'] == 0
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

        assert summary['total_margin_used'] == 60000.0
        assert summary['avg_margin_requirement'] == 20000.0

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

        # PnL metrics
        assert summary['total_net_pnl'] == 550.0
        assert summary['avg_trade_net_pnl'] == 55.0
        assert summary['avg_win_net'] == round((120 + 85 + 210 + 150 + 95 + 180) / 6, 2)
        assert summary['avg_loss_net'] == round((-45 - 75 - 110 - 60) / 4, 2)

        # Duration metrics
        assert summary['avg_trade_duration_hours'] == 6.8  # (4+6+12+3+5+8+10+7+4+9)/10

        # Risk metrics
        assert summary['profit_factor'] > 1.0  # Profitable strategy
        assert summary['max_drawdown'] > 0
        assert summary['maximum_drawdown_percentage'] > 0
        assert summary['return_to_drawdown_ratio'] > 0

        # Performance ratios
        assert 'sharpe_ratio' in summary
        assert 'sortino_ratio' in summary
        assert 'calmar_ratio' in summary

        # Consecutive wins/losses
        assert 'max_consecutive_wins' in summary
        assert 'max_consecutive_losses' in summary
        assert summary['max_consecutive_wins'] >= 1
        assert summary['max_consecutive_losses'] >= 1

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

        # PnL metrics
        total_expected_pnl = 200 + 150 - 80 - 120 - 90 - 150 - 100 + 120 + 180 + 250 + 300
        assert summary['total_net_pnl'] == total_expected_pnl
        assert summary['avg_trade_net_pnl'] == round(total_expected_pnl / 11, 2)

        # Risk metrics
        # Significant drawdown expected
        assert summary['max_drawdown'] >= 540.0  # From peak of 350 to lowest point of -190
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

        # PnL metrics
        total_expected_pnl = 50 - 30 + 40 + 100 - 70 + 200 - 150 + 500 - 300
        assert summary['total_net_pnl'] == total_expected_pnl
        assert summary['avg_trade_net_pnl'] == round(total_expected_pnl / 9, 2)

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

        # PnL metrics
        total_expected_pnl = 100 - 50 + 5000 + 200 - 150 - 3000 + 300 - 100
        assert summary['total_net_pnl'] == total_expected_pnl
        assert summary['avg_trade_net_pnl'] == round(total_expected_pnl / 8, 2)

        # Risk metrics
        # Extreme values should be reflected in the metrics
        assert summary['avg_win_net'] > 1000  # Skewed by the 5000 profit
        assert summary['avg_loss_net'] < -500  # Skewed by the -3000 loss
        assert summary['max_drawdown'] >= 3000  # The extreme loss should create a significant drawdown
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

        # PnL metrics
        total_expected_pnl = 150 - 120 + 200 - 180 + 250 - 220 + 300 - 270 + 350 - 320
        assert summary['total_net_pnl'] == total_expected_pnl
        assert summary['avg_trade_net_pnl'] == round(total_expected_pnl / 10, 2)

        # Duration metrics
        total_duration = 2 + 1 + 3 + 2 + 4 + 3 + 5 + 4 + 6 + 5
        assert summary['avg_trade_duration_hours'] == round(total_duration / 10, 2)

        # Risk metrics
        # Volatility should be reflected in the drawdown metrics
        assert summary['max_drawdown'] > 0
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

        # PnL metrics
        total_expected_pnl = (200 + 150 + 250 - 50) + (-120 - 180 + 100 - 150) + (130 - 90 + 160 - 70)
        assert summary['total_net_pnl'] == total_expected_pnl
        assert summary['avg_trade_net_pnl'] == round(total_expected_pnl / 12, 2)

        # Risk metrics
        # The seasonal pattern should create a significant drawdown during the correction period
        assert summary['max_drawdown'] > 0
        assert summary['maximum_drawdown_percentage'] > 0
        # Overall, the strategy should still be profitable
        assert summary['total_return_percentage_of_margin'] > 0
        assert summary['profit_factor'] > 1.0


class TestPrintSummaryMetrics:
    """Tests for the print_summary_metrics function."""

    def test_print_positive_summary(self):
        """Test printing of a positive summary."""
        summary = {
            'total_trades': 10,
            'winning_trades': 7,
            'losing_trades': 3,
            'win_rate': 70.0,
            'avg_trade_duration_hours': 24.0,
            'max_consecutive_wins': 4,
            'max_consecutive_losses': 2,
            'total_margin_used': 100000.0,
            'avg_margin_requirement': 10000.0,
            'total_net_pnl': 5000.0,
            'avg_trade_net_pnl': 500.0,
            'avg_win_net': 1000.0,
            'avg_loss_net': -500.0,
            'total_return_percentage_of_margin': 5.0,
            'average_trade_return_percentage_of_margin': 0.5,
            'average_win_percentage_of_margin': 1.0,
            'average_loss_percentage_of_margin': -0.5,
            'total_commission_paid': 40.0,
            'commission_percentage_of_margin': 0.04,
            'profit_factor': 3.5,
            'max_drawdown': 1000.0,
            'maximum_drawdown_percentage': 1.0,
            'return_to_drawdown_ratio': 5.0,
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
        assert "Total Trades: 10" in output
        assert "Winning Trades: 7 (70.0%)" in output
        assert "Losing Trades: 3" in output
        assert "Max Consecutive Wins: 4" in output
        assert "Max Consecutive Losses: 2" in output
        assert "Total Net PnL: $5,000.00" in output
        assert "Profit Factor: 3.5" in output
        assert "Max Drawdown: $1,000.00" in output
        assert "Maximum Drawdown Percentage: 1.0%" in output
        assert "Return to Drawdown Ratio: 5.0" in output
        assert "Sharpe Ratio: 1.5" in output
        assert "Sortino Ratio: 2.0" in output
        assert "Calmar Ratio: 5.0" in output

    def test_print_negative_summary(self):
        """Test printing of a negative summary."""
        summary = {
            'total_trades': 10,
            'winning_trades': 3,
            'losing_trades': 7,
            'win_rate': 30.0,
            'avg_trade_duration_hours': 24.0,
            'max_consecutive_wins': 2,
            'max_consecutive_losses': 5,
            'total_margin_used': 100000.0,
            'avg_margin_requirement': 10000.0,
            'total_net_pnl': -5000.0,
            'avg_trade_net_pnl': -500.0,
            'avg_win_net': 500.0,
            'avg_loss_net': -1000.0,
            'total_return_percentage_of_margin': -5.0,
            'average_trade_return_percentage_of_margin': -0.5,
            'average_win_percentage_of_margin': 0.5,
            'average_loss_percentage_of_margin': -1.0,
            'total_commission_paid': 40.0,
            'commission_percentage_of_margin': 0.04,
            'profit_factor': 0.3,
            'max_drawdown': 5000.0,
            'maximum_drawdown_percentage': 5.0,
            'return_to_drawdown_ratio': -1.0,
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
        assert "Total Trades: 10" in output
        assert "Winning Trades: 3 (30.0%)" in output
        assert "Losing Trades: 7" in output
        assert "Max Consecutive Wins: 2" in output
        assert "Max Consecutive Losses: 5" in output
        assert "Total Net PnL: $-5,000.00" in output
        assert "Profit Factor: 0.3" in output
        assert "Max Drawdown: $5,000.00" in output
        assert "Maximum Drawdown Percentage: 5.0%" in output
        assert "Return to Drawdown Ratio: -1.0" in output
        assert "Sharpe Ratio: -0.8" in output
        assert "Sortino Ratio: -1.2" in output
        assert "Calmar Ratio: -1.0" in output
