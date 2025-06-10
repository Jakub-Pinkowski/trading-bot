import io
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app.backtesting.per_trade_metrics import calculate_trade_metrics, print_trade_metrics, COMMISSION_PER_TRADE


# Helper function to create a sample trade
def create_sample_trade(side='long', entry_price=100.0, exit_price=110.0, hours_duration=24):
    entry_time = datetime.now()
    exit_time = entry_time + timedelta(hours=hours_duration)

    return {
        'entry_time': entry_time,
        'exit_time': exit_time,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'side': side
    }


class TestCalculateTradeMetrics:
    """Tests for the calculate_trade_metrics function."""

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 50})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'ES': 16889.88})
    def test_long_trade_calculation(self):
        """Test calculation of metrics for a long trade."""
        # Create a sample-long trade
        trade = create_sample_trade(side='long', entry_price=4200.0, exit_price=4210.0)

        # Calculate metrics
        metrics = calculate_trade_metrics(trade, 'ES')

        # Verify the calculated metrics
        assert metrics['side'] == 'long'
        assert metrics['entry_price'] == 4200.0
        assert metrics['exit_price'] == 4210.0
        assert metrics['pnl_points'] == 10.0
        assert metrics['gross_pnl'] == 500.0  # 10 points * 50 multiplier
        assert metrics['commission'] == COMMISSION_PER_TRADE
        assert metrics['net_pnl'] == 500.0 - COMMISSION_PER_TRADE
        assert metrics['margin_requirement'] == 16889.88
        assert metrics['return_percentage_of_margin'] == round(((500.0 - COMMISSION_PER_TRADE) / 16889.88) * 100, 2)
        assert metrics['return_percentage_of_contract'] == round(((500.0 - COMMISSION_PER_TRADE) / (4200.0 * 50)) * 100,
                                                                 2)
        assert metrics['duration_hours'] == 24

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'CL': 1000})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'CL': 16250})
    def test_short_trade_calculation(self):
        """Test calculation of metrics for a short trade."""
        # Create a sample short trade
        trade = create_sample_trade(side='short', entry_price=75.0, exit_price=73.0)

        # Calculate metrics
        metrics = calculate_trade_metrics(trade, 'CL')

        # Verify the calculated metrics
        assert metrics['side'] == 'short'
        assert metrics['entry_price'] == 75.0
        assert metrics['exit_price'] == 73.0
        assert metrics['pnl_points'] == 2.0  # For short: entry_price - exit_price
        assert metrics['gross_pnl'] == 2000.0  # 2 points * 1000 multiplier
        assert metrics['commission'] == COMMISSION_PER_TRADE
        assert metrics['net_pnl'] == 2000.0 - COMMISSION_PER_TRADE
        assert metrics['margin_requirement'] == 16250
        assert metrics['return_percentage_of_margin'] == round(((2000.0 - COMMISSION_PER_TRADE) / 16250) * 100, 2)
        assert metrics['return_percentage_of_contract'] == round(((2000.0 - COMMISSION_PER_TRADE) / (
                75.0 * 1000)) * 100, 2)

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'GC': 100})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'GC': 25338.86})
    def test_breakeven_trade(self):
        """Test calculation of metrics for a breakeven trade."""
        # Create a sample breakeven trade (commission will make it slightly negative)
        trade = create_sample_trade(side='long', entry_price=1800.0, exit_price=1800.0)

        # Calculate metrics
        metrics = calculate_trade_metrics(trade, 'GC')

        # Verify the calculated metrics
        assert metrics['pnl_points'] == 0.0
        assert metrics['gross_pnl'] == 0.0
        assert metrics['net_pnl'] == -COMMISSION_PER_TRADE
        assert metrics['return_percentage_of_margin'] < 0  # Should be negative due to commission

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'NQ': 20})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'NQ': 24458.12})
    def test_losing_trade(self):
        """Test calculation of metrics for a losing trade."""
        # Create a sample losing trade
        trade = create_sample_trade(side='long', entry_price=15000.0, exit_price=14950.0)

        # Calculate metrics
        metrics = calculate_trade_metrics(trade, 'NQ')

        # Verify the calculated metrics
        assert metrics['pnl_points'] == -50.0
        assert metrics['gross_pnl'] == -1000.0  # -50 points * 20 multiplier
        assert metrics['net_pnl'] == -1000.0 - COMMISSION_PER_TRADE
        assert metrics['return_percentage_of_margin'] < 0

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'ES': 16889.88})
    def test_missing_contract_multiplier(self):
        """Test error handling when contract multiplier is missing."""
        trade = create_sample_trade()

        # Verify that ValueError is raised
        with pytest.raises(ValueError, match="No contract multiplier found for symbol: ES"):
            calculate_trade_metrics(trade, 'ES')

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 50})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {})
    def test_missing_margin_requirement(self):
        """Test error handling when margin requirement is missing."""
        trade = create_sample_trade()

        # Verify that ValueError is raised
        with pytest.raises(ValueError, match="No margin requirement found for symbol: ES"):
            calculate_trade_metrics(trade, 'ES')

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 0})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'ES': 16889.88})
    def test_zero_contract_multiplier(self):
        """Test error handling when contract multiplier is zero."""
        trade = create_sample_trade()

        # Verify that ValueError is raised
        with pytest.raises(ValueError, match="No contract multiplier found for symbol: ES"):
            calculate_trade_metrics(trade, 'ES')

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 50})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'ES': 0})
    def test_zero_margin_requirement(self):
        """Test error handling when margin requirement is zero."""
        trade = create_sample_trade()

        # Verify that ValueError is raised
        with pytest.raises(ValueError, match="No margin requirement found for symbol: ES"):
            calculate_trade_metrics(trade, 'ES')

    def test_invalid_trade_side(self):
        """Test error handling when trade side is invalid."""
        # Create a trade with an invalid side
        trade = create_sample_trade(side='invalid')

        # Mock the contract multiplier and margin requirement
        with patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 50}):
            with patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'ES': 16889.88}):
                # Verify that ValueError is raised
                with pytest.raises(ValueError, match="Unknown trade side: invalid"):
                    calculate_trade_metrics(trade, 'ES')

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 50})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'ES': 16889.88})
    def test_trade_duration_calculation(self):
        """Test calculation of trade duration."""
        # Create trades with different durations
        short_duration_trade = create_sample_trade(hours_duration=2)
        long_duration_trade = create_sample_trade(hours_duration=48)

        # Calculate metrics
        short_metrics = calculate_trade_metrics(short_duration_trade, 'ES')
        long_metrics = calculate_trade_metrics(long_duration_trade, 'ES')

        # Verify the calculated durations
        assert short_metrics['duration_hours'] == 2
        assert long_metrics['duration_hours'] == 48

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 50})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'ES': 16889.88})
    def test_trade_copy_not_modified(self):
        """Test that the original trade dictionary is not modified."""
        # Create a sample trade
        original_trade = create_sample_trade()
        original_trade_copy = original_trade.copy()

        # Calculate metrics
        calculate_trade_metrics(original_trade, 'ES')

        # Verify that the original trade was not modified
        assert original_trade == original_trade_copy

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 50})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'ES': 16889.88})
    def test_extreme_price_values(self):
        """Test calculation of metrics with extreme price values."""
        # Test with very large prices
        large_price_trade = create_sample_trade(
            side='long',
            entry_price=100000.0,
            exit_price=100100.0
        )
        large_price_metrics = calculate_trade_metrics(large_price_trade, 'ES')

        # Verify calculations with large prices
        assert large_price_metrics['pnl_points'] == 100.0
        assert large_price_metrics['gross_pnl'] == 5000.0  # 100 points * 50 multiplier
        assert large_price_metrics['net_pnl'] == 5000.0 - COMMISSION_PER_TRADE
        assert large_price_metrics['return_percentage_of_margin'] == round(((
                                                                                    5000.0 - COMMISSION_PER_TRADE) / 16889.88) * 100,
                                                                           2)
        assert large_price_metrics['return_percentage_of_contract'] == round(((5000.0 - COMMISSION_PER_TRADE) / (
                100000.0 * 50)) * 100, 2)

        # Test with very small prices
        small_price_trade = create_sample_trade(
            side='short',
            entry_price=0.01,
            exit_price=0.005
        )
        small_price_metrics = calculate_trade_metrics(small_price_trade, 'ES')

        # Verify calculations with small prices
        assert small_price_metrics['pnl_points'] == 0.01  # Rounded to 2 decimal places
        assert small_price_metrics['gross_pnl'] == 0.25  # 0.005 points * 50 multiplier
        assert small_price_metrics['net_pnl'] == 0.25 - COMMISSION_PER_TRADE
        assert small_price_metrics['return_percentage_of_margin'] == round(((
                                                                                    0.25 - COMMISSION_PER_TRADE) / 16889.88) * 100,
                                                                           2)
        assert small_price_metrics['return_percentage_of_contract'] == round(((0.25 - COMMISSION_PER_TRADE) / (
                0.01 * 50)) * 100, 2)

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 50})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'ES': 16889.88})
    def test_extreme_duration_trades(self):
        """Test calculation of metrics with extreme duration values."""
        # Test with a very short duration (seconds)
        very_short_trade = create_sample_trade(hours_duration=0.01)  # 36 seconds
        very_short_metrics = calculate_trade_metrics(very_short_trade, 'ES')

        # Verify the calculated duration
        assert very_short_metrics['duration_hours'] == 0

        # Test with a very long duration (months)
        very_long_trade = create_sample_trade(hours_duration=24 * 30 * 3)  # ~3 months
        very_long_metrics = calculate_trade_metrics(very_long_trade, 'ES')

        # Verify the calculated duration
        assert very_long_metrics['duration_hours'] == 24 * 30 * 3

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS',
           {'ES': 50, 'NQ': 20, 'CL': 1000, 'GC': 100, 'ZB': 1000})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS',
           {'ES': 16889.88, 'NQ': 24458.12, 'CL': 16250, 'GC': 25338.86, 'ZB': 5060.0})
    def test_multiple_symbols(self):
        """Test calculation of metrics for different symbols with different multipliers and margin requirements."""
        # Create sample trades for different symbols
        es_trade = create_sample_trade(side='long', entry_price=4200.0, exit_price=4210.0)
        nq_trade = create_sample_trade(side='short', entry_price=15000.0, exit_price=14950.0)
        cl_trade = create_sample_trade(side='long', entry_price=75.0, exit_price=77.0)
        gc_trade = create_sample_trade(side='short', entry_price=1800.0, exit_price=1790.0)
        zb_trade = create_sample_trade(side='long', entry_price=110.0, exit_price=110.5)

        # Calculate metrics for each symbol
        es_metrics = calculate_trade_metrics(es_trade, 'ES')
        nq_metrics = calculate_trade_metrics(nq_trade, 'NQ')
        cl_metrics = calculate_trade_metrics(cl_trade, 'CL')
        gc_metrics = calculate_trade_metrics(gc_trade, 'GC')
        zb_metrics = calculate_trade_metrics(zb_trade, 'ZB')

        # Verify the calculated metrics for each symbol
        # ES
        assert es_metrics['pnl_points'] == 10.0
        assert es_metrics['gross_pnl'] == 500.0  # 10 points * 50 multiplier
        assert es_metrics['margin_requirement'] == 16889.88

        # NQ
        assert nq_metrics['pnl_points'] == 50.0
        assert nq_metrics['gross_pnl'] == 1000.0  # 50 points * 20 multiplier
        assert nq_metrics['margin_requirement'] == 24458.12

        # CL
        assert cl_metrics['pnl_points'] == 2.0
        assert cl_metrics['gross_pnl'] == 2000.0  # 2 points * 1000 multiplier
        assert cl_metrics['margin_requirement'] == 16250

        # GC
        assert gc_metrics['pnl_points'] == 10.0
        assert gc_metrics['gross_pnl'] == 1000.0  # 10 points * 100 multiplier
        assert gc_metrics['margin_requirement'] == 25338.86

        # ZB
        assert zb_metrics['pnl_points'] == 0.5
        assert zb_metrics['gross_pnl'] == 500.0  # 0.5 points * 1000 multiplier
        assert zb_metrics['margin_requirement'] == 5060.0

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 50})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'ES': 16889.88})
    def test_multiple_trades_analysis(self):
        """Test analysis of multiple trades to calculate aggregate metrics."""
        # Create a series of trades
        trades = [
            create_sample_trade(side='long', entry_price=4200.0, exit_price=4210.0),  # +10 points
            create_sample_trade(side='short', entry_price=4220.0, exit_price=4200.0),  # +20 points
            create_sample_trade(side='long', entry_price=4190.0, exit_price=4180.0),  # -10 points
            create_sample_trade(side='short', entry_price=4170.0, exit_price=4190.0),  # -20 points
            create_sample_trade(side='long', entry_price=4200.0, exit_price=4215.0),  # +15 points
        ]

        # Calculate metrics for each trade
        trade_metrics = [calculate_trade_metrics(trade, 'ES') for trade in trades]

        # Calculate aggregate metrics
        total_gross_pnl = sum(metric['gross_pnl'] for metric in trade_metrics)
        total_net_pnl = sum(metric['net_pnl'] for metric in trade_metrics)
        total_commission = sum(metric['commission'] for metric in trade_metrics)
        win_count = sum(1 for metric in trade_metrics if metric['gross_pnl'] > 0)
        loss_count = sum(1 for metric in trade_metrics if metric['gross_pnl'] < 0)
        win_rate = win_count / len(trades) if len(trades) > 0 else 0

        # Verify aggregate metrics
        assert len(trade_metrics) == 5
        assert win_count == 3
        assert loss_count == 2
        assert win_rate == 0.6
        assert total_gross_pnl == 750.0  # (10 + 20 - 10 - 20 + 15) * 50
        assert total_commission == COMMISSION_PER_TRADE * 5
        assert total_net_pnl == total_gross_pnl - total_commission

        # Calculate average metrics
        avg_gross_pnl = total_gross_pnl / len(trades)
        avg_net_pnl = total_net_pnl / len(trades)
        avg_return_percentage = sum(metric['return_percentage_of_margin'] for metric in trade_metrics) / len(trades)

        # Verify average metrics
        assert avg_gross_pnl == 150.0
        assert avg_net_pnl == avg_gross_pnl - COMMISSION_PER_TRADE

        # Calculate profit factor
        gross_profit = sum(metric['gross_pnl'] for metric in trade_metrics if metric['gross_pnl'] > 0)
        gross_loss = abs(sum(metric['gross_pnl'] for metric in trade_metrics if metric['gross_pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Verify profit factor
        assert gross_profit == 2250.0  # (10 + 20 + 15) * 50
        assert gross_loss == 1500.0  # (10 + 20) * 50
        assert profit_factor == 1.5

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 50})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS', {'ES': 16889.88})
    def test_specific_trade_patterns(self):
        """Test calculation of metrics for specific trade patterns."""
        # Test a trade with a large gap (e.g., overnight gap)
        gap_trade = create_sample_trade(side='long', entry_price=4200.0, exit_price=4250.0)
        gap_metrics = calculate_trade_metrics(gap_trade, 'ES')

        assert gap_metrics['pnl_points'] == 50.0
        assert gap_metrics['gross_pnl'] == 2500.0  # 50 points * 50 multiplier

        # Test a trade with a very small profit (just covering commission)
        small_profit_trade = create_sample_trade(
            side='long',
            entry_price=4200.0,
            exit_price=4200.0 + (COMMISSION_PER_TRADE / 50)  # Just enough to cover commission
        )
        small_profit_metrics = calculate_trade_metrics(small_profit_trade, 'ES')

        assert small_profit_metrics['gross_pnl'] == COMMISSION_PER_TRADE
        assert small_profit_metrics['net_pnl'] == 0.0

        # Test a trade with a very large loss
        large_loss_trade = create_sample_trade(side='long', entry_price=4200.0, exit_price=4100.0)
        large_loss_metrics = calculate_trade_metrics(large_loss_trade, 'ES')

        assert large_loss_metrics['pnl_points'] == -100.0
        assert large_loss_metrics['gross_pnl'] == -5000.0  # -100 points * 50 multiplier

        # Test a trade with a very large profit
        large_profit_trade = create_sample_trade(side='short', entry_price=4300.0, exit_price=4100.0)
        large_profit_metrics = calculate_trade_metrics(large_profit_trade, 'ES')

        assert large_profit_metrics['pnl_points'] == 200.0
        assert large_profit_metrics['gross_pnl'] == 10000.0  # 200 points * 50 multiplier

    @patch('app.backtesting.per_trade_metrics.CONTRACT_MULTIPLIERS', {'ES': 50, 'NQ': 20, 'CL': 1000, 'GC': 100})
    @patch('app.backtesting.per_trade_metrics.MARGIN_REQUIREMENTS',
           {'ES': 16889.88, 'NQ': 24458.12, 'CL': 16250, 'GC': 25338.86})
    def test_real_life_trading_scenarios(self):
        """Test calculation of metrics for real-life trading scenarios."""

        # Scenario 1: Risk-Reward Ratio Trade (2:1)
        # A common trading strategy is to aim for a risk-reward ratio of at least 2:1
        # Here we simulate a trade where the potential profit is twice the potential loss
        risk_points = 10.0  # Willing to risk 10 points
        reward_points = 20.0  # Aiming for 20 points profit

        # Long trade with 2:1 risk-reward that hits profit target
        long_rr_trade_win = create_sample_trade(
            side='long',
            entry_price=4200.0,
            exit_price=4200.0 + reward_points  # Hit profit target
        )
        long_rr_metrics_win = calculate_trade_metrics(long_rr_trade_win, 'ES')

        assert long_rr_metrics_win['pnl_points'] == reward_points
        assert long_rr_metrics_win['gross_pnl'] == reward_points * 50  # 20 points * 50 multiplier = $1000

        # Long trade with 2:1 risk-reward that hits stop loss
        long_rr_trade_loss = create_sample_trade(
            side='long',
            entry_price=4200.0,
            exit_price=4200.0 - risk_points  # Hit stop loss
        )
        long_rr_metrics_loss = calculate_trade_metrics(long_rr_trade_loss, 'ES')

        assert long_rr_metrics_loss['pnl_points'] == -risk_points
        assert long_rr_metrics_loss['gross_pnl'] == -risk_points * 50  # -10 points * 50 multiplier = -$500

        # Scenario 2: Volatility Breakout Trade
        # In volatile markets, prices can move quickly in one direction after breaking a key level
        # Here we simulate a volatility breakout trade with a large price movement

        # Breakout trade with high volatility (large price movement in short time)
        breakout_trade = create_sample_trade(
            side='long',
            entry_price=4200.0,
            exit_price=4260.0,  # 60 point move
            hours_duration=2  # Short duration (2 hours)
        )
        breakout_metrics = calculate_trade_metrics(breakout_trade, 'ES')

        assert breakout_metrics['pnl_points'] == 60.0
        assert breakout_metrics['gross_pnl'] == 3000.0  # 60 points * 50 multiplier
        assert breakout_metrics['duration_hours'] == 2

        # Scenario 3: Trend Following Trade
        # Trend following involves entering in the direction of an established trend
        # Here we simulate a trend following trade with multiple entries

        # First entry in the trend
        trend_entry1 = create_sample_trade(
            side='short',
            entry_price=4300.0,
            exit_price=4250.0,  # 50 point move
            hours_duration=24
        )
        trend_metrics1 = calculate_trade_metrics(trend_entry1, 'ES')

        # Second entry in the same trend (price continued lower)
        trend_entry2 = create_sample_trade(
            side='short',
            entry_price=4250.0,
            exit_price=4200.0,  # 50 point move
            hours_duration=24
        )
        trend_metrics2 = calculate_trade_metrics(trend_entry2, 'ES')

        # Combined results of the trend following strategy
        total_trend_pnl = trend_metrics1['gross_pnl'] + trend_metrics2['gross_pnl']
        total_trend_commission = trend_metrics1['commission'] + trend_metrics2['commission']
        total_trend_net_pnl = trend_metrics1['net_pnl'] + trend_metrics2['net_pnl']

        assert trend_metrics1['pnl_points'] == 50.0
        assert trend_metrics2['pnl_points'] == 50.0
        assert total_trend_pnl == 5000.0  # (50 + 50) points * 50 multiplier
        assert total_trend_commission == COMMISSION_PER_TRADE * 2
        assert total_trend_net_pnl == total_trend_pnl - total_trend_commission

        # Scenario 4: Mean Reversion Trade
        # Mean reversion involves betting that prices will return to their average after moving away
        # Here we simulate a mean reversion trade after a price spike

        # Price spikes up and then reverts back to the mean
        mean_reversion_trade = create_sample_trade(
            side='short',
            entry_price=4250.0,  # Enter short after price spike
            exit_price=4200.0,  # Exit when price returns to average
            hours_duration=12
        )
        mean_reversion_metrics = calculate_trade_metrics(mean_reversion_trade, 'ES')

        assert mean_reversion_metrics['pnl_points'] == 50.0
        assert mean_reversion_metrics['gross_pnl'] == 2500.0  # 50 points * 50 multiplier

        # Scenario 5: Multi-Day Position with Weekend Gap
        # Holding positions over weekends can result in price gaps
        # Here we simulate a trade held over a weekend with a gap up

        # Friday entry, Monday exit with gap up
        weekend_gap_trade = create_sample_trade(
            side='long',
            entry_price=4200.0,
            exit_price=4240.0,  # Gap up on Monday
            hours_duration=72  # 3 days (Friday to Monday)
        )
        weekend_gap_metrics = calculate_trade_metrics(weekend_gap_trade, 'ES')

        assert weekend_gap_metrics['pnl_points'] == 40.0
        assert weekend_gap_metrics['gross_pnl'] == 2000.0  # 40 points * 50 multiplier
        assert weekend_gap_metrics['duration_hours'] == 72

        # Scenario 6: Trading Different Markets
        # Traders often trade multiple markets with different characteristics
        # Here we compare trades in different markets (ES, NQ, CL, GC)

        # ES trade (S&P 500 futures)
        es_trade = create_sample_trade(side='long', entry_price=4200.0, exit_price=4210.0)
        es_metrics = calculate_trade_metrics(es_trade, 'ES')

        # NQ trade (Nasdaq futures)
        nq_trade = create_sample_trade(side='long', entry_price=14500.0, exit_price=14550.0)
        nq_metrics = calculate_trade_metrics(nq_trade, 'NQ')

        # CL trade (Crude Oil futures)
        cl_trade = create_sample_trade(side='short', entry_price=80.0, exit_price=79.0)
        cl_metrics = calculate_trade_metrics(cl_trade, 'CL')

        # GC trade (Gold futures)
        gc_trade = create_sample_trade(side='long', entry_price=1900.0, exit_price=1910.0)
        gc_metrics = calculate_trade_metrics(gc_trade, 'GC')

        # Compare return percentages across different markets
        es_return = es_metrics['return_percentage_of_margin']
        nq_return = nq_metrics['return_percentage_of_margin']
        cl_return = cl_metrics['return_percentage_of_margin']
        gc_return = gc_metrics['return_percentage_of_margin']

        # Verify each market's metrics
        assert es_metrics['gross_pnl'] == 500.0  # 10 points * 50 multiplier
        assert nq_metrics['gross_pnl'] == 1000.0  # 50 points * 20 multiplier
        assert cl_metrics['gross_pnl'] == 1000.0  # 1 point * 1000 multiplier
        assert gc_metrics['gross_pnl'] == 1000.0  # 10 points * 100 multiplier

        # Scenario 7: Scaling In and Out of Positions
        # Traders often scale into and out of positions to manage risk and maximize profits
        # Here we simulate a trader scaling into a position and then scaling out

        # Initial position entry
        scale_entry1 = create_sample_trade(
            side='long',
            entry_price=4200.0,
            exit_price=4220.0,  # Partial exit at first target
            hours_duration=12
        )
        scale_metrics1 = calculate_trade_metrics(scale_entry1, 'ES')

        # Adding to the position (scaling in)
        scale_entry2 = create_sample_trade(
            side='long',
            entry_price=4210.0,  # Better average price
            exit_price=4240.0,  # Exit at second target
            hours_duration=24
        )
        scale_metrics2 = calculate_trade_metrics(scale_entry2, 'ES')

        # Calculate combined metrics for the scaling strategy
        # In real trading, position sizes might vary, but for simplicity we use equal sizes here
        avg_entry_price = (4200.0 + 4210.0) / 2
        avg_exit_price = (4220.0 + 4240.0) / 2
        total_scale_pnl = scale_metrics1['gross_pnl'] + scale_metrics2['gross_pnl']
        total_scale_commission = scale_metrics1['commission'] + scale_metrics2['commission']
        total_scale_net_pnl = scale_metrics1['net_pnl'] + scale_metrics2['net_pnl']

        # Verify the individual trades
        assert scale_metrics1['pnl_points'] == 20.0
        assert scale_metrics1['gross_pnl'] == 1000.0  # 20 points * 50 multiplier
        assert scale_metrics2['pnl_points'] == 30.0
        assert scale_metrics2['gross_pnl'] == 1500.0  # 30 points * 50 multiplier

        # Verify the combined results
        assert total_scale_pnl == 2500.0  # 1000 + 1500
        assert total_scale_commission == COMMISSION_PER_TRADE * 2
        assert total_scale_net_pnl == total_scale_pnl - total_scale_commission

        # Calculate what the result would be if it was a single trade with average prices
        avg_trade = create_sample_trade(
            side='long',
            entry_price=avg_entry_price,
            exit_price=avg_exit_price,
            hours_duration=24  # Using the longer duration
        )
        avg_metrics = calculate_trade_metrics(avg_trade, 'ES')

        # Verify the average trade metrics
        assert avg_metrics['pnl_points'] == 25.0  # (4230 - 4205)
        assert avg_metrics['gross_pnl'] == 1250.0  # 25 points * 50 multiplier

        # Compare scaling strategy vs single entry strategy
        # The scaling strategy should have higher gross PnL but also higher commission
        assert total_scale_pnl > avg_metrics['gross_pnl']  # 2500 > 1250
        assert total_scale_commission > avg_metrics['commission']  # 8 > 4


class TestPrintTradeMetrics:
    """Tests for the print_trade_metrics function."""

    def test_print_profitable_trade(self):
        """Test printing of a profitable trade."""
        # Create a sample profitable trade metrics
        trade_metrics = {
            'entry_time': datetime.now(),
            'exit_time': datetime.now() + timedelta(hours=24),
            'duration': timedelta(hours=24),
            'duration_hours': 24,
            'side': 'long',
            'entry_price': 4200.0,
            'exit_price': 4210.0,
            'return_percentage_of_margin': 2.5,
            'return_percentage_of_contract': 0.5,
            'margin_requirement': 16889.88,
            'commission': 4.0,
            'pnl_points': 10.0,
            'gross_pnl': 500.0,
            'net_pnl': 496.0
        }

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Print the trade metrics
        print_trade_metrics(trade_metrics)

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Get the output
        output = captured_output.getvalue()

        # Verify the output contains key information
        assert "TRADE METRICS" in output
        assert "Entry Time:" in output
        assert "Exit Time:" in output
        assert "Side: long" in output
        assert "Entry Price: 4200.0" in output
        assert "Exit Price: 4210.0" in output

        # Check for text without exact color codes
        assert "Net Return % of Margin:" in output
        assert "2.5%" in output
        assert "Return % of Contract:" in output
        assert "0.5%" in output
        assert "Margin Requirement: $16889.88" in output
        assert "Commission (dollars): $4.0" in output
        assert "PnL (points):" in output
        assert "10.0" in output
        assert "Gross PnL (dollars):" in output
        assert "$500.0" in output
        assert "Net PnL (dollars):" in output
        assert "$496.0" in output

    def test_print_losing_trade(self):
        """Test printing of a losing trade."""
        # Create a sample losing trade metrics
        trade_metrics = {
            'entry_time': datetime.now(),
            'exit_time': datetime.now() + timedelta(hours=24),
            'duration': timedelta(hours=24),
            'duration_hours': 24,
            'side': 'long',
            'entry_price': 4200.0,
            'exit_price': 4190.0,
            'return_percentage_of_margin': -0.5,
            'return_percentage_of_contract': -0.1,
            'margin_requirement': 16889.88,
            'commission': 4.0,
            'pnl_points': -10.0,
            'gross_pnl': -500.0,
            'net_pnl': -504.0
        }

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Print the trade metrics
        print_trade_metrics(trade_metrics)

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Get the output
        output = captured_output.getvalue()

        # Verify the output contains key information
        assert "TRADE METRICS" in output

        # Check for text without exact color codes
        assert "Net Return % of Margin:" in output
        assert "-0.5%" in output
        assert "PnL (points):" in output
        assert "-10.0" in output
        assert "Gross PnL (dollars):" in output
        assert "$-500.0" in output
        assert "Net PnL (dollars):" in output
        assert "$-504.0" in output
