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
            'commission_percentage': 0.2,  # This field is expected by print_trade_metrics
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
        assert "Commission % of Return:" in output
        assert "0.2%" in output
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
            'commission_percentage': 0.2,  # This field is expected by print_trade_metrics
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
