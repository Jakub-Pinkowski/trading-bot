"""
Tests for Per-Trade Metrics Module.

Tests cover:
- Trade metric calculation for long and short positions
- Commission and margin requirement calculations
- Return percentage calculations (margin-based and contract-based)
- Trade duration calculations
- Symbol category mapping and margin ratios
- Edge cases (zero PnL, very short trades, very long trades)
- Real trade data from strategy backtests

All tests use realistic trade scenarios with actual futures symbols.
"""
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app.backtesting.metrics.per_trade_metrics import (
    calculate_trade_metrics,
    print_trade_metrics,
    _get_symbol_category,
    _estimate_margin,
    MARGIN_RATIOS,
    COMMISSION_PER_TRADE
)
from config import CONTRACT_MULTIPLIERS


# ==================== Fixtures ====================

@pytest.fixture
def sample_long_trade_zs():
    """Sample long trade in ZS (soybeans) for testing."""
    return {
        'entry_time': datetime(2024, 1, 15, 10, 0),
        'exit_time': datetime(2024, 1, 15, 14, 30),
        'entry_price': 1200.0,  # cents per bushel
        'exit_price': 1210.0,  # cents per bushel (profitable)
        'side': 'long'
    }


@pytest.fixture
def sample_short_trade_cl():
    """Sample short trade in CL (crude oil) for testing."""
    return {
        'entry_time': datetime(2024, 2, 20, 9, 30),
        'exit_time': datetime(2024, 2, 20, 15, 45),
        'entry_price': 75.50,  # dollars per barrel
        'exit_price': 74.80,  # dollars per barrel (profitable short)
        'side': 'short'
    }


@pytest.fixture
def sample_losing_trade_es():
    """Sample losing long trade in ES (E-mini S&P 500) for testing."""
    return {
        'entry_time': datetime(2024, 3, 10, 13, 0),
        'exit_time': datetime(2024, 3, 10, 16, 0),
        'entry_price': 5000.00,
        'exit_price': 4985.00,  # Loss of 15 points
        'side': 'long'
    }


@pytest.fixture
def breakeven_trade_gc():
    """Sample breakeven trade in GC (gold) for testing."""
    return {
        'entry_time': datetime(2024, 4, 5, 11, 0),
        'exit_time': datetime(2024, 4, 5, 12, 0),
        'entry_price': 2050.0,
        'exit_price': 2050.0,  # No price movement
        'side': 'long'
    }


# ==================== Test Classes ====================

class TestTradeMetricsCalculation:
    """Test basic trade metric calculations."""

    def test_long_trade_metrics_calculation(self, sample_long_trade_zs):
        """Test metrics calculation for profitable long trade."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        # Original trade data preserved
        assert metrics['entry_time'] == sample_long_trade_zs['entry_time']
        assert metrics['exit_time'] == sample_long_trade_zs['exit_time']
        assert metrics['entry_price'] == sample_long_trade_zs['entry_price']
        assert metrics['exit_price'] == sample_long_trade_zs['exit_price']
        assert metrics['side'] == 'long'

        # Duration calculated correctly
        assert metrics['duration'] == timedelta(hours=4, minutes=30)
        assert metrics['duration_hours'] == 4.5

        # Commission applied
        assert metrics['commission'] == COMMISSION_PER_TRADE

        # PnL calculated correctly
        # ZS: (1210 - 1200) * 5000 = 50,000 cents = $500
        # Net PnL: $500 - $4 commission = $496
        assert metrics['net_pnl'] == 496.0

        # Margin requirement calculated
        # ZS is in 'grains' category (8% margin ratio)
        # Contract value: 1200 * 5000 / 100 = $60,000
        # Margin: $60,000 * 0.08 = $4,800
        assert metrics['margin_requirement'] == 4800.0

        # Return percentages calculated
        # Return % of margin: ($496 / $4,800) * 100 = 10.33%
        assert round(metrics['return_percentage_of_margin'], 2) == 10.33

    def test_short_trade_metrics_calculation(self, sample_short_trade_cl):
        """Test metrics calculation for profitable short trade."""
        metrics = calculate_trade_metrics(sample_short_trade_cl, 'CL')

        # Original trade data preserved
        assert metrics['side'] == 'short'
        assert metrics['entry_price'] == 75.50
        assert metrics['exit_price'] == 74.80

        # Duration calculated
        expected_duration = timedelta(hours=6, minutes=15)
        assert metrics['duration'] == expected_duration
        assert metrics['duration_hours'] == 6.25

        # PnL calculated correctly for short
        # CL: (75.50 - 74.80) * 1000 = $700
        # Net PnL: $700 - $4 = $696
        assert round(metrics['net_pnl'], 2) == 696.0

        # Margin requirement (CL is 'energies' - 25% ratio)
        # Contract value: 75.50 * 1000 = $75,500
        # Margin: $75,500 * 0.25 = $18,875
        assert metrics['margin_requirement'] == 18875.0

        # Return percentage
        # ($696 / $18,875) * 100 = 3.69%
        assert round(metrics['return_percentage_of_margin'], 2) == 3.69

    def test_losing_trade_metrics(self, sample_losing_trade_es):
        """Test metrics calculation for losing trade."""
        metrics = calculate_trade_metrics(sample_losing_trade_es, 'ES')

        # PnL should be negative
        # ES: (4985 - 5000) * 50 = -$750
        # Net PnL: -$750 - $4 = -$754
        assert metrics['net_pnl'] == -754.0

        # Return percentage should be negative
        # ES is 'indices' (8% margin ratio)
        # Contract value: 5000 * 50 = $250,000
        # Margin: $250,000 * 0.08 = $20,000
        # Return: (-$754 / $20,000) * 100 = -3.77%
        assert round(metrics['return_percentage_of_margin'], 2) == -3.77

    def test_breakeven_trade_metrics(self, breakeven_trade_gc):
        """Test metrics calculation for breakeven trade (no price movement)."""
        metrics = calculate_trade_metrics(breakeven_trade_gc, 'GC')

        # Gross PnL is zero
        # Net PnL: 0 - $4 commission = -$4
        assert metrics['net_pnl'] == -4.0

        # Return percentage should be small negative (commission only)
        # GC is 'metals' (12% margin ratio)
        # Contract value: 2050 * 100 = $205,000
        # Margin: $205,000 * 0.12 = $24,600
        # Return: (-$4 / $24,600) * 100 = -0.02%
        assert round(metrics['return_percentage_of_margin'], 2) == -0.02


class TestReturnPercentageCalculations:
    """Test return percentage calculations."""

    def test_return_percentage_of_margin(self, sample_long_trade_zs):
        """Test return percentage based on margin requirement."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        # Return % of margin = (net_pnl / margin_requirement) * 100
        expected_return = (metrics['net_pnl'] / metrics['margin_requirement']) * 100
        assert round(metrics['return_percentage_of_margin'], 2) == round(expected_return, 2)

    def test_return_percentage_of_contract_value(self, sample_long_trade_zs):
        """Test return percentage based on contract value."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        # Contract value = entry_price * contract_multiplier
        contract_value = 1200.0 * CONTRACT_MULTIPLIERS['ZS']
        expected_return = (metrics['net_pnl'] / contract_value) * 100

        assert round(metrics['return_percentage_of_contract'], 2) == round(expected_return, 2)

    def test_both_return_percentages_present(self, sample_short_trade_cl):
        """Test both return percentage types are calculated."""
        metrics = calculate_trade_metrics(sample_short_trade_cl, 'CL')

        assert 'return_percentage_of_margin' in metrics
        assert 'return_percentage_of_contract' in metrics
        assert isinstance(metrics['return_percentage_of_margin'], float)
        assert isinstance(metrics['return_percentage_of_contract'], float)

    def test_margin_based_return_higher_than_contract_based(self, sample_long_trade_zs):
        """Test margin-based return is higher due to leverage."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        # Margin-based return should be higher because margin < contract value
        assert metrics['return_percentage_of_margin'] > metrics['return_percentage_of_contract']


class TestCommissionHandling:
    """Test commission calculations."""

    def test_commission_applied_to_trade(self, sample_long_trade_zs):
        """Test fixed commission is applied to every trade."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        assert metrics['commission'] == COMMISSION_PER_TRADE
        assert metrics['commission'] == 4.0

    def test_commission_reduces_profit(self, sample_long_trade_zs):
        """Test commission reduces net PnL for profitable trade."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        # Calculate gross PnL
        pnl_points = 1210.0 - 1200.0  # 10 cents
        gross_pnl = pnl_points * CONTRACT_MULTIPLIERS['ZS']  # $500

        # Net PnL should be gross - commission
        expected_net = gross_pnl - COMMISSION_PER_TRADE
        assert metrics['net_pnl'] == expected_net

    def test_commission_increases_loss(self, sample_losing_trade_es):
        """Test commission increases net loss for losing trade."""
        metrics = calculate_trade_metrics(sample_losing_trade_es, 'ES')

        # Calculate gross loss
        pnl_points = 4985.0 - 5000.0  # -15 points
        gross_pnl = pnl_points * CONTRACT_MULTIPLIERS['ES']  # -$750

        # Net loss should be more negative due to commission
        expected_net = gross_pnl - COMMISSION_PER_TRADE  # -$754
        assert metrics['net_pnl'] == expected_net
        assert metrics['net_pnl'] < gross_pnl  # More negative


class TestTradeDurationCalculations:
    """Test trade duration calculations."""

    def test_duration_timedelta_object(self, sample_long_trade_zs):
        """Test duration is returned as timedelta object."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        assert isinstance(metrics['duration'], timedelta)
        assert metrics['duration'] == timedelta(hours=4, minutes=30)

    def test_duration_hours_float(self, sample_short_trade_cl):
        """Test duration_hours is float with 2 decimal places."""
        metrics = calculate_trade_metrics(sample_short_trade_cl, 'CL')

        assert isinstance(metrics['duration_hours'], float)
        assert metrics['duration_hours'] == 6.25

    def test_very_short_duration(self):
        """Test trade with very short duration (minutes)."""
        short_trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 10, 15),  # 15 minutes
            'entry_price': 1200.0,
            'exit_price': 1205.0,
            'side': 'long'
        }

        metrics = calculate_trade_metrics(short_trade, 'ZS')

        assert metrics['duration'] == timedelta(minutes=15)
        assert metrics['duration_hours'] == 0.25

    def test_very_long_duration(self):
        """Test trade with very long duration (days)."""
        long_trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 20, 14, 30),  # ~5 days
            'entry_price': 75.0,
            'exit_price': 78.0,
            'side': 'long'
        }

        metrics = calculate_trade_metrics(long_trade, 'CL')

        expected_hours = (5 * 24) + 4.5  # 5 days + 4.5 hours
        assert metrics['duration_hours'] == expected_hours


class TestMarginRequirementCalculations:
    """Test margin requirement calculations."""

    def test_margin_requirement_grains(self, sample_long_trade_zs):
        """Test margin calculation for grains category."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        # ZS is grains (8% margin ratio)
        contract_value = 1200.0 * CONTRACT_MULTIPLIERS['ZS']
        expected_margin = contract_value * MARGIN_RATIOS['grains']

        assert metrics['margin_requirement'] == expected_margin

    def test_margin_requirement_energies(self, sample_short_trade_cl):
        """Test margin calculation for energies category."""
        metrics = calculate_trade_metrics(sample_short_trade_cl, 'CL')

        # CL is energies (25% margin ratio)
        contract_value = 75.50 * CONTRACT_MULTIPLIERS['CL']
        expected_margin = contract_value * MARGIN_RATIOS['energies']

        assert metrics['margin_requirement'] == expected_margin

    def test_margin_requirement_indices(self, sample_losing_trade_es):
        """Test margin calculation for indices category."""
        metrics = calculate_trade_metrics(sample_losing_trade_es, 'ES')

        # ES is indices (8% margin ratio)
        contract_value = 5000.0 * CONTRACT_MULTIPLIERS['ES']
        expected_margin = contract_value * MARGIN_RATIOS['indices']

        assert metrics['margin_requirement'] == expected_margin

    def test_margin_requirement_metals(self, breakeven_trade_gc):
        """Test margin calculation for metals category."""
        metrics = calculate_trade_metrics(breakeven_trade_gc, 'GC')

        # GC is metals (12% margin ratio)
        contract_value = 2050.0 * CONTRACT_MULTIPLIERS['GC']
        expected_margin = contract_value * MARGIN_RATIOS['metals']

        assert metrics['margin_requirement'] == expected_margin


class TestSymbolCategoryMapping:
    """Test symbol category mapping for margin calculation."""

    def test_get_symbol_category_grains(self):
        """Test grain symbols are categorized correctly."""
        assert _get_symbol_category('ZS') == 'grains'
        assert _get_symbol_category('ZC') == 'grains'
        assert _get_symbol_category('ZW') == 'grains'
        assert _get_symbol_category('ZL') == 'grains'

    def test_get_symbol_category_energies(self):
        """Test energy symbols are categorized correctly."""
        assert _get_symbol_category('CL') == 'energies'
        assert _get_symbol_category('NG') == 'energies'
        assert _get_symbol_category('MCL') == 'energies'

    def test_get_symbol_category_metals(self):
        """Test metal symbols are categorized correctly."""
        assert _get_symbol_category('GC') == 'metals'
        assert _get_symbol_category('SI') == 'metals'
        assert _get_symbol_category('HG') == 'metals'

    def test_get_symbol_category_indices(self):
        """Test index symbols are categorized correctly."""
        assert _get_symbol_category('ES') == 'indices'
        assert _get_symbol_category('NQ') == 'indices'
        assert _get_symbol_category('YM') == 'indices'

    def test_get_symbol_category_forex(self):
        """Test forex symbols are categorized correctly."""
        assert _get_symbol_category('6E') == 'forex'
        assert _get_symbol_category('6J') == 'forex'
        assert _get_symbol_category('6B') == 'forex'

    def test_get_symbol_category_crypto(self):
        """Test crypto symbols are categorized correctly."""
        assert _get_symbol_category('BTC') == 'crypto'
        assert _get_symbol_category('ETH') == 'crypto'

    def test_get_symbol_category_softs(self):
        """Test soft commodity symbols are categorized correctly."""
        assert _get_symbol_category('SB') == 'softs'
        assert _get_symbol_category('KC') == 'softs'
        assert _get_symbol_category('CC') == 'softs'

    def test_get_symbol_category_unknown(self):
        """Test unknown symbols return default category."""
        assert _get_symbol_category('UNKNOWN') == 'default'
        assert _get_symbol_category('XXX') == 'default'


class TestEstimateMargin:
    """Test margin estimation function."""

    def test_estimate_margin_grains(self):
        """Test margin estimation for grain futures."""
        symbol = 'ZS'
        entry_price = 1200.0
        multiplier = CONTRACT_MULTIPLIERS['ZS']

        margin = _estimate_margin(symbol, entry_price, multiplier)

        contract_value = entry_price * multiplier
        expected_margin = contract_value * MARGIN_RATIOS['grains']

        assert margin == expected_margin

    def test_estimate_margin_energies(self):
        """Test margin estimation for energy futures."""
        symbol = 'CL'
        entry_price = 75.0
        multiplier = CONTRACT_MULTIPLIERS['CL']

        margin = _estimate_margin(symbol, entry_price, multiplier)

        contract_value = entry_price * multiplier
        expected_margin = contract_value * MARGIN_RATIOS['energies']

        assert margin == expected_margin

    def test_estimate_margin_scales_with_price(self):
        """Test margin estimate increases with higher prices."""
        symbol = 'GC'
        multiplier = CONTRACT_MULTIPLIERS['GC']

        margin_low = _estimate_margin(symbol, 1800.0, multiplier)
        margin_high = _estimate_margin(symbol, 2200.0, multiplier)

        assert margin_high > margin_low


class TestEdgeCases:
    """Test edge cases in trade metric calculations."""

    def test_same_entry_exit_price(self, breakeven_trade_gc):
        """Test trade with no price movement (only commission loss)."""
        metrics = calculate_trade_metrics(breakeven_trade_gc, 'GC')

        # Gross PnL should be 0
        # Net PnL should be -commission
        assert metrics['net_pnl'] == -COMMISSION_PER_TRADE

    def test_very_small_profit(self):
        """Test trade with very small profit."""
        small_profit_trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 11, 0),
            'entry_price': 1200.00,
            'exit_price': 1200.25,  # 0.25 cent profit
            'side': 'long'
        }

        metrics = calculate_trade_metrics(small_profit_trade, 'ZS')

        # Gross: 0.25 * 5000 = $12.50
        # Net: $12.50 - $4 = $8.50
        assert metrics['net_pnl'] == 8.50

    def test_very_large_profit(self):
        """Test trade with very large profit."""
        large_profit_trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 16, 10, 0),
            'entry_price': 50.00,
            'exit_price': 65.00,  # $15 move in CL
            'side': 'long'
        }

        metrics = calculate_trade_metrics(large_profit_trade, 'CL')

        # Gross: 15 * 1000 = $15,000
        # Net: $15,000 - $4 = $14,996
        assert metrics['net_pnl'] == 14996.0

    def test_invalid_symbol_raises_error(self, sample_long_trade_zs):
        """Test invalid symbol raises ValueError."""
        with pytest.raises(ValueError, match="No contract multiplier found"):
            calculate_trade_metrics(sample_long_trade_zs, 'INVALID')

    def test_invalid_side_raises_error(self):
        """Test invalid trade side raises ValueError."""
        invalid_trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 11, 0),
            'entry_price': 1200.0,
            'exit_price': 1210.0,
            'side': 'invalid'  # Invalid side
        }

        with pytest.raises(ValueError, match="Unknown trade side"):
            calculate_trade_metrics(invalid_trade, 'ZS')

    def test_zero_duration_trade(self):
        """Test trade with same entry and exit time."""
        zero_duration_trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 10, 0),  # Same time
            'entry_price': 1200.0,
            'exit_price': 1205.0,
            'side': 'long'
        }

        metrics = calculate_trade_metrics(zero_duration_trade, 'ZS')

        assert metrics['duration'] == timedelta(0)
        assert metrics['duration_hours'] == 0.0


class TestTradeDataPreservation:
    """Test that original trade data is preserved in metrics."""

    def test_all_original_fields_preserved(self, sample_long_trade_zs):
        """Test all original trade fields are in metrics output."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        # All original fields should be present
        for key in sample_long_trade_zs.keys():
            assert key in metrics
            assert metrics[key] == sample_long_trade_zs[key]

    def test_original_trade_not_modified(self, sample_long_trade_zs):
        """Test original trade dict is not modified."""
        original_keys = set(sample_long_trade_zs.keys())
        original_values = {k: v for k, v in sample_long_trade_zs.items()}

        calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        # Original trade should be unchanged
        assert set(sample_long_trade_zs.keys()) == original_keys
        for key, value in original_values.items():
            assert sample_long_trade_zs[key] == value

    def test_additional_metrics_added(self, sample_long_trade_zs):
        """Test additional metric fields are added to output."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        # Additional fields that should be present
        additional_fields = [
            'duration', 'duration_hours', 'return_percentage_of_margin',
            'return_percentage_of_contract', 'net_pnl', 'margin_requirement',
            'commission'
        ]

        for field in additional_fields:
            assert field in metrics


class TestMultipleSymbols:
    """Test metrics calculation across different futures symbols."""

    @pytest.mark.parametrize("symbol,entry_price,category,margin_ratio", [
        ('ZS', 1200.0, 'grains', 0.08),
        ('CL', 75.0, 'energies', 0.25),
        ('ES', 5000.0, 'indices', 0.08),
        ('GC', 2050.0, 'metals', 0.12),
    ])
    def test_metrics_for_various_symbols(self, symbol, entry_price, category, margin_ratio):
        """Test metrics calculation works for various symbols."""
        trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 14, 0),
            'entry_price': entry_price,
            'exit_price': entry_price * 1.01,  # 1% gain
            'side': 'long'
        }

        metrics = calculate_trade_metrics(trade, symbol)

        # Basic checks
        assert metrics['net_pnl'] > 0  # Should be profitable after 1% gain
        assert metrics['margin_requirement'] > 0
        assert _get_symbol_category(symbol) == category

        # Verify margin ratio is applied correctly
        contract_value = entry_price * CONTRACT_MULTIPLIERS[symbol]
        expected_margin = contract_value * margin_ratio
        assert metrics['margin_requirement'] == expected_margin


class TestPrintTradeMetrics:
    """Test print_trade_metrics function output."""

    def test_print_profitable_trade(self, sample_long_trade_zs, capsys):
        """Test printing metrics for profitable trade (green color)."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        print_trade_metrics(metrics)

        captured = capsys.readouterr()
        output = captured.out

        # Verify key information is printed
        assert 'TRADE METRICS' in output
        assert 'Entry Time:' in output
        assert 'Exit Time:' in output
        assert 'Duration:' in output
        assert 'Side: long' in output
        assert 'Entry Price: 1200.0' in output
        assert 'Exit Price: 1210.0' in output
        assert 'Return % of Contract:' in output

        # Verify green color code is used for profitable trade
        assert '\033[92m' in output  # Green color

    def test_print_losing_trade(self, sample_losing_trade_es, capsys):
        """Test printing metrics for losing trade (red color)."""
        metrics = calculate_trade_metrics(sample_losing_trade_es, 'ES')

        print_trade_metrics(metrics)

        captured = capsys.readouterr()
        output = captured.out

        # Verify key information is printed
        assert 'TRADE METRICS' in output
        assert 'Side: long' in output

        # Verify red color code is used for losing trade
        assert '\033[91m' in output  # Red color

    def test_print_breakeven_trade(self, breakeven_trade_gc, capsys):
        """Test printing metrics for breakeven trade (no color)."""
        metrics = calculate_trade_metrics(breakeven_trade_gc, 'GC')

        # Manually set return to exactly 0 for this test
        metrics['return_percentage_of_contract'] = 0.0

        print_trade_metrics(metrics)

        captured = capsys.readouterr()
        output = captured.out

        # Verify information is printed
        assert 'TRADE METRICS' in output

    def test_print_trade_with_duration(self, sample_long_trade_zs, capsys):
        """Test duration information is printed correctly."""
        metrics = calculate_trade_metrics(sample_long_trade_zs, 'ZS')

        print_trade_metrics(metrics)

        captured = capsys.readouterr()
        output = captured.out

        # Verify duration is displayed
        assert '4.5' in output or '4.50' in output  # Duration hours
        assert 'hours' in output

    def test_print_trade_handles_missing_fields(self, capsys):
        """Test print function handles missing fields gracefully."""
        incomplete_trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 14, 0),
            'return_percentage_of_contract': 5.0
        }

        # Should not raise error even with missing fields
        print_trade_metrics(incomplete_trade)

        captured = capsys.readouterr()
        output = captured.out

        assert 'TRADE METRICS' in output
        assert 'N/A' in output  # Missing fields show as N/A

    def test_print_trade_with_duration_hours_only(self, capsys):
        """Test print function when only duration_hours is present (no duration timedelta)."""
        trade_with_duration_hours = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 14, 0),
            'duration_hours': 4.5,  # Only duration_hours, no duration
            'side': 'long',
            'entry_price': 1200.0,
            'exit_price': 1210.0,
            'return_percentage_of_contract': 2.5
        }

        print_trade_metrics(trade_with_duration_hours)

        captured = capsys.readouterr()
        output = captured.out

        # Verify duration_hours is displayed
        assert 'Duration:' in output
        assert '4.50' in output or '4.5' in output
        assert 'hours' in output


class TestErrorHandling:
    """Test error handling and logging."""

    def test_zero_contract_multiplier_raises_error(self):
        """Test that zero contract multiplier raises ValueError."""
        trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 11, 0),
            'entry_price': 1200.0,
            'exit_price': 1210.0,
            'side': 'long'
        }

        # Mock CONTRACT_MULTIPLIERS to return 0
        with patch.dict('app.backtesting.metrics.per_trade_metrics.CONTRACT_MULTIPLIERS', {'TEST': 0}):
            with pytest.raises(ValueError, match="No contract multiplier found"):
                calculate_trade_metrics(trade, 'TEST')

    def test_none_contract_multiplier_raises_error(self):
        """Test that None contract multiplier raises ValueError."""
        trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 11, 0),
            'entry_price': 1200.0,
            'exit_price': 1210.0,
            'side': 'long'
        }

        with pytest.raises(ValueError, match="No contract multiplier found"):
            calculate_trade_metrics(trade, 'NONEXISTENT')

    def test_negative_margin_raises_error(self):
        """Test that negative margin requirement raises ValueError."""
        trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 11, 0),
            'entry_price': -1200.0,  # Negative price leads to negative margin
            'exit_price': -1210.0,
            'side': 'long'
        }

        with pytest.raises(ValueError, match="Margin requirement must be positive"):
            calculate_trade_metrics(trade, 'ZS')

    def test_zero_margin_raises_error(self):
        """Test that zero margin requirement raises ValueError."""
        trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 11, 0),
            'entry_price': 0.0,  # Zero price leads to zero margin
            'exit_price': 10.0,
            'side': 'long'
        }

        with pytest.raises(ValueError, match="Margin requirement must be positive"):
            calculate_trade_metrics(trade, 'ZS')


class TestLoggerCalls:
    """Test that logger is called appropriately for errors."""

    def test_logger_called_for_invalid_symbol(self, sample_long_trade_zs):
        """Test logger.error is called for invalid symbol."""
        with patch('app.backtesting.metrics.per_trade_metrics.logger') as mock_logger:
            with pytest.raises(ValueError):
                calculate_trade_metrics(sample_long_trade_zs, 'INVALID')

            # Verify logger was called
            mock_logger.error.assert_called_once()
            assert 'No contract multiplier found' in mock_logger.error.call_args[0][0]

    def test_logger_called_for_invalid_margin(self):
        """Test logger.error is called for invalid margin."""
        trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 11, 0),
            'entry_price': -100.0,
            'exit_price': -90.0,
            'side': 'long'
        }

        with patch('app.backtesting.metrics.per_trade_metrics.logger') as mock_logger:
            with pytest.raises(ValueError):
                calculate_trade_metrics(trade, 'ZS')

            # Verify logger was called
            mock_logger.error.assert_called_once()
            assert 'Invalid margin requirement' in mock_logger.error.call_args[0][0]

    def test_logger_called_for_invalid_side(self):
        """Test logger.error is called for invalid trade side."""
        trade = {
            'entry_time': datetime(2024, 1, 15, 10, 0),
            'exit_time': datetime(2024, 1, 15, 11, 0),
            'entry_price': 1200.0,
            'exit_price': 1210.0,
            'side': 'sideways'
        }

        with patch('app.backtesting.metrics.per_trade_metrics.logger') as mock_logger:
            with pytest.raises(ValueError):
                calculate_trade_metrics(trade, 'ZS')

            # Verify logger was called
            mock_logger.error.assert_called_once()
            assert 'Unknown trade side' in mock_logger.error.call_args[0][0]
