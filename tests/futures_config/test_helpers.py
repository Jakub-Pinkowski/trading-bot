"""
Tests for Helper Functions Module.

Tests cover:
- get_exchange_for_symbol function
- get_category_for_symbol function
- get_tick_size function with fallback behavior
- get_contract_multiplier function
- get_margin_requirement function
- Error handling for unknown symbols
- Edge cases and boundary conditions

All tests use actual futures symbols from SYMBOL_SPECS.
"""
import pytest

from futures_config.helpers import (
    get_exchange_for_symbol,
    get_category_for_symbol,
    get_tick_size,
    get_contract_multiplier,
    get_margin_requirement,
)
from futures_config.symbol_specs import DEFAULT_TICK_SIZE


class TestGetExchangeForSymbol:
    """Test get_exchange_for_symbol function."""

    @pytest.mark.parametrize("symbol,expected_exchange", [
        ('ZS', 'CBOT'),
        ('ZC', 'CBOT'),
        ('CL', 'NYMEX'),
        ('NG', 'NYMEX'),
        ('GC', 'COMEX'),
        ('SI', 'COMEX'),
        ('ES', 'CME_MINI'),
        ('BTC', 'CME'),
        ('6E', 'CME'),
    ])
    def test_major_symbols(self, symbol, expected_exchange):
        """Test exchange retrieval for major symbols."""
        assert get_exchange_for_symbol(symbol) == expected_exchange

    def test_unknown_symbol_raises_error(self):
        """Test that unknown symbol raises ValueError."""
        with pytest.raises(ValueError, match='Unknown symbol'):
            get_exchange_for_symbol('UNKNOWN')

    def test_mini_grains_exchange(self):
        """Test exchange for mini grains."""
        assert get_exchange_for_symbol('XC') == 'CBOT'
        assert get_exchange_for_symbol('XW') == 'CBOT'
        assert get_exchange_for_symbol('XK') == 'CBOT'

    def test_micro_symbols_exchange(self):
        """Test exchange for micro symbols."""
        assert get_exchange_for_symbol('MZC') == 'CBOT'
        assert get_exchange_for_symbol('MCL') == 'NYMEX'
        assert get_exchange_for_symbol('MGC') == 'COMEX_MINI'


class TestGetCategoryForSymbol:
    """Test get_category_for_symbol function."""

    @pytest.mark.parametrize("symbol,expected_category", [
        ('ZS', 'Grains'),
        ('ZC', 'Grains'),
        ('CL', 'Energy'),
        ('NG', 'Energy'),
        ('GC', 'Metals'),
        ('SI', 'Metals'),
        ('BTC', 'Crypto'),
        ('ETH', 'Crypto'),
        ('ES', 'Index'),
        ('YM', 'Index'),
        ('6E', 'Forex'),
        ('SB', 'Softs'),
    ])
    def test_category_retrieval(self, symbol, expected_category):
        """Test category retrieval for various symbols."""
        assert get_category_for_symbol(symbol) == expected_category

    def test_unknown_symbol_raises_error(self):
        """Test that unknown symbol raises ValueError."""
        with pytest.raises(ValueError, match='Unknown symbol'):
            get_category_for_symbol('UNKNOWN')

    def test_mini_grains_category(self):
        """Test category for mini grains."""
        assert get_category_for_symbol('XC') == 'Grains'
        assert get_category_for_symbol('XW') == 'Grains'
        assert get_category_for_symbol('XK') == 'Grains'

    def test_micro_metals_category(self):
        """Test category for micro metals."""
        assert get_category_for_symbol('MGC') == 'Metals'
        assert get_category_for_symbol('SIL') == 'Metals'


class TestGetTickSize:
    """Test get_tick_size function."""

    @pytest.mark.parametrize("symbol,expected_tick_size", [
        ('ZS', 0.25),
        ('ZC', 0.25),
        ('CL', 0.01),
        ('NG', 0.001),
        ('GC', 0.10),
        ('SI', 0.005),
        ('XC', 0.125),
        ('MGC', 0.10),
    ])
    def test_tick_size_retrieval(self, symbol, expected_tick_size):
        """Test tick size retrieval for various symbols."""
        assert get_tick_size(symbol) == expected_tick_size

    def test_unknown_symbol_returns_default(self):
        """Test that unknown symbol returns DEFAULT_TICK_SIZE."""
        result = get_tick_size('UNKNOWN')
        assert result == DEFAULT_TICK_SIZE

    def test_symbol_with_none_tick_size_returns_default(self):
        """Test that symbol with None tick_size returns default."""
        # Find a symbol with None tick_size (if any)
        from futures_config.symbol_specs import SYMBOL_SPECS
        symbols_with_none = [
            s for s, spec in SYMBOL_SPECS.items()
            if spec['tick_size'] is None
        ]

        if symbols_with_none:
            symbol = symbols_with_none[0]
            result = get_tick_size(symbol)
            assert result == DEFAULT_TICK_SIZE

    def test_default_tick_size_value(self):
        """Test that DEFAULT_TICK_SIZE has expected value."""
        assert DEFAULT_TICK_SIZE == 0.01


class TestGetContractMultiplier:
    """Test get_contract_multiplier function."""

    @pytest.mark.parametrize("symbol,expected_multiplier", [
        ('ZS', 50),
        ('ZC', 50),
        ('CL', 1000),
        ('NG', 10000),
        ('GC', 100),
        ('SI', 5000),
        ('XC', 1000),
        ('MGC', 10),
    ])
    def test_multiplier_retrieval(self, symbol, expected_multiplier):
        """Test contract multiplier retrieval for various symbols."""
        assert get_contract_multiplier(symbol) == expected_multiplier

    def test_unknown_symbol_raises_error(self):
        """Test that unknown symbol raises ValueError."""
        with pytest.raises(ValueError, match='Unknown symbol'):
            get_contract_multiplier('UNKNOWN')

    def test_symbol_with_none_multiplier(self):
        """Test that symbol with None multiplier returns None."""
        from futures_config.symbol_specs import SYMBOL_SPECS
        symbols_with_none = [
            s for s, spec in SYMBOL_SPECS.items()
            if spec['multiplier'] is None
        ]

        if symbols_with_none:
            symbol = symbols_with_none[0]
            result = get_contract_multiplier(symbol)
            assert result is None

    def test_all_multipliers_positive(self):
        """Test that all non-None multipliers are positive."""
        from futures_config.symbol_specs import SYMBOL_SPECS
        for symbol, specs in SYMBOL_SPECS.items():
            multiplier = get_contract_multiplier(symbol)
            if multiplier is not None:
                assert multiplier > 0, f"Symbol {symbol} multiplier is not positive"


class TestGetMarginRequirement:
    """Test get_margin_requirement function."""

    @pytest.mark.parametrize("symbol", ['ZS', 'CL', 'GC', 'ES'])
    def test_margin_retrieval(self, symbol):
        """Test margin requirement retrieval for major symbols."""
        margin = get_margin_requirement(symbol)
        assert margin is not None
        assert margin > 0

    def test_unknown_symbol_raises_error(self):
        """Test that unknown symbol raises ValueError."""
        with pytest.raises(ValueError, match='Unknown symbol'):
            get_margin_requirement('UNKNOWN')

    def test_symbol_with_none_margin(self):
        """Test that symbol with None margin returns None."""
        from futures_config.symbol_specs import SYMBOL_SPECS
        symbols_with_none = [
            s for s, spec in SYMBOL_SPECS.items()
            if spec['margin'] is None
        ]

        if symbols_with_none:
            symbol = symbols_with_none[0]
            result = get_margin_requirement(symbol)
            assert result is None

    def test_all_margins_positive(self):
        """Test that all non-None margins are positive."""
        from futures_config.symbol_specs import SYMBOL_SPECS
        for symbol in SYMBOL_SPECS.keys():
            margin = get_margin_requirement(symbol)
            if margin is not None:
                assert margin > 0, f"Symbol {symbol} margin is not positive"


class TestErrorHandling:
    """Test error handling across all helper functions."""

    @pytest.mark.parametrize("func", [
        get_exchange_for_symbol,
        get_category_for_symbol,
        get_contract_multiplier,
        get_margin_requirement,
    ])
    def test_unknown_symbol_error_message(self, func):
        """Test that error message includes symbol name."""
        with pytest.raises(ValueError, match='UNKNOWN'):
            func('UNKNOWN')

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            get_exchange_for_symbol('')

    def test_none_input_raises_error(self):
        """Test that None input raises appropriate error."""
        with pytest.raises(Exception):  # Could be ValueError or TypeError
            get_exchange_for_symbol(None)

    def test_lowercase_symbol_raises_error(self):
        """Test that lowercase symbols raise error (case sensitive)."""
        with pytest.raises(ValueError):
            get_exchange_for_symbol('zs')  # Should be 'ZS'


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_get_all_info_for_symbol(self):
        """Test getting all information for a symbol."""
        symbol = 'ZS'

        exchange = get_exchange_for_symbol(symbol)
        category = get_category_for_symbol(symbol)
        tick_size = get_tick_size(symbol)
        multiplier = get_contract_multiplier(symbol)
        margin = get_margin_requirement(symbol)

        assert exchange == 'CBOT'
        assert category == 'Grains'
        assert tick_size == 0.25
        assert multiplier == 50
        assert margin > 0

    def test_get_tick_size_with_fallback(self):
        """Test tick size retrieval with fallback for unknown symbols."""
        # Known symbol
        known_tick_size = get_tick_size('ZS')
        assert known_tick_size == 0.25

        # Unknown symbol falls back to default
        unknown_tick_size = get_tick_size('UNKNOWN_SYMBOL')
        assert unknown_tick_size == DEFAULT_TICK_SIZE

    def test_calculate_contract_value(self):
        """Test calculating contract value using helper functions."""
        symbol = 'ZS'
        price = 1200.0  # cents per bushel

        multiplier = get_contract_multiplier(symbol)
        tick_size = get_tick_size(symbol)

        # Contract value = price * multiplier / 100 (cents to dollars)
        contract_value = (price * multiplier) / 100

        assert multiplier == 50
        assert tick_size == 0.25
        assert contract_value == 600.0  # $600

    def test_validate_trading_setup(self):
        """Test validating a trading setup using helpers."""
        symbol = 'CL'

        exchange = get_exchange_for_symbol(symbol)
        margin = get_margin_requirement(symbol)

        assert exchange == 'NYMEX'
        assert margin is not None
        assert margin > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_symbols_with_numbers(self):
        """Test symbols with numbers in name."""
        # These symbols have numbers but should work fine
        assert get_exchange_for_symbol('6E') == 'CME'
        assert get_category_for_symbol('6E') == 'Forex'

    def test_micro_prefix_symbols(self):
        """Test symbols with M prefix (micro contracts)."""
        micro_symbols = ['MZC', 'MCL', 'MGC', 'MES', 'MYM']

        for symbol in micro_symbols:
            # Should not raise error
            exchange = get_exchange_for_symbol(symbol)
            category = get_category_for_symbol(symbol)

            assert exchange is not None
            assert category is not None

    def test_all_defined_symbols_work(self):
        """Test that all symbols in SYMBOL_SPECS work with helpers."""
        from futures_config.symbol_specs import SYMBOL_SPECS

        for symbol in SYMBOL_SPECS.keys():
            # All these should work without raising errors
            get_exchange_for_symbol(symbol)
            get_category_for_symbol(symbol)
            get_tick_size(symbol)  # May return default
            get_contract_multiplier(symbol)  # May return None
            get_margin_requirement(symbol)  # May return None

    def test_special_character_symbols(self):
        """Test symbols with special characters."""
        # 6E, 6J, etc. have numbers which could be considered special
        forex_symbols = ['6E', '6J', '6B', '6A', '6C', '6S']

        for symbol in forex_symbols:
            category = get_category_for_symbol(symbol)
            assert category == 'Forex'
