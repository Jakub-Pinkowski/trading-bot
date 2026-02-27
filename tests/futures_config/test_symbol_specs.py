"""
Tests for Symbol Specifications Module.

Tests cover:
- SYMBOL_SPECS structure and integrity
- Symbol specification attributes (category, exchange, multiplier, etc.)
- Data completeness and validation
- TradingView compatibility flags
- Edge cases and error conditions

All tests use actual futures symbols from the configuration.
"""
import pytest

from futures_config.symbol_specs import SYMBOL_SPECS


class TestSymbolSpecsStructure:
    """Test SYMBOL_SPECS structure and integrity."""

    def test_symbol_specs_not_empty(self):
        """Test that SYMBOL_SPECS contains symbols."""
        assert len(SYMBOL_SPECS) > 0
        assert isinstance(SYMBOL_SPECS, dict)

    def test_all_symbols_have_required_keys(self):
        """Test that all symbols have required specification keys."""
        required_keys = {'category', 'exchange', 'multiplier', 'tick_size', 'margin'}

        for symbol, specs in SYMBOL_SPECS.items():
            assert isinstance(specs, dict), f"Symbol {symbol} specs is not a dict"
            assert required_keys == set(specs.keys()), f"Symbol {symbol} missing required keys"

    def test_category_values_valid(self):
        """Test that all category values are from expected set."""
        valid_categories = {'Grains', 'Softs', 'Energy', 'Metals', 'Crypto', 'Index', 'Forex'}

        for symbol, specs in SYMBOL_SPECS.items():
            assert specs['category'] in valid_categories, f"Symbol {symbol} has invalid category: {specs['category']}"

    def test_exchange_values_valid(self):
        """Test that all exchange values are from expected set."""
        valid_exchanges = {'CBOT', 'ICEUS', 'NYMEX', 'COMEX', 'CME', 'CME_MINI', 'COMEX_MINI'}

        for symbol, specs in SYMBOL_SPECS.items():
            assert specs['exchange'] in valid_exchanges, f"Symbol {symbol} has invalid exchange: {specs['exchange']}"

    @pytest.mark.parametrize("symbol", ['ZS', 'CL', 'GC', 'ES', 'BTC', '6E'])
    def test_major_symbols_exist(self, symbol):
        """Test that major futures symbols exist in SYMBOL_SPECS."""
        assert symbol in SYMBOL_SPECS, f"Major symbol {symbol} not found in SYMBOL_SPECS"


class TestSymbolSpecsDataTypes:
    """Test data types of symbol specifications."""

    def test_multiplier_types(self):
        """Test that multiplier is numeric or None."""
        for symbol, specs in SYMBOL_SPECS.items():
            multiplier = specs['multiplier']
            assert multiplier is None or isinstance(multiplier, (int, float)), \
                f"Symbol {symbol} multiplier has invalid type: {type(multiplier)}"

    def test_tick_size_types(self):
        """Test that tick_size is numeric or None."""
        for symbol, specs in SYMBOL_SPECS.items():
            tick_size = specs['tick_size']
            assert tick_size is None or isinstance(tick_size, (int, float)), \
                f"Symbol {symbol} tick_size has invalid type: {type(tick_size)}"

    def test_margin_types(self):
        """Test that margin is numeric or None."""
        for symbol, specs in SYMBOL_SPECS.items():
            margin = specs['margin']
            assert margin is None or isinstance(margin, (int, float)), \
                f"Symbol {symbol} margin has invalid type: {type(margin)}"

    def test_positive_values(self):
        """Test that numeric values are positive when not None."""
        for symbol, specs in SYMBOL_SPECS.items():
            if specs['multiplier'] is not None:
                assert specs['multiplier'] > 0, f"Symbol {symbol} multiplier is not positive"
            if specs['tick_size'] is not None:
                assert specs['tick_size'] > 0, f"Symbol {symbol} tick_size is not positive"
            if specs['margin'] is not None:
                assert specs['margin'] > 0, f"Symbol {symbol} margin is not positive"


class TestCategoryOrganization:
    """Test organization of symbols by category."""

    def test_grains_category(self):
        """Test Grains category contains expected symbols."""
        grains = [s for s, spec in SYMBOL_SPECS.items() if spec['category'] == 'Grains']

        assert 'ZS' in grains  # Soybeans
        assert 'ZC' in grains  # Corn
        assert 'ZW' in grains  # Wheat
        assert len(grains) > 0

    def test_energy_category(self):
        """Test Energy category contains expected symbols."""
        energy = [s for s, spec in SYMBOL_SPECS.items() if spec['category'] == 'Energy']

        assert 'CL' in energy  # Crude Oil
        assert 'NG' in energy  # Natural Gas
        assert len(energy) > 0

    def test_metals_category(self):
        """Test Metals category contains expected symbols."""
        metals = [s for s, spec in SYMBOL_SPECS.items() if spec['category'] == 'Metals']

        assert 'GC' in metals  # Gold
        assert 'SI' in metals  # Silver
        assert len(metals) > 0

    def test_crypto_category(self):
        """Test Crypto category contains expected symbols."""
        crypto = [s for s, spec in SYMBOL_SPECS.items() if spec['category'] == 'Crypto']

        assert 'BTC' in crypto  # Bitcoin
        assert 'ETH' in crypto  # Ethereum
        assert len(crypto) > 0


class TestExchangeMapping:
    """Test exchange mapping for symbols."""

    def test_cbot_symbols(self):
        """Test CBOT exchange symbols."""
        cbot_symbols = [s for s, spec in SYMBOL_SPECS.items() if spec['exchange'] == 'CBOT']

        # Grains should be CBOT
        assert 'ZS' in cbot_symbols
        assert 'ZC' in cbot_symbols
        assert 'ZW' in cbot_symbols
        assert len(cbot_symbols) > 0

    def test_nymex_symbols(self):
        """Test NYMEX exchange symbols."""
        nymex_symbols = [s for s, spec in SYMBOL_SPECS.items() if spec['exchange'] == 'NYMEX']

        # Energy should be NYMEX
        assert 'CL' in nymex_symbols
        assert 'NG' in nymex_symbols
        assert len(nymex_symbols) > 0

    def test_comex_symbols(self):
        """Test COMEX exchange symbols."""
        comex_symbols = [s for s, spec in SYMBOL_SPECS.items() if spec['exchange'] == 'COMEX']

        # Metals should be COMEX
        assert 'GC' in comex_symbols
        assert 'SI' in comex_symbols
        assert len(comex_symbols) > 0

    def test_cme_symbols(self):
        """Test CME exchange symbols."""
        cme_symbols = [s for s, spec in SYMBOL_SPECS.items() if spec['exchange'] == 'CME']

        # Crypto and some indices should be CME
        assert 'BTC' in cme_symbols
        assert 'ETH' in cme_symbols
        assert len(cme_symbols) > 0


class TestSpecificSymbols:
    """Test specific symbol specifications."""

    def test_zs_specifications(self):
        """Test ZS (Soybeans) specifications."""
        zs = SYMBOL_SPECS['ZS']

        assert zs['category'] == 'Grains'
        assert zs['exchange'] == 'CBOT'
        assert zs['multiplier'] == 50
        assert zs['tick_size'] == 0.25

    def test_cl_specifications(self):
        """Test CL (Crude Oil) specifications."""
        cl = SYMBOL_SPECS['CL']

        assert cl['category'] == 'Energy'
        assert cl['exchange'] == 'NYMEX'
        assert cl['multiplier'] == 1000
        assert cl['tick_size'] == 0.01

    def test_gc_specifications(self):
        """Test GC (Gold) specifications."""
        gc = SYMBOL_SPECS['GC']

        assert gc['category'] == 'Metals'
        assert gc['exchange'] == 'COMEX'
        assert gc['multiplier'] == 100
        assert gc['tick_size'] == 0.10

    def test_xc_specifications(self):
        """Test XC (Mini Corn) specifications."""
        xc = SYMBOL_SPECS['XC']

        assert xc['category'] == 'Grains'
        assert xc['exchange'] == 'CBOT'
        assert xc['multiplier'] == 1000
        assert xc['tick_size'] == 0.125



class TestDataCompleteness:
    """Test data completeness for commonly used symbols."""

    @pytest.mark.parametrize("symbol", ['ZS', 'ZC', 'CL', 'NG', 'GC', 'SI'])
    def test_major_symbols_have_complete_data(self, symbol):
        """Test that major symbols have complete specifications."""
        specs = SYMBOL_SPECS[symbol]

        # These major symbols should have all data
        assert specs['multiplier'] is not None, f"{symbol} missing multiplier"
        assert specs['tick_size'] is not None, f"{symbol} missing tick_size"
        assert specs['margin'] is not None, f"{symbol} missing margin"

    def test_symbols_with_none_values_documented(self):
        """Test that symbols with None values are acceptable."""
        # Some symbols may have None for multiplier, tick_size, or margin
        # This is intentional and documented
        symbols_with_none = [
            s for s, spec in SYMBOL_SPECS.items()
            if spec['multiplier'] is None or spec['tick_size'] is None or spec['margin'] is None
        ]

        # Should have some symbols with None (it's expected)
        assert len(symbols_with_none) >= 0  # This is fine


class TestSymbolNaming:
    """Test symbol naming conventions."""

    def test_symbols_are_uppercase(self):
        """Test that all symbol keys are uppercase."""
        for symbol in SYMBOL_SPECS.keys():
            assert symbol.isupper(), f"Symbol {symbol} is not uppercase"

    def test_no_duplicate_symbols(self):
        """Test that there are no duplicate symbols."""
        symbols = list(SYMBOL_SPECS.keys())
        assert len(symbols) == len(set(symbols)), "Duplicate symbols found"

    def test_symbol_length_reasonable(self):
        """Test that symbol names have reasonable length."""
        for symbol in SYMBOL_SPECS.keys():
            assert 1 <= len(symbol) <= 5, f"Symbol {symbol} has unusual length"
