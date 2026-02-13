"""
Integration Tests for Futures Config Package.

Tests cover:
- Package-level imports
- Cross-module integration
- Real-world usage scenarios
- End-to-end workflows
- Public API completeness

All tests validate the futures_config package as a cohesive unit.
"""
import pytest

import futures_config
from futures_config import (
    SYMBOL_SPECS,
    TV_TO_IBKR_MAPPING,
    map_tv_to_ibkr,
    map_ibkr_to_tv,
    GRAINS,
    SOFTS,
    ENERGY,
    METALS,
    CRYPTO,
    INDEX,
    FOREX,
    CATEGORIES,
    get_exchange_for_symbol,
    get_category_for_symbol,
    get_tick_size,
    get_contract_multiplier,
    get_margin_requirement,
    is_tradingview_compatible,
)


class TestPackageImports:
    """Test package-level imports."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        expected_exports = [
            'SYMBOL_SPECS',
            'DEFAULT_TICK_SIZE',
            'TV_TO_IBKR_MAPPING',
            'IBKR_TO_TV_MAPPING',
            'map_tv_to_ibkr',
            'map_ibkr_to_tv',
            'GRAINS',
            'SOFTS',
            'ENERGY',
            'METALS',
            'CRYPTO',
            'INDEX',
            'FOREX',
            'CATEGORIES',
            'get_exchange_for_symbol',
            'get_category_for_symbol',
            'get_tick_size',
            'get_contract_multiplier',
            'get_margin_requirement',
            'is_tradingview_compatible',
        ]

        for export in expected_exports:
            assert hasattr(futures_config, export), f"Missing export: {export}"

    def test_package_has_all_attribute(self):
        """Test that package has __all__ defined."""
        assert hasattr(futures_config, '__all__')
        assert len(futures_config.__all__) > 0

    def test_direct_import_works(self):
        """Test that direct imports from package work."""
        from futures_config import SYMBOL_SPECS as specs
        from futures_config import map_tv_to_ibkr as mapper

        assert specs is not None
        assert mapper is not None
        assert callable(mapper)


class TestCrossModuleIntegration:
    """Test integration between different modules."""

    def test_categories_match_symbol_specs(self):
        """Test that all category symbols exist in SYMBOL_SPECS."""
        all_category_symbols = GRAINS + SOFTS + ENERGY + METALS + CRYPTO + INDEX + FOREX

        for symbol in all_category_symbols:
            assert symbol in SYMBOL_SPECS, f"Category symbol {symbol} not in SYMBOL_SPECS"

    def test_mapped_symbols_in_symbol_specs(self):
        """Test that all mapped symbols exist in SYMBOL_SPECS."""
        for tv_symbol in TV_TO_IBKR_MAPPING.keys():
            assert tv_symbol in SYMBOL_SPECS, f"Mapped TV symbol {tv_symbol} not in SYMBOL_SPECS"

    def test_helper_functions_use_symbol_specs(self):
        """Test that helper functions work with symbols from SYMBOL_SPECS."""
        for symbol in list(SYMBOL_SPECS.keys())[:10]:  # Test first 10
            # Should not raise errors
            get_exchange_for_symbol(symbol)
            get_category_for_symbol(symbol)
            is_tradingview_compatible(symbol)

    def test_categories_dict_consistency(self):
        """Test that CATEGORIES dict is consistent with individual lists."""
        assert CATEGORIES['Grains'] == GRAINS
        assert CATEGORIES['Energy'] == ENERGY
        assert CATEGORIES['Metals'] == METALS


class TestTradingViewWorkflow:
    """Test complete TradingView to IBKR workflow."""

    def test_alert_processing_workflow(self):
        """Test processing TradingView alert for IBKR order placement."""
        # TradingView alert comes in with XC
        tv_symbol = 'XC'

        # Step 1: Verify symbol is TradingView compatible
        assert is_tradingview_compatible(tv_symbol)

        # Step 2: Get symbol information
        category = get_category_for_symbol(tv_symbol)
        exchange = get_exchange_for_symbol(tv_symbol)

        assert category == 'Grains'
        assert exchange == 'CBOT'

        # Step 3: Map to IBKR symbol for order placement
        ibkr_symbol = map_tv_to_ibkr(tv_symbol)
        assert ibkr_symbol == 'YC'

        # Step 4: Get contract specifications for order
        multiplier = get_contract_multiplier(tv_symbol)
        tick_size = get_tick_size(tv_symbol)
        margin = get_margin_requirement(tv_symbol)

        assert multiplier == 1000
        assert tick_size == 0.125
        assert margin > 0

    def test_unmapped_symbol_workflow(self):
        """Test workflow for symbols that don't require mapping."""
        # TradingView alert with ZS (no mapping needed)
        tv_symbol = 'ZS'

        # Verify TradingView compatible
        assert is_tradingview_compatible(tv_symbol)

        # Map to IBKR (should return same symbol)
        ibkr_symbol = map_tv_to_ibkr(tv_symbol)
        assert ibkr_symbol == 'ZS'

        # Get specifications
        exchange = get_exchange_for_symbol(tv_symbol)
        category = get_category_for_symbol(tv_symbol)

        assert exchange == 'CBOT'
        assert category == 'Grains'

    def test_micro_silver_workflow(self):
        """Test complete workflow for micro silver (SIL -> QI)."""
        tv_symbol = 'SIL'

        # Verify compatibility
        assert is_tradingview_compatible(tv_symbol)

        # Map to IBKR
        ibkr_symbol = map_tv_to_ibkr(tv_symbol)
        assert ibkr_symbol == 'QI'

        # Get specifications
        category = get_category_for_symbol(tv_symbol)
        exchange = get_exchange_for_symbol(tv_symbol)

        assert category == 'Metals'
        assert exchange == 'COMEX'


class TestPositionTrackingWorkflow:
    """Test position tracking from IBKR to TradingView."""

    def test_ibkr_position_to_tv_symbol(self):
        """Test converting IBKR position data to TradingView symbol."""
        # IBKR reports position with YC
        ibkr_symbol = 'YC'

        # Map to TradingView symbol
        tv_symbol = map_ibkr_to_tv(ibkr_symbol)
        assert tv_symbol == 'XC'

        # Verify it's TV compatible
        assert is_tradingview_compatible(tv_symbol)

        # Get category for grouping
        category = get_category_for_symbol(tv_symbol)
        assert category == 'Grains'

    def test_unmapped_ibkr_position(self):
        """Test IBKR position that doesn't require mapping."""
        # IBKR reports ZS position
        ibkr_symbol = 'ZS'

        # Map to TV (should return same)
        tv_symbol = map_ibkr_to_tv(ibkr_symbol)
        assert tv_symbol == 'ZS'

    def test_position_value_calculation(self):
        """Test calculating position value using helpers."""
        symbol = 'CL'
        position_size = 2  # 2 contracts
        entry_price = 75.50
        current_price = 76.00

        # Get contract specifications
        multiplier = get_contract_multiplier(symbol)
        tick_size = get_tick_size(symbol)

        # Calculate P&L
        price_diff = current_price - entry_price
        pnl = price_diff * multiplier * position_size

        assert multiplier == 1000
        assert tick_size == 0.01
        assert pnl == 1000.0  # $1000 profit


class TestCategoryFiltering:
    """Test filtering and grouping by categories."""

    def test_get_all_energy_symbols(self):
        """Test getting all energy symbols."""
        energy_symbols = CATEGORIES['Energy']

        assert 'CL' in energy_symbols
        assert 'NG' in energy_symbols
        assert 'ZS' not in energy_symbols  # Not energy

    def test_filter_grains_by_size(self):
        """Test filtering grains by contract size."""
        # Get all grains
        all_grains = GRAINS

        # Separate by size prefix
        normal_grains = [s for s in all_grains if not s.startswith(('X', 'M'))]
        mini_grains = [s for s in all_grains if s.startswith('X')]
        micro_grains = [s for s in all_grains if s.startswith('M')]

        assert 'ZS' in normal_grains
        assert 'XC' in mini_grains
        assert 'MZC' in micro_grains

    def test_get_tv_compatible_by_category(self):
        """Test that all category symbols are TV compatible."""
        for category_name, symbols in CATEGORIES.items():
            for symbol in symbols:
                assert is_tradingview_compatible(symbol), \
                    f"Symbol {symbol} in {category_name} not TV compatible"


class TestRiskManagement:
    """Test risk management calculations."""

    def test_margin_requirements_by_category(self):
        """Test getting margin requirements for portfolio."""
        # Get one symbol from each category
        portfolio = {
            'ZS': 1,  # Grains
            'CL': 2,  # Energy
            'GC': 1,  # Metals
        }

        total_margin = 0
        for symbol, contracts in portfolio.items():
            margin = get_margin_requirement(symbol)
            if margin:
                total_margin += margin * contracts

        assert total_margin > 0

    def test_tick_value_calculation(self):
        """Test calculating tick value for position sizing."""
        symbol = 'ZS'

        tick_size = get_tick_size(symbol)
        multiplier = get_contract_multiplier(symbol)

        # Tick value = tick_size * multiplier / 100 (cents to dollars)
        tick_value = (tick_size * multiplier) / 100

        assert tick_size == 0.25
        assert multiplier == 50
        assert tick_value == 0.125  # $0.125 per tick


class TestDataQuality:
    """Test overall data quality and consistency."""

    def test_no_orphaned_mappings(self):
        """Test that all mappings reference valid symbols."""
        for tv_symbol, ibkr_symbol in TV_TO_IBKR_MAPPING.items():
            # TV symbol must be in SYMBOL_SPECS
            assert tv_symbol in SYMBOL_SPECS, f"TV symbol {tv_symbol} not in SYMBOL_SPECS"

    def test_all_tv_compatible_symbols_queryable(self):
        """Test that all TV-compatible symbols can be queried."""
        tv_symbols = [s for s, spec in SYMBOL_SPECS.items() if spec['tv_compatible']]

        for symbol in tv_symbols:
            # All should work without errors
            get_exchange_for_symbol(symbol)
            get_category_for_symbol(symbol)
            get_tick_size(symbol)

    def test_categories_cover_all_tv_symbols(self):
        """Test that categories include all TV-compatible symbols."""
        all_category_symbols = set(GRAINS + SOFTS + ENERGY + METALS + CRYPTO + INDEX + FOREX)
        tv_compatible_symbols = {s for s, spec in SYMBOL_SPECS.items() if spec['tv_compatible']}

        assert all_category_symbols == tv_compatible_symbols

    def test_symbol_specs_completeness(self):
        """Test that SYMBOL_SPECS has reasonable completeness."""
        assert len(SYMBOL_SPECS) >= 40  # Should have at least 40 symbols

        # Should have symbols in all major categories
        categories = {spec['category'] for spec in SYMBOL_SPECS.values()}
        assert 'Grains' in categories
        assert 'Energy' in categories
        assert 'Metals' in categories
        assert 'Crypto' in categories


class TestBackwardCompatibility:
    """Test backward compatibility with old futures_config.py."""

    def test_get_contract_multiplier_available(self):
        """Test that get_contract_multiplier function exists."""
        assert callable(get_contract_multiplier)
        assert get_contract_multiplier('ZS') == 50

    def test_get_exchange_for_symbol_available(self):
        """Test that get_exchange_for_symbol function exists."""
        assert callable(get_exchange_for_symbol)
        assert get_exchange_for_symbol('ZS') == 'CBOT'

    def test_symbol_specs_dict_available(self):
        """Test that SYMBOL_SPECS dictionary is available."""
        assert isinstance(SYMBOL_SPECS, dict)
        assert 'ZS' in SYMBOL_SPECS
        assert 'CL' in SYMBOL_SPECS


class TestRealWorldScenarios:
    """Test complete real-world scenarios."""

    @pytest.mark.parametrize("tv_symbol,expected_category,expected_exchange", [
        ('ZS', 'Grains', 'CBOT'),
        ('CL', 'Energy', 'NYMEX'),
        ('GC', 'Metals', 'COMEX'),
        ('XC', 'Grains', 'CBOT'),
        ('SIL', 'Metals', 'COMEX'),
    ])
    def test_complete_symbol_lookup(self, tv_symbol, expected_category, expected_exchange):
        """Test complete symbol lookup workflow."""
        # Verify TV compatible
        assert is_tradingview_compatible(tv_symbol)

        # Get category and exchange
        category = get_category_for_symbol(tv_symbol)
        exchange = get_exchange_for_symbol(tv_symbol)

        assert category == expected_category
        assert exchange == expected_exchange

        # Map to IBKR if needed
        ibkr_symbol = map_tv_to_ibkr(tv_symbol)
        assert ibkr_symbol is not None

    def test_multi_symbol_portfolio_setup(self):
        """Test setting up multi-symbol portfolio."""
        portfolio_symbols = ['ZS', 'CL', 'GC', 'XC']

        portfolio_info = {}
        for symbol in portfolio_symbols:
            portfolio_info[symbol] = {
                'tv_compatible': is_tradingview_compatible(symbol),
                'category': get_category_for_symbol(symbol),
                'exchange': get_exchange_for_symbol(symbol),
                'ibkr_symbol': map_tv_to_ibkr(symbol),
                'tick_size': get_tick_size(symbol),
                'multiplier': get_contract_multiplier(symbol),
            }

        # All should be TV compatible
        assert all(info['tv_compatible'] for info in portfolio_info.values())

        # Should have diverse categories
        categories = {info['category'] for info in portfolio_info.values()}
        assert len(categories) >= 2  # At least 2 different categories
