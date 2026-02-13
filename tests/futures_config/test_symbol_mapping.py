"""
Tests for Symbol Mapping Module.

Tests cover:
- TradingView to IBKR symbol mapping
- IBKR to TradingView symbol mapping (reverse)
- Mapping function behavior
- Bidirectional mapping consistency
- Edge cases and unmapped symbols

All tests use actual symbol mappings between TradingView and IBKR.
"""
import pytest

from futures_config.symbol_mapping import (
    TV_TO_IBKR_MAPPING,
    IBKR_TO_TV_MAPPING,
    map_tv_to_ibkr,
    map_ibkr_to_tv,
)


class TestMappingDictionaries:
    """Test mapping dictionary structure and content."""

    def test_tv_to_ibkr_mapping_not_empty(self):
        """Test that TV_TO_IBKR_MAPPING contains mappings."""
        assert len(TV_TO_IBKR_MAPPING) > 0
        assert isinstance(TV_TO_IBKR_MAPPING, dict)

    def test_ibkr_to_tv_mapping_not_empty(self):
        """Test that IBKR_TO_TV_MAPPING contains mappings."""
        assert len(IBKR_TO_TV_MAPPING) > 0
        assert isinstance(IBKR_TO_TV_MAPPING, dict)

    def test_mapping_counts_equal(self):
        """Test that both mappings have same number of entries."""
        assert len(TV_TO_IBKR_MAPPING) == len(IBKR_TO_TV_MAPPING)

    def test_mappings_are_inverse(self):
        """Test that mappings are inverses of each other."""
        for tv_symbol, ibkr_symbol in TV_TO_IBKR_MAPPING.items():
            assert IBKR_TO_TV_MAPPING[ibkr_symbol] == tv_symbol, \
                f"Mapping not inverse for {tv_symbol} -> {ibkr_symbol}"


class TestMiniGrainsMappings:
    """Test mini grains symbol mappings."""

    def test_xc_maps_to_yc(self):
        """Test that XC (Mini Corn) maps to YC."""
        assert TV_TO_IBKR_MAPPING['XC'] == 'YC'
        assert IBKR_TO_TV_MAPPING['YC'] == 'XC'

    def test_xw_maps_to_yw(self):
        """Test that XW (Mini Wheat) maps to YW."""
        assert TV_TO_IBKR_MAPPING['XW'] == 'YW'
        assert IBKR_TO_TV_MAPPING['YW'] == 'XW'

    def test_xk_maps_to_yk(self):
        """Test that XK (Mini Soybeans) maps to YK."""
        assert TV_TO_IBKR_MAPPING['XK'] == 'YK'
        assert IBKR_TO_TV_MAPPING['YK'] == 'XK'

    def test_all_mini_grains_present(self):
        """Test that all mini grains mappings are present."""
        assert 'XC' in TV_TO_IBKR_MAPPING
        assert 'XW' in TV_TO_IBKR_MAPPING
        assert 'XK' in TV_TO_IBKR_MAPPING


class TestMicroMetalsMappings:
    """Test micro metals symbol mappings."""

    def test_sil_maps_to_qi(self):
        """Test that SIL (Micro Silver) maps to QI."""
        assert TV_TO_IBKR_MAPPING['SIL'] == 'QI'
        assert IBKR_TO_TV_MAPPING['QI'] == 'SIL'

    def test_mgc_not_in_mapping(self):
        """Test that MGC is not in mapping (same on both platforms)."""
        assert 'MGC' not in TV_TO_IBKR_MAPPING
        assert 'MGC' not in IBKR_TO_TV_MAPPING


class TestMapTvToIbkr:
    """Test map_tv_to_ibkr function."""

    @pytest.mark.parametrize("tv_symbol,ibkr_symbol", [
        ('XC', 'YC'),
        ('XW', 'YW'),
        ('XK', 'YK'),
        ('SIL', 'QI'),
    ])
    def test_mapped_symbols(self, tv_symbol, ibkr_symbol):
        """Test that mapped symbols convert correctly."""
        assert map_tv_to_ibkr(tv_symbol) == ibkr_symbol

    @pytest.mark.parametrize("symbol", [
        'ZS', 'ZC', 'CL', 'NG', 'GC', 'SI', 'ES', 'NQ', 'BTC', 'ETH', 'MGC'
    ])
    def test_unmapped_symbols_return_unchanged(self, symbol):
        """Test that unmapped symbols return unchanged."""
        assert map_tv_to_ibkr(symbol) == symbol

    def test_unknown_symbol_returns_unchanged(self):
        """Test that unknown symbols return unchanged."""
        assert map_tv_to_ibkr('UNKNOWN') == 'UNKNOWN'
        assert map_tv_to_ibkr('TEST') == 'TEST'

    def test_none_input(self):
        """Test behavior with None input."""
        result = map_tv_to_ibkr(None)
        assert result is None

    def test_empty_string_input(self):
        """Test behavior with empty string input."""
        result = map_tv_to_ibkr('')
        assert result == ''

    def test_lowercase_symbol(self):
        """Test that lowercase symbols are not mapped."""
        # Mapping is case-sensitive
        assert map_tv_to_ibkr('xc') == 'xc'  # Not 'YC'


class TestMapIbkrToTv:
    """Test map_ibkr_to_tv function."""

    @pytest.mark.parametrize("ibkr_symbol,tv_symbol", [
        ('YC', 'XC'),
        ('YW', 'XW'),
        ('YK', 'XK'),
        ('QI', 'SIL'),
    ])
    def test_mapped_symbols(self, ibkr_symbol, tv_symbol):
        """Test that mapped symbols convert correctly."""
        assert map_ibkr_to_tv(ibkr_symbol) == tv_symbol

    @pytest.mark.parametrize("symbol", [
        'ZS', 'ZC', 'CL', 'NG', 'GC', 'SI', 'ES', 'NQ', 'BTC', 'ETH', 'MGC'
    ])
    def test_unmapped_symbols_return_unchanged(self, symbol):
        """Test that unmapped symbols return unchanged."""
        assert map_ibkr_to_tv(symbol) == symbol

    def test_unknown_symbol_returns_unchanged(self):
        """Test that unknown symbols return unchanged."""
        assert map_ibkr_to_tv('UNKNOWN') == 'UNKNOWN'
        assert map_ibkr_to_tv('TEST') == 'TEST'

    def test_none_input(self):
        """Test behavior with None input."""
        result = map_ibkr_to_tv(None)
        assert result is None

    def test_empty_string_input(self):
        """Test behavior with empty string input."""
        result = map_ibkr_to_tv('')
        assert result == ''


class TestBidirectionalMapping:
    """Test bidirectional mapping consistency."""

    def test_round_trip_tv_to_ibkr_to_tv(self):
        """Test that TV -> IBKR -> TV returns original."""
        for tv_symbol in TV_TO_IBKR_MAPPING.keys():
            ibkr_symbol = map_tv_to_ibkr(tv_symbol)
            result = map_ibkr_to_tv(ibkr_symbol)
            assert result == tv_symbol, f"Round trip failed for {tv_symbol}"

    def test_round_trip_ibkr_to_tv_to_ibkr(self):
        """Test that IBKR -> TV -> IBKR returns original."""
        for ibkr_symbol in IBKR_TO_TV_MAPPING.keys():
            tv_symbol = map_ibkr_to_tv(ibkr_symbol)
            result = map_tv_to_ibkr(tv_symbol)
            assert result == ibkr_symbol, f"Round trip failed for {ibkr_symbol}"

    def test_unmapped_symbol_round_trip(self):
        """Test that unmapped symbols survive round trips."""
        unmapped_symbols = ['ZS', 'CL', 'GC', 'ES']

        for symbol in unmapped_symbols:
            # TV -> IBKR -> TV
            assert map_ibkr_to_tv(map_tv_to_ibkr(symbol)) == symbol
            # IBKR -> TV -> IBKR
            assert map_tv_to_ibkr(map_ibkr_to_tv(symbol)) == symbol


class TestMappingUseCases:
    """Test real-world mapping use cases."""

    def test_order_placement_scenario(self):
        """Test mapping for order placement from TradingView to IBKR."""
        # TradingView sends alert with XC
        tv_symbol = 'XC'
        ibkr_symbol = map_tv_to_ibkr(tv_symbol)

        # Should be converted to YC for IBKR order
        assert ibkr_symbol == 'YC'

    def test_position_tracking_scenario(self):
        """Test mapping for position tracking from IBKR to TradingView."""
        # IBKR reports position with YC
        ibkr_symbol = 'YC'
        tv_symbol = map_ibkr_to_tv(ibkr_symbol)

        # Should be converted to XC for TradingView comparison
        assert tv_symbol == 'XC'

    def test_unmapped_order_placement(self):
        """Test that unmapped symbols pass through for order placement."""
        # TradingView sends alert with ZS
        tv_symbol = 'ZS'
        ibkr_symbol = map_tv_to_ibkr(tv_symbol)

        # Should remain ZS (no mapping needed)
        assert ibkr_symbol == 'ZS'

    def test_micro_silver_order_placement(self):
        """Test micro silver mapping for order placement."""
        # TradingView uses SIL
        tv_symbol = 'SIL'
        ibkr_symbol = map_tv_to_ibkr(tv_symbol)

        # IBKR expects QI
        assert ibkr_symbol == 'QI'


class TestMappingCompleteness:
    """Test mapping completeness and consistency."""

    def test_no_self_mappings(self):
        """Test that no symbol maps to itself."""
        for tv_symbol, ibkr_symbol in TV_TO_IBKR_MAPPING.items():
            assert tv_symbol != ibkr_symbol, f"Symbol {tv_symbol} maps to itself"

    def test_no_duplicate_targets(self):
        """Test that no two TV symbols map to same IBKR symbol."""
        ibkr_targets = list(TV_TO_IBKR_MAPPING.values())
        assert len(ibkr_targets) == len(set(ibkr_targets)), "Duplicate IBKR targets found"

    def test_no_duplicate_sources(self):
        """Test that no two IBKR symbols map to same TV symbol."""
        tv_targets = list(IBKR_TO_TV_MAPPING.values())
        assert len(tv_targets) == len(set(tv_targets)), "Duplicate TV targets found"

    def test_all_symbols_uppercase(self):
        """Test that all mapped symbols are uppercase."""
        for tv_symbol, ibkr_symbol in TV_TO_IBKR_MAPPING.items():
            assert tv_symbol.isupper(), f"TV symbol {tv_symbol} not uppercase"
            assert ibkr_symbol.isupper(), f"IBKR symbol {ibkr_symbol} not uppercase"


class TestMappingDocumentation:
    """Test that mappings match documented differences."""

    def test_mini_grains_have_x_to_y_prefix(self):
        """Test that mini grains follow X -> Y prefix pattern."""
        mini_grains_tv = ['XC', 'XW', 'XK']

        for tv_symbol in mini_grains_tv:
            ibkr_symbol = TV_TO_IBKR_MAPPING[tv_symbol]
            # Should start with Y instead of X
            assert tv_symbol.startswith('X')
            assert ibkr_symbol.startswith('Y')
            assert tv_symbol[1:] == ibkr_symbol[1:], f"Suffix mismatch for {tv_symbol}"

    def test_only_known_differences_mapped(self):
        """Test that only known symbol differences are in mapping."""
        # Only 4 known differences: XC, XW, XK, SIL
        expected_tv_symbols = {'XC', 'XW', 'XK', 'SIL'}
        actual_tv_symbols = set(TV_TO_IBKR_MAPPING.keys())

        assert actual_tv_symbols == expected_tv_symbols, \
            f"Unexpected mappings found: {actual_tv_symbols - expected_tv_symbols}"

    def test_mapping_count(self):
        """Test that we have exactly 4 mappings."""
        assert len(TV_TO_IBKR_MAPPING) == 4
        assert len(IBKR_TO_TV_MAPPING) == 4
