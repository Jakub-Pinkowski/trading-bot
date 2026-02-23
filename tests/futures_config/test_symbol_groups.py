"""
Tests for Symbol Groups Module.

Test coverage for symbol grouping functionality that prevents pseudo-replication
in strategy analysis from correlated mini/micro contracts.
"""

from futures_config.symbol_groups import (
    SYMBOL_GROUPS,
    SYMBOL_TO_GROUP,
    get_group_for_symbol,
    get_symbols_in_group,
    are_symbols_correlated,
    get_representative_symbol,
    filter_to_one_per_group
)


# ==================== Test Data Fixtures ====================

class TestSymbolGroupsStructure:
    """Test the structure and integrity of SYMBOL_GROUPS."""

    def test_symbol_groups_not_empty(self):
        """Test that SYMBOL_GROUPS is not empty."""
        assert len(SYMBOL_GROUPS) > 0

    def test_all_groups_have_symbols(self):
        """Test that all groups contain at least one symbol."""
        for group_name, symbols in SYMBOL_GROUPS.items():
            assert len(symbols) > 0, f"Group {group_name} has no symbols"

    def test_corn_group_complete(self):
        """Test that corn group contains standard, mini, and micro."""
        corn_symbols = SYMBOL_GROUPS.get('Corn', [])
        assert 'ZC' in corn_symbols  # Standard
        assert 'XC' in corn_symbols  # Mini
        assert 'MZC' in corn_symbols  # Micro

    def test_grains_groups_present(self):
        """Test that all grain groups are present."""
        assert 'Corn' in SYMBOL_GROUPS
        assert 'Wheat' in SYMBOL_GROUPS
        assert 'Soybeans' in SYMBOL_GROUPS
        assert 'Soybean_Oil' in SYMBOL_GROUPS

    def test_energy_groups_present(self):
        """Test that energy groups are present."""
        assert 'Crude_Oil' in SYMBOL_GROUPS
        assert 'Natural_Gas' in SYMBOL_GROUPS

    def test_softs_groups_present(self):
        """Test that softs groups are present."""
        assert 'Sugar' in SYMBOL_GROUPS
        assert 'Coffee' in SYMBOL_GROUPS
        assert 'Cocoa' in SYMBOL_GROUPS

    def test_metals_groups_present(self):
        """Test that metal groups are present."""
        assert 'Gold' in SYMBOL_GROUPS
        assert 'Silver' in SYMBOL_GROUPS
        assert 'Copper' in SYMBOL_GROUPS
        assert 'Platinum' in SYMBOL_GROUPS

    def test_index_groups_present(self):
        """Test that index groups are present."""
        assert 'SP500' in SYMBOL_GROUPS
        assert 'NASDAQ' in SYMBOL_GROUPS
        assert 'Dow' in SYMBOL_GROUPS
        assert 'Russell' in SYMBOL_GROUPS
        assert 'T_Bond' in SYMBOL_GROUPS

    def test_forex_groups_present(self):
        """Test that forex groups are present."""
        assert 'Euro' in SYMBOL_GROUPS
        assert 'Japanese_Yen' in SYMBOL_GROUPS
        assert 'British_Pound' in SYMBOL_GROUPS
        assert 'Australian_Dollar' in SYMBOL_GROUPS
        assert 'Canadian_Dollar' in SYMBOL_GROUPS
        assert 'Swiss_Franc' in SYMBOL_GROUPS

    def test_all_symbol_specs_covered(self):
        """Test that every symbol in SYMBOL_SPECS has a group."""
        from futures_config.symbol_specs import SYMBOL_SPECS
        for symbol in SYMBOL_SPECS:
            assert symbol in SYMBOL_TO_GROUP, f"{symbol} from SYMBOL_SPECS has no group"

    def test_no_duplicate_symbols_across_groups(self):
        """Test that no symbol appears in multiple groups."""
        all_symbols = []
        for symbols in SYMBOL_GROUPS.values():
            all_symbols.extend(symbols)

        assert len(all_symbols) == len(set(all_symbols)), "Duplicate symbols found across groups"


class TestSymbolToGroupMapping:
    """Test the reverse mapping from symbols to groups."""

    def test_symbol_to_group_not_empty(self):
        """Test that SYMBOL_TO_GROUP is not empty."""
        assert len(SYMBOL_TO_GROUP) > 0

    def test_corn_symbols_map_to_corn(self):
        """Test that all corn symbols map to Corn group."""
        assert SYMBOL_TO_GROUP.get('ZC') == 'Corn'
        assert SYMBOL_TO_GROUP.get('XC') == 'Corn'
        assert SYMBOL_TO_GROUP.get('MZC') == 'Corn'

    def test_index_symbols_map_correctly(self):
        """Test that index symbols map to correct groups."""
        assert SYMBOL_TO_GROUP.get('ES') == 'SP500'
        assert SYMBOL_TO_GROUP.get('MES') == 'SP500'
        assert SYMBOL_TO_GROUP.get('NQ') == 'NASDAQ'
        assert SYMBOL_TO_GROUP.get('MNQ') == 'NASDAQ'

    def test_forex_symbols_map_correctly(self):
        """Test that forex symbols map to correct groups."""
        assert SYMBOL_TO_GROUP.get('6E') == 'Euro'
        assert SYMBOL_TO_GROUP.get('M6E') == 'Euro'
        assert SYMBOL_TO_GROUP.get('6B') == 'British_Pound'
        assert SYMBOL_TO_GROUP.get('M6B') == 'British_Pound'
        assert SYMBOL_TO_GROUP.get('6J') == 'Japanese_Yen'
        assert SYMBOL_TO_GROUP.get('6C') == 'Canadian_Dollar'

    def test_softs_symbols_map_correctly(self):
        """Test that softs symbols map to correct groups."""
        assert SYMBOL_TO_GROUP.get('SB') == 'Sugar'
        assert SYMBOL_TO_GROUP.get('KC') == 'Coffee'
        assert SYMBOL_TO_GROUP.get('CC') == 'Cocoa'

    def test_all_group_symbols_in_reverse_mapping(self):
        """Test that all symbols in SYMBOL_GROUPS are in SYMBOL_TO_GROUP."""
        for group_name, symbols in SYMBOL_GROUPS.items():
            for symbol in symbols:
                assert symbol in SYMBOL_TO_GROUP, f"{symbol} from {group_name} not in SYMBOL_TO_GROUP"
                assert SYMBOL_TO_GROUP[symbol] == group_name


# ==================== Function Tests ====================

class TestGetGroupForSymbol:
    """Test get_group_for_symbol function."""

    def test_get_group_for_corn_symbols(self):
        """Test getting group for corn symbols."""
        assert get_group_for_symbol('ZC') == 'Corn'
        assert get_group_for_symbol('XC') == 'Corn'
        assert get_group_for_symbol('MZC') == 'Corn'

    def test_get_group_for_gold_symbols(self):
        """Test getting group for gold symbols."""
        assert get_group_for_symbol('GC') == 'Gold'
        assert get_group_for_symbol('MGC') == 'Gold'

    def test_get_group_for_unknown_symbol(self):
        """Test getting group for symbol not in any group."""
        assert get_group_for_symbol('UNKNOWN') is None
        assert get_group_for_symbol('XYZ') is None


class TestGetSymbolsInGroup:
    """Test get_symbols_in_group function."""

    def test_get_corn_symbols(self):
        """Test getting all corn symbols."""
        corn_symbols = get_symbols_in_group('Corn')
        assert 'ZC' in corn_symbols
        assert 'XC' in corn_symbols
        assert 'MZC' in corn_symbols
        assert len(corn_symbols) == 3

    def test_get_sp500_symbols(self):
        """Test getting S&P 500 symbols."""
        sp500_symbols = get_symbols_in_group('SP500')
        assert 'ES' in sp500_symbols
        assert 'MES' in sp500_symbols

    def test_get_unknown_group(self):
        """Test getting symbols for non-existent group."""
        assert get_symbols_in_group('NonExistent') == []


class TestAreSymbolsCorrelated:
    """Test are_symbols_correlated function."""

    def test_corn_symbols_are_correlated(self):
        """Test that corn symbols are identified as correlated."""
        assert are_symbols_correlated('ZC', 'XC') is True
        assert are_symbols_correlated('ZC', 'MZC') is True
        assert are_symbols_correlated('XC', 'MZC') is True

    def test_different_markets_not_correlated(self):
        """Test that symbols from different markets are not correlated."""
        assert are_symbols_correlated('ZC', 'CL') is False  # Corn vs Oil
        assert are_symbols_correlated('ES', 'GC') is False  # S&P vs Gold
        assert are_symbols_correlated('NQ', 'ZW') is False  # NASDAQ vs Wheat

    def test_unknown_symbol_not_correlated(self):
        """Test that unknown symbols return False."""
        assert are_symbols_correlated('ZC', 'UNKNOWN') is False
        assert are_symbols_correlated('UNKNOWN1', 'UNKNOWN2') is False

    def test_same_symbol_is_correlated(self):
        """Test that a symbol is correlated with itself."""
        assert are_symbols_correlated('ZC', 'ZC') is True
        assert are_symbols_correlated('ES', 'ES') is True


class TestGetRepresentativeSymbol:
    """Test get_representative_symbol function."""

    def test_get_representative_corn(self):
        """Test getting representative corn symbol."""
        # Should return first symbol (typically standard size)
        rep = get_representative_symbol('Corn')
        assert rep in ['ZC', 'XC', 'MZC']
        # First symbol should be standard contract
        assert rep == 'ZC'

    def test_get_representative_sp500(self):
        """Test getting representative S&P 500 symbol."""
        rep = get_representative_symbol('SP500')
        assert rep in ['ES', 'MES']
        # First should be E-mini
        assert rep == 'ES'

    def test_get_representative_unknown_group(self):
        """Test getting representative for non-existent group."""
        assert get_representative_symbol('NonExistent') is None


class TestFilterToOnePerGroup:
    """Test filter_to_one_per_group function."""

    def test_filter_removes_duplicate_corn(self):
        """Test that duplicate corn contracts are filtered."""
        symbols = ['ZC', 'CL', 'MZC', 'ES']
        filtered = filter_to_one_per_group(symbols)

        # Should keep only one corn symbol
        corn_count = sum(1 for s in filtered if s in ['ZC', 'XC', 'MZC'])
        assert corn_count == 1

        # Should keep non-grouped symbols
        assert 'CL' in filtered
        assert 'ES' in filtered

    def test_filter_removes_duplicate_index(self):
        """Test that duplicate index contracts are filtered."""
        symbols = ['ES', 'MES', 'NQ', 'MNQ']
        filtered = filter_to_one_per_group(symbols)

        # Should have at most one S&P symbol
        sp500_count = sum(1 for s in filtered if s in ['ES', 'MES'])
        assert sp500_count == 1

        # Should have at most one NASDAQ symbol
        nasdaq_count = sum(1 for s in filtered if s in ['NQ', 'MNQ'])
        assert nasdaq_count == 1

    def test_filter_keeps_first_from_each_group(self):
        """Test that representative (standard) symbol from each group is kept."""
        symbols = ['ZC', 'XC', 'MZC', 'ES', 'MES', 'CL', 'MCL']
        filtered = filter_to_one_per_group(symbols)

        # Should keep representative of each group (always standard, not mini/micro)
        assert 'ZC' in filtered  # Standard corn (not XC, MZC)
        assert 'ES' in filtered  # Standard S&P (not MES)
        assert 'CL' in filtered  # Standard crude oil (not MCL)

        # Should not keep mini/micro variants
        assert 'XC' not in filtered
        assert 'MZC' not in filtered
        assert 'MES' not in filtered
        assert 'MCL' not in filtered

        # Test with mini/micro listed first - should still get standard
        symbols_reversed = ['MCL', 'CL', 'MES', 'ES', 'MZC', 'XC', 'ZC']
        filtered_reversed = filter_to_one_per_group(symbols_reversed)
        assert set(filtered) == set(filtered_reversed)
        assert 'ZC' in filtered_reversed
        assert 'ES' in filtered_reversed
        assert 'CL' in filtered_reversed
        assert 'MZC' not in filtered
        assert 'MES' not in filtered
        assert 'MCL' not in filtered

    def test_filter_empty_list(self):
        """Test filtering empty list."""
        assert filter_to_one_per_group([]) == []

    def test_filter_single_symbol(self):
        """Test filtering single symbol."""
        assert filter_to_one_per_group(['ZC']) == ['ZC']

    def test_filter_no_duplicates(self):
        """Test filtering list with no correlated symbols."""
        symbols = ['ZC', 'CL', 'GC', 'ES']  # All different markets
        filtered = filter_to_one_per_group(symbols)
        assert set(filtered) == set(symbols)

    def test_filter_preserves_order(self):
        """Test that filtering returns representative symbols deterministically."""
        symbols = ['CL', 'ZC', 'ES', 'MZC', 'NQ', 'MES']
        filtered = filter_to_one_per_group(symbols)

        # Should always return standard contracts: ZC (not MZC), ES (not MES)
        assert 'ZC' in filtered
        assert 'MZC' not in filtered
        assert 'ES' in filtered
        assert 'MES' not in filtered
        assert 'CL' in filtered
        assert 'NQ' in filtered

        # Result should be deterministic regardless of input order
        symbols_reversed = ['MES', 'NQ', 'MZC', 'ES', 'ZC', 'CL']
        filtered_reversed = filter_to_one_per_group(symbols_reversed)
        assert set(filtered) == set(filtered_reversed)

    def test_filter_real_world_scenario(self):
        """Test realistic scenario with multiple symbols."""
        symbols = [
            'ZC', 'XC', 'MZC',  # Corn (3 variations)
            'ZW', 'XW', 'MZW',  # Wheat (3 variations)
            'CL', 'MCL',  # Crude Oil (2 variations)
            'ES', 'MES',  # S&P 500 (2 variations)
            'GC', 'MGC',  # Gold (2 variations)
            'ZS',  # Soybeans (single)
        ]

        filtered = filter_to_one_per_group(symbols)

        # Should have exactly 6 symbols (one per market)
        assert len(filtered) == 6

        # Should have one from each market
        assert any(s in filtered for s in ['ZC', 'XC', 'MZC'])
        assert any(s in filtered for s in ['ZW', 'XW', 'MZW'])
        assert any(s in filtered for s in ['CL', 'MCL'])
        assert any(s in filtered for s in ['ES', 'MES'])
        assert any(s in filtered for s in ['GC', 'MGC'])
        assert 'ZS' in filtered


class TestSymbolGroupsIntegration:
    """Integration tests for symbol groups functionality."""

    def test_all_mini_contracts_identified(self):
        """Test that all mini contracts are properly grouped."""
        mini_symbols = ['XC', 'XW', 'XK']  # Mini grains

        for symbol in mini_symbols:
            group = get_group_for_symbol(symbol)
            assert group is not None, f"Mini symbol {symbol} not in any group"

            # Should be grouped with standard contracts
            group_symbols = get_symbols_in_group(group)
            assert len(group_symbols) > 1, f"Mini {symbol} should have standard version"

    def test_all_micro_contracts_identified(self):
        """Test that all micro contracts are properly grouped."""
        micro_symbols = ['MZC', 'MZW', 'MZS', 'MCL', 'MGC', 'MES', 'MNQ', 'M6E', 'M6A', 'M6B']

        for symbol in micro_symbols:
            group = get_group_for_symbol(symbol)
            assert group is not None, f"Micro symbol {symbol} not in any group"

            # Should be grouped with standard contracts
            group_symbols = get_symbols_in_group(group)
            assert len(group_symbols) > 1, f"Micro {symbol} should have standard version"

    def test_forex_micro_correlated_with_standard(self):
        """Test that micro forex symbols are correlated with their standard versions."""
        assert are_symbols_correlated('6E', 'M6E') is True
        assert are_symbols_correlated('6B', 'M6B') is True
        assert are_symbols_correlated('6A', 'M6A') is True

    def test_single_symbol_groups(self):
        """Test that standalone symbols (no mini/micro) form their own group."""
        standalone = ['SB', 'KC', 'CC', 'PL', 'ZB', '6J', '6C', '6S']

        for symbol in standalone:
            group = get_group_for_symbol(symbol)
            assert group is not None, f"{symbol} has no group"
            group_symbols = get_symbols_in_group(group)
            assert symbol in group_symbols
