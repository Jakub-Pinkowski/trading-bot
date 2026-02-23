"""
Tests for Categories Module.

Tests cover:
- Category list generation and content
- TradingView compatibility filtering
- Category organization and completeness
- CATEGORIES dictionary structure
- Symbol presence in correct categories

All tests validate auto-generated category lists from SYMBOL_SPECS.
"""
import pytest

from futures_config.categories import (
    GRAINS,
    SOFTS,
    ENERGY,
    METALS,
    CRYPTO,
    INDEX,
    FOREX,
    CATEGORIES,
)
from futures_config.symbol_specs import SYMBOL_SPECS


class TestCategoryLists:
    """Test category list structure and content."""

    def test_all_category_lists_exist(self):
        """Test that all category lists are defined."""
        assert GRAINS is not None
        assert SOFTS is not None
        assert ENERGY is not None
        assert METALS is not None
        assert CRYPTO is not None
        assert INDEX is not None
        assert FOREX is not None

    def test_category_lists_are_lists(self):
        """Test that all categories are lists."""
        assert isinstance(GRAINS, list)
        assert isinstance(SOFTS, list)
        assert isinstance(ENERGY, list)
        assert isinstance(METALS, list)
        assert isinstance(CRYPTO, list)
        assert isinstance(INDEX, list)
        assert isinstance(FOREX, list)

    def test_category_lists_not_empty(self):
        """Test that category lists are not empty."""
        assert len(GRAINS) > 0
        assert len(SOFTS) > 0
        assert len(ENERGY) > 0
        assert len(METALS) > 0
        assert len(CRYPTO) > 0
        assert len(INDEX) > 0
        assert len(FOREX) > 0

    def test_category_lists_are_sorted(self):
        """Test that category lists are sorted alphabetically."""
        assert GRAINS == sorted(GRAINS)
        assert SOFTS == sorted(SOFTS)
        assert ENERGY == sorted(ENERGY)
        assert METALS == sorted(METALS)
        assert CRYPTO == sorted(CRYPTO)
        assert INDEX == sorted(INDEX)
        assert FOREX == sorted(FOREX)


class TestGrainsCategory:
    """Test Grains category."""

    def test_grains_major_symbols(self):
        """Test that major grains symbols are present."""
        assert 'ZS' in GRAINS  # Soybeans
        assert 'ZC' in GRAINS  # Corn
        assert 'ZW' in GRAINS  # Wheat

    def test_grains_mini_symbols(self):
        """Test that mini grains symbols are present."""
        assert 'XC' in GRAINS  # Mini Corn
        assert 'XW' in GRAINS  # Mini Wheat
        assert 'XK' in GRAINS  # Mini Soybeans

    def test_grains_micro_symbols(self):
        """Test that micro grains symbols are present."""
        assert 'MZC' in GRAINS  # Micro Corn
        assert 'MZW' in GRAINS  # Micro Wheat
        assert 'MZS' in GRAINS  # Micro Soybeans

    def test_all_grains_from_symbol_specs(self):
        """Test that all grains match SYMBOL_SPECS."""
        expected_grains = sorted([
            s for s, spec in SYMBOL_SPECS.items()
            if spec['category'] == 'Grains' and spec['tv_compatible']
        ])
        assert GRAINS == expected_grains


class TestSoftsCategory:
    """Test Softs category."""

    def test_softs_major_symbols(self):
        """Test that major softs symbols are present."""
        assert 'SB' in SOFTS  # Sugar
        assert 'KC' in SOFTS  # Coffee
        assert 'CC' in SOFTS  # Cocoa

    def test_all_softs_from_symbol_specs(self):
        """Test that all softs match SYMBOL_SPECS."""
        expected_softs = sorted([
            s for s, spec in SYMBOL_SPECS.items()
            if spec['category'] == 'Softs' and spec['tv_compatible']
        ])
        assert SOFTS == expected_softs


class TestEnergyCategory:
    """Test Energy category."""

    def test_energy_major_symbols(self):
        """Test that major energy symbols are present."""
        assert 'CL' in ENERGY  # Crude Oil
        assert 'NG' in ENERGY  # Natural Gas

    def test_energy_micro_symbols(self):
        """Test that micro energy symbols are present."""
        assert 'MCL' in ENERGY  # Micro Crude Oil
        assert 'MNG' in ENERGY  # Micro Natural Gas

    def test_all_energy_from_symbol_specs(self):
        """Test that all energy symbols match SYMBOL_SPECS."""
        expected_energy = sorted([
            s for s, spec in SYMBOL_SPECS.items()
            if spec['category'] == 'Energy' and spec['tv_compatible']
        ])
        assert ENERGY == expected_energy


class TestMetalsCategory:
    """Test Metals category."""

    def test_metals_major_symbols(self):
        """Test that major metals symbols are present."""
        assert 'GC' in METALS  # Gold
        assert 'HG' in METALS  # Copper
        assert 'PL' in METALS  # Platinum

    def test_metals_micro_symbols(self):
        """Test that micro metals symbols are present."""
        assert 'SIL' in METALS  # Micro Silver

    def test_all_metals_from_symbol_specs(self):
        """Test that all metals match SYMBOL_SPECS."""
        expected_metals = sorted([
            s for s, spec in SYMBOL_SPECS.items()
            if spec['category'] == 'Metals' and spec['tv_compatible']
        ])
        assert METALS == expected_metals


class TestCryptoCategory:
    """Test Crypto category."""

    def test_crypto_major_symbols(self):
        """Test that major crypto symbols are present."""
        assert 'BTC' in CRYPTO  # Bitcoin
        assert 'ETH' in CRYPTO  # Ethereum

    def test_crypto_micro_symbols(self):
        """Test that micro crypto symbols are present."""
        assert 'MET' in CRYPTO  # Micro Ethereum

    def test_all_crypto_from_symbol_specs(self):
        """Test that all crypto symbols match SYMBOL_SPECS."""
        expected_crypto = sorted([
            s for s, spec in SYMBOL_SPECS.items()
            if spec['category'] == 'Crypto' and spec['tv_compatible']
        ])
        assert CRYPTO == expected_crypto


class TestIndexCategory:
    """Test Index category."""

    def test_index_major_symbols(self):
        """Test that major index symbols are present."""
        assert 'YM' in INDEX  # E-mini Dow
        assert 'ZB' in INDEX  # 30-Year T-Bond

    def test_index_micro_symbols(self):
        """Test that micro index symbols are present."""
        assert 'MYM' in INDEX  # Micro E-mini Dow

    def test_es_not_in_index(self):
        """Test that ES is not in INDEX (not TV compatible)."""
        assert 'ES' not in INDEX

    def test_all_index_from_symbol_specs(self):
        """Test that all index symbols match SYMBOL_SPECS."""
        expected_index = sorted([
            s for s, spec in SYMBOL_SPECS.items()
            if spec['category'] == 'Index' and spec['tv_compatible']
        ])
        assert INDEX == expected_index


class TestForexCategory:
    """Test Forex category."""

    def test_forex_major_symbols(self):
        """Test that major forex symbols are present."""
        assert '6E' in FOREX  # Euro FX
        assert '6J' in FOREX  # Japanese Yen
        assert '6B' in FOREX  # British Pound
        assert '6A' in FOREX  # Australian Dollar

    def test_all_forex_from_symbol_specs(self):
        """Test that all forex symbols match SYMBOL_SPECS."""
        expected_forex = sorted([
            s for s, spec in SYMBOL_SPECS.items()
            if spec['category'] == 'Forex' and spec['tv_compatible']
        ])
        assert FOREX == expected_forex


class TestCategoriesDictionary:
    """Test CATEGORIES dictionary."""

    def test_categories_dict_exists(self):
        """Test that CATEGORIES dictionary is defined."""
        assert CATEGORIES is not None
        assert isinstance(CATEGORIES, dict)

    def test_categories_dict_has_all_keys(self):
        """Test that CATEGORIES has all expected keys."""
        expected_keys = {'Grains', 'Softs', 'Energy', 'Metals', 'Crypto', 'Index', 'Forex'}
        assert set(CATEGORIES.keys()) == expected_keys

    def test_categories_dict_values_match_lists(self):
        """Test that CATEGORIES values match individual lists."""
        assert CATEGORIES['Grains'] == GRAINS
        assert CATEGORIES['Softs'] == SOFTS
        assert CATEGORIES['Energy'] == ENERGY
        assert CATEGORIES['Metals'] == METALS
        assert CATEGORIES['Crypto'] == CRYPTO
        assert CATEGORIES['Index'] == INDEX
        assert CATEGORIES['Forex'] == FOREX

    @pytest.mark.parametrize("category_name", [
        'Grains', 'Softs', 'Energy', 'Metals', 'Crypto', 'Index', 'Forex'
    ])
    def test_category_access_by_name(self, category_name):
        """Test that categories can be accessed by name."""
        assert category_name in CATEGORIES
        assert isinstance(CATEGORIES[category_name], list)
        assert len(CATEGORIES[category_name]) > 0


class TestTvCompatibilityFiltering:
    """Test that only TV-compatible symbols are in categories."""

    def test_only_tv_compatible_in_grains(self):
        """Test that all grains are TV compatible."""
        for symbol in GRAINS:
            assert SYMBOL_SPECS[symbol]['tv_compatible'] is True

    def test_only_tv_compatible_in_softs(self):
        """Test that all softs are TV compatible."""
        for symbol in SOFTS:
            assert SYMBOL_SPECS[symbol]['tv_compatible'] is True

    def test_only_tv_compatible_in_energy(self):
        """Test that all energy symbols are TV compatible."""
        for symbol in ENERGY:
            assert SYMBOL_SPECS[symbol]['tv_compatible'] is True

    def test_only_tv_compatible_in_metals(self):
        """Test that all metals are TV compatible."""
        for symbol in METALS:
            assert SYMBOL_SPECS[symbol]['tv_compatible'] is True

    def test_only_tv_compatible_in_crypto(self):
        """Test that all crypto symbols are TV compatible."""
        for symbol in CRYPTO:
            assert SYMBOL_SPECS[symbol]['tv_compatible'] is True

    def test_only_tv_compatible_in_index(self):
        """Test that all index symbols are TV compatible."""
        for symbol in INDEX:
            assert SYMBOL_SPECS[symbol]['tv_compatible'] is True

    def test_only_tv_compatible_in_forex(self):
        """Test that all forex symbols are TV compatible."""
        for symbol in FOREX:
            assert SYMBOL_SPECS[symbol]['tv_compatible'] is True


class TestCategoryCompleteness:
    """Test that categories cover all TV-compatible symbols."""

    def test_all_tv_compatible_symbols_in_categories(self):
        """Test that every TV-compatible symbol is in a category."""
        all_category_symbols = set(GRAINS + SOFTS + ENERGY + METALS + CRYPTO + INDEX + FOREX)
        tv_compatible_symbols = {
            s for s, spec in SYMBOL_SPECS.items() if spec['tv_compatible']
        }

        assert all_category_symbols == tv_compatible_symbols

    def test_no_symbol_in_multiple_categories(self):
        """Test that no symbol appears in multiple categories."""
        all_symbols = GRAINS + SOFTS + ENERGY + METALS + CRYPTO + INDEX + FOREX
        assert len(all_symbols) == len(set(all_symbols)), "Symbol appears in multiple categories"

    def test_category_counts_sum_to_total(self):
        """Test that sum of category counts equals total TV-compatible symbols."""
        total_in_categories = (
                len(GRAINS) + len(SOFTS) + len(ENERGY) + len(METALS) +
                len(CRYPTO) + len(INDEX) + len(FOREX)
        )
        tv_compatible_count = sum(
            1 for spec in SYMBOL_SPECS.values() if spec['tv_compatible']
        )

        assert total_in_categories == tv_compatible_count


class TestCategoryUseCases:
    """Test real-world category usage scenarios."""

    def test_get_all_grains_symbols(self):
        """Test getting all grains symbols."""
        grains_symbols = CATEGORIES['Grains']
        assert 'ZS' in grains_symbols
        assert len(grains_symbols) > 0

    def test_iterate_through_all_categories(self):
        """Test iterating through all categories."""
        for category_name, symbols in CATEGORIES.items():
            assert isinstance(category_name, str)
            assert isinstance(symbols, list)
            assert len(symbols) > 0

    def test_check_if_symbol_in_category(self):
        """Test checking if a symbol is in a specific category."""
        assert 'ZS' in GRAINS
        assert 'CL' in ENERGY
        assert 'GC' in METALS
        assert 'BTC' in CRYPTO

    def test_filter_symbols_by_category(self):
        """Test filtering symbols by category."""
        # Get all energy symbols
        energy_symbols = CATEGORIES['Energy']
        assert 'CL' in energy_symbols
        assert 'NG' in energy_symbols
        assert 'ZS' not in energy_symbols  # Not an energy symbol
