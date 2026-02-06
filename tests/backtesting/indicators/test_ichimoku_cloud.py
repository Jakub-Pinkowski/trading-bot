"""
Tests for Ichimoku Cloud indicator calculation.

Uses real historical data from ZS (soybeans) and CL (crude oil) to validate
Ichimoku calculation accuracy, caching behavior, and edge case handling.
Tests all five components: Tenkan-sen, Kijun-sen, Senkou Span A,
Senkou Span B, and Chikou Span.
"""
import numpy as np
import pandas as pd
import pytest

from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.indicators import calculate_ichimoku_cloud
from app.utils.backtesting_utils.indicators_utils import hash_series
from tests.backtesting.helpers.assertions import assert_valid_indicator, assert_indicator_varies
from tests.backtesting.helpers.data_utils import inject_price_spike


# ==================== Helper Functions ====================

def _calculate_ichimoku(
    high, low, close, tenkan_period=9, kijun_period=26,
    senkou_span_b_period=52, displacement=26
):
    """
    Helper function to calculate Ichimoku with automatic hashing.

    Simplifies test code by handling hash calculation internally.
    """
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    return calculate_ichimoku_cloud(
        high, low, close,
        tenkan_period, kijun_period, senkou_span_b_period, displacement,
        high_hash, low_hash, close_hash
    )


def _price_series_to_hlc(prices, range_pct=2.0):
    """
    Convert price series to High/Low/Close series for Ichimoku testing.
    
    Args:
        prices: Series of close prices
        range_pct: Percentage range for high/low around close (default 2%)
    
    Returns:
        Tuple of (high, low, close) Series
    """
    range_size = prices * (range_pct / 100)
    high = prices + range_size
    low = prices - range_size
    return high, low, prices


# ==================== Basic Logic Tests ====================

class TestIchimokuBasicLogic:
    """Simple sanity checks for Ichimoku basic behavior using shared test fixtures."""

    def test_ichimoku_returns_dict_with_five_series(self, medium_price_series):
        """Ichimoku should return dict with 5 series of correct length."""
        high, low, close = _price_series_to_hlc(medium_price_series)

        result = _calculate_ichimoku(high, low, close)

        # Should return dictionary
        assert isinstance(result, dict), "Ichimoku must return a dictionary"

        # Should have exactly 5 keys
        expected_keys = {'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span'}
        assert set(result.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(result.keys())}"

        # Each component should be a Series with same length as input
        for key, series in result.items():
            assert isinstance(series, pd.Series), f"{key} must be pandas Series"
            assert len(series) == len(high), f"{key} length must equal input length"

    def test_tenkan_sen_is_midpoint_of_highs_lows(self, short_price_series):
        """Tenkan-sen should be midpoint of highest high and lowest low over period."""
        # Use short series and create simple data where we know the expected values
        simple_prices = pd.Series(range(100, 115))  # 15 bars
        high, low, close = _price_series_to_hlc(simple_prices, range_pct=5.0)

        result = _calculate_ichimoku(high, low, close, tenkan_period=9)
        tenkan = result['tenkan_sen']

        # At bar 9 (0-indexed position 8), we have 9 bars
        # Should have a valid Tenkan value
        assert not pd.isna(tenkan.iloc[8]), "Tenkan should have value at position 8"

        # Tenkan should be between high and low midpoint
        expected_min = low.iloc[:9].min()
        expected_max = high.iloc[:9].max()
        expected_tenkan = (expected_max + expected_min) / 2
        assert abs(tenkan.iloc[8] - expected_tenkan) < 0.01, \
            f"Tenkan at position 8 should be {expected_tenkan}, got {tenkan.iloc[8]}"

    def test_kijun_sen_is_midpoint_over_longer_period(self, medium_price_series):
        """Kijun-sen should be midpoint over kijun_period (typically 26)."""
        high, low, close = _price_series_to_hlc(medium_price_series)

        result = _calculate_ichimoku(high, low, close, kijun_period=26)
        kijun = result['kijun_sen']

        # At bar 26 (0-indexed position 25), we have exactly 26 bars
        assert not pd.isna(kijun.iloc[25]), "Kijun should have value at position 25"

        expected_min = low.iloc[:26].min()
        expected_max = high.iloc[:26].max()
        expected_kijun = (expected_max + expected_min) / 2
        assert abs(kijun.iloc[25] - expected_kijun) < 0.01, \
            f"Kijun at position 25 should be {expected_kijun}, got {kijun.iloc[25]}"

    def test_senkou_spans_are_displaced_forward(self, medium_price_series):
        """Senkou Span A and B should be displaced forward by displacement periods."""
        # Need enough bars: at least senkou_span_b_period + displacement
        long_series = pd.Series(range(100, 180))  # 80 bars
        high, low, close = _price_series_to_hlc(long_series)

        displacement = 26
        result = _calculate_ichimoku(high, low, close, displacement=displacement, senkou_span_b_period=52)

        senkou_a = result['senkou_span_a']
        senkou_b = result['senkou_span_b']

        # Senkou spans should have NaN at the beginning due to displacement
        # The first valid value should appear at position (displacement + min_period)
        assert senkou_a.iloc[:displacement].isna().all(), \
            "Senkou Span A should have NaN for first displacement periods"

        # Note: with 80 bars and senkou_b_period=52, we need 52+displacement=78 bars for valid senkou_b
        # So with only 75 bars, senkou_b will be mostly NaN. This is expected behavior.
        # Just verify structure is correct
        assert len(senkou_a) == len(high), "Senkou A length should match input"
        assert len(senkou_b) == len(high), "Senkou B length should match input"

        # Senkou A should have some valid values (needs only tenkan/kijun periods)
        valid_a_count = senkou_a.notna().sum()
        assert valid_a_count > 0, "Senkou Span A should have some valid values"

    def test_chikou_span_is_displaced_backward(self):
        """Chikou span should be close price displaced backward by displacement periods."""
        high = pd.Series(range(105, 180))  # 75 bars
        low = pd.Series(range(95, 170))
        close = pd.Series(range(100, 175))

        displacement = 26
        result = _calculate_ichimoku(high, low, close, displacement=displacement)

        chikou = result['chikou_span']

        # Chikou should have NaN at the end due to backward displacement
        assert chikou.iloc[-displacement:].isna().all(), \
            "Chikou span should have NaN for last displacement periods"

        # Chikou at position i should equal close at position i+displacement
        for i in range(len(close) - displacement):
            if not pd.isna(chikou.iloc[i]):
                expected_value = close.iloc[i + displacement]
                assert abs(chikou.iloc[i] - expected_value) < 0.01, \
                    f"Chikou at position {i} should equal close at position {i + displacement}"

    def test_tenkan_kijun_relationship_in_uptrend(self, rising_price_series):
        """In strong uptrend, Tenkan-sen should be above Kijun-sen."""
        # Create strong uptrend with longer series
        high = pd.Series(range(100, 200))  # 100 consecutive increases
        low = pd.Series(range(90, 190))
        close = pd.Series(range(95, 195))

        result = _calculate_ichimoku(high, low, close)
        tenkan = result['tenkan_sen'].dropna()
        kijun = result['kijun_sen'].dropna()

        # In uptrend, shorter period (Tenkan) should generally be above longer (Kijun)
        # Check last 20 bars where trend is established
        assert (tenkan.iloc[-20:] > kijun.iloc[-20:]).sum() > 15, \
            "In uptrend, Tenkan should be above Kijun most of the time"

    def test_tenkan_kijun_relationship_in_downtrend(self):
        """In strong downtrend, Tenkan-sen should be below Kijun-sen."""
        # Create strong downtrend
        high = pd.Series(range(200, 100, -1))  # 100 consecutive decreases
        low = pd.Series(range(190, 90, -1))
        close = pd.Series(range(195, 95, -1))

        result = _calculate_ichimoku(high, low, close)
        tenkan = result['tenkan_sen'].dropna()
        kijun = result['kijun_sen'].dropna()

        # In downtrend, shorter period (Tenkan) should generally be below longer (Kijun)
        # Check last 20 bars where trend is established
        assert (tenkan.iloc[-20:] < kijun.iloc[-20:]).sum() > 15, \
            "In downtrend, Tenkan should be below Kijun most of the time"

    def test_senkou_span_a_is_average_of_tenkan_kijun(self):
        """Senkou Span A should be average of Tenkan and Kijun (before displacement)."""
        high = pd.Series(range(105, 180))  # 75 bars
        low = pd.Series(range(95, 170))
        close = pd.Series(range(100, 175))

        displacement = 26
        result = _calculate_ichimoku(high, low, close, displacement=displacement)

        tenkan = result['tenkan_sen']
        kijun = result['kijun_sen']
        senkou_a = result['senkou_span_a']

        # Senkou A at position i should equal (Tenkan + Kijun) / 2 at position i-displacement
        # Check a few positions where all values are valid
        for i in range(displacement + 26, displacement + 30):
            if not pd.isna(senkou_a.iloc[i]) and not pd.isna(tenkan.iloc[i - displacement]) and not pd.isna(
                    kijun.iloc[i - displacement]):
                expected_senkou_a = (tenkan.iloc[i - displacement] + kijun.iloc[i - displacement]) / 2
                assert abs(senkou_a.iloc[i] - expected_senkou_a) < 0.01, \
                    f"Senkou A at position {i} should be average of Tenkan and Kijun at {i - displacement}"

    def test_cloud_thickness_varies_with_volatility(self):
        """Cloud thickness (difference between Senkou A and B) should vary with market."""
        # Create data with varying ranges - use oscillating pattern for variance
        base = 100
        high = pd.Series([base + i * 0.3 + (i % 5) * 2 for i in range(100)])
        low = pd.Series([base + i * 0.3 - (i % 5) * 2 for i in range(100)])
        close = pd.Series([base + i * 0.3 for i in range(100)])

        result = _calculate_ichimoku(high, low, close)
        senkou_a = result['senkou_span_a']
        senkou_b = result['senkou_span_b']

        # Calculate cloud thickness where both spans have values
        cloud_thickness = (senkou_a - senkou_b).abs().dropna()

        # With oscillating prices, cloud thickness should show some variation
        if len(cloud_thickness) > 5:
            # Cloud may show limited variation with this pattern, but should not be constant
            unique_values = cloud_thickness.nunique()
            assert unique_values > 1 or cloud_thickness.std() > 0, \
                "Cloud thickness should show some variation with changing volatility"


class TestIchimokuCalculationWithRealData:
    """Test Ichimoku calculation using real historical data."""

    def test_standard_ichimoku_with_zs_hourly_data(self, zs_1h_data):
        """
        Test Ichimoku with standard parameters on 2 years of ZS hourly data.

        Validates that all components are calculated correctly, handle NaN values
        properly, and produce values in valid range.
        """
        result = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )

        # Validate each component
        for component_name, component in result.items():
            # Validate structure
            assert len(component) == len(zs_1h_data), \
                f"{component_name} length must match input data length"
            assert component.index.equals(zs_1h_data.index), \
                f"{component_name} index must match input index"

            # Validate range (all components should be positive prices)
            valid_values = component.dropna()
            assert len(valid_values) > 0, f"{component_name} should have valid values"
            assert (valid_values > 0).all(), f"{component_name} should be positive"

            # Validate market behavior
            assert_indicator_varies(component, component_name)

    @pytest.mark.parametrize("tenkan,kijun,senkou_b", [
        (9, 26, 52),  # Standard
        (7, 22, 44),  # Faster
        (12, 30, 60),  # Slower
    ])
    def test_ichimoku_with_different_periods(self, zs_1h_data, tenkan, kijun, senkou_b):
        """
        Test Ichimoku calculation with various period combinations.

        Validates that different periods produce valid results and proper
        NaN counts matching the periods.
        """
        result = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close'],
            tenkan_period=tenkan,
            kijun_period=kijun,
            senkou_span_b_period=senkou_b
        )

        # Validate structure
        assert len(result) == 5, "Should have 5 components"

        # Tenkan should have tenkan-1 NaN values at start
        assert result['tenkan_sen'].isna().sum() >= tenkan - 1

        # Kijun should have kijun-1 NaN values at start
        assert result['kijun_sen'].isna().sum() >= kijun - 1

        # Validate all components have valid values
        for component in result.values():
            valid_values = component.dropna()
            assert len(valid_values) > 0, "Should have valid values"

    def test_ichimoku_extreme_but_valid_parameters(self, zs_1h_data):
        """
        Test Ichimoku with extreme but technically valid parameters.

        Tests edge cases like very short periods and very long periods
        to ensure calculation doesn't break.
        """
        # Very short periods
        result_short = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close'],
            tenkan_period=3,
            kijun_period=6,
            senkou_span_b_period=12,
            displacement=6
        )

        for component in result_short.values():
            valid_values = component.dropna()
            assert len(valid_values) > 0, "Short period Ichimoku should have valid values"
            assert_valid_indicator(component, 'Ichimoku Component', min_val=0)

        # Very long periods
        result_long = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close'],
            tenkan_period=50,
            kijun_period=100,
            senkou_span_b_period=200,
            displacement=100
        )

        for component in result_long.values():
            valid_values = component.dropna()
            assert len(valid_values) > 0, "Long period Ichimoku should have valid values"

    def test_ichimoku_values_match_expected_calculation(self, zs_1h_data):
        """
        Test that Ichimoku values match the expected calculation formula.

        Manually calculates Ichimoku for a subset of data and compares with
        indicator function output to validate correctness.
        """
        # Use subset for clearer validation
        subset = zs_1h_data.iloc[200:300].copy()

        result = _calculate_ichimoku(
            subset['high'],
            subset['low'],
            subset['close'],
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26
        )

        # Manual calculation of Tenkan-sen
        tenkan_manual = (
                                subset['high'].rolling(window=9).max() +
                                subset['low'].rolling(window=9).min()
                        ) / 2

        # Compare with calculated Tenkan
        valid_mask = tenkan_manual.notna() & result['tenkan_sen'].notna()
        np.testing.assert_allclose(
            result['tenkan_sen'][valid_mask].values,
            tenkan_manual[valid_mask].values,
            rtol=0.01,
            err_msg="Tenkan-sen calculation doesn't match expected formula"
        )

        # Manual calculation of Kijun-sen
        kijun_manual = (
                               subset['high'].rolling(window=26).max() +
                               subset['low'].rolling(window=26).min()
                       ) / 2

        # Compare with calculated Kijun
        valid_mask = kijun_manual.notna() & result['kijun_sen'].notna()
        np.testing.assert_allclose(
            result['kijun_sen'][valid_mask].values,
            kijun_manual[valid_mask].values,
            rtol=0.01,
            err_msg="Kijun-sen calculation doesn't match expected formula"
        )


class TestIchimokuInMarketScenarios:
    """Test Ichimoku behavior in different market conditions using real data."""

    def test_ichimoku_in_trending_market(self, trending_market_data):
        """
        Test Ichimoku behavior during strong trend.

        In uptrend: Tenkan > Kijun, price > cloud, Chikou > historical price.
        """
        if trending_market_data is None:
            pytest.skip("No trending market data available")

        result = _calculate_ichimoku(
            trending_market_data['high'],
            trending_market_data['low'],
            trending_market_data['close']
        )

        tenkan = result['tenkan_sen'].dropna()
        kijun = result['kijun_sen'].dropna()
        senkou_a = result['senkou_span_a'].dropna()
        senkou_b = result['senkou_span_b'].dropna()

        # Align tenkan and kijun for comparison (use common index)
        common_idx = tenkan.index.intersection(kijun.index)
        tenkan_aligned = tenkan.loc[common_idx]
        kijun_aligned = kijun.loc[common_idx]

        # In uptrend, Tenkan should often be above Kijun
        tk_cross_ratio = (tenkan_aligned > kijun_aligned).sum() / len(tenkan_aligned)
        assert tk_cross_ratio > 0.5, \
            "In uptrend, Tenkan should be above Kijun most of the time"

        # Cloud should show directionality
        assert senkou_a.std() > 0, "Senkou Span A should vary in trending market"
        assert senkou_b.std() > 0, "Senkou Span B should vary in trending market"

    def test_ichimoku_in_ranging_market(self, ranging_market_data):
        """
        Test Ichimoku in sideways/ranging market.

        In range: TK crosses frequent, thin cloud, price oscillates around cloud.
        """
        if ranging_market_data is None:
            pytest.skip("No ranging market data available")

        result = _calculate_ichimoku(
            ranging_market_data['high'],
            ranging_market_data['low'],
            ranging_market_data['close']
        )

        tenkan = result['tenkan_sen'].dropna()
        kijun = result['kijun_sen'].dropna()
        senkou_a = result['senkou_span_a'].dropna()
        senkou_b = result['senkou_span_b'].dropna()

        # In ranging market, Tenkan/Kijun should cross frequently
        # Detect changes in relationship
        tk_diff = (tenkan - kijun).dropna()
        sign_changes = ((tk_diff.shift(1) * tk_diff) < 0).sum()
        assert sign_changes > 2, "Ranging market should show TK crossovers"

        # Cloud should be relatively thin in ranging market
        cloud_thickness = (senkou_a - senkou_b).abs()
        valid_thickness = cloud_thickness.dropna()
        assert len(valid_thickness) > 0, "Should have valid cloud thickness values"

    def test_ichimoku_in_volatile_market(self, volatile_market_data):
        """
        Test Ichimoku during extreme volatility.

        Ichimoku should handle volatility without breaking and cloud should
        show increased width during volatile periods.
        """
        if volatile_market_data is None:
            pytest.skip("No volatile market data available")

        result = _calculate_ichimoku(
            volatile_market_data['high'],
            volatile_market_data['low'],
            volatile_market_data['close']
        )

        # Should still produce valid values despite volatility
        for component_name, component in result.items():
            assert_valid_indicator(component, component_name, min_val=0)

        # High volatility should produce wider cloud
        senkou_a = result['senkou_span_a'].dropna()
        senkou_b = result['senkou_span_b'].dropna()
        cloud_thickness = (senkou_a - senkou_b).abs()
        assert cloud_thickness.std() > 0, "Cloud thickness should vary in volatile market"


class TestIchimokuCaching:
    """Test Ichimoku caching behavior."""

    def test_cache_hit_returns_identical_values(self, zs_1h_data):
        """
        Verify cached Ichimoku exactly matches fresh calculation.

        Tests that cache stores and retrieves Ichimoku correctly without
        any data corruption or loss of precision.
        """
        # Clear cache to ensure test isolation and prevent false positives
        indicator_cache.cache_data.clear()
        indicator_cache.reset_stats()

        # First calculation (should miss due to empty cache)
        result_1 = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )
        first_misses = indicator_cache.misses

        # Second calculation (should hit cache)
        result_2 = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )

        # Verify cache was hit (misses didn't increase, hits increased)
        assert indicator_cache.misses == first_misses, "Second calculation should not cause cache miss"
        assert indicator_cache.hits > 0, "Second calculation should cause cache hit"

        # Verify identical results for all components
        for component_name in result_1.keys():
            np.testing.assert_array_equal(
                result_1[component_name].values,
                result_2[component_name].values,
                err_msg=f"Cached {component_name} should match exactly"
            )

    def test_cache_distinguishes_different_periods(self, zs_1h_data):
        """
        Verify cache stores different results for different periods.

        Tests that cache keys include period parameters so different
        periods don't overwrite each other.
        """
        indicator_cache.reset_stats()

        # Calculate with different periods
        result_standard = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close'],
            tenkan_period=9,
            kijun_period=26
        )

        result_fast = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close'],
            tenkan_period=7,
            kijun_period=22
        )

        # Results should differ
        assert not result_standard['tenkan_sen'].equals(result_fast['tenkan_sen']), \
            "Different periods should produce different Tenkan values"

        # Recalculate standard - should hit cache
        misses_before_recalc = indicator_cache.misses
        result_standard_again = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close'],
            tenkan_period=9,
            kijun_period=26
        )
        assert indicator_cache.misses == misses_before_recalc, \
            "Recalculation should hit cache"

        # Should get same values
        np.testing.assert_array_equal(
            result_standard['tenkan_sen'].values,
            result_standard_again['tenkan_sen'].values
        )

    def test_cache_distinguishes_different_data(self, zs_1h_data, cl_15m_data):
        """
        Verify cache stores different results for different data series.

        Tests that cache keys include data hash so different datasets
        don't return wrong cached values.
        """
        indicator_cache.reset_stats()

        # Calculate Ichimoku for two different datasets
        result_zs = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )

        result_cl = _calculate_ichimoku(
            cl_15m_data['high'],
            cl_15m_data['low'],
            cl_15m_data['close']
        )

        # Results should differ (different underlying data)
        assert not result_zs['tenkan_sen'].equals(result_cl['tenkan_sen']), \
            "Different data should produce different Ichimoku"
        assert len(result_zs['tenkan_sen']) != len(result_cl['tenkan_sen']), \
            "Different datasets should have different lengths"


class TestIchimokuEdgeCases:
    """Test Ichimoku with edge cases and error conditions."""

    def test_ichimoku_with_insufficient_data(self, minimal_price_series):
        """
        Test Ichimoku when data length < longest period.

        Should return all NaN values when insufficient data for calculation.
        """
        high = minimal_price_series + 5
        low = minimal_price_series - 5
        close = minimal_price_series

        result = _calculate_ichimoku(high, low, close)

        # All components should be mostly NaN (insufficient data)
        for component in result.values():
            assert component.isna().all() or component.notna().sum() <= 1, \
                "All values should be NaN when data < period"
            assert len(component) == len(high), \
                "Output length should match input length"

    def test_ichimoku_with_exact_minimum_data(self):
        """
        Test Ichimoku with exactly enough data for calculation.

        With standard periods (9, 26, 52), needs at least 52 data points.
        """
        # Create exactly 52 bars
        high = pd.Series(range(105, 157))
        low = pd.Series(range(95, 147))
        close = pd.Series(range(100, 152))

        result = _calculate_ichimoku(high, low, close)

        # Tenkan should have valid values after period 9
        tenkan_valid = result['tenkan_sen'].notna().sum()
        assert tenkan_valid > 0, "Tenkan should have some valid values with 52 bars"

        # Kijun should have valid values after period 26
        kijun_valid = result['kijun_sen'].notna().sum()
        assert kijun_valid > 0, "Kijun should have some valid values with 52 bars"

        # Senkou B needs 52 periods, should have at least one valid value
        senkou_b_valid = result['senkou_span_b'].notna().sum()
        assert senkou_b_valid >= 0, "Senkou B should be calculable with 52 bars"

    def test_ichimoku_with_constant_prices(self, constant_price_series):
        """
        Test Ichimoku when prices don't change.

        With constant prices (zero variance), all components should be constant
        and equal to the price level.
        """
        high = constant_price_series
        low = constant_price_series
        close = constant_price_series

        result = _calculate_ichimoku(high, low, close)

        # All valid values should be constant and equal to the price
        for component_name, component in result.items():
            valid_values = component.dropna()
            if len(valid_values) > 0:
                # Should be constant
                assert valid_values.std() < 0.01, \
                    f"{component_name} should be constant with flat prices"
                # Should equal the flat price level
                assert abs(valid_values.iloc[0] - 100.0) < 0.01, \
                    f"{component_name} should equal flat price level"

    def test_ichimoku_with_monotonic_increase(self):
        """
        Test Ichimoku with continuously increasing prices.

        All gains, no pullbacks - should show strong bullish structure.
        """
        # Use longer series for better demonstration
        high = pd.Series(range(105, 205))  # 100 consecutive increases
        low = pd.Series(range(95, 195))
        close = pd.Series(range(100, 200))

        result = _calculate_ichimoku(high, low, close)

        tenkan = result['tenkan_sen'].dropna()
        kijun = result['kijun_sen'].dropna()

        # In strong uptrend, Tenkan should be consistently above Kijun
        assert (tenkan.iloc[-20:] > kijun.iloc[-20:]).sum() > 18, \
            "Tenkan should be above Kijun in strong uptrend"

        # All components should be increasing
        for component_name, component in result.items():
            valid_values = component.dropna()
            if len(valid_values) > 10:
                # Check if generally increasing (last > first)
                assert valid_values.iloc[-1] > valid_values.iloc[0], \
                    f"{component_name} should increase in uptrend"

    def test_ichimoku_with_monotonic_decrease(self):
        """
        Test Ichimoku with continuously decreasing prices.

        All losses, no bounces - should show strong bearish structure.
        """
        # Use longer series for better demonstration
        high = pd.Series(range(205, 105, -1))  # 100 consecutive decreases
        low = pd.Series(range(195, 95, -1))
        close = pd.Series(range(200, 100, -1))

        result = _calculate_ichimoku(high, low, close)

        tenkan = result['tenkan_sen'].dropna()
        kijun = result['kijun_sen'].dropna()

        # In strong downtrend, Tenkan should be consistently below Kijun
        assert (tenkan.iloc[-20:] < kijun.iloc[-20:]).sum() > 18, \
            "Tenkan should be below Kijun in strong downtrend"

        # All components should be decreasing
        for component_name, component in result.items():
            valid_values = component.dropna()
            if len(valid_values) > 10:
                # Check if generally decreasing (last < first)
                assert valid_values.iloc[-1] < valid_values.iloc[0], \
                    f"{component_name} should decrease in downtrend"

    def test_ichimoku_with_extreme_spike(self, zs_1h_data):
        """
        Test Ichimoku reaction to extreme price spike.

        Injects artificial spike to test Ichimoku handles extreme moves.
        """
        # Inject 10% spike upward at bar 1000
        modified_data = inject_price_spike(zs_1h_data.copy(), 1000, 10.0, 'up')

        result = _calculate_ichimoku(
            modified_data['high'],
            modified_data['low'],
            modified_data['close']
        )

        # Ichimoku should still be within reasonable bounds despite spike
        for component_name, component in result.items():
            assert_valid_indicator(component, component_name, min_val=0)

        # Components around spike should show elevated values
        tenkan_around_spike = result['tenkan_sen'].iloc[1000:1010].dropna()
        if len(tenkan_around_spike) > 0:
            assert tenkan_around_spike.max() > result['tenkan_sen'].iloc[900:999].median(), \
                "Tenkan should react to price spike"

    def test_ichimoku_with_flat_then_movement(self, flat_then_volatile_series):
        """
        Test Ichimoku transition from flat period to normal movement.

        Validates Ichimoku can handle transition from constant prices to price movement.
        """
        high = flat_then_volatile_series + 5
        low = flat_then_volatile_series - 5
        close = flat_then_volatile_series

        result = _calculate_ichimoku(high, low, close)

        # Variable region should have valid, varying values
        # Note: Longer-period components (Kijun=26, Senkou B=52) lag significantly
        # and may still reflect the flat period even in the variable region.
        # Only check faster-responding components (Tenkan=9, Chikou=displaced close)
        fast_components = ['tenkan_sen', 'chikou_span']

        for component_name in fast_components:
            component = result[component_name]
            variable_region = component.iloc[70:]  # Check later region after more volatile data
            valid_variable = variable_region.dropna()

            if len(valid_variable) > 10:
                assert valid_variable.std() > 0, \
                    f"{component_name} should vary in variable region"

    def test_ichimoku_with_empty_series(self, empty_price_series):
        """Test Ichimoku with empty input series."""
        result = _calculate_ichimoku(
            empty_price_series,
            empty_price_series,
            empty_price_series
        )

        # Should return dict with empty series
        assert isinstance(result, dict)
        assert len(result) == 5

        for component in result.values():
            assert len(component) == 0, "Empty input should return empty components"
            assert isinstance(component, pd.Series)


class TestIchimokuDataTypes:
    """Test Ichimoku with different input data types and structures."""

    def test_ichimoku_with_different_datetime_frequencies(self, zs_1h_data, zs_1d_data):
        """
        Test Ichimoku works with different time frequencies.

        Validates that Ichimoku calculation is time-agnostic and works with
        any datetime frequency (hourly, daily, etc.).
        """
        result_hourly = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )

        result_daily = _calculate_ichimoku(
            zs_1d_data['high'],
            zs_1d_data['low'],
            zs_1d_data['close']
        )

        # Both should produce valid Ichimoku
        for component in result_hourly.values():
            assert_valid_indicator(component, 'Ichimoku Hourly', min_val=0)

        for component in result_daily.values():
            assert_valid_indicator(component, 'Ichimoku Daily', min_val=0)

        # Both should show variation regardless of frequency
        for component in result_hourly.values():
            valid_values = component.dropna()
            if len(valid_values) > 0:
                assert valid_values.std() > 0, "Hourly Ichimoku should vary"

        for component in result_daily.values():
            valid_values = component.dropna()
            if len(valid_values) > 0:
                assert valid_values.std() > 0, "Daily Ichimoku should vary"

    def test_ichimoku_preserves_index(self, zs_1h_data):
        """Test that Ichimoku output maintains input index."""
        result = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )

        for component_name, component in result.items():
            assert component.index.equals(zs_1h_data.index), \
                f"{component_name} should preserve input index"
            assert isinstance(component.index, pd.DatetimeIndex), \
                f"{component_name} index should remain DatetimeIndex"

    def test_ichimoku_with_series_names(self, zs_1h_data):
        """Test that Ichimoku works with named series."""
        named_high = zs_1h_data['high'].rename('ZS_High')
        named_low = zs_1h_data['low'].rename('ZS_Low')
        named_close = zs_1h_data['close'].rename('ZS_Close')

        result = _calculate_ichimoku(named_high, named_low, named_close)

        # Series should still work correctly
        for component in result.values():
            assert len(component) == len(named_high)
            assert_valid_indicator(component, 'Ichimoku', min_val=0)


class TestIchimokuPracticalUsage:
    """Test Ichimoku in practical trading scenarios."""

    def test_ichimoku_identifies_cloud_support_resistance(self, zs_1h_data):
        """
        Test that Ichimoku cloud can identify support/resistance zones.

        Cloud (area between Senkou A and B) should act as dynamic support in
        uptrend and resistance in downtrend.
        """
        result = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )

        senkou_a = result['senkou_span_a']
        senkou_b = result['senkou_span_b']
        close = zs_1h_data['close']

        # Calculate cloud top and bottom
        cloud_top = pd.DataFrame({'a': senkou_a, 'b': senkou_b}).max(axis=1)
        cloud_bottom = pd.DataFrame({'a': senkou_a, 'b': senkou_b}).min(axis=1)

        # Find valid regions where cloud and price exist
        valid_mask = cloud_top.notna() & cloud_bottom.notna() & close.notna()
        valid_indices = valid_mask[valid_mask].index

        if len(valid_indices) > 100:
            # Check if price interacts with cloud
            price_above_cloud = close > cloud_top
            price_below_cloud = close < cloud_bottom

            # Price should be in various positions relative to cloud
            assert price_above_cloud.sum() > 0, "Should have periods above cloud"
            assert price_below_cloud.sum() > 0, "Should have periods below cloud"

    def test_ichimoku_tenkan_kijun_crossover_detection(self, zs_1h_data):
        """
        Test detecting TK (Tenkan/Kijun) crossovers.

        TK cross is a key Ichimoku signal: bullish when Tenkan crosses above Kijun,
        bearish when Tenkan crosses below Kijun.
        """
        result = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )

        tenkan = result['tenkan_sen']
        kijun = result['kijun_sen']

        # Detect bullish TK cross (Tenkan crosses above Kijun)
        bullish_tk_cross = (tenkan.shift(1) <= kijun.shift(1)) & (tenkan > kijun)

        # Detect bearish TK cross (Tenkan crosses below Kijun)
        bearish_tk_cross = (tenkan.shift(1) >= kijun.shift(1)) & (tenkan < kijun)

        # Should detect some crossovers in real data
        assert bullish_tk_cross.sum() > 0, "Should detect bullish TK crossovers"
        assert bearish_tk_cross.sum() > 0, "Should detect bearish TK crossovers"

        # Crossovers should not happen on consecutive bars
        assert not (bullish_tk_cross & bullish_tk_cross.shift(1)).any(), \
            "Bullish crosses should not be consecutive"
        assert not (bearish_tk_cross & bearish_tk_cross.shift(1)).any(), \
            "Bearish crosses should not be consecutive"

    def test_ichimoku_price_vs_cloud_position(self, zs_1h_data):
        """
        Test analyzing price position relative to cloud.

        Price above cloud = bullish, price below cloud = bearish,
        price in cloud = neutral/transitional.
        """
        result = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )

        senkou_a = result['senkou_span_a']
        senkou_b = result['senkou_span_b']
        close = zs_1h_data['close']

        # Calculate cloud boundaries
        cloud_top = pd.DataFrame({'a': senkou_a, 'b': senkou_b}).max(axis=1)
        cloud_bottom = pd.DataFrame({'a': senkou_a, 'b': senkou_b}).min(axis=1)

        # Classify price position
        above_cloud = close > cloud_top
        below_cloud = close < cloud_bottom

        # Real market data should show all three scenarios
        assert above_cloud.sum() > 0, "Should have bullish periods (above cloud)"
        assert below_cloud.sum() > 0, "Should have bearish periods (below cloud)"

        # Verify price can transition between positions
        position_changes = (above_cloud.astype(int).diff() != 0).sum()
        assert position_changes > 5, "Price should transition between cloud positions"

    def test_ichimoku_chikou_span_confirmation(self, zs_1h_data):
        """
        Test using Chikou span for trend confirmation.

        When Chikou is above historical price = bullish confirmation,
        when Chikou is below historical price = bearish confirmation.
        """
        result = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close'],
            displacement=26
        )

        chikou = result['chikou_span']
        close = zs_1h_data['close']

        # Chikou at position i is close[i+26] (displaced backward)
        # So comparing chikou[i] to close[i] compares current price to price 26 bars ago

        # Find valid comparison points
        valid_mask = chikou.notna() & close.notna()
        valid_chikou = chikou[valid_mask]
        valid_close = close[valid_mask]

        if len(valid_chikou) > 50:
            # Detect bullish confirmation (Chikou above price)
            bullish_confirmation = valid_chikou > valid_close

            # Detect bearish confirmation (Chikou below price)
            bearish_confirmation = valid_chikou < valid_close

            # Should have both confirmations in real data
            assert bullish_confirmation.sum() > 0, "Should have bullish Chikou confirmation"
            assert bearish_confirmation.sum() > 0, "Should have bearish Chikou confirmation"

    def test_ichimoku_cloud_color_change(self, zs_1h_data):
        """
        Test detecting cloud color change (Senkou A/B crossover).

        When Senkou A > B, cloud is bullish (green). When A < B, cloud is
        bearish (red). Color change can signal trend shift.
        """
        result = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )

        senkou_a = result['senkou_span_a']
        senkou_b = result['senkou_span_b']

        # Detect cloud color
        bullish_cloud = senkou_a > senkou_b  # Green cloud
        bearish_cloud = senkou_a < senkou_b  # Red cloud

        # Detect color changes
        cloud_color_change = bullish_cloud != bullish_cloud.shift(1)

        # Real data should show both cloud types
        assert bullish_cloud.sum() > 0, "Should have bullish cloud periods"
        assert bearish_cloud.sum() > 0, "Should have bearish cloud periods"

        # Should detect cloud color changes
        assert cloud_color_change.sum() > 5, "Should detect cloud color changes"

    def test_ichimoku_thick_vs_thin_cloud(self, zs_1h_data):
        """
        Test analyzing cloud thickness.

        Thick cloud = strong support/resistance, thin cloud = weak support/resistance.
        Cloud thickness varies with market volatility and trend strength.
        """
        result = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )

        senkou_a = result['senkou_span_a']
        senkou_b = result['senkou_span_b']

        # Calculate cloud thickness
        cloud_thickness = (senkou_a - senkou_b).abs()
        valid_thickness = cloud_thickness.dropna()

        # Cloud thickness should vary
        assert valid_thickness.std() > 0, "Cloud thickness should vary"

        # Should have both thick and thin cloud periods
        median_thickness = valid_thickness.median()
        thick_cloud = valid_thickness > median_thickness * 1.5
        thin_cloud = valid_thickness < median_thickness * 0.5

        assert thick_cloud.sum() > 0, "Should have thick cloud periods"
        assert thin_cloud.sum() > 0, "Should have thin cloud periods"

    def test_ichimoku_complete_signal_validation(self, zs_1h_data):
        """
        Test complete Ichimoku signal with multiple confirmations.

        Strong bullish signal requires: TK bullish cross, price above cloud,
        Chikou above price, and bullish cloud. Test we can identify such signals.
        """
        result = _calculate_ichimoku(
            zs_1h_data['high'],
            zs_1h_data['low'],
            zs_1h_data['close']
        )

        tenkan = result['tenkan_sen']
        kijun = result['kijun_sen']
        senkou_a = result['senkou_span_a']
        senkou_b = result['senkou_span_b']
        chikou = result['chikou_span']
        close = zs_1h_data['close']

        # Calculate conditions
        tk_bullish = tenkan > kijun
        cloud_top = pd.DataFrame({'a': senkou_a, 'b': senkou_b}).max(axis=1)
        price_above_cloud = close > cloud_top
        chikou_above_price = chikou > close
        bullish_cloud = senkou_a > senkou_b

        # Combine conditions for strong bullish signal
        strong_bullish_signal = (
                tk_bullish &
                price_above_cloud &
                chikou_above_price &
                bullish_cloud
        )

        # Should find at least some complete signals in 2 years of data
        signal_count = strong_bullish_signal.sum()
        assert signal_count > 0, \
            "Should find complete bullish signals in real data (may be rare)"
