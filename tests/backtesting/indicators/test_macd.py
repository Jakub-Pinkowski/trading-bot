"""
Tests for MACD indicator calculation.

Uses real historical data from ZS (soybeans) and CL (crude oil) to validate
MACD calculation accuracy, caching behavior, and edge case handling.
"""
import numpy as np
import pandas as pd
import pytest

from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.indicators import calculate_macd
from app.utils.backtesting_utils.indicators_utils import hash_series
from tests.backtesting.helpers.assertions import assert_indicator_varies
from tests.backtesting.indicators.indicator_test_utils import (
    setup_cache_test,
    assert_cache_was_hit,
    assert_cache_hit_on_second_call,
    assert_indicator_structure,
    assert_longer_period_smoother,
    assert_different_params_use_different_cache,
    assert_cache_distinguishes_different_data,
)


# ==================== Helper Function ====================

def _calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Helper function to calculate MACD with automatic hashing.

    Simplifies test code by handling hash calculation internally.
    """
    prices_hash = hash_series(prices)
    return calculate_macd(prices, fast_period, slow_period, signal_period, prices_hash)


# ==================== Basic Logic Tests ====================

class TestMACDBasicLogic:
    """Simple sanity checks for MACD basic behavior using shared test fixtures."""

    def test_macd_returns_dataframe_with_correct_columns(self, medium_price_series):
        """MACD should return DataFrame with macd_line, signal_line, histogram."""
        macd = _calculate_macd(medium_price_series, fast_period=12, slow_period=26, signal_period=9)

        assert_indicator_structure(
            macd,
            len(medium_price_series),
            'dataframe',
            ['macd_line', 'signal_line', 'histogram'],
            'MACD'
        )

    def test_histogram_equals_macd_minus_signal(self, medium_price_series):
        """Histogram should equal macd_line - signal_line at all points."""
        macd = _calculate_macd(medium_price_series, fast_period=12, slow_period=26, signal_period=9)

        # Calculate expected histogram
        expected_histogram = macd['macd_line'] - macd['signal_line']

        # Compare (use np.testing for NaN-safe comparison)
        np.testing.assert_array_almost_equal(
            macd['histogram'].values,
            expected_histogram.values,
            decimal=10,
            err_msg="Histogram must equal macd_line - signal_line"
        )

    def test_rising_prices_give_positive_macd(self, rising_price_series):
        """Continuously rising prices should produce positive MACD values."""
        # Need longer series for MACD calculation (slow period is 26)
        long_rising = pd.Series(range(100, 200))  # 100 consecutive increases
        macd = _calculate_macd(long_rising, fast_period=12, slow_period=26, signal_period=9)

        valid_macd = macd['macd_line'].dropna()
        # After warmup, rising prices should have positive MACD
        recent_macd = valid_macd.iloc[-10:].mean()
        assert recent_macd > 0, f"Rising prices should give positive MACD, got {recent_macd:.4f}"

    def test_falling_prices_give_negative_macd(self, falling_price_series):
        """Continuously falling prices should produce negative MACD values."""
        # Need longer series for MACD calculation
        long_falling = pd.Series(range(200, 100, -1))  # 100 consecutive decreases
        macd = _calculate_macd(long_falling, fast_period=12, slow_period=26, signal_period=9)

        valid_macd = macd['macd_line'].dropna()
        # After warmup, falling prices should have negative MACD
        recent_macd = valid_macd.iloc[-10:].mean()
        assert recent_macd < 0, f"Falling prices should give negative MACD, got {recent_macd:.4f}"

    def test_macd_line_more_volatile_than_signal_line(self, volatile_price_series):
        """MACD line should be more responsive than signal line."""
        # Need more data for MACD
        extended_volatile = pd.Series(list(volatile_price_series) * 2)  # Repeat to get ~60 bars
        macd = _calculate_macd(extended_volatile, fast_period=12, slow_period=26, signal_period=9)

        valid_macd = macd['macd_line'].dropna()
        valid_signal = macd['signal_line'].dropna()

        # MACD line should have higher volatility than signal line
        if len(valid_macd) > 0 and len(valid_signal) > 0:
            macd_volatility = valid_macd.std()
            signal_volatility = valid_signal.std()

            assert macd_volatility > signal_volatility, \
                f"MACD line should be more volatile than signal: {macd_volatility:.4f} vs {signal_volatility:.4f}"

    def test_first_slow_period_values_are_nan(self, medium_price_series):
        """First 'slow_period' values should be NaN (need warmup)."""
        slow_period = 26
        macd = _calculate_macd(medium_price_series, fast_period=12, slow_period=slow_period, signal_period=9)

        # First slow_period-1 values in MACD line should be NaN
        # (EWM with min_periods=26 produces value at index 25)
        assert macd['macd_line'].iloc[:slow_period - 1].isna().all(), \
            f"First {slow_period - 1} MACD values should be NaN"
        # At slow_period, should have first valid value
        assert not pd.isna(macd['macd_line'].iloc[slow_period - 1]), \
            "Should have valid MACD at slow_period-1"

    def test_signal_line_lags_macd_line(self, medium_price_series):
        """Signal line should lag behind MACD line changes."""
        # Create longer price series with stable then uptrending pattern
        stable = pd.Series([100] * 30)
        trending = pd.Series(range(100, 150))
        prices = pd.concat([stable, trending], ignore_index=True)
        macd = _calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9)

        # Find where MACD starts rising strongly
        macd_line = macd['macd_line'].dropna()
        signal_line = macd['signal_line'].dropna()

        # In uptrend region, MACD should be above signal (signal lags)
        uptrend_region = slice(-20, -1)
        assert (macd_line.iloc[uptrend_region] > signal_line.iloc[uptrend_region]).mean() > 0.7, \
            "MACD should be above signal during uptrend (signal lags)"

    def test_histogram_changes_sign_at_crossovers(self):
        """Histogram should cross zero when MACD crosses signal line."""
        prices = pd.Series([
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
            120, 121, 120, 119, 118, 117, 116, 115, 114, 113,
            112, 111, 110, 109, 108, 107, 106, 105, 104, 103,
            102, 101, 102, 103, 104, 105, 106, 107, 108, 109
        ])
        macd = _calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9)

        histogram = macd['histogram'].dropna()

        # Histogram should cross zero (change signs) with this specific oscillating data
        sign_changes = (histogram.shift(1) * histogram < 0).sum()
        assert sign_changes >= 1, f"Histogram should cross zero at least once, got {sign_changes} crossings"


class TestMACDCalculationWithRealData:
    """Test MACD calculation using real historical data."""

    def test_standard_macd_with_zs_hourly_data(self, zs_1h_data):
        """
        Test MACD(12,26,9) calculation on 2 years of ZS hourly data.

        Validates that MACD is calculated correctly across all data points,
        handles NaN values properly, and produces reasonable values.
        """
        macd = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)

        # Validate structure
        assert len(macd) == len(zs_1h_data), "MACD length must match input data length"
        assert macd.index.equals(zs_1h_data.index), "MACD index must match input index"

        # Validate NaN handling (first slow_period-1 values should be NaN)
        assert macd['macd_line'].isna().sum() == 25, "First 25 values should be NaN for MACD(12,26,9)"

        # Validate all three components vary
        assert_indicator_varies(macd['macd_line'], 'MACD Line')
        assert_indicator_varies(macd['signal_line'], 'Signal Line')
        assert_indicator_varies(macd['histogram'], 'Histogram')

        # Validate histogram calculation
        valid_idx = ~macd['macd_line'].isna()
        calculated_histogram = macd.loc[valid_idx, 'macd_line'] - macd.loc[valid_idx, 'signal_line']
        np.testing.assert_array_almost_equal(
            macd.loc[valid_idx, 'histogram'].values,
            calculated_histogram.values,
            decimal=8
        )

    @pytest.mark.parametrize("fast_period,slow_period,signal_period", [
        (5, 13, 5),  # Faster MACD
        (12, 26, 9),  # Standard MACD
        (19, 39, 9),  # Slower MACD
        (8, 17, 9),  # Custom MACD
    ])
    def test_macd_with_different_periods(self, zs_1h_data, fast_period, slow_period, signal_period):
        """
        Test MACD calculation with various period combinations.

        Validates that different periods produce valid results and proper
        NaN counts matching the slow period.
        """
        macd = _calculate_macd(zs_1h_data['close'], fast_period, slow_period, signal_period)

        # Validate structure
        assert len(macd) == len(zs_1h_data)
        assert macd['macd_line'].isna().sum() == slow_period - 1, \
            f"Should have {slow_period - 1} NaN values for slow_period={slow_period}"

        # Validate all components exist and vary
        valid_macd = macd['macd_line'].dropna()
        valid_signal = macd['signal_line'].dropna()
        valid_histogram = macd['histogram'].dropna()

        assert len(valid_macd) > 0, "Should have valid MACD values"
        assert len(valid_signal) > 0, "Should have valid signal values"
        assert len(valid_histogram) > 0, "Should have valid histogram values"

    def test_macd_extreme_but_valid_parameters(self, zs_1h_data):
        """
        Test MACD with extreme but technically valid parameters.

        Tests edge cases like very short periods and very long periods
        to ensure calculation doesn't break.
        """
        # Very short periods
        macd_short = _calculate_macd(zs_1h_data['close'], fast_period=3, slow_period=6, signal_period=3)
        assert_indicator_varies(macd_short['macd_line'], 'Fast MACD Line')
        valid_short = macd_short['histogram'].dropna()
        assert len(valid_short) > 0

        # Very long periods
        macd_long = _calculate_macd(zs_1h_data['close'], fast_period=50, slow_period=100, signal_period=20)
        assert_indicator_varies(macd_long['macd_line'], 'Slow MACD Line')
        valid_long = macd_long['histogram'].dropna()
        assert len(valid_long) > 0

        # Longer period should be smoother (less volatile changes)
        # Align the series to compare the same time periods
        common_idx = valid_short.index.intersection(valid_long.index)
        assert len(common_idx) > 100, "Need sufficient overlap to compare volatility meaningfully"

        short_aligned = valid_short.loc[common_idx]
        long_aligned = valid_long.loc[common_idx]

        # Use utility to compare smoothness
        assert_longer_period_smoother(short_aligned, long_aligned, 'MACD Histogram')

    def test_macd_values_match_expected_calculation(self, zs_1h_data):
        """
        Test that MACD values match the expected calculation formula.

        Manually calculates MACD for a subset of data and compares with
        indicator function output to validate correctness.
        """
        fast_period = 12
        slow_period = 26
        signal_period = 9
        subset = zs_1h_data['close'].iloc[100:200]  # 100 bars for stable calculation

        macd = _calculate_macd(subset, fast_period, slow_period, signal_period)

        # Manual calculation
        fast_ema = subset.ewm(span=fast_period, min_periods=fast_period, adjust=False).mean()
        slow_ema = subset.ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()
        expected_macd = fast_ema - slow_ema
        expected_signal = expected_macd.ewm(span=signal_period, adjust=False).mean()
        expected_histogram = expected_macd - expected_signal

        # Compare all valid values
        valid_idx = ~macd['macd_line'].isna()

        np.testing.assert_allclose(
            macd.loc[valid_idx, 'macd_line'].values,
            expected_macd.loc[valid_idx].values,
            rtol=0.0001,
            err_msg="MACD line calculation doesn't match expected formula"
        )

        np.testing.assert_allclose(
            macd.loc[valid_idx, 'signal_line'].values,
            expected_signal.loc[valid_idx].values,
            rtol=0.0001,
            err_msg="Signal line calculation doesn't match expected formula"
        )

        np.testing.assert_allclose(
            macd.loc[valid_idx, 'histogram'].values,
            expected_histogram.loc[valid_idx].values,
            rtol=0.0001,
            err_msg="Histogram calculation doesn't match expected formula"
        )


class TestMACDInMarketScenarios:
    """Test MACD behavior in different market conditions using real data."""

    def test_macd_in_trending_market(self, trending_market_data):
        """
        Test MACD behavior during strong trend.

        In uptrend, MACD should be positive and diverging from zero.
        Histogram should show sustained positive or negative values.
        """
        if trending_market_data is None:
            pytest.skip("No trending market data available")

        macd = _calculate_macd(trending_market_data['close'], fast_period=12, slow_period=26, signal_period=9)
        valid_macd = macd['macd_line'].dropna()

        # In strong trend, MACD should diverge from zero
        recent_macd = valid_macd.iloc[-20:]
        assert abs(recent_macd.mean()) > 0.1, "MACD should diverge from zero in trending market"

        # Should show sustained direction (not constantly crossing zero)
        positive_pct = (recent_macd > 0).sum() / len(recent_macd)
        assert positive_pct > 0.7 or positive_pct < 0.3, \
            "MACD should maintain direction in trending market"

    def test_macd_in_ranging_market(self, ranging_market_data):
        """
        Test MACD in sideways/ranging market.

        MACD should oscillate around zero in ranging markets with
        frequent crossovers of signal line.
        """
        if ranging_market_data is None:
            pytest.skip("No ranging market data available")

        macd = _calculate_macd(ranging_market_data['close'], fast_period=12, slow_period=26, signal_period=9)
        valid_macd = macd['macd_line'].dropna()
        valid_histogram = macd['histogram'].dropna()

        # In ranging market, MACD should oscillate around zero
        assert abs(valid_macd.mean()) < valid_macd.std(), \
            "MACD should oscillate around zero in ranging market"

        # Histogram should cross zero frequently (crossovers)
        histogram_sign_changes = (valid_histogram.shift(1) * valid_histogram < 0).sum()
        assert histogram_sign_changes > len(valid_histogram) * 0.05, \
            "MACD should show frequent crossovers in ranging market"

    def test_macd_in_volatile_market(self, volatile_market_data):
        """
        Test MACD during extreme volatility.

        MACD should handle volatility without breaking and show
        larger swings but remain functional.
        """
        if volatile_market_data is None:
            pytest.skip("No volatile market data available")

        macd = _calculate_macd(volatile_market_data['close'], fast_period=12, slow_period=26, signal_period=9)

        # Should still produce valid values despite volatility
        assert_indicator_varies(macd['macd_line'], 'MACD Line')
        assert_indicator_varies(macd['signal_line'], 'Signal Line')

        # High volatility should produce higher MACD volatility
        valid_histogram = macd['histogram'].dropna()
        assert valid_histogram.std() > 0, "MACD should show variation in volatile market"


class TestMACDCaching:
    """Test MACD caching behavior."""

    def test_cache_hit_returns_identical_values(self, zs_1h_data):
        """
        Verify cached MACD exactly matches fresh calculation.

        Tests that cache stores and retrieves MACD correctly without
        any data corruption or loss of precision.
        """
        # Setup: Clear cache and reset stats
        setup_cache_test()

        # First calculation (should miss due to empty cache)
        macd_1 = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)
        misses_after_first = indicator_cache.misses

        # Second calculation (should hit cache)
        macd_2 = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)

        # Verify cache was hit and results match
        assert_cache_was_hit(misses_after_first)
        assert_cache_hit_on_second_call(macd_1, macd_2, 'dataframe')

    def test_cache_distinguishes_different_periods(self, zs_1h_data):
        """
        Verify cache stores different results for different periods.

        Tests that cache keys include period parameters so different
        periods don't overwrite each other.
        """
        indicator_cache.reset_stats()

        # Calculate with different periods
        macd_standard = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)
        macd_fast = _calculate_macd(zs_1h_data['close'], fast_period=5, slow_period=13, signal_period=5)

        # Validate different parameters produce different results
        assert_different_params_use_different_cache(macd_standard, macd_fast)

        # Recalculate - should hit cache
        misses_before = indicator_cache.misses
        macd_standard_again = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)
        assert indicator_cache.misses == misses_before, "Recalculation should hit cache"
        pd.testing.assert_frame_equal(macd_standard, macd_standard_again)

    def test_cache_distinguishes_different_data(self, zs_1h_data, cl_15m_data):
        """
        Verify cache stores different results for different data series.

        Tests that cache keys include data hash so different datasets
        don't return wrong cached values.
        """
        indicator_cache.reset_stats()

        # Calculate MACD for two different datasets
        macd_zs = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)
        macd_cl = _calculate_macd(cl_15m_data['close'], fast_period=12, slow_period=26, signal_period=9)

        # Use utility to validate different data behavior
        assert_cache_distinguishes_different_data(
            macd_zs, macd_cl,
            len(zs_1h_data), len(cl_15m_data),
            'MACD'
        )


class TestMACDEdgeCases:
    """Test MACD with edge cases and error conditions."""

    def test_macd_with_insufficient_data(self, minimal_price_series):
        """
        Test MACD when data length < slow_period.

        Should return all NaN values when insufficient data for calculation.
        """
        macd = _calculate_macd(minimal_price_series, fast_period=12, slow_period=26, signal_period=9)

        # All MACD line values should be NaN
        assert macd['macd_line'].isna().all(), "All MACD values should be NaN when data < slow_period"
        assert len(macd) == len(minimal_price_series), "MACD length should match input length"

    def test_macd_with_constant_prices(self, constant_price_series):
        """
        Test MACD when prices don't change.

        With constant prices, MACD should be zero (no divergence between EMAs).
        """
        macd = _calculate_macd(constant_price_series, fast_period=12, slow_period=26, signal_period=9)

        valid_macd = macd['macd_line'].dropna()
        valid_signal = macd['signal_line'].dropna()
        valid_histogram = macd['histogram'].dropna()

        # MACD, signal, and histogram should all be zero (or very close)
        assert abs(valid_macd.mean()) < 0.0001, "MACD should be ~0 for constant prices"
        assert abs(valid_signal.mean()) < 0.0001, "Signal should be ~0 for constant prices"
        assert abs(valid_histogram.mean()) < 0.0001, "Histogram should be ~0 for constant prices"

    def test_macd_with_empty_series(self, empty_price_series):
        """Test MACD with empty input series."""
        macd = _calculate_macd(empty_price_series, fast_period=12, slow_period=26, signal_period=9)

        assert len(macd) == 0, "Empty input should return empty DataFrame"
        assert isinstance(macd, pd.DataFrame)
        assert list(macd.columns) == ['macd_line', 'signal_line', 'histogram']


class TestMACDDataTypes:
    """Test MACD with different input data types and structures."""

    def test_macd_with_series_name(self, zs_1h_data):
        """Test that MACD works correctly with named series."""
        named_series = zs_1h_data['close'].rename('ZS_Close')
        macd = _calculate_macd(named_series, fast_period=12, slow_period=26, signal_period=9)

        # Series should still work correctly
        assert len(macd) == len(named_series)
        assert_indicator_varies(macd['macd_line'], 'MACD Line')

    def test_macd_with_different_datetime_frequencies(self, zs_1h_data, zs_1d_data):
        """
        Test MACD works with different time frequencies.

        Validates that MACD calculation is time-agnostic and works with
        any datetime frequency (hourly, daily, etc.).
        """
        macd_hourly = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)
        macd_daily = _calculate_macd(zs_1d_data['close'], fast_period=12, slow_period=26, signal_period=9)

        # Both should produce valid MACD
        assert_indicator_varies(macd_hourly['macd_line'], 'MACD(12,26,9) Hourly')
        assert_indicator_varies(macd_daily['macd_line'], 'MACD(12,26,9) Daily')

        # Both should show variation regardless of frequency
        assert macd_hourly['histogram'].dropna().std() > 0, "Hourly MACD histogram should vary"
        assert macd_daily['histogram'].dropna().std() > 0, "Daily MACD histogram should vary"

    def test_macd_preserves_index(self, zs_1h_data):
        """Test that MACD output maintains input index."""
        macd = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)

        assert macd.index.equals(zs_1h_data.index), "MACD should preserve input index"
        assert isinstance(macd.index, pd.DatetimeIndex), "Index should remain DatetimeIndex"


class TestMACDPracticalUsage:
    """Test MACD in practical trading scenarios."""

    def test_macd_crossover_detection_bullish(self, zs_1h_data):
        """
        Test detection of bullish MACD crossovers (MACD crosses above signal).

        Practical use: Generating long entry signals when MACD crosses above signal line.
        """
        macd = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)

        # Detect bullish crossovers (MACD crosses above signal)
        bullish_cross = (macd['macd_line'].shift(1) <= macd['signal_line'].shift(1)) & \
                        (macd['macd_line'] > macd['signal_line'])

        crossover_count = bullish_cross.sum()

        # Should detect some bullish crossovers in real data
        assert crossover_count > 0, "Should detect bullish MACD crossovers in real data"
        assert crossover_count < len(macd) * 0.1, "Crossovers should be minority of time"

        # Verify actual crossovers
        for idx in bullish_cross[bullish_cross].index:
            idx_loc = macd.index.get_loc(idx)
            if idx_loc > 0:
                prev_idx = macd.index[idx_loc - 1]
                # Previous: MACD <= signal, Current: MACD > signal
                assert macd.loc[prev_idx, 'macd_line'] <= macd.loc[prev_idx, 'signal_line']
                assert macd.loc[idx, 'macd_line'] > macd.loc[idx, 'signal_line']

    def test_macd_crossover_detection_bearish(self, zs_1h_data):
        """
        Test detection of bearish MACD crossovers (MACD crosses below signal).

        Practical use: Generating short entry signals when MACD crosses below signal line.
        """
        macd = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)

        # Detect bearish crossovers (MACD crosses below signal)
        bearish_cross = (macd['macd_line'].shift(1) >= macd['signal_line'].shift(1)) & \
                        (macd['macd_line'] < macd['signal_line'])

        crossover_count = bearish_cross.sum()

        # Should detect some bearish crossovers in real data
        assert crossover_count > 0, "Should detect bearish MACD crossovers in real data"
        assert crossover_count < len(macd) * 0.1, "Crossovers should be minority of time"

        # Verify actual crossovers
        for idx in bearish_cross[bearish_cross].index:
            idx_loc = macd.index.get_loc(idx)
            if idx_loc > 0:
                prev_idx = macd.index[idx_loc - 1]
                # Previous: MACD >= signal, Current: MACD < signal
                assert macd.loc[prev_idx, 'macd_line'] >= macd.loc[prev_idx, 'signal_line']
                assert macd.loc[idx, 'macd_line'] < macd.loc[idx, 'signal_line']

    def test_macd_zero_line_crossover(self, zs_1h_data):
        """
        Test detection of MACD zero line crossovers.

        Practical use: Zero line crossovers indicate momentum shift.
        MACD crossing above zero = bullish momentum, below zero = bearish momentum.
        """
        macd = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)

        # Detect zero line crossovers
        cross_above_zero = (macd['macd_line'].shift(1) <= 0) & (macd['macd_line'] > 0)
        cross_below_zero = (macd['macd_line'].shift(1) >= 0) & (macd['macd_line'] < 0)

        # Should detect some zero line crossovers
        assert cross_above_zero.sum() > 0, "Should detect MACD crossing above zero"
        assert cross_below_zero.sum() > 0, "Should detect MACD crossing below zero"

    def test_macd_histogram_divergence_detection(self, zs_1h_data):
        """
        Test histogram analysis for divergence patterns.

        Practical use: Histogram shows momentum strength. Decreasing histogram
        amplitude can signal weakening trend even if MACD is still positive/negative.
        """
        macd = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)

        histogram = macd['histogram'].dropna()

        # Calculate histogram peaks and troughs
        histogram_abs = histogram.abs()

        # Histogram should show varying strength
        assert histogram_abs.std() > 0, "Histogram amplitude should vary"

        # Histogram should cross zero (indicating MACD/signal crossovers)
        zero_crosses = (histogram.shift(1) * histogram < 0).sum()
        assert zero_crosses > 0, "Histogram should cross zero (MACD/signal crossovers)"

    def test_macd_trend_strength_analysis(self, zs_1h_data):
        """
        Test using MACD to measure trend strength.

        Practical use: Larger MACD values indicate stronger trends.
        MACD near zero indicates weak trend or consolidation.
        """
        macd = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)

        macd_line = macd['macd_line'].dropna()

        # Categorize MACD values by strength
        strong_trend = abs(macd_line) > macd_line.std()
        weak_trend = abs(macd_line) < macd_line.std() * 0.5

        # Should have periods of both strong and weak trends
        assert strong_trend.sum() > 0, "Should have periods of strong trend"
        assert weak_trend.sum() > 0, "Should have periods of weak trend"

        # Strong trend periods should be less common than weak
        assert strong_trend.sum() < len(macd_line) * 0.5, \
            "Strong trends should not dominate entire dataset"

    def test_macd_histogram_peak_analysis(self, zs_1h_data):
        """
        Test histogram peak/trough detection.

        Practical use: Histogram peaks can signal end of momentum expansion.
        Declining peaks suggest trend exhaustion even before crossover occurs.
        """
        macd = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)

        histogram = macd['histogram'].dropna()

        # Find local peaks (histogram higher than neighbors)
        local_peaks = (histogram > histogram.shift(1)) & (histogram > histogram.shift(-1))
        peak_count = local_peaks.sum()

        # Find local troughs (histogram lower than neighbors)
        local_troughs = (histogram < histogram.shift(1)) & (histogram < histogram.shift(-1))
        trough_count = local_troughs.sum()

        # Should find multiple peaks and troughs in real data
        assert peak_count > 5, "Should find multiple histogram peaks"
        assert trough_count > 5, "Should find multiple histogram troughs"

    def test_macd_signal_filtering_by_position(self, zs_1h_data):
        """
        Test filtering MACD signals based on MACD line position relative to zero.

        Practical use: Only take bullish crossovers when MACD > 0 (with trend),
        only take bearish crossovers when MACD < 0 (with trend).
        This filters out weak counter-trend signals.
        """
        macd = _calculate_macd(zs_1h_data['close'], fast_period=12, slow_period=26, signal_period=9)

        # Bullish crossovers above zero (stronger signals)
        bullish_cross = (macd['macd_line'].shift(1) <= macd['signal_line'].shift(1)) & \
                        (macd['macd_line'] > macd['signal_line'])
        bullish_above_zero = bullish_cross & (macd['macd_line'] > 0)

        # Bearish crossovers below zero (stronger signals)
        bearish_cross = (macd['macd_line'].shift(1) >= macd['signal_line'].shift(1)) & \
                        (macd['macd_line'] < macd['signal_line'])
        bearish_below_zero = bearish_cross & (macd['macd_line'] < 0)

        # Filtered signals should be subset of all signals
        assert bullish_above_zero.sum() <= bullish_cross.sum(), \
            "Filtered bullish signals should be subset"
        assert bearish_below_zero.sum() <= bearish_cross.sum(), \
            "Filtered bearish signals should be subset"

        # Should still have some filtered signals
        assert bullish_above_zero.sum() > 0, "Should find bullish signals above zero"
        assert bearish_below_zero.sum() > 0, "Should find bearish signals below zero"
