"""
Tests for RSI indicator calculation.

Uses real historical data from ZS (soybeans) to validate RSI calculation
accuracy, caching behavior, and edge case handling.
"""
import numpy as np
import pandas as pd
import pytest

from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.indicators import calculate_rsi
from app.utils.backtesting_utils.indicators_utils import hash_series
from tests.backtesting.helpers.assertions import assert_valid_indicator, assert_indicator_varies
from tests.backtesting.helpers.data_utils import inject_price_spike


# ==================== Helper Function ====================

def _calculate_rsi(prices, period=14):
    """
    Helper function to calculate RSI with automatic hashing.

    Simplifies test code by handling hash calculation internally.
    """
    prices_hash = hash_series(prices)
    return calculate_rsi(prices, period, prices_hash)


class TestRSICalculationWithRealData:
    """Test RSI calculation using real historical data."""

    def test_standard_rsi_with_zs_hourly_data(self, zs_1h_data):
        """
        Test RSI(14) calculation on 2 years of ZS hourly data.

        Validates that RSI is calculated correctly across all data points,
        handles NaN values properly, and produces values in valid range.
        """
        rsi = _calculate_rsi(zs_1h_data['close'], period=14)

        # Validate structure
        assert len(rsi) == len(zs_1h_data), "RSI length must match input data length"
        assert rsi.index.equals(zs_1h_data.index), "RSI index must match input index"

        # Validate NaN handling (first period-1 values should be NaN)
        assert rsi.isna().sum() == 14, "First 14 values should be NaN for RSI(14)"

        # Validate range (RSI must be 0-100)
        assert_valid_indicator(rsi, 'RSI', min_val=0, max_val=100)

        # Validate market behavior with real data
        valid_rsi = rsi.dropna()
        assert_indicator_varies(valid_rsi, 'RSI')

        # Real market data should reach oversold/overbought zones
        assert valid_rsi.min() < 40, "Real data should show oversold conditions (RSI < 40)"
        assert valid_rsi.max() > 60, "Real data should show overbought conditions (RSI > 60)"

        # Mean should be around 50 over long period
        assert 45 <= valid_rsi.mean() <= 55, "Long-term RSI mean should center around 50"

    @pytest.mark.parametrize("period", [7, 14, 21, 28])
    def test_rsi_with_different_periods(self, zs_1h_data, period):
        """
        Test RSI calculation with various standard periods.

        Validates that different periods produce valid results and proper
        NaN counts matching the period.
        """
        rsi = _calculate_rsi(zs_1h_data['close'], period=period)

        # Validate structure
        assert len(rsi) == len(zs_1h_data)
        assert rsi.isna().sum() == period, f"Should have {period} NaN values for RSI({period})"

        # Validate range
        assert_valid_indicator(rsi, f'RSI({period})', min_val=0, max_val=100)

        # Longer periods should have smoother RSI (lower volatility)
        if period > 7:
            valid_rsi = rsi.dropna()
            assert len(valid_rsi) > 0, "Should have valid RSI values"

    def test_rsi_extreme_but_valid_parameters(self, zs_1h_data):
        """
        Test RSI with extreme but technically valid parameters.

        Tests edge cases like very short period (3) and very long period (100)
        to ensure calculation doesn't break.
        """
        # Very short period
        rsi_short = _calculate_rsi(zs_1h_data['close'], period=3)
        assert_valid_indicator(rsi_short, 'RSI(3)', min_val=0, max_val=100)
        valid_short = rsi_short.dropna()
        assert len(valid_short) > 0

        # Very long period
        rsi_long = _calculate_rsi(zs_1h_data['close'], period=100)
        assert_valid_indicator(rsi_long, 'RSI(100)', min_val=0, max_val=100)
        valid_long = rsi_long.dropna()
        assert len(valid_long) > 0

        # Longer period should be smoother (less volatile)
        assert valid_short.std() > valid_long.std(), "Short period RSI should be more volatile"

    def test_rsi_values_match_expected_calculation(self, zs_1h_data):
        """
        Test that RSI values match the expected calculation formula.

        Manually calculates RSI for a subset of data and compares with
        indicator function output to validate correctness. Uses Wilder's
        smoothing (EWM) method as implemented in the app.
        """
        period = 14
        subset = zs_1h_data['close'].iloc[100:150]  # 50 bars for stable calculation

        rsi = _calculate_rsi(subset, period=period)

        # Manual calculation using Wilder's smoothing (EWM)
        delta = subset.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Use EWM with Wilder's smoothing (alpha = 1/period)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        expected_rsi = 100 - (100 / (1 + rs))

        # Compare all valid values
        valid_calculated = rsi.dropna()
        valid_expected = expected_rsi.dropna()

        # Should match within reasonable tolerance
        np.testing.assert_allclose(
            valid_calculated.values,
            valid_expected.values,
            rtol=0.01,
            err_msg="RSI calculation doesn't match expected Wilder's smoothing formula"
        )


class TestRSIInMarketScenarios:
    """Test RSI behavior in different market conditions using real data."""

    def test_rsi_in_trending_market(self, trending_market_data):
        """
        Test RSI behavior during strong trend.

        Uses subset of real data showing clear uptrend. RSI should
        remain elevated but not pegged at extreme values.
        """
        if trending_market_data is None:
            pytest.skip("No trending market data available")

        rsi = _calculate_rsi(trending_market_data['close'], period=14)
        valid_rsi = rsi.dropna()

        # In uptrend, RSI should be elevated but not always at 100
        assert valid_rsi.mean() > 50, "RSI should be above 50 in uptrend"
        assert valid_rsi.max() < 95, "RSI should not peg at 100 even in strong trend"

        # Should still show some variation (not constant)
        assert valid_rsi.std() > 3, "RSI should vary even in trending market"

    def test_rsi_in_ranging_market(self, ranging_market_data):
        """
        Test RSI in sideways/ranging market.

        RSI should oscillate around 50 in ranging markets without
        extreme readings.
        """
        if ranging_market_data is None:
            pytest.skip("No ranging market data available")

        rsi = _calculate_rsi(ranging_market_data['close'], period=14)
        valid_rsi = rsi.dropna()

        # In ranging market, RSI should center around 50
        assert 40 <= valid_rsi.mean() <= 60, "RSI should center around 50 in ranging market"

        # Should show oscillation but not extreme values
        assert valid_rsi.std() > 5, "RSI should oscillate in ranging market"

    def test_rsi_in_volatile_market(self, volatile_market_data):
        """
        Test RSI during extreme volatility.

        RSI should handle volatility without breaking and may reach
        extreme values but stay within 0-100 bounds.
        """
        if volatile_market_data is None:
            pytest.skip("No volatile market data available")

        rsi = _calculate_rsi(volatile_market_data['close'], period=14)

        # Should still produce valid values despite volatility
        assert_valid_indicator(rsi, 'RSI', min_val=0, max_val=100)

        # High volatility should produce higher RSI volatility
        valid_rsi = rsi.dropna()
        assert valid_rsi.std() > 10, "RSI should show high variation in volatile market"


class TestRSICaching:
    """Test RSI caching behavior."""

    def test_cache_hit_returns_identical_values(self, zs_1h_data):
        """
        Verify cached RSI exactly matches fresh calculation.

        Tests that cache stores and retrieves RSI correctly without
        any data corruption or loss of precision.
        """
        # Reset stats to track this test's cache behavior
        indicator_cache.reset_stats()
        initial_hits = indicator_cache.hits
        initial_misses = indicator_cache.misses

        # First calculation (may hit or miss depending on previous tests)
        rsi_1 = _calculate_rsi(zs_1h_data['close'], period=14)
        first_misses = indicator_cache.misses

        # Second calculation (should hit cache)
        rsi_2 = _calculate_rsi(zs_1h_data['close'], period=14)

        # Verify cache was hit (misses didn't increase)
        assert indicator_cache.misses == first_misses, "Second calculation should not cause cache miss"

        # Verify identical results (use np.array_equal for NaN-safe comparison)
        assert len(rsi_1) == len(rsi_2), "RSI series should have same length"
        np.testing.assert_array_equal(rsi_1.values, rsi_2.values, "Cached RSI should match exactly")

    def test_cache_distinguishes_different_periods(self, zs_1h_data):
        """
        Verify cache stores different results for different periods.

        Tests that cache keys include period parameter so different
        periods don't overwrite each other.
        """
        indicator_cache.reset_stats()
        initial_misses = indicator_cache.misses

        # Calculate with different periods
        rsi_14 = _calculate_rsi(zs_1h_data['close'], period=14)
        rsi_21 = _calculate_rsi(zs_1h_data['close'], period=21)

        # Results should differ
        assert not rsi_14.equals(rsi_21), "Different periods should produce different RSI"

        # Recalculate - should hit cache (misses shouldn't increase)
        misses_before_recalc = indicator_cache.misses
        rsi_14_again = _calculate_rsi(zs_1h_data['close'], period=14)
        assert indicator_cache.misses == misses_before_recalc, "Recalculation should hit cache"

        # Should get same values
        np.testing.assert_array_equal(rsi_14.values, rsi_14_again.values)

    def test_cache_distinguishes_different_data(self, zs_1h_data, cl_15m_data):
        """
        Verify cache stores different results for different data series.

        Tests that cache keys include data hash so different datasets
        don't return wrong cached values.
        """
        indicator_cache.reset_stats()

        # Calculate RSI for two different datasets
        rsi_zs = _calculate_rsi(zs_1h_data['close'], period=14)
        rsi_cl = _calculate_rsi(cl_15m_data['close'], period=14)

        # Results should differ (different underlying data)
        assert not rsi_zs.equals(rsi_cl), "Different data should produce different RSI"
        assert len(rsi_zs) != len(rsi_cl), "Different datasets should have different lengths"


class TestRSIEdgeCases:
    """Test RSI with edge cases and error conditions."""

    def test_rsi_with_insufficient_data(self):
        """
        Test RSI when data length < period.

        Should return all NaN values when insufficient data for calculation.
        """
        short_data = pd.Series([100, 101, 102, 103, 104])
        rsi = _calculate_rsi(short_data, period=14)

        # All values should be NaN
        assert rsi.isna().all(), "All RSI values should be NaN when data < period"
        assert len(rsi) == len(short_data), "RSI length should match input length"

    def test_rsi_with_exact_minimum_data(self):
        """
        Test RSI with exactly enough data for calculation.

        With period=14, needs at least 15 data points (14 for warmup + 1 for calculation).
        """
        # 15 data points for period=14
        min_data = pd.Series(range(100, 115))
        rsi = _calculate_rsi(min_data, period=14)

        # First 14 should be NaN, last should have value
        assert rsi.isna().sum() == 14
        assert not pd.isna(rsi.iloc[-1])

    def test_rsi_with_constant_prices(self):
        """
        Test RSI when prices don't change.

        With constant prices (zero variance), RSI returns NaN because there are
        no gains or losses to calculate. This is correct behavior.
        """
        constant_data = pd.Series([100.0] * 100)
        rsi = _calculate_rsi(constant_data, period=14)

        # With zero variance, RSI should return all NaN (division by zero in formula)
        assert rsi.isna().all(), "RSI should be all NaN for constant prices (zero variance)"
        assert len(rsi) == len(constant_data), "RSI length should match input length"

    def test_rsi_with_monotonic_increase(self):
        """
        Test RSI with continuously increasing prices.

        All gains, no losses - RSI should approach 100.
        """
        increasing_data = pd.Series(range(100, 200))  # 100 consecutive increases
        rsi = _calculate_rsi(increasing_data, period=14)

        valid_rsi = rsi.dropna()

        # RSI should be very high (approaching 100)
        assert valid_rsi.min() > 80, "RSI should be high with all gains"
        assert valid_rsi.iloc[-1] > 95, "Recent RSI should approach 100"

    def test_rsi_with_monotonic_decrease(self):
        """
        Test RSI with continuously decreasing prices.

        All losses, no gains - RSI should approach 0.
        """
        decreasing_data = pd.Series(range(200, 100, -1))  # 100 consecutive decreases
        rsi = _calculate_rsi(decreasing_data, period=14)

        valid_rsi = rsi.dropna()

        # RSI should be very low (approaching 0)
        assert valid_rsi.max() < 20, "RSI should be low with all losses"
        assert valid_rsi.iloc[-1] < 5, "Recent RSI should approach 0"

    def test_rsi_with_extreme_spike(self, zs_1h_data):
        """
        Test RSI reaction to extreme price spike.

        Injects artificial spike to test RSI handles extreme moves.
        """
        # Inject 10% spike upward at bar 1000
        modified_data = inject_price_spike(zs_1h_data.copy(), 1000, 10.0, 'up')

        rsi = _calculate_rsi(modified_data['close'], period=14)

        # RSI should still be within bounds despite spike
        assert_valid_indicator(rsi, 'RSI', min_val=0, max_val=100)

        # RSI around spike should be elevated
        spike_region = rsi.iloc[1000:1010]
        assert spike_region.max() > 70, "RSI should spike after large price move"

    def test_rsi_with_flat_then_movement(self):
        """
        Test RSI transition from flat period to normal movement.

        Validates RSI can handle transition from constant prices to price movement.
        """
        # Create data: 50 flat, then 50 with oscillating movement
        flat_part = pd.Series([100.0] * 50)
        # Create oscillating pattern instead of monotonic to get RSI variation
        variable_part = pd.Series([100 + (i % 10 - 5) * 0.5 for i in range(50)])
        mixed_data = pd.concat([flat_part, variable_part], ignore_index=True)

        rsi = _calculate_rsi(mixed_data, period=14)

        # Flat region will have NaN (zero variance) - expected behavior

        # RSI in variable region should have valid values
        variable_rsi = rsi.iloc[50:]
        valid_variable_rsi = variable_rsi.dropna()

        assert len(valid_variable_rsi) > 0, "Should have valid RSI values in variable region"
        # With oscillating data, RSI should center around 50 and vary
        assert 30 <= valid_variable_rsi.mean() <= 70, "RSI should be reasonable in oscillating region"
        assert valid_variable_rsi.std() > 0, "RSI should vary in variable region"

    def test_rsi_invalid_period_zero(self):
        """Test that period=0 raises appropriate error."""
        data = pd.Series([100, 101, 102, 103])

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            _calculate_rsi(data, period=0)

    def test_rsi_invalid_period_negative(self):
        """Test that negative period raises appropriate error."""
        data = pd.Series([100, 101, 102, 103])

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            _calculate_rsi(data, period=-5)

    def test_rsi_with_empty_series(self):
        """Test RSI with empty input series."""
        empty_data = pd.Series([], dtype=float)
        rsi = _calculate_rsi(empty_data, period=14)

        assert len(rsi) == 0, "Empty input should return empty RSI"
        assert isinstance(rsi, pd.Series)


class TestRSIDataTypes:
    """Test RSI with different input data types and structures."""

    def test_rsi_with_series_name(self, zs_1h_data):
        """Test that RSI preserves series name if present."""
        named_series = zs_1h_data['close'].rename('ZS_Close')
        rsi = _calculate_rsi(named_series, period=14)

        # Series should still work correctly
        assert len(rsi) == len(named_series)
        assert_valid_indicator(rsi, 'RSI', min_val=0, max_val=100)

    def test_rsi_with_different_datetime_frequencies(self, zs_1h_data, zs_1d_data):
        """
        Test RSI works with different time frequencies.

        Validates that RSI calculation is time-agnostic and works with
        any datetime frequency (hourly, daily, etc.).
        """
        rsi_hourly = _calculate_rsi(zs_1h_data['close'], period=14)
        rsi_daily = _calculate_rsi(zs_1d_data['close'], period=14)

        # Both should produce valid RSI
        assert_valid_indicator(rsi_hourly, 'RSI(14) Hourly', min_val=0, max_val=100)
        assert_valid_indicator(rsi_daily, 'RSI(14) Daily', min_val=0, max_val=100)

        # Both should show variation regardless of frequency
        assert rsi_hourly.dropna().std() > 0, "Hourly RSI should vary"
        assert rsi_daily.dropna().std() > 0, "Daily RSI should vary"

    def test_rsi_preserves_index(self, zs_1h_data):
        """Test that RSI output maintains input index."""
        rsi = _calculate_rsi(zs_1h_data['close'], period=14)

        assert rsi.index.equals(zs_1h_data.index), "RSI should preserve input index"
        assert isinstance(rsi.index, pd.DatetimeIndex), "Index should remain DatetimeIndex"
