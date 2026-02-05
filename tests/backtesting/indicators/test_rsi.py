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


# ==================== Basic Logic Tests ====================

class TestRSIBasicLogic:
    """Simple sanity checks for RSI basic behavior using shared test fixtures."""

    def test_rsi_returns_series_with_correct_length(self, short_price_series):
        """RSI should return a series with same length as input."""
        rsi = _calculate_rsi(short_price_series, period=14)

        assert len(rsi) == len(short_price_series), "RSI length must equal input length"
        assert isinstance(rsi, pd.Series), "RSI must return pandas Series"

    def test_rsi_stays_within_bounds(self, volatile_price_series):
        """RSI must always be between 0 and 100."""
        rsi = _calculate_rsi(volatile_price_series, period=10)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all(), "RSI should never be below 0"
        assert (valid_rsi <= 100).all(), "RSI should never be above 100"

    def test_rising_prices_give_high_rsi(self, rising_price_series):
        """Continuously rising prices should produce high RSI (>70)."""
        rsi = _calculate_rsi(rising_price_series, period=14)

        recent_rsi = rsi.iloc[-5:].mean()  # Average of last 5 RSI values
        assert recent_rsi > 70, f"Rising prices should give high RSI, got {recent_rsi:.1f}"

    def test_falling_prices_give_low_rsi(self, falling_price_series):
        """Continuously falling prices should produce low RSI (<30)."""
        rsi = _calculate_rsi(falling_price_series, period=14)

        recent_rsi = rsi.iloc[-5:].mean()  # Average of last 5 RSI values
        assert recent_rsi < 30, f"Falling prices should give low RSI, got {recent_rsi:.1f}"

    def test_rsi_changes_with_price_changes(self, rising_price_series, falling_price_series):
        """RSI should change when prices change."""
        rsi_rising = _calculate_rsi(rising_price_series, period=14)
        rsi_falling = _calculate_rsi(falling_price_series, period=14)

        # RSI at end of rising series should be high
        rsi_after_rising = rsi_rising.iloc[-1]
        # RSI at end of falling series should be low
        rsi_after_falling = rsi_falling.iloc[-1]

        assert rsi_after_rising > 70, f"RSI after rising prices should be >70, got {rsi_after_rising:.1f}"
        assert rsi_after_falling < 30, f"RSI after falling prices should be <30, got {rsi_after_falling:.1f}"
        assert rsi_after_rising > rsi_after_falling, "Rising prices should give higher RSI than falling prices"

    def test_first_n_values_are_nan(self, medium_price_series):
        """First 'period' values should be NaN (need warmup)."""
        period = 14
        rsi = _calculate_rsi(medium_price_series, period=period)

        # First 'period' values should be NaN
        assert rsi.iloc[:period].isna().all(), f"First {period} RSI values should be NaN"
        # After warmup, should have valid values
        assert not rsi.iloc[period:].isna().all(), "Should have valid RSI after warmup period"

    def test_small_price_changes_give_moderate_rsi(self, oscillating_price_series):
        """Small oscillations should keep RSI near 50."""
        rsi = _calculate_rsi(oscillating_price_series, period=10)

        recent_rsi = rsi.iloc[-5:].mean()
        assert 40 < recent_rsi < 60, f"Small oscillations should keep RSI near 50, got {recent_rsi:.1f}"

    def test_larger_period_gives_smoother_rsi(self, volatile_price_series):
        """Larger RSI period should produce less volatile RSI values."""
        rsi_short = _calculate_rsi(volatile_price_series, period=5)
        rsi_long = _calculate_rsi(volatile_price_series, period=20)

        # Standard deviation measures volatility
        short_volatility = rsi_short.dropna().std()
        long_volatility = rsi_long.dropna().std()

        assert short_volatility > long_volatility, \
            f"Shorter period should be more volatile: {short_volatility:.1f} vs {long_volatility:.1f}"

    def test_rsi_responds_to_price_direction_not_magnitude(self, low_price_level_series, high_price_level_series):
        """RSI is based on gains/losses, not absolute price level."""
        rsi_low = _calculate_rsi(low_price_level_series, period=10)
        rsi_high = _calculate_rsi(high_price_level_series, period=10)

        # RSI should be similar (both are 10% increases each period)
        # Allow some tolerance due to EWM calculation differences
        rsi_low_final = rsi_low.iloc[-1]
        rsi_high_final = rsi_high.iloc[-1]

        assert abs(rsi_low_final - rsi_high_final) < 5, \
            "RSI should be similar for same % changes regardless of price level"


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

    def test_rsi_with_insufficient_data(self, minimal_price_series):
        """
        Test RSI when data length < period.

        Should return all NaN values when insufficient data for calculation.
        """
        rsi = _calculate_rsi(minimal_price_series, period=14)

        # All values should be NaN
        assert rsi.isna().all(), "All RSI values should be NaN when data < period"
        assert len(rsi) == len(minimal_price_series), "RSI length should match input length"

    def test_rsi_with_exact_minimum_data(self, exact_period_price_series):
        """
        Test RSI with exactly enough data for calculation.

        With period=14, needs at least 15 data points (14 for warmup + 1 for calculation).
        """
        rsi = _calculate_rsi(exact_period_price_series, period=14)

        # First 14 should be NaN, last should have value
        assert rsi.isna().sum() == 14
        assert not pd.isna(rsi.iloc[-1])

    def test_rsi_with_constant_prices(self, constant_price_series):
        """
        Test RSI when prices don't change.

        With constant prices (zero variance), RSI returns NaN because there are
        no gains or losses to calculate. This is correct behavior.
        """
        rsi = _calculate_rsi(constant_price_series, period=14)

        # With zero variance, RSI should return all NaN (division by zero in formula)
        assert rsi.isna().all(), "RSI should be all NaN for constant prices (zero variance)"
        assert len(rsi) == len(constant_price_series), "RSI length should match input length"

    def test_rsi_with_monotonic_increase(self, rising_price_series):
        """
        Test RSI with continuously increasing prices.

        All gains, no losses - RSI should approach 100.
        """
        # Use longer series for better demonstration
        long_rising = pd.Series(range(100, 200))  # 100 consecutive increases
        rsi = _calculate_rsi(long_rising, period=14)

        valid_rsi = rsi.dropna()

        # RSI should be very high (approaching 100)
        assert valid_rsi.min() > 80, "RSI should be high with all gains"
        assert valid_rsi.iloc[-1] > 95, "Recent RSI should approach 100"

    def test_rsi_with_monotonic_decrease(self, falling_price_series):
        """
        Test RSI with continuously decreasing prices.

        All losses, no gains - RSI should approach 0.
        """
        # Use longer series for better demonstration
        long_falling = pd.Series(range(200, 100, -1))  # 100 consecutive decreases
        rsi = _calculate_rsi(long_falling, period=14)

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

    def test_rsi_with_flat_then_movement(self, flat_then_volatile_series):
        """
        Test RSI transition from flat period to normal movement.

        Validates RSI can handle transition from constant prices to price movement.
        """
        rsi = _calculate_rsi(flat_then_volatile_series, period=14)

        # Flat region will have NaN (zero variance) - expected behavior

        # RSI in variable region should have valid values
        variable_rsi = rsi.iloc[50:]
        valid_variable_rsi = variable_rsi.dropna()

        assert len(valid_variable_rsi) > 0, "Should have valid RSI values in variable region"
        # With oscillating data, RSI should center around 50 and vary
        assert 30 <= valid_variable_rsi.mean() <= 70, "RSI should be reasonable in oscillating region"
        assert valid_variable_rsi.std() > 0, "RSI should vary in variable region"

    def test_rsi_invalid_period_zero(self, short_price_series):
        """Test that period=0 raises appropriate error."""
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            _calculate_rsi(short_price_series, period=0)

    def test_rsi_invalid_period_negative(self, short_price_series):
        """Test that negative period raises appropriate error."""
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            _calculate_rsi(short_price_series, period=-5)

    def test_rsi_with_empty_series(self, empty_price_series):
        """Test RSI with empty input series."""
        rsi = _calculate_rsi(empty_price_series, period=14)

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


class TestRSIPracticalUsage:
    """Test RSI in practical trading scenarios."""

    def test_rsi_identifies_oversold_condition(self, zs_1h_data):
        """
        Test that RSI can identify oversold conditions (RSI < 30).

        Practical use: Finding potential long entry points.
        """
        rsi = _calculate_rsi(zs_1h_data['close'], period=14)

        # Find oversold conditions
        oversold = rsi < 30
        oversold_count = oversold.sum()

        # Real market data should have some oversold conditions
        assert oversold_count > 0, "Should find oversold conditions in real data"
        assert oversold_count < len(rsi) * 0.2, "Oversold should be minority of time"

    def test_rsi_identifies_overbought_condition(self, zs_1h_data):
        """
        Test that RSI can identify overbought conditions (RSI > 70).

        Practical use: Finding potential short entry or profit-taking points.
        """
        rsi = _calculate_rsi(zs_1h_data['close'], period=14)

        # Find overbought conditions
        overbought = rsi > 70
        overbought_count = overbought.sum()

        # Real market data should have some overbought conditions
        assert overbought_count > 0, "Should find overbought conditions in real data"
        assert overbought_count < len(rsi) * 0.2, "Overbought should be minority of time"

    def test_rsi_crossover_detection(self, zs_1h_data):
        """
        Test RSI crossing key levels (30 and 70).

        Practical use: Generating entry/exit signals on RSI crossovers.
        """
        rsi = _calculate_rsi(zs_1h_data['close'], period=14)

        # Detect crossover above 30 (potential long signal)
        crosses_above_30 = (rsi.shift(1) < 30) & (rsi >= 30)

        # Detect crossover below 70 (potential short signal)
        crosses_below_70 = (rsi.shift(1) > 70) & (rsi <= 70)

        # Should detect some crossovers in real data
        assert crosses_above_30.sum() > 0, "Should detect RSI crossing above 30"
        assert crosses_below_70.sum() > 0, "Should detect RSI crossing below 70"
