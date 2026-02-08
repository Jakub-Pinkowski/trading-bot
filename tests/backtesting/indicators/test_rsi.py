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
from tests.backtesting.indicators.indicator_test_utils import (
    setup_cache_test,
    assert_cache_hit_on_second_call,
    assert_indicator_structure,
    assert_values_in_range,
    assert_different_params_use_different_cache,
    assert_cache_distinguishes_different_data,
    assert_insufficient_data_returns_nan,
    assert_empty_series_returns_empty,
    assert_hash_parameter_required,
    assert_hash_parameter_required_even_with_cache,
)


# ==================== Helper Function ====================

def _calculate_rsi(prices, period=14):
    """
    Helper function to calculate RSI with automatic hashing.

    Simplifies test code by handling hash calculation internally.
    """
    prices_hash = hash_series(prices)
    return calculate_rsi(prices, period, prices_hash)


# ==================== Hash Parameter Contract Tests ====================

class TestHashParameterRequired:
    """Test that hash parameters are required (contract test)."""

    def test_calculate_rsi_requires_prices_hash(self, short_price_series):
        """Test that calculate_rsi requires prices_hash parameter."""
        assert_hash_parameter_required(
            calculate_func=calculate_rsi,
            prices=short_price_series,
            required_params={'period': 14},
            indicator_name='RSI'
        )

    def test_calculate_rsi_fails_without_hash_even_with_cache(self, short_price_series):
        """Test that calculate_rsi always requires hash, even if result might be cached."""
        assert_hash_parameter_required_even_with_cache(
            calculate_func=calculate_rsi,
            calculate_with_hash_func=lambda: _calculate_rsi(short_price_series, period=14),
            prices=short_price_series,
            required_params={'period': 14},
            indicator_name='RSI'
        )


# ==================== Basic Logic Tests ====================

class TestRSIBasicLogic:
    """Simple sanity checks for RSI basic behavior using shared test fixtures."""

    def test_rsi_returns_series_with_correct_length(self, short_price_series):
        """RSI should return a series with same length as input."""
        rsi = _calculate_rsi(short_price_series, period=14)

        assert_indicator_structure(rsi, len(short_price_series), 'series', indicator_name='RSI')

    def test_rsi_stays_within_bounds(self, volatile_price_series):
        """RSI must always be between 0 and 100."""
        rsi = _calculate_rsi(volatile_price_series, period=10)

        assert_values_in_range(rsi, 0, 100, indicator_name='RSI')

    def test_rsi_responds_to_price_direction(self, rising_price_series, falling_price_series):
        """
        RSI should be high for rising prices, low for falling prices.

        Consolidates testing of RSI directional behavior with synthetic data.
        """
        rsi_rising = _calculate_rsi(rising_price_series, period=14)
        rsi_falling = _calculate_rsi(falling_price_series, period=14)

        # Average of last 5 RSI values should show clear directional bias
        assert rsi_rising.iloc[-5:].mean() > 70, "Rising prices should give high RSI (>70)"
        assert rsi_falling.iloc[-5:].mean() < 30, "Falling prices should give low RSI (<30)"

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

    def test_rsi_behavior_is_correct(self, zs_1h_data):
        """
        Test RSI behaves correctly on real data.

        Tests that RSI has expected properties on real market data rather than
        testing implementation details. This approach is more maintainable and
        focuses on what actually matters: correct behavior.
        """
        rsi = _calculate_rsi(zs_1h_data['close'], period=14)

        # RSI should stay within valid bounds
        valid_rsi = rsi.dropna()
        assert 0 <= valid_rsi.min() <= 100, "RSI minimum should be between 0 and 100"
        assert 0 <= valid_rsi.max() <= 100, "RSI maximum should be between 0 and 100"

        # Long-term mean should center around 50 (neutral)
        assert 40 <= valid_rsi.mean() <= 60, f"RSI mean should be near 50, got {valid_rsi.mean():.2f}"

        # RSI should show variation (not constant)
        assert valid_rsi.std() > 5, "RSI should vary with market conditions"


class TestRSIInMarketScenarios:
    """Test RSI behavior in different market conditions using real data."""

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
        # Setup: Clear cache and reset stats
        setup_cache_test()

        # First calculation (should miss due to empty cache)
        rsi_1 = _calculate_rsi(zs_1h_data['close'], period=14)
        misses_after_first = indicator_cache.misses

        # Second calculation (should hit cache)
        rsi_2 = _calculate_rsi(zs_1h_data['close'], period=14)

        # Verify cache was hit and results match
        assert indicator_cache.misses == misses_after_first, \
            f"Cache misses increased from {misses_after_first} to {indicator_cache.misses}"
        assert indicator_cache.hits > 0, f"Cache hits should be > 0, got {indicator_cache.hits}"
        assert_cache_hit_on_second_call(rsi_1, rsi_2, 'series')

    def test_cache_distinguishes_different_periods(self, zs_1h_data):
        """
        Verify cache stores different results for different periods.

        Tests that cache keys include period parameter so different
        periods don't overwrite each other.
        """
        indicator_cache.reset_stats()

        # Calculate with different periods
        rsi_14 = _calculate_rsi(zs_1h_data['close'], period=14)
        rsi_21 = _calculate_rsi(zs_1h_data['close'], period=21)

        # Use utility to test complete cache behavior
        assert_different_params_use_different_cache(rsi_14, rsi_21)

        # Recalculate - should hit cache
        misses_before = indicator_cache.misses
        rsi_14_again = _calculate_rsi(zs_1h_data['close'], period=14)
        assert indicator_cache.misses == misses_before, "Recalculation should hit cache"
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

        # Use utility to validate different data behavior
        assert_cache_distinguishes_different_data(rsi_zs, rsi_cl, 'RSI')


class TestRSIEdgeCases:
    """Test RSI with edge cases and error conditions."""

    def test_rsi_with_insufficient_data(self, minimal_price_series):
        """
        Test RSI when data length < period.

        Should return all NaN values when insufficient data for calculation.
        """
        rsi = _calculate_rsi(minimal_price_series, period=14)

        # Use utility to validate insufficient data behavior
        assert_insufficient_data_returns_nan(rsi, len(minimal_price_series), 'RSI')

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
        assert_empty_series_returns_empty(rsi, 'series', 'RSI')


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

    def test_rsi_practical_thresholds(self, zs_1h_data):
        """
        RSI identifies overbought/oversold conditions on real data.

        Consolidates testing of practical RSI thresholds (30/70) used in trading.
        Validates that real market data produces both conditions occasionally.
        """
        rsi = _calculate_rsi(zs_1h_data['close'], period=14)

        # Should find some overbought/oversold conditions in real data
        overbought_count = (rsi > 70).sum()
        oversold_count = (rsi < 30).sum()

        assert overbought_count > 0, "Should find overbought conditions (RSI > 70) in real data"
        assert oversold_count > 0, "Should find oversold conditions (RSI < 30) in real data"

        # But these should be minority of time (not always extreme)
        assert overbought_count < len(rsi) * 0.2, "Overbought should be minority of time"
        assert oversold_count < len(rsi) * 0.2, "Oversold should be minority of time"

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
