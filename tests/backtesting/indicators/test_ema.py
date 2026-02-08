"""
Tests for EMA indicator calculation.

Uses real historical data from ZS (soybeans) to validate EMA calculation
accuracy, caching behavior, and edge case handling.
"""
import numpy as np
import pandas as pd
import pytest

from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.indicators import calculate_ema
from app.utils.backtesting_utils.indicators_utils import hash_series
from tests.backtesting.helpers.assertions import assert_valid_indicator, assert_indicator_varies
from tests.backtesting.indicators.indicator_test_utils import (
    setup_cache_test,
    assert_cache_hit_on_second_call,
    assert_indicator_structure,
    assert_longer_period_smoother,
    assert_different_params_use_different_cache,
    assert_cache_distinguishes_different_data,
    assert_empty_series_returns_empty,
)


# ==================== Helper Function ====================

def _calculate_ema(prices, period=14):
    """
    Helper function to calculate EMA with automatic hashing.

    Simplifies test code by handling hash calculation internally.
    """
    prices_hash = hash_series(prices)
    return calculate_ema(prices, period, prices_hash)


# ==================== Basic Logic Tests ====================

class TestEMABasicLogic:
    """Simple sanity checks for EMA basic behavior using shared test fixtures."""

    def test_ema_returns_series_with_correct_length(self, short_price_series):
        """EMA should return a series with same length as input."""
        ema = _calculate_ema(short_price_series, period=9)

        assert_indicator_structure(ema, len(short_price_series), 'series', indicator_name='EMA')

    def test_ema_lags_rising_prices(self, rising_price_series):
        """EMA should lag behind continuously rising prices."""
        ema = _calculate_ema(rising_price_series, period=9)

        # After warmup period, EMA should be below current price
        for i in range(10, len(rising_price_series)):
            assert ema.iloc[i] < rising_price_series.iloc[i], \
                f"EMA should lag below rising prices at index {i}"

    def test_ema_lags_falling_prices(self, falling_price_series):
        """EMA should lag behind continuously falling prices."""
        ema = _calculate_ema(falling_price_series, period=9)

        # After warmup period, EMA should be above current price
        for i in range(10, len(falling_price_series)):
            assert ema.iloc[i] > falling_price_series.iloc[i], \
                f"EMA should lag above falling prices at index {i}"

    def test_ema_changes_with_price_changes(self, medium_price_series):
        """EMA should change when prices change."""
        ema = _calculate_ema(medium_price_series, period=9)

        assert_indicator_varies(ema, 'EMA')

    def test_ema_has_no_initial_nans(self, short_price_series):
        """EMA calculation starts immediately (no initial NaN warmup)."""
        period = 9
        ema = _calculate_ema(short_price_series, period=period)

        # EMA starts calculating from first value (no NaN period)
        assert not ema.iloc[0:period].isna().all(), "EMA should not have all NaN in warmup"

    def test_shorter_period_more_responsive(self, volatile_price_series):
        """Shorter EMA period should be more responsive to price changes."""
        ema_short = _calculate_ema(volatile_price_series, period=5)
        ema_long = _calculate_ema(volatile_price_series, period=15)

        # Shorter period should be more volatile (higher std)
        short_std = ema_short.std()
        long_std = ema_long.std()

        assert short_std > long_std, \
            f"Shorter period should be more volatile: {short_std:.2f} vs {long_std:.2f}"

    def test_ema_approaches_constant_price(self, constant_price_series):
        """EMA should converge to price level when prices are constant."""
        ema = _calculate_ema(constant_price_series, period=9)

        # By end of constant period, EMA should be very close to constant price
        final_ema = ema.iloc[-1]
        constant_price = constant_price_series.iloc[0]
        assert abs(final_ema - constant_price) < 0.1, \
            f"EMA should converge to constant price, got {final_ema}"

    def test_ema_responds_to_price_magnitude(self, low_price_level_series, high_price_level_series):
        """EMA values should scale with price level."""
        ema_low = _calculate_ema(low_price_level_series, period=5)
        ema_high = _calculate_ema(high_price_level_series, period=5)

        # EMA values should scale proportionally
        # High prices are 10x low prices, so EMAs should follow
        ratio = ema_high.iloc[-1] / ema_low.iloc[-1]
        assert 9 < ratio < 11, \
            f"EMA should scale with price level, ratio {ratio:.1f} not near 10"


class TestEMACalculationWithRealData:
    """Test EMA calculation using real historical data."""

    def test_standard_ema_with_zs_hourly_data(self, zs_1h_data):
        """
        Test EMA(9) calculation on 2 years of ZS hourly data.

        Validates that EMA is calculated correctly across all data points
        and produces values in valid range.
        """
        ema = _calculate_ema(zs_1h_data['close'], period=9)

        # Validate structure
        assert len(ema) == len(zs_1h_data), "EMA length must match input data length"
        assert ema.index.equals(zs_1h_data.index), "EMA index must match input index"

        # Validate no NaN values (EMA calculates from start)
        assert not ema.isna().all(), "EMA should have valid values from start"

        # Validate range (EMA should be in reasonable range of prices)
        close_min = zs_1h_data['close'].min()
        close_max = zs_1h_data['close'].max()
        assert ema.min() >= close_min * 0.9, "EMA shouldn't go too far below price range"
        assert ema.max() <= close_max * 1.1, "EMA shouldn't go too far above price range"

        # EMA should track prices reasonably
        assert_indicator_varies(ema, 'EMA(9)')

    @pytest.mark.parametrize("period", [5, 9, 21, 50])
    def test_ema_with_different_periods(self, zs_1h_data, period):
        """
        Test EMA calculation with various standard periods.

        Validates that different periods produce valid results.
        """
        ema = _calculate_ema(zs_1h_data['close'], period=period)

        # Validate structure and variation
        assert_indicator_structure(ema, len(zs_1h_data), 'series', indicator_name=f'EMA({period})')
        assert_valid_indicator(ema, f'EMA({period})', min_val=0)
        assert_indicator_varies(ema, f'EMA({period})')

    def test_ema_extreme_but_valid_parameters(self, zs_1h_data):
        """
        Test EMA with extreme but technically valid parameters.

        Tests edge cases like very short period (2) and very long period (200).
        """
        # Very short period
        ema_short = _calculate_ema(zs_1h_data['close'], period=2)
        assert_valid_indicator(ema_short, 'EMA(2)', min_val=0)

        # Very long period
        ema_long = _calculate_ema(zs_1h_data['close'], period=200)
        assert_valid_indicator(ema_long, 'EMA(200)', min_val=0)

        # Longer period should be smoother (less volatile changes)
        assert_longer_period_smoother(ema_short, ema_long, 'EMA')

    def test_ema_values_match_expected_calculation(self, zs_1h_data):
        """
        Test that EMA values match the expected calculation formula.

        Manually calculates EMA for a subset and compares with indicator output.
        """
        period = 9
        subset = zs_1h_data['close'].iloc[100:150]  # 50 bars

        ema = _calculate_ema(subset, period=period)

        # Manual calculation using pandas ewm
        expected_ema = subset.ewm(span=period, adjust=False).mean()

        # Compare values
        np.testing.assert_allclose(
            ema.values,
            expected_ema.values,
            rtol=0.001,
            err_msg="EMA calculation doesn't match expected EWM formula"
        )


class TestEMAInMarketScenarios:
    """Test EMA behavior in different market conditions using real data."""

    def test_ema_in_trending_market(self, trending_market_data):
        """
        Test EMA behavior during strong trend.

        In uptrend, EMA should generally trend upward and lag below price.
        """
        if trending_market_data is None:
            pytest.skip("No trending market data available")

        ema = _calculate_ema(trending_market_data['close'], period=9)
        prices = trending_market_data['close']

        # EMA should trend in same direction as prices
        ema_trend = ema.iloc[-1] - ema.iloc[0]
        price_trend = prices.iloc[-1] - prices.iloc[0]

        assert (ema_trend > 0 and price_trend > 0) or (ema_trend < 0 and price_trend < 0), \
            "EMA should trend in same direction as prices"

        # EMA should show smooth progression
        assert ema.std() > 0, "EMA should vary in trending market"

    def test_ema_in_ranging_market(self, ranging_market_data):
        """
        Test EMA in sideways/ranging market.

        EMA should oscillate around the range midpoint.
        """
        if ranging_market_data is None:
            pytest.skip("No ranging market data available")

        ema = _calculate_ema(ranging_market_data['close'], period=9)
        prices = ranging_market_data['close']

        # EMA mean should be similar to price mean
        price_mean = prices.mean()
        ema_mean = ema.mean()

        assert abs(ema_mean - price_mean) / price_mean < 0.05, \
            "EMA mean should be close to price mean in ranging market"

    def test_ema_in_volatile_market(self, volatile_market_data):
        """
        Test EMA during extreme volatility.

        EMA should handle volatility and provide smoothing.
        """
        if volatile_market_data is None:
            pytest.skip("No volatile market data available")

        ema = _calculate_ema(volatile_market_data['close'], period=9)
        prices = volatile_market_data['close']

        # EMA should be smoother than raw prices
        price_std = prices.std()
        ema_std = ema.std()

        assert ema_std < price_std, \
            "EMA should smooth out volatility (lower std than prices)"


class TestEMACaching:
    """Test EMA caching behavior."""

    def test_cache_hit_returns_identical_values(self, zs_1h_data):
        """
        Verify cached EMA exactly matches fresh calculation.

        Tests that cache stores and retrieves EMA correctly.
        """
        # Setup: Clear cache and reset stats
        setup_cache_test()

        # First calculation (should miss due to empty cache)
        ema_1 = _calculate_ema(zs_1h_data['close'], period=9)
        misses_after_first = indicator_cache.misses

        # Second calculation (should hit cache)
        ema_2 = _calculate_ema(zs_1h_data['close'], period=9)

        # Verify cache was hit and results match
        assert indicator_cache.misses == misses_after_first, \
            f"Cache misses increased from {misses_after_first} to {indicator_cache.misses}"
        assert indicator_cache.hits > 0, f"Cache hits should be > 0, got {indicator_cache.hits}"
        assert_cache_hit_on_second_call(ema_1, ema_2, 'series')

    def test_cache_distinguishes_different_periods(self, zs_1h_data):
        """
        Verify cache stores different results for different periods.
        """
        indicator_cache.reset_stats()

        # Calculate with different periods
        ema_9 = _calculate_ema(zs_1h_data['close'], period=9)
        ema_21 = _calculate_ema(zs_1h_data['close'], period=21)

        # Validate different parameters produce different results
        assert_different_params_use_different_cache(ema_9, ema_21)

        # Recalculate - should hit cache
        misses_before = indicator_cache.misses
        ema_9_again = _calculate_ema(zs_1h_data['close'], period=9)
        assert indicator_cache.misses == misses_before, "Recalculation should hit cache"
        np.testing.assert_array_equal(ema_9.values, ema_9_again.values)

    def test_cache_distinguishes_different_data(self, zs_1h_data, cl_15m_data):
        """
        Verify cache stores different results for different data series.
        """
        indicator_cache.reset_stats()

        # Calculate EMA for two different datasets
        ema_zs = _calculate_ema(zs_1h_data['close'], period=9)
        ema_cl = _calculate_ema(cl_15m_data['close'], period=9)

        # Use utility to validate different data behavior
        assert_cache_distinguishes_different_data(ema_zs, ema_cl, 'EMA')


class TestEMAEdgeCases:
    """Test EMA with edge cases and error conditions."""

    def test_ema_with_insufficient_data(self, minimal_price_series):
        """
        Test EMA when data length is very small.

        EMA should still calculate but may have limited values.
        """
        ema = _calculate_ema(minimal_price_series, period=9)

        # Should return series of same length
        assert len(ema) == len(minimal_price_series), "EMA length should match input"
        assert isinstance(ema, pd.Series)

    def test_ema_with_constant_prices(self, constant_price_series):
        """
        Test EMA when prices don't change.

        With constant prices, EMA should equal the constant price.
        """
        ema = _calculate_ema(constant_price_series, period=9)

        # EMA should converge to constant price
        unique_price = constant_price_series.iloc[0]
        # After warmup, all values should be close to constant price
        final_values = ema.iloc[-5:]
        assert all(abs(val - unique_price) < 0.01 for val in final_values), \
            "EMA should equal constant price"

    def test_ema_invalid_period_zero(self, short_price_series):
        """Test that period=0 raises appropriate error."""
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            _calculate_ema(short_price_series, period=0)

    def test_ema_invalid_period_negative(self, short_price_series):
        """Test that negative period raises appropriate error."""
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            _calculate_ema(short_price_series, period=-5)

    def test_ema_with_empty_series(self, empty_price_series):
        """Test EMA with empty input series."""
        ema = _calculate_ema(empty_price_series, period=9)
        assert_empty_series_returns_empty(ema, 'series', 'EMA')


class TestEMADataTypes:
    """Test EMA with different input data types and structures."""

    def test_ema_with_series_name(self, zs_1h_data):
        """Test that EMA works with named series."""
        named_series = zs_1h_data['close'].rename('ZS_Close')
        ema = _calculate_ema(named_series, period=9)

        # Series should work correctly
        assert len(ema) == len(named_series)
        assert_valid_indicator(ema, 'EMA', min_val=0)

    def test_ema_with_different_datetime_frequencies(self, zs_1h_data, zs_1d_data):
        """
        Test EMA works with different time frequencies.

        Validates that EMA calculation is time-agnostic.
        """
        ema_hourly = _calculate_ema(zs_1h_data['close'], period=9)
        ema_daily = _calculate_ema(zs_1d_data['close'], period=9)

        # Both should produce valid EMA
        assert_valid_indicator(ema_hourly, 'EMA(9) Hourly', min_val=0)
        assert_valid_indicator(ema_daily, 'EMA(9) Daily', min_val=0)

        # Both should show variation
        assert ema_hourly.std() > 0, "Hourly EMA should vary"
        assert ema_daily.std() > 0, "Daily EMA should vary"

    def test_ema_preserves_index(self, zs_1h_data):
        """Test that EMA output maintains input index."""
        ema = _calculate_ema(zs_1h_data['close'], period=9)

        assert ema.index.equals(zs_1h_data.index), "EMA should preserve input index"
        assert isinstance(ema.index, pd.DatetimeIndex), "Index should remain DatetimeIndex"


class TestEMAPracticalUsage:
    """Test EMA in practical trading scenarios."""

    def test_ema_crossover_detection(self, zs_1h_data):
        """
        Test detecting EMA crossovers (fast crossing slow).

        Practical use: Generating entry/exit signals on EMA crossovers.
        """
        # Calculate fast and slow EMAs
        ema_fast = _calculate_ema(zs_1h_data['close'], period=9)
        ema_slow = _calculate_ema(zs_1h_data['close'], period=21)

        # Detect bullish crossover (fast crosses above slow)
        bullish_cross = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))

        # Detect bearish crossover (fast crosses below slow)
        bearish_cross = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

        # Should detect some crossovers in 2 years of data
        assert bullish_cross.sum() > 0, "Should detect bullish EMA crossovers"
        assert bearish_cross.sum() > 0, "Should detect bearish EMA crossovers"

    def test_ema_as_support_resistance(self, zs_1h_data):
        """
        Test EMA as dynamic support/resistance.

        Practical use: EMA can act as support in uptrends, resistance in downtrends.
        """
        ema = _calculate_ema(zs_1h_data['close'], period=21)
        prices = zs_1h_data['close']

        # Count times price bounces off EMA (comes close then moves away)
        # This is a simplified test - real bounce detection would be more complex
        price_near_ema = abs(prices - ema) / ema < 0.01  # Within 1% of EMA

        # Should have some instances where price approaches EMA
        assert price_near_ema.sum() > 0, "Price should approach EMA at times"

    def test_ema_trend_identification(self, zs_1h_data):
        """
        Test using EMA for trend identification.

        Practical use: Price above EMA suggests uptrend, below suggests downtrend.
        """
        ema = _calculate_ema(zs_1h_data['close'], period=50)
        prices = zs_1h_data['close']

        # Identify trend periods
        uptrend = prices > ema
        downtrend = prices < ema

        # Over 2 years, should have both uptrend and downtrend periods
        uptrend_pct = uptrend.sum() / len(prices) * 100
        downtrend_pct = downtrend.sum() / len(prices) * 100

        assert 20 < uptrend_pct < 80, "Should have reasonable uptrend periods"
        assert 20 < downtrend_pct < 80, "Should have reasonable downtrend periods"
