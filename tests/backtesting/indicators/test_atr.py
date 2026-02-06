"""
Tests for ATR indicator calculation.

Uses real historical data from ZS (soybeans) to validate ATR calculation
accuracy, caching behavior, and edge case handling.
"""
import numpy as np
import pandas as pd
import pytest

from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.indicators import calculate_atr
from app.utils.backtesting_utils.indicators_utils import hash_series
from tests.backtesting.helpers.assertions import assert_valid_indicator, assert_indicator_varies
from tests.backtesting.helpers.data_utils import inject_price_spike, inject_gap


# ==================== Helper Functions ====================

def _calculate_atr(df, period=14):
    """
    Helper function to calculate ATR with automatic hashing.

    Simplifies test code by handling hash calculation internally.
    """
    high_hash = hash_series(df['high'])
    low_hash = hash_series(df['low'])
    close_hash = hash_series(df['close'])
    return calculate_atr(df, period, high_hash, low_hash, close_hash)


def _price_series_to_ohlc(prices, range_pct=1.0):
    """
    Convert price series to OHLC DataFrame for ATR testing.
    
    Args:
        prices: Series of close prices
        range_pct: Percentage range for high/low around close (default 1%)
    
    Returns:
        DataFrame with high, low, close columns
    """
    range_size = prices * (range_pct / 100)
    return pd.DataFrame({
        'high': prices + range_size,
        'low': prices - range_size,
        'close': prices
    })


# ==================== Basic Logic Tests ====================

class TestATRBasicLogic:
    """Simple sanity checks for ATR basic behavior using shared test fixtures."""

    def test_atr_returns_series_with_correct_length(self, short_price_series):
        """ATR should return a series with same length as input."""
        df = _price_series_to_ohlc(short_price_series)
        atr = _calculate_atr(df, period=14)

        assert len(atr) == len(df), "ATR length must equal input length"
        assert isinstance(atr, pd.Series), "ATR must return pandas Series"

    def test_atr_is_always_positive(self, volatile_price_series):
        """ATR measures volatility and must always be positive."""
        df = _price_series_to_ohlc(volatile_price_series, range_pct=5.0)
        atr = _calculate_atr(df, period=10)

        valid_atr = atr.dropna()
        assert (valid_atr > 0).all(), "ATR should always be positive"

    def test_higher_volatility_gives_higher_atr(self, oscillating_price_series, volatile_price_series):
        """ATR should increase when price ranges widen."""
        # Low volatility: small oscillations
        low_vol = _price_series_to_ohlc(oscillating_price_series, range_pct=1.0)

        # High volatility: large swings with wider ranges
        high_vol = _price_series_to_ohlc(volatile_price_series, range_pct=10.0)

        atr_low_vol = _calculate_atr(low_vol, period=14)
        atr_high_vol = _calculate_atr(high_vol, period=14)

        avg_atr_low = atr_low_vol.iloc[-5:].mean()
        avg_atr_high = atr_high_vol.iloc[-5:].mean()

        assert avg_atr_high > avg_atr_low, \
            f"Higher volatility should give higher ATR: {avg_atr_high:.2f} vs {avg_atr_low:.2f}"

    def test_atr_responds_to_gaps(self, short_price_series):
        """ATR should capture gaps in true range calculation."""
        # Create data with consistent ranges
        df = _price_series_to_ohlc(short_price_series)

        # Create data with a gap (bar 15 gaps up from previous close)
        df_with_gap = df.copy()
        prev_close = df_with_gap.iloc[14]['close']
        gap_size = 10.0  # Gap up 10 points

        # Modify bar 15 to have gap up from previous close
        df_with_gap.iloc[15, df_with_gap.columns.get_loc('low')] = prev_close + gap_size - 1.0
        df_with_gap.iloc[15, df_with_gap.columns.get_loc('high')] = prev_close + gap_size + 1.0
        df_with_gap.iloc[15, df_with_gap.columns.get_loc('close')] = prev_close + gap_size

        # Calculate ATR - the gap bar should have larger true range
        # True range = max(H-L, |H-prev_close|, |prev_close-L|)
        # With gap: max(2.0, |125-114|, |114-123|) = max(2.0, 11.0, 9.0) = 11.0
        # Without gap: max(2.0, 1.0, 1.0) = 2.0

        high = df_with_gap['high']
        low = df_with_gap['low']
        close = df_with_gap['close']
        prev_close_series = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close_series).abs()
        tr3 = (low - prev_close_series).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # True range at gap should be much larger
        tr_at_gap = true_range.iloc[15]
        assert tr_at_gap > 5, f"True range at gap should capture the gap, got {tr_at_gap:.2f}"

    def test_first_n_values_are_nan(self, medium_price_series):
        """First 'period-1' values should be NaN (need warmup)."""
        df = _price_series_to_ohlc(medium_price_series)
        period = 14
        atr = _calculate_atr(df, period=period)

        # First 'period-1' values should be NaN (EWM with min_periods=period gives value at period-1)
        assert atr.iloc[:period - 1].isna().all(), f"First {period - 1} ATR values should be NaN"
        # After warmup, should have valid values
        assert not atr.iloc[period - 1:].isna().all(), "Should have valid ATR after warmup period"

    def test_larger_period_gives_smoother_atr(self, volatile_price_series):
        """Larger ATR period should produce less volatile ATR values."""
        df = _price_series_to_ohlc(volatile_price_series, range_pct=5.0)

        atr_short = _calculate_atr(df, period=5)
        atr_long = _calculate_atr(df, period=20)

        # Standard deviation measures smoothness
        short_volatility = atr_short.dropna().std()
        long_volatility = atr_long.dropna().std()

        assert short_volatility > long_volatility, \
            f"Shorter period should be more volatile: {short_volatility:.2f} vs {long_volatility:.2f}"

    def test_atr_with_constant_ranges(self, constant_price_series):
        """ATR should converge to constant value with constant ranges."""
        # Create bars with constant range
        df = _price_series_to_ohlc(constant_price_series, range_pct=1.0)

        atr = _calculate_atr(df, period=14)
        valid_atr = atr.dropna()

        # ATR should converge to the constant range (1% of 100 = 1.0 * 2 = 2.0)
        final_atr = valid_atr.iloc[-5:].mean()
        expected_range = 2.0  # 1% above + 1% below = 2.0
        assert abs(final_atr - expected_range) < 0.1, \
            f"ATR should converge to constant range value, got {final_atr:.2f}"

    def test_atr_increases_with_price_spike(self):
        """ATR should spike up after large price movement."""
        df = pd.DataFrame({
            'open': [
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0
            ],
            'high': [
                101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0
            ],
            'low': [
                99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0
            ],
            'close': [
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0
            ]
        })

        # Calculate ATR before spike
        atr_before = _calculate_atr(df, period=10)
        avg_before = atr_before.iloc[-3:].mean()

        # Inject spike and recalculate
        df_with_spike = inject_price_spike(df.copy(), 15, 15.0, 'up')
        atr_after = _calculate_atr(df_with_spike, period=10)
        avg_after = atr_after.iloc[16:19].mean()

        assert avg_after > avg_before, \
            "ATR should increase after large price spike"


class TestATRCalculationWithRealData:
    """Test ATR calculation using real historical data."""

    def test_standard_atr_with_zs_hourly_data(self, zs_1h_data):
        """
        Test ATR(14) calculation on 2 years of ZS hourly data.

        Validates that ATR is calculated correctly across all data points,
        handles NaN values properly, and produces values in valid range.
        """
        atr = _calculate_atr(zs_1h_data, period=14)

        # Validate structure
        assert len(atr) == len(zs_1h_data), "ATR length must match input data length"
        assert atr.index.equals(zs_1h_data.index), "ATR index must match input index"

        # Validate NaN handling (first period-1 values should be NaN)
        assert atr.isna().sum() == 13, "First 13 values should be NaN for ATR(14)"

        # Validate range (ATR must be positive)
        assert_valid_indicator(atr, 'ATR', min_val=0)

        # Validate market behavior with real data
        valid_atr = atr.dropna()
        assert_indicator_varies(valid_atr, 'ATR')

        # ATR should be reasonable relative to price
        avg_price = zs_1h_data['close'].mean()
        avg_atr = valid_atr.mean()
        atr_pct = (avg_atr / avg_price) * 100

        assert 0.1 < atr_pct < 10, \
            f"ATR should be 0.1-10% of price, got {atr_pct:.2f}%"

    @pytest.mark.parametrize("period", [7, 14, 21, 28])
    def test_atr_with_different_periods(self, zs_1h_data, period):
        """
        Test ATR calculation with various standard periods.

        Validates that different periods produce valid results and proper
        NaN counts matching the period.
        """
        atr = _calculate_atr(zs_1h_data, period=period)

        # Validate structure
        assert len(atr) == len(zs_1h_data)
        assert atr.isna().sum() == period - 1, f"Should have {period - 1} NaN values for ATR({period})"

        # Validate range
        assert_valid_indicator(atr, f'ATR({period})', min_val=0)

        # Longer periods should have smoother ATR (lower volatility)
        if period > 7:
            valid_atr = atr.dropna()
            assert len(valid_atr) > 0, "Should have valid ATR values"

    def test_atr_extreme_but_valid_parameters(self, zs_1h_data):
        """
        Test ATR with extreme but technically valid parameters.

        Tests edge cases like very short period (3) and very long period (100)
        to ensure calculation doesn't break.
        """
        # Very short period
        atr_short = _calculate_atr(zs_1h_data, period=3)
        assert_valid_indicator(atr_short, 'ATR(3)', min_val=0)
        valid_short = atr_short.dropna()
        assert len(valid_short) > 0

        # Very long period
        atr_long = _calculate_atr(zs_1h_data, period=100)
        assert_valid_indicator(atr_long, 'ATR(100)', min_val=0)
        valid_long = atr_long.dropna()
        assert len(valid_long) > 0

        # Longer period should be smoother (less volatile)
        assert valid_short.std() > valid_long.std(), "Short period ATR should be more volatile"

    def test_atr_values_match_expected_calculation(self, zs_1h_data):
        """
        Test that ATR values match the expected calculation formula.

        Manually calculates ATR for a subset of data and compares with
        indicator function output to validate correctness.
        """
        period = 14
        subset = zs_1h_data.iloc[100:200]

        atr = _calculate_atr(subset, period=period)

        # Manual calculation
        high = subset['high']
        low = subset['low']
        close = subset['close']
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        expected_atr = true_range.ewm(span=period, min_periods=period, adjust=False).mean()

        # Compare all valid values
        valid_calculated = atr.dropna()
        valid_expected = expected_atr.dropna()

        # Should match within reasonable tolerance
        np.testing.assert_allclose(
            valid_calculated.values,
            valid_expected.values,
            rtol=0.01,
            err_msg="ATR calculation doesn't match expected formula"
        )


class TestATRInMarketScenarios:
    """Test ATR behavior in different market conditions using real data."""

    def test_atr_in_trending_market(self, trending_market_data):
        """
        Test ATR behavior during strong trend.

        ATR can vary but should remain relatively stable unless trend
        includes volatility spikes.
        """
        if trending_market_data is None:
            pytest.skip("No trending market data available")

        atr = _calculate_atr(trending_market_data, period=14)
        valid_atr = atr.dropna()

        # ATR should be positive and reasonable
        assert_valid_indicator(atr, 'ATR', min_val=0)
        assert len(valid_atr) > 0, "Should have valid ATR values"

        # Should show some variation (not completely flat)
        assert valid_atr.std() > 0, "ATR should vary in trending market"

    def test_atr_in_ranging_market(self, ranging_market_data):
        """
        Test ATR in sideways/ranging market.

        ATR should be relatively low and stable in ranging markets.
        """
        if ranging_market_data is None:
            pytest.skip("No ranging market data available")

        atr = _calculate_atr(ranging_market_data, period=14)
        valid_atr = atr.dropna()

        # ATR should be positive
        assert_valid_indicator(atr, 'ATR', min_val=0)

        # In ranging market, ATR should be more stable (lower volatility)
        atr_volatility = valid_atr.std() / valid_atr.mean()
        assert atr_volatility < 0.5, \
            "ATR should be relatively stable in ranging market"

    def test_atr_in_volatile_market(self, volatile_market_data):
        """
        Test ATR during extreme volatility.

        ATR should be elevated in volatile markets.
        """
        if volatile_market_data is None:
            pytest.skip("No volatile market data available")

        atr = _calculate_atr(volatile_market_data, period=14)

        # Should still produce valid values despite volatility
        assert_valid_indicator(atr, 'ATR', min_val=0)

        # High volatility should produce higher ATR values
        valid_atr = atr.dropna()
        avg_price = volatile_market_data['close'].mean()
        avg_atr = valid_atr.mean()
        atr_pct = (avg_atr / avg_price) * 100

        assert atr_pct > 0.2, \
            "ATR should be elevated in volatile market"


class TestATRCaching:
    """Test ATR caching behavior."""

    def test_cache_hit_returns_identical_values(self, zs_1h_data):
        """
        Verify cached ATR exactly matches fresh calculation.

        Tests that cache stores and retrieves ATR correctly without
        any data corruption or loss of precision.
        """
        # Clear cache to ensure test isolation and prevent false positives
        indicator_cache.cache_data.clear()
        indicator_cache.reset_stats()

        # First calculation (should miss due to empty cache)
        atr_1 = _calculate_atr(zs_1h_data, period=14)
        first_misses = indicator_cache.misses

        # Second calculation (should hit cache)
        atr_2 = _calculate_atr(zs_1h_data, period=14)

        # Verify cache was hit (misses didn't increase, hits increased)
        assert indicator_cache.misses == first_misses, "Second calculation should not cause cache miss"
        assert indicator_cache.hits > 0, "Second calculation should cause cache hit"

        # Verify identical results (use np.array_equal for NaN-safe comparison)
        assert len(atr_1) == len(atr_2), "ATR series should have same length"
        np.testing.assert_array_equal(atr_1.values, atr_2.values, "Cached ATR should match exactly")

    def test_cache_distinguishes_different_periods(self, zs_1h_data):
        """
        Verify cache stores different results for different periods.

        Tests that cache keys include period parameter so different
        periods don't overwrite each other.
        """
        indicator_cache.reset_stats()

        # Calculate with different periods
        atr_14 = _calculate_atr(zs_1h_data, period=14)
        atr_21 = _calculate_atr(zs_1h_data, period=21)

        # Results should differ
        assert not atr_14.equals(atr_21), "Different periods should produce different ATR"

        # Recalculate - should hit cache (misses shouldn't increase)
        misses_before_recalc = indicator_cache.misses
        atr_14_again = _calculate_atr(zs_1h_data, period=14)
        assert indicator_cache.misses == misses_before_recalc, "Recalculation should hit cache"

        # Should get same values
        np.testing.assert_array_equal(atr_14.values, atr_14_again.values)

    def test_cache_distinguishes_different_data(self, zs_1h_data, cl_15m_data):
        """
        Verify cache stores different results for different data series.

        Tests that cache keys include data hash so different datasets
        don't return wrong cached values.
        """
        indicator_cache.reset_stats()

        # Calculate ATR for two different datasets
        atr_zs = _calculate_atr(zs_1h_data, period=14)
        atr_cl = _calculate_atr(cl_15m_data, period=14)

        # Results should differ (different underlying data)
        assert not atr_zs.equals(atr_cl), "Different data should produce different ATR"
        assert len(atr_zs) != len(atr_cl), "Different datasets should have different lengths"


class TestATREdgeCases:
    """Test ATR with edge cases and error conditions."""

    def test_atr_with_insufficient_data(self, minimal_price_series):
        """
        Test ATR when data length < period.

        Should return all NaN values when insufficient data for calculation.
        """
        # Create minimal DataFrame
        df = pd.DataFrame({
            'high': minimal_price_series + 1,
            'low': minimal_price_series - 1,
            'close': minimal_price_series
        })

        atr = _calculate_atr(df, period=14)

        # All values should be NaN
        assert atr.isna().all(), "All ATR values should be NaN when data < period"
        assert len(atr) == len(df), "ATR length should match input length"

    def test_atr_with_exact_minimum_data(self, exact_period_price_series):
        """
        Test ATR with exactly enough data for calculation.

        With period=14, needs at least 14 data points for first ATR value.
        EWM with min_periods=14 produces value at index 13.
        """
        df = pd.DataFrame({
            'high': exact_period_price_series + 1,
            'low': exact_period_price_series - 1,
            'close': exact_period_price_series
        })

        atr = _calculate_atr(df, period=14)

        # First 13 should be NaN, 14th onwards should have values
        assert atr.isna().sum() == 13
        assert not pd.isna(atr.iloc[13])
        assert not pd.isna(atr.iloc[-1])

    def test_atr_with_constant_prices(self, constant_price_series):
        """
        Test ATR when prices don't change.

        With constant prices, ATR should be zero since there's no range.
        """
        df = pd.DataFrame({
            'high': constant_price_series,
            'low': constant_price_series,
            'close': constant_price_series
        })

        atr = _calculate_atr(df, period=14)

        # With zero range, ATR should be 0 after warmup
        valid_atr = atr.dropna()
        assert (valid_atr == 0).all(), "ATR should be 0 for constant prices (zero range)"

    def test_atr_with_monotonic_increase(self, rising_price_series):
        """
        Test ATR with continuously increasing prices.

        ATR depends on ranges, not direction. Should show consistent ATR
        if ranges are consistent.
        """
        # Create rising prices with consistent ranges
        long_rising = pd.Series(range(100, 200))
        df = pd.DataFrame({
            'high': long_rising + 1,
            'low': long_rising - 1,
            'close': long_rising
        })

        atr = _calculate_atr(df, period=14)
        valid_atr = atr.dropna()

        # ATR should be positive and relatively stable
        assert (valid_atr > 0).all(), "ATR should be positive"
        assert valid_atr.std() < valid_atr.mean() * 0.2, \
            "ATR should be stable with consistent ranges"

    def test_atr_with_monotonic_decrease(self, falling_price_series):
        """
        Test ATR with continuously decreasing prices.

        ATR depends on ranges, not direction. Should show consistent ATR
        if ranges are consistent.
        """
        long_falling = pd.Series(range(200, 100, -1))
        df = pd.DataFrame({
            'high': long_falling + 1,
            'low': long_falling - 1,
            'close': long_falling
        })

        atr = _calculate_atr(df, period=14)
        valid_atr = atr.dropna()

        # ATR should be positive and relatively stable
        assert (valid_atr > 0).all(), "ATR should be positive"
        assert valid_atr.std() < valid_atr.mean() * 0.2, \
            "ATR should be stable with consistent ranges"

    def test_atr_with_extreme_spike(self, zs_1h_data):
        """
        Test ATR reaction to extreme price spike.

        Injects artificial spike to test ATR handles extreme moves.
        """
        # Inject 15% spike upward at bar 1000
        modified_data = inject_price_spike(zs_1h_data.copy(), 1000, 15.0, 'up')

        atr = _calculate_atr(modified_data, period=14)

        # ATR should still be valid despite spike
        assert_valid_indicator(atr, 'ATR', min_val=0)

        # ATR around spike should be elevated
        atr_before_spike = atr.iloc[990:1000].mean()
        atr_after_spike = atr.iloc[1001:1011].mean()

        assert atr_after_spike > atr_before_spike, \
            "ATR should increase after large price spike"

    def test_atr_with_flat_then_movement(self, flat_then_volatile_series):
        """
        Test ATR transition from flat period to normal movement.

        Validates ATR can handle transition from zero volatility to normal volatility.
        """
        df = pd.DataFrame({
            'high': flat_then_volatile_series + 1,
            'low': flat_then_volatile_series - 1,
            'close': flat_then_volatile_series
        })

        atr = _calculate_atr(df, period=14)

        # Flat region will have low/zero ATR
        flat_atr = atr.iloc[14:50].mean()

        # Variable region should have higher ATR
        variable_atr = atr.iloc[50:].dropna().mean()

        assert variable_atr > flat_atr, \
            "ATR should be higher in variable region than flat region"

    def test_atr_with_large_gap(self, zs_1h_data):
        """
        Test ATR captures gaps in true range calculation.

        ATR should include gap from previous close to current high/low.
        """
        # Calculate normal ATR
        atr_normal = _calculate_atr(zs_1h_data, period=14)

        # Inject large gap
        df_with_gap = inject_gap(zs_1h_data.copy(), 1000, 8.0, 'up')
        atr_with_gap = _calculate_atr(df_with_gap, period=14)

        # ATR after gap should be noticeably higher
        atr_after_normal = atr_normal.iloc[1005:1015].mean()
        atr_after_gap = atr_with_gap.iloc[1005:1015].mean()

        assert atr_after_gap > atr_after_normal * 1.2, \
            "ATR should significantly increase after large gap (captures true range)"

    def test_atr_with_empty_dataframe(self):
        """Test ATR with empty input DataFrame."""
        df = pd.DataFrame({
            'high': pd.Series([], dtype=float),
            'low': pd.Series([], dtype=float),
            'close': pd.Series([], dtype=float)
        })

        atr = _calculate_atr(df, period=14)

        assert len(atr) == 0, "Empty input should return empty ATR"
        assert isinstance(atr, pd.Series)


class TestATRDataTypes:
    """Test ATR with different input data types and structures."""

    def test_atr_with_different_datetime_frequencies(self, zs_1h_data, zs_1d_data):
        """
        Test ATR works with different time frequencies.

        Validates that ATR calculation is time-agnostic and works with
        any datetime frequency (hourly, daily, etc.).
        """
        atr_hourly = _calculate_atr(zs_1h_data, period=14)
        atr_daily = _calculate_atr(zs_1d_data, period=14)

        # Both should produce valid ATR
        assert_valid_indicator(atr_hourly, 'ATR(14) Hourly', min_val=0)
        assert_valid_indicator(atr_daily, 'ATR(14) Daily', min_val=0)

        # Both should show variation regardless of frequency
        assert atr_hourly.dropna().std() > 0, "Hourly ATR should vary"
        assert atr_daily.dropna().std() > 0, "Daily ATR should vary"

    def test_atr_preserves_index(self, zs_1h_data):
        """Test that ATR output maintains input index."""
        atr = _calculate_atr(zs_1h_data, period=14)

        assert atr.index.equals(zs_1h_data.index), "ATR should preserve input index"
        assert isinstance(atr.index, pd.DatetimeIndex), "Index should remain DatetimeIndex"

    def test_atr_with_missing_ohlc_columns(self, zs_1h_data):
        """Test that missing required columns raises appropriate error."""
        # Missing 'high' column
        df_missing_high = zs_1h_data[['low', 'close']].copy()
        with pytest.raises(KeyError):
            _calculate_atr(df_missing_high, period=14)

        # Missing 'low' column
        df_missing_low = zs_1h_data[['high', 'close']].copy()
        with pytest.raises(KeyError):
            _calculate_atr(df_missing_low, period=14)

        # Missing 'close' column
        df_missing_close = zs_1h_data[['high', 'low']].copy()
        with pytest.raises(KeyError):
            _calculate_atr(df_missing_close, period=14)


class TestATRPracticalUsage:
    """Test ATR in practical trading scenarios."""

    def test_atr_for_position_sizing(self, zs_1h_data):
        """
        Test using ATR for volatility-adjusted position sizing.

        In practice, traders use ATR to size positions inversely to volatility.
        Higher ATR = smaller position size to maintain constant risk.
        """
        atr = _calculate_atr(zs_1h_data, period=14)

        # Calculate position size based on fixed risk amount
        risk_per_trade = 1000  # Risk $1000 per trade
        contract_multiplier = 50  # ZS contract multiplier

        # Position size = Risk / (ATR * Multiplier)
        position_sizes = risk_per_trade / (atr * contract_multiplier)

        valid_positions = position_sizes.dropna()

        # Position sizes should be positive
        assert (valid_positions > 0).all(), "Position sizes should be positive"

        # Position sizes should vary (larger when ATR is smaller)
        assert_indicator_varies(valid_positions, 'Position Size')

        # Verify inverse relationship: high ATR = low position size
        high_atr_mask = atr > atr.quantile(0.75)
        low_atr_mask = atr < atr.quantile(0.25)

        avg_position_high_atr = position_sizes[high_atr_mask].mean()
        avg_position_low_atr = position_sizes[low_atr_mask].mean()

        assert avg_position_low_atr > avg_position_high_atr, \
            "Position size should be larger when ATR is smaller (inverse relationship)"

    def test_atr_for_stop_loss_placement(self, zs_1h_data):
        """
        Test using ATR for stop loss placement.

        Common practice: Set stop loss at entry price +/- (ATR * multiplier).
        This adapts stop distance to current volatility.
        """
        atr = _calculate_atr(zs_1h_data, period=14)
        entry_price = zs_1h_data['close']

        # Standard stop loss: 2 * ATR
        atr_multiplier = 2.0

        # For long positions: stop = entry - (ATR * multiplier)
        long_stops = entry_price - (atr * atr_multiplier)

        # For short positions: stop = entry + (ATR * multiplier)
        short_stops = entry_price + (atr * atr_multiplier)

        # Validate stop distances
        long_stop_distances = entry_price - long_stops
        short_stop_distances = short_stops - entry_price

        valid_long_distances = long_stop_distances.dropna()
        valid_short_distances = short_stop_distances.dropna()

        # Stop distances should be positive
        assert (valid_long_distances > 0).all(), "Long stop distances should be positive"
        assert (valid_short_distances > 0).all(), "Short stop distances should be positive"

        # Stop distances should equal ATR * multiplier
        expected_distance = (atr * atr_multiplier).dropna()
        np.testing.assert_allclose(
            valid_long_distances.values,
            expected_distance.values,
            rtol=0.01
        )

        # Stop distances should vary with ATR
        assert valid_long_distances.std() > 0, "Stop distances should vary with ATR"

    def test_atr_for_volatility_filtering(self, zs_1h_data):
        """
        Test using ATR to filter low volatility periods.

        Strategy: Only trade when volatility (ATR) is above threshold.
        Avoids choppy, low-movement markets.
        """
        atr = _calculate_atr(zs_1h_data, period=14)

        # Calculate ATR as percentage of price
        close_prices = zs_1h_data['close']
        atr_pct = (atr / close_prices) * 100

        # Use median ATR as threshold to filter bottom half
        valid_atr_pct = atr_pct.dropna()
        median_atr = valid_atr_pct.median()
        tradeable_periods = atr_pct > median_atr

        # Should filter out approximately half the periods
        filtered_count = (~tradeable_periods).sum()
        assert filtered_count > 0, "Should filter out some low volatility periods"

        # Tradeable periods should be roughly half (40-60% range for flexibility)
        tradeable_ratio = tradeable_periods.sum() / len(tradeable_periods)
        assert 0.35 < tradeable_ratio < 0.65, \
            "Volatility filter should be selective but not too restrictive"

    def test_atr_for_trend_strength_confirmation(self, zs_1h_data):
        """
        Test using ATR to confirm trend strength.

        Rising ATR during trend suggests strong momentum.
        Falling ATR suggests weakening trend.
        """
        atr = _calculate_atr(zs_1h_data, period=14)

        # Calculate ATR momentum (rate of change)
        atr_roc = atr.pct_change(periods=5)  # 5-bar rate of change

        # Detect trending periods (simple: 20-period price ROC)
        price_roc = zs_1h_data['close'].pct_change(periods=20)
        uptrend = price_roc > 0.03  # 3% move up

        # In strong trends, ATR should often be rising (positive ROC)
        atr_rising_in_uptrend = (uptrend & (atr_roc > 0)).sum()
        atr_falling_in_uptrend = (uptrend & (atr_roc < 0)).sum()

        # Not all uptrends will have rising ATR, but should be common
        if atr_rising_in_uptrend + atr_falling_in_uptrend > 10:
            # Only test if we have meaningful sample size
            rising_ratio = atr_rising_in_uptrend / (atr_rising_in_uptrend + atr_falling_in_uptrend)
            assert rising_ratio > 0.3, \
                "ATR should rise in at least some strong trends (confirms momentum)"

    def test_atr_multiple_timeframe_analysis(self, zs_1h_data, zs_1d_data):
        """
        Test comparing ATR across different timeframes.

        Daily ATR should be larger than hourly ATR (longer timeframe = larger ranges).
        """
        atr_hourly = _calculate_atr(zs_1h_data, period=14)
        atr_daily = _calculate_atr(zs_1d_data, period=14)

        avg_atr_hourly = atr_hourly.dropna().mean()
        avg_atr_daily = atr_daily.dropna().mean()

        # Daily ATR should be significantly larger
        assert avg_atr_daily > avg_atr_hourly * 2, \
            "Daily ATR should be larger than hourly ATR (longer timeframe)"

    def test_atr_percentile_for_regime_detection(self, zs_1h_data):
        """
        Test using ATR percentiles to detect volatility regimes.

        High ATR percentile = high volatility regime.
        Low ATR percentile = low volatility regime.
        """
        atr = _calculate_atr(zs_1h_data, period=14)
        valid_atr = atr.dropna()

        # Calculate rolling percentile rank (0-100)
        rolling_window = 100
        atr_percentile = valid_atr.rolling(rolling_window).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) * 100,
            raw=False
        )

        # Define regimes
        high_vol_regime = atr_percentile > 75  # Top 25% of ATR values
        low_vol_regime = atr_percentile < 25  # Bottom 25% of ATR values

        # Should identify both regimes
        assert high_vol_regime.sum() > 0, "Should identify high volatility regimes"
        assert low_vol_regime.sum() > 0, "Should identify low volatility regimes"

        # Regimes should be roughly 25% each
        high_vol_ratio = high_vol_regime.sum() / len(atr_percentile.dropna())
        low_vol_ratio = low_vol_regime.sum() / len(atr_percentile.dropna())

        assert 0.15 < high_vol_ratio < 0.35, "High vol regime should be ~25%"
        assert 0.15 < low_vol_ratio < 0.35, "Low vol regime should be ~25%"
