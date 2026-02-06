"""
Tests for Bollinger Bands indicator calculation.

Uses real historical data from ZS (soybeans) to validate Bollinger Bands calculation
accuracy, caching behavior, and edge case handling.
"""
import numpy as np
import pandas as pd
import pytest

from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.indicators import calculate_bollinger_bands
from app.utils.backtesting_utils.indicators_utils import hash_series
from tests.backtesting.helpers.assertions import assert_indicator_varies


# ==================== Helper Function ====================

def _calculate_bollinger_bands(prices, period=20, num_std=2.0):
    """
    Helper function to calculate Bollinger Bands with automatic hashing.

    Simplifies test code by handling hash calculation internally.
    """
    prices_hash = hash_series(prices)
    return calculate_bollinger_bands(prices, period, num_std, prices_hash)


# ==================== Basic Logic Tests ====================

class TestBollingerBandsBasicLogic:
    """Simple sanity checks for Bollinger Bands basic behavior using shared test fixtures."""

    def test_bollinger_bands_returns_dataframe_with_correct_columns(self, medium_price_series):
        """Bollinger Bands should return DataFrame with three band columns."""
        bb = _calculate_bollinger_bands(medium_price_series, period=20, num_std=2.0)

        assert isinstance(bb, pd.DataFrame), "Bollinger Bands must return DataFrame"
        assert len(bb) == len(medium_price_series), "BB length must equal input length"

        expected_columns = ['middle_band', 'upper_band', 'lower_band']
        assert list(bb.columns) == expected_columns, f"Expected columns {expected_columns}"

    def test_upper_greater_than_middle_greater_than_lower(self, volatile_price_series):
        """Upper band must always be >= middle band >= lower band."""
        bb = _calculate_bollinger_bands(volatile_price_series, period=20, num_std=2.0)

        valid_bb = bb.dropna()
        assert (valid_bb['upper_band'] >= valid_bb['middle_band']).all(), \
            "Upper band must be >= middle band"
        assert (valid_bb['middle_band'] >= valid_bb['lower_band']).all(), \
            "Middle band must be >= lower band"

    def test_middle_band_is_sma(self, rising_price_series):
        """Middle band should equal simple moving average."""
        period = 20
        bb = _calculate_bollinger_bands(rising_price_series, period=period, num_std=2.0)

        manual_sma = rising_price_series.rolling(window=period).mean()

        # Compare valid values
        valid_indices = bb['middle_band'].dropna().index
        np.testing.assert_allclose(
            bb.loc[valid_indices, 'middle_band'].values,
            manual_sma.loc[valid_indices].values,
            rtol=0.001,
            err_msg="Middle band should equal SMA"
        )

    def test_band_width_proportional_to_volatility(self, oscillating_price_series, volatile_price_series):
        """Band width should increase with price volatility."""
        bb_stable = _calculate_bollinger_bands(oscillating_price_series, period=20, num_std=2.0)
        bb_volatile = _calculate_bollinger_bands(volatile_price_series, period=20, num_std=2.0)

        # Calculate average band width
        stable_width = (bb_stable['upper_band'] - bb_stable['lower_band']).dropna().mean()
        volatile_width = (bb_volatile['upper_band'] - bb_volatile['lower_band']).dropna().mean()

        assert volatile_width > stable_width, \
            f"Volatile market should have wider bands: {volatile_width:.2f} vs {stable_width:.2f}"

    def test_band_width_increases_with_num_std(self, volatile_price_series):
        """Bands should be wider with higher number of standard deviations."""
        bb_1std = _calculate_bollinger_bands(volatile_price_series, period=20, num_std=1.0)
        bb_2std = _calculate_bollinger_bands(volatile_price_series, period=20, num_std=2.0)
        bb_3std = _calculate_bollinger_bands(volatile_price_series, period=20, num_std=3.0)

        # Get valid values (last 5 bars for averaging)
        valid_idx = bb_1std['middle_band'].dropna().index[-5:]

        width_1std = (bb_1std.loc[valid_idx, 'upper_band'] - bb_1std.loc[valid_idx, 'lower_band']).mean()
        width_2std = (bb_2std.loc[valid_idx, 'upper_band'] - bb_2std.loc[valid_idx, 'lower_band']).mean()
        width_3std = (bb_3std.loc[valid_idx, 'upper_band'] - bb_3std.loc[valid_idx, 'lower_band']).mean()

        assert width_2std > width_1std, "2 std should be wider than 1 std"
        assert width_3std > width_2std, "3 std should be wider than 2 std"

    def test_first_n_values_are_nan(self):
        """First 'period-1' values should be NaN (need warmup)."""
        prices = pd.Series(range(100, 150))
        period = 20
        bb = _calculate_bollinger_bands(prices, period=period, num_std=2.0)

        # First 'period-1' values should be NaN
        assert bb['middle_band'].iloc[:period - 1].isna().all(), \
            f"First {period - 1} middle band values should be NaN"
        assert bb['upper_band'].iloc[:period - 1].isna().all(), \
            f"First {period - 1} upper band values should be NaN"
        assert bb['lower_band'].iloc[:period - 1].isna().all(), \
            f"First {period - 1} lower band values should be NaN"

        # After warmup, should have valid values
        assert not bb['middle_band'].iloc[period:].isna().all(), \
            "Should have valid middle band after warmup period"


class TestBollingerBandsCalculationWithRealData:
    """Test Bollinger Bands calculation using real historical data."""

    def test_standard_bollinger_bands_with_zs_hourly_data(self, zs_1h_data):
        """
        Test BB(20,2) calculation on 2 years of ZS hourly data.

        Validates that BB is calculated correctly across all data points,
        handles NaN values properly, and produces values in valid range.
        """
        bb = _calculate_bollinger_bands(zs_1h_data['close'], period=20, num_std=2.0)

        # Validate structure
        assert len(bb) == len(zs_1h_data), "BB length must match input data length"
        assert bb.index.equals(zs_1h_data.index), "BB index must match input index"

        # Validate NaN handling (first period-1 values should be NaN)
        assert bb['middle_band'].isna().sum() == 19, "First 19 values should be NaN for BB(20)"

        # Validate all bands are positive
        valid_bb = bb.dropna()
        assert (valid_bb['middle_band'] > 0).all(), "Middle band must be positive"
        assert (valid_bb['upper_band'] > 0).all(), "Upper band must be positive"
        assert (valid_bb['lower_band'] > 0).all(), "Lower band must be positive"

        # Validate band relationships
        assert (valid_bb['upper_band'] >= valid_bb['middle_band']).all()
        assert (valid_bb['middle_band'] >= valid_bb['lower_band']).all()

        # Validate bands vary (not constant)
        assert_indicator_varies(valid_bb['middle_band'], 'Middle Band')
        assert_indicator_varies(valid_bb['upper_band'], 'Upper Band')
        assert_indicator_varies(valid_bb['lower_band'], 'Lower Band')

    @pytest.mark.parametrize("period", [10, 20, 50, 100])
    def test_bollinger_bands_with_different_periods(self, zs_1h_data, period):
        """
        Test BB calculation with various standard periods.

        Validates that different periods produce valid results and proper
        NaN counts matching the period.
        """
        bb = _calculate_bollinger_bands(zs_1h_data['close'], period=period, num_std=2.0)

        # Validate structure
        assert len(bb) == len(zs_1h_data)
        assert bb['middle_band'].isna().sum() == period - 1, \
            f"Should have {period - 1} NaN values for BB({period})"

        # Validate band relationships
        valid_bb = bb.dropna()
        assert len(valid_bb) > 0, "Should have valid BB values"
        assert (valid_bb['upper_band'] >= valid_bb['middle_band']).all()
        assert (valid_bb['middle_band'] >= valid_bb['lower_band']).all()


class TestBollingerBandsInMarketScenarios:
    """Test Bollinger Bands behavior in different market conditions using real data."""

    def test_bollinger_bands_in_trending_market(self, trending_market_data):
        """
        Test BB behavior during strong trend.

        In trending markets, prices often move along one band. The bands
        should still maintain proper relationships and not break.
        """
        if trending_market_data is None:
            pytest.skip("No trending market data available")

        bb = _calculate_bollinger_bands(trending_market_data['close'], period=20, num_std=2.0)
        valid_bb = bb.dropna()

        # Bands should maintain proper order
        assert (valid_bb['upper_band'] >= valid_bb['middle_band']).all()
        assert (valid_bb['middle_band'] >= valid_bb['lower_band']).all()

        # Middle band should trend with price
        assert valid_bb['middle_band'].std() > 0, "Middle band should vary in trending market"


class TestBollingerBandsCaching:
    """Test Bollinger Bands caching behavior."""

    def test_cache_hit_returns_identical_values(self, zs_1h_data):
        """
        Verify cached BB exactly matches fresh calculation.

        Tests that cache stores and retrieves BB correctly without
        any data corruption or loss of precision.
        """
        # Clear cache to ensure test isolation and prevent false positives
        indicator_cache.cache_data.clear()
        indicator_cache.reset_stats()

        # First calculation (should miss due to empty cache)
        bb_1 = _calculate_bollinger_bands(zs_1h_data['close'], period=20, num_std=2.0)
        assert indicator_cache.misses == 1, "First calculation should cause cache miss"

        # Second calculation (should hit cache)
        bb_2 = _calculate_bollinger_bands(zs_1h_data['close'], period=20, num_std=2.0)

        # Verify cache was hit (misses remained at 1)
        assert indicator_cache.misses == 1, \
            "Second calculation should not cause cache miss"
        assert indicator_cache.hits == 1, \
            "Second calculation should cause cache hit"

        # Verify identical results
        assert len(bb_1) == len(bb_2), "BB DataFrames should have same length"
        for col in ['middle_band', 'upper_band', 'lower_band']:
            np.testing.assert_array_equal(
                bb_1[col].values,
                bb_2[col].values,
                err_msg=f"Cached BB {col} should match exactly"
            )


class TestBollingerBandsEdgeCases:
    """Test Bollinger Bands with edge cases and error conditions."""

    def test_bollinger_bands_with_insufficient_data(self, minimal_price_series):
        """
        Test BB when data length < period.

        Should return all NaN values when insufficient data for calculation.
        """
        bb = _calculate_bollinger_bands(minimal_price_series, period=20, num_std=2.0)

        # All values should be NaN
        assert bb['middle_band'].isna().all(), \
            "All middle band values should be NaN when data < period"
        assert bb['upper_band'].isna().all(), \
            "All upper band values should be NaN when data < period"
        assert bb['lower_band'].isna().all(), \
            "All lower band values should be NaN when data < period"
        assert len(bb) == len(minimal_price_series), \
            "BB length should match input length"

    def test_bollinger_bands_with_constant_prices(self, constant_price_series):
        """
        Test BB when prices don't change.

        With constant prices (zero variance), bands should collapse to
        middle band (zero width). This is correct behavior.
        """
        bb = _calculate_bollinger_bands(constant_price_series, period=20, num_std=2.0)

        valid_bb = bb.dropna()

        # All bands should be equal (or very close)
        assert len(valid_bb) > 0, "Should have valid values"

        band_width = valid_bb['upper_band'] - valid_bb['lower_band']
        assert (band_width < 0.001).all(), \
            "Constant prices should produce near-zero band width"

    def test_bollinger_bands_with_empty_series(self, empty_price_series):
        """Test BB with empty input series."""
        bb = _calculate_bollinger_bands(empty_price_series, period=20, num_std=2.0)

        assert len(bb) == 0, "Empty input should return empty BB"
        assert isinstance(bb, pd.DataFrame)
        assert list(bb.columns) == ['middle_band', 'upper_band', 'lower_band']


class TestBollingerBandsDataTypes:
    """Test Bollinger Bands with different input data types and structures."""

    def test_bollinger_bands_with_series_name(self, zs_1h_data):
        """Test that BB works correctly with named series."""
        named_series = zs_1h_data['close'].rename('ZS_Close')
        bb = _calculate_bollinger_bands(named_series, period=20, num_std=2.0)

        # Series should still work correctly
        assert len(bb) == len(named_series)
        assert (bb.dropna()['upper_band'] >= bb.dropna()['middle_band']).all()

    def test_bollinger_bands_with_different_datetime_frequencies(self, zs_1h_data, zs_1d_data):
        """
        Test BB works with different time frequencies.

        Validates that BB calculation is time-agnostic and works with
        any datetime frequency (hourly, daily, etc.).
        """
        bb_hourly = _calculate_bollinger_bands(zs_1h_data['close'], period=20, num_std=2.0)
        bb_daily = _calculate_bollinger_bands(zs_1d_data['close'], period=20, num_std=2.0)

        # Both should produce valid BB
        valid_hourly = bb_hourly.dropna()
        valid_daily = bb_daily.dropna()

        assert len(valid_hourly) > 0, "Hourly BB should have valid values"
        assert len(valid_daily) > 0, "Daily BB should have valid values"

        # Both should maintain proper band relationships
        assert (valid_hourly['upper_band'] >= valid_hourly['middle_band']).all()
        assert (valid_daily['upper_band'] >= valid_daily['middle_band']).all()


class TestBollingerBandsPracticalUsage:
    """Test Bollinger Bands in practical trading scenarios."""

    def test_bollinger_bands_squeeze_detection(self, zs_1h_data):
        """
        Test that BB can detect squeeze (narrow bands).

        Squeeze occurs when volatility contracts, indicating potential
        breakout. Practical use: Identifying consolidation before moves.
        """
        bb = _calculate_bollinger_bands(zs_1h_data['close'], period=20, num_std=2.0)

        # Calculate band width as percentage of middle band
        valid_bb = bb.dropna()
        band_width_pct = ((valid_bb['upper_band'] - valid_bb['lower_band']) /
                          valid_bb['middle_band']) * 100

        # Use bottom 20th percentile as squeeze threshold (relative to actual data)
        squeeze_threshold = band_width_pct.quantile(0.20)
        squeeze = band_width_pct < squeeze_threshold
        squeeze_count = squeeze.sum()

        # Real market data should have some squeeze periods
        assert squeeze_count > 0, "Should find squeeze conditions in real data"
        # Squeeze should be minority of time (by definition, 20%)
        assert squeeze_count < len(valid_bb) * 0.25, \
            "Squeeze should be minority of time"

    def test_bollinger_bands_breakout_detection(self, zs_1h_data):
        """
        Test BB can identify price breakouts beyond bands.

        Practical use: Finding strong momentum moves when price breaks
        outside the bands (typically 2 standard deviations).
        """
        prices = zs_1h_data['close']
        bb = _calculate_bollinger_bands(prices, period=20, num_std=2.0)

        # Detect price breaking above upper band
        breaks_above = prices > bb['upper_band']

        # Detect price breaking below lower band
        breaks_below = prices < bb['lower_band']

        # Real market data should have some breakouts
        assert breaks_above.sum() > 0, "Should find breaks above upper band"
        assert breaks_below.sum() > 0, "Should find breaks below lower band"

        # Breakouts should be relatively rare (outside 2 std dev)
        total_breakouts = breaks_above.sum() + breaks_below.sum()
        valid_bars = bb['upper_band'].notna().sum()
        breakout_pct = (total_breakouts / valid_bars) * 100

        assert breakout_pct < 15, \
            f"Breakouts should be <15% of time, got {breakout_pct:.1f}%"

    def test_bollinger_bands_mean_reversion_signals(self, zs_1h_data):
        """
        Test BB can identify mean reversion opportunities.

        Practical use: When price touches bands, it often reverts to middle.
        Identify potential reversal points.
        """
        prices = zs_1h_data['close']
        bb = _calculate_bollinger_bands(prices, period=20, num_std=2.0)

        # Calculate distance from middle band
        valid_idx = bb['middle_band'].notna()
        distance_from_middle = (prices[valid_idx] - bb.loc[valid_idx, 'middle_band']).abs()
        band_width = (bb.loc[valid_idx, 'upper_band'] - bb.loc[valid_idx, 'middle_band'])

        # When price is near bands (> 80% of distance to band)
        near_bands = distance_from_middle > (band_width * 0.8)

        # Should find some mean reversion opportunities
        assert near_bands.sum() > 0, "Should find prices near bands"
