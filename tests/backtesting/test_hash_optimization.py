"""
These tests verify that the pre-computed hash optimization works correctly.
All hash parameters are REQUIRED - no backward compatibility.
"""

import numpy as np
import pandas as pd
import pytest

from app.backtesting.indicators import calculate_rsi, calculate_ema, calculate_macd, calculate_ichimoku
from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy
from app.backtesting.strategies.ichimoku_cloud import IchimokuCloudStrategy
from app.backtesting.strategies.rsi import RSIStrategy
from app.utils.backtesting_utils.indicators_utils import hash_series


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    dates = pd.date_range(start='2020-01-01', periods=200, freq='1D')

    close = np.cumsum(np.random.randn(200) * 2) + 100
    high = close + np.random.rand(200) * 5
    low = close - np.random.rand(200) * 5
    open_price = close + np.random.randn(200) * 2
    volume = np.random.randint(1000, 100000, 200)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df


class TestHashOptimizationRequired:
    """Test that hash parameters are required (no backward compatibility)"""

    def test_rsi_requires_hash(self, sample_dataframe):
        """Test that RSI requires prices_hash parameter"""
        with pytest.raises(TypeError):
            # Should fail - missing required prices_hash parameter
            calculate_rsi(sample_dataframe['close'], period=14)

    def test_ema_requires_hash(self, sample_dataframe):
        """Test that EMA requires prices_hash parameter"""
        with pytest.raises(TypeError):
            # Should fail - missing required prices_hash parameter
            calculate_ema(sample_dataframe['close'], period=9)


class TestHashOptimizationCorrectness:
    """Test that indicators work correctly with required pre-computed hashes"""

    def test_rsi_with_precomputed_hash(self, sample_dataframe):
        """Test RSI produces correct results with pre-computed hash"""
        close_hash = hash_series(sample_dataframe['close'])
        result = calculate_rsi(sample_dataframe['close'], period=14, prices_hash=close_hash)

        assert len(result) == len(sample_dataframe)
        assert not result.isna().all()
        # RSI should be between 0 and 100
        assert result.dropna().min() >= 0
        assert result.dropna().max() <= 100

    def test_ema_with_precomputed_hash(self, sample_dataframe):
        """Test EMA produces correct results with pre-computed hash"""
        close_hash = hash_series(sample_dataframe['close'])
        result = calculate_ema(sample_dataframe['close'], period=9, prices_hash=close_hash)

        assert len(result) == len(sample_dataframe)
        assert not result.isna().all()

    def test_macd_with_precomputed_hash(self, sample_dataframe):
        """Test MACD produces correct results with pre-computed hash"""
        close_hash = hash_series(sample_dataframe['close'])
        result = calculate_macd(sample_dataframe['close'], fast_period=12,
                                slow_period=26, signal_period=9, prices_hash=close_hash)

        assert len(result) == len(sample_dataframe)
        assert 'macd_line' in result.columns
        assert 'signal_line' in result.columns
        assert 'histogram' in result.columns

    def test_ichimoku_with_precomputed_hashes(self, sample_dataframe):
        """Test Ichimoku produces correct results with pre-computed hashes"""
        high_hash = hash_series(sample_dataframe['high'])
        low_hash = hash_series(sample_dataframe['low'])
        close_hash = hash_series(sample_dataframe['close'])

        result = calculate_ichimoku(
            sample_dataframe['high'],
            sample_dataframe['low'],
            sample_dataframe['close'],
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            high_hash=high_hash,
            low_hash=low_hash,
            close_hash=close_hash
        )

        assert 'tenkan_sen' in result
        assert 'kijun_sen' in result


class TestBaseStrategyPrecomputeHashes:
    """Test the _precompute_hashes() utility method"""

    def test_precompute_hashes_returns_all_columns(self, sample_dataframe):
        """Test that _precompute_hashes returns hashes for all OHLCV columns"""
        strategy = RSIStrategy(rsi_period=14, lower=30, upper=70)
        hashes = strategy._precompute_hashes(sample_dataframe)

        # Should have hashes for all OHLCV columns
        assert 'close' in hashes
        assert 'high' in hashes
        assert 'low' in hashes
        assert 'open' in hashes
        assert 'volume' in hashes

        # All values should be strings (MD5 hashes)
        for key, value in hashes.items():
            assert isinstance(value, str)
            assert len(value) == 32  # MD5 hash length

    def test_precompute_hashes_consistency(self, sample_dataframe):
        """Test that _precompute_hashes produces consistent results"""
        strategy = RSIStrategy(rsi_period=14, lower=30, upper=70)

        hashes1 = strategy._precompute_hashes(sample_dataframe)
        hashes2 = strategy._precompute_hashes(sample_dataframe)

        # Same DataFrame should produce same hashes
        assert hashes1 == hashes2


class TestStrategyHashOptimization:
    """Test that strategies use hash optimization correctly"""

    def test_rsi_strategy_uses_precomputed_hash(self, sample_dataframe):
        """Test that RSIStrategy uses _precompute_hashes()"""
        strategy = RSIStrategy(rsi_period=14, lower=30, upper=70)

        # add_indicators should work without errors
        df_with_indicators = strategy.add_indicators(sample_dataframe.copy())

        # RSI column should be added
        assert 'rsi' in df_with_indicators.columns
        assert not df_with_indicators['rsi'].isna().all()

    def test_ema_strategy_uses_precomputed_hash(self, sample_dataframe):
        """Test that EMACrossoverStrategy uses _precompute_hashes()"""
        strategy = EMACrossoverStrategy(ema_short=9, ema_long=21)

        # add_indicators should work without errors
        df_with_indicators = strategy.add_indicators(sample_dataframe.copy())

        # EMA columns should be added
        assert 'ema_short' in df_with_indicators.columns
        assert 'ema_long' in df_with_indicators.columns

    def test_ichimoku_strategy_uses_precomputed_hashes(self, sample_dataframe):
        """Test that IchimokuCloudStrategy uses _precompute_hashes()"""
        strategy = IchimokuCloudStrategy()

        # add_indicators should work without errors
        df_with_indicators = strategy.add_indicators(sample_dataframe.copy())

        # Ichimoku columns should be added
        assert 'tenkan_sen' in df_with_indicators.columns
        assert 'kijun_sen' in df_with_indicators.columns


class TestHashOptimizationPerformance:
    """Test that hash optimization allows sharing hashes across indicators"""

    def test_multiple_indicators_share_same_hash(self, sample_dataframe):
        """Test that multiple indicators can use the same pre-computed hash"""
        # Pre-compute hash ONCE
        hashes = RSIStrategy()._precompute_hashes(sample_dataframe)

        # Clear cache
        from app.backtesting.cache.indicators_cache import indicator_cache
        indicator_cache.cache_data.clear()

        # Use same hash for all indicators - the optimization!
        rsi = calculate_rsi(sample_dataframe['close'], period=14,
                            prices_hash=hashes['close'])
        rsi_long = calculate_rsi(sample_dataframe['close'], period=21,
                                 prices_hash=hashes['close'])
        ema = calculate_ema(sample_dataframe['close'], period=9,
                            prices_hash=hashes['close'])

        # All should produce valid results
        assert not rsi.isna().all()
        assert not rsi_long.isna().all()
        assert not ema.isna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
