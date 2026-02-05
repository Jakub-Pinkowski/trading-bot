import numpy as np
import pandas as pd

from app.backtesting.indicators import calculate_ichimoku_cloud
from app.utils.backtesting_utils.indicators_utils import hash_series


def compute_hashes(df):
    """Helper function to compute hashes for Ichimoku testing"""
    return {
        'high_hash': hash_series(df['high']),
        'low_hash': hash_series(df['low']),
        'close_hash': hash_series(df['close'])
    }


def test_calculate_ichimoku_with_valid_prices():
    """Test Ichimoku calculation with valid price data"""
    # Create sample price data
    dates = pd.date_range(start='2020-01-01', periods=100)
    high = pd.Series(np.random.randint(100, 150, size=100), index=dates)
    low = pd.Series(np.random.randint(50, 100, size=100), index=dates)
    close = pd.Series(np.random.randint(75, 125, size=100), index=dates)

    # Calculate Ichimoku
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku = calculate_ichimoku_cloud(high,
                                        low,
                                        close,
                                        tenkan_period=9,
                                        kijun_period=26,
                                        senkou_span_b_period=52,
                                        displacement=26,
                                        high_hash=high_hash,
                                        low_hash=low_hash,
                                        close_hash=close_hash)

    # Check that all components are present
    assert 'tenkan_sen' in ichimoku
    assert 'kijun_sen' in ichimoku
    assert 'senkou_span_a' in ichimoku
    assert 'senkou_span_b' in ichimoku
    assert 'chikou_span' in ichimoku

    # Check that the components have the correct length
    assert len(ichimoku['tenkan_sen']) == len(close)
    assert len(ichimoku['kijun_sen']) == len(close)
    assert len(ichimoku['senkou_span_a']) == len(close)
    assert len(ichimoku['senkou_span_b']) == len(close)
    assert len(ichimoku['chikou_span']) == len(close)


def test_calculate_ichimoku_with_custom_parameters():
    """Test Ichimoku calculation with custom parameters"""
    # Create sample price data
    dates = pd.date_range(start='2020-01-01', periods=100)
    high = pd.Series(np.random.randint(100, 150, size=100), index=dates)
    low = pd.Series(np.random.randint(50, 100, size=100), index=dates)
    close = pd.Series(np.random.randint(75, 125, size=100), index=dates)

    # Calculate Ichimoku with custom parameters
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku = calculate_ichimoku_cloud(
        high, low, close,
        tenkan_period=5,
        kijun_period=15,
        senkou_span_b_period=30,
        displacement=15
        ,
        high_hash=high_hash, low_hash=low_hash, close_hash=close_hash
    )

    # Check that all components are present
    assert 'tenkan_sen' in ichimoku
    assert 'kijun_sen' in ichimoku
    assert 'senkou_span_a' in ichimoku
    assert 'senkou_span_b' in ichimoku
    assert 'chikou_span' in ichimoku


def test_calculate_ichimoku_with_not_enough_data():
    """Test Ichimoku calculation with price data less than the required periods"""
    # Create sample price data with only a few points
    dates = pd.date_range(start='2020-01-01', periods=5)
    high = pd.Series(np.random.randint(100, 150, size=5), index=dates)
    low = pd.Series(np.random.randint(50, 100, size=5), index=dates)
    close = pd.Series(np.random.randint(75, 125, size=5), index=dates)

    # Calculate Ichimoku
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku = calculate_ichimoku_cloud(high,
                                        low,
                                        close,
                                        tenkan_period=9,
                                        kijun_period=26,
                                        senkou_span_b_period=52,
                                        displacement=26,
                                        high_hash=high_hash,
                                        low_hash=low_hash,
                                        close_hash=close_hash)

    # Check that the components have the correct length
    assert len(ichimoku['tenkan_sen']) == len(close)

    # Check that early values are NaN due to not enough data
    assert ichimoku['tenkan_sen'].iloc[:8].isna().all()
    assert ichimoku['kijun_sen'].iloc[:25].isna().all()
    assert ichimoku['senkou_span_b'].iloc[:51].isna().all()


def test_calculate_ichimoku_with_constant_prices():
    """Test Ichimoku calculation when prices remain constant"""
    # Create sample price data with constant values
    dates = pd.date_range(start='2020-01-01', periods=100)
    high = pd.Series(np.ones(100) * 100, index=dates)
    low = pd.Series(np.ones(100) * 100, index=dates)
    close = pd.Series(np.ones(100) * 100, index=dates)

    # Calculate Ichimoku
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku = calculate_ichimoku_cloud(high,
                                        low,
                                        close,
                                        tenkan_period=9,
                                        kijun_period=26,
                                        senkou_span_b_period=52,
                                        displacement=26,
                                        high_hash=high_hash,
                                        low_hash=low_hash,
                                        close_hash=close_hash)

    # Check that the components have constant values where not NaN
    non_nan_tenkan = ichimoku['tenkan_sen'].dropna()
    non_nan_kijun = ichimoku['kijun_sen'].dropna()

    assert (non_nan_tenkan == 100).all()
    assert (non_nan_kijun == 100).all()


def test_calculate_ichimoku_handles_empty_prices():
    """Test Ichimoku calculation with empty price data"""
    # Create empty price data
    high = pd.Series(dtype='float64')
    low = pd.Series(dtype='float64')
    close = pd.Series(dtype='float64')

    # Calculate Ichimoku
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku = calculate_ichimoku_cloud(high,
                                        low,
                                        close,
                                        tenkan_period=9,
                                        kijun_period=26,
                                        senkou_span_b_period=52,
                                        displacement=26,
                                        high_hash=high_hash,
                                        low_hash=low_hash,
                                        close_hash=close_hash)

    # Check that all components are empty
    assert ichimoku['tenkan_sen'].empty
    assert ichimoku['kijun_sen'].empty
    assert ichimoku['senkou_span_a'].empty
    assert ichimoku['senkou_span_b'].empty
    assert ichimoku['chikou_span'].empty


def test_calculate_ichimoku_with_uptrend():
    """Test Ichimoku calculation with consistently increasing prices"""
    # Create sample price data with an uptrend
    dates = pd.date_range(start='2020-01-01', periods=100)
    high = pd.Series(np.arange(100, 200), index=dates)
    low = pd.Series(np.arange(90, 190), index=dates)
    close = pd.Series(np.arange(95, 195), index=dates)

    # Calculate Ichimoku
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku = calculate_ichimoku_cloud(high,
                                        low,
                                        close,
                                        tenkan_period=9,
                                        kijun_period=26,
                                        senkou_span_b_period=52,
                                        displacement=26,
                                        high_hash=high_hash,
                                        low_hash=low_hash,
                                        close_hash=close_hash)

    # In an uptrend, we expect:
    # 1. Tenkan-sen > Kijun-sen (faster line above slower line)
    # 2. Price > Senkou Span A > Senkou Span B (price above cloud, cloud is bullish)
    # 3. Chikou Span > Price from displacement periods ago

    # Skip initial NaN values
    # We need to use an index that's past all initialization periods including displacement
    # senkou_span_b has the longest period (52) and is displaced by 26, so we need at least 78 periods
    valid_idx = 80  # Ensure we're past all initialization periods

    assert ichimoku['tenkan_sen'].iloc[valid_idx] > ichimoku['kijun_sen'].iloc[valid_idx]
    assert close.iloc[valid_idx] > ichimoku['senkou_span_a'].iloc[valid_idx]
    assert ichimoku['senkou_span_a'].iloc[valid_idx] > ichimoku['senkou_span_b'].iloc[valid_idx]


def test_calculate_ichimoku_with_downtrend():
    """Test Ichimoku calculation with consistently decreasing prices"""
    # Create sample price data with a downtrend
    dates = pd.date_range(start='2020-01-01', periods=100)
    high = pd.Series(np.arange(200, 100, -1), index=dates)
    low = pd.Series(np.arange(190, 90, -1), index=dates)
    close = pd.Series(np.arange(195, 95, -1), index=dates)

    # Calculate Ichimoku
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku = calculate_ichimoku_cloud(high,
                                        low,
                                        close,
                                        tenkan_period=9,
                                        kijun_period=26,
                                        senkou_span_b_period=52,
                                        displacement=26,
                                        high_hash=high_hash,
                                        low_hash=low_hash,
                                        close_hash=close_hash)

    # In a downtrend, we expect:
    # 1. Tenkan-sen < Kijun-sen (faster line below slower line)
    # 2. Price < Senkou Span A < Senkou Span B (price below cloud, cloud is bearish)
    # 3. Chikou Span < Price from displacement periods ago

    # Skip initial NaN values
    # We need to use an index that's past all initialization periods including displacement
    # senkou_span_b has the longest period (52) and is displaced by 26, so we need at least 78 periods
    valid_idx = 80  # Ensure we're past all initialization periods

    assert ichimoku['tenkan_sen'].iloc[valid_idx] < ichimoku['kijun_sen'].iloc[valid_idx]
    assert close.iloc[valid_idx] < ichimoku['senkou_span_a'].iloc[valid_idx]
    assert ichimoku['senkou_span_a'].iloc[valid_idx] < ichimoku['senkou_span_b'].iloc[valid_idx]


def test_calculate_ichimoku_with_sideways_market():
    """Test Ichimoku calculation with sideways market (range-bound)"""
    # Create sample price data with a sideways market
    dates = pd.date_range(start='2020-01-01', periods=100)

    # Create oscillating prices within a range
    high_values = []
    low_values = []
    close_values = []

    for i in range(100):
        # Oscillate between 90 and 110
        phase = i % 10
        if phase < 5:
            high_val = 110 - phase
            low_val = 90 - phase
            close_val = 100 - phase
        else:
            high_val = 90 + (phase - 5)
            low_val = 70 + (phase - 5)
            close_val = 80 + (phase - 5)

        high_values.append(high_val)
        low_values.append(low_val)
        close_values.append(close_val)

    high = pd.Series(high_values, index=dates)
    low = pd.Series(low_values, index=dates)
    close = pd.Series(close_values, index=dates)

    # Calculate Ichimoku
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku = calculate_ichimoku_cloud(high,
                                        low,
                                        close,
                                        tenkan_period=9,
                                        kijun_period=26,
                                        senkou_span_b_period=52,
                                        displacement=26,
                                        high_hash=high_hash,
                                        low_hash=low_hash,
                                        close_hash=close_hash)

    # In a sideways market, we expect:
    # 1. Tenkan-sen and Kijun-sen to be close to each other
    # 2. Senkou Span A and Senkou Span B to be close to each other

    # Skip initial NaN values
    # We need to use an index that's past all initialization periods including displacement
    # senkou_span_b has the longest period (52) and is displaced by 26, so we need at least 78 periods
    valid_idx = 80  # Ensure we're past all initialization periods

    # Check that the difference between Tenkan-sen and Kijun-sen is small
    assert abs(ichimoku['tenkan_sen'].iloc[valid_idx] - ichimoku['kijun_sen'].iloc[valid_idx]) < 10

    # Check that the difference between Senkou Span A and Senkou Span B is small
    assert abs(ichimoku['senkou_span_a'].iloc[valid_idx] - ichimoku['senkou_span_b'].iloc[valid_idx]) < 10


def test_calculate_ichimoku_with_market_reversal():
    """Test Ichimoku calculation during a market reversal scenario"""
    # Create sample price data with a market reversal (uptrend to downtrend)
    dates = pd.date_range(start='2020-01-01', periods=150)

    # First 75 days: uptrend
    high_up = np.arange(100, 175)
    low_up = np.arange(90, 165)
    close_up = np.arange(95, 170)

    # Next 75 days: downtrend
    high_down = np.arange(175, 100, -1)
    low_down = np.arange(165, 90, -1)
    close_down = np.arange(170, 95, -1)

    high = pd.Series(np.concatenate([high_up, high_down]), index=dates)
    low = pd.Series(np.concatenate([low_up, low_down]), index=dates)
    close = pd.Series(np.concatenate([close_up, close_down]), index=dates)

    # Calculate Ichimoku
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku = calculate_ichimoku_cloud(high,
                                        low,
                                        close,
                                        tenkan_period=9,
                                        kijun_period=26,
                                        senkou_span_b_period=52,
                                        displacement=26,
                                        high_hash=high_hash,
                                        low_hash=low_hash,
                                        close_hash=close_hash)

    # At the reversal point, we expect:
    # 1. Tenkan-sen to cross below Kijun-sen
    # 2. Price to cross below the cloud

    # Check for Tenkan-sen crossing below Kijun-sen around the reversal point
    reversal_idx = 75
    before_reversal = reversal_idx - 5
    after_reversal = reversal_idx + 15  # Allow some time for the crossover to occur

    assert ichimoku['tenkan_sen'].iloc[before_reversal] > ichimoku['kijun_sen'].iloc[before_reversal]
    assert ichimoku['tenkan_sen'].iloc[after_reversal] < ichimoku['kijun_sen'].iloc[after_reversal]


def test_calculate_ichimoku_with_high_volatility():
    """Test Ichimoku calculation with high volatility market conditions"""
    # Create sample price data with high volatility
    dates = pd.date_range(start='2020-01-01', periods=100)

    # Base trend with high volatility
    base = np.linspace(100, 200, 100)
    volatility = np.random.normal(0, 20, 100)  # High standard deviation

    high_values = base + volatility + 10
    low_values = base + volatility - 10
    close_values = base + volatility

    high = pd.Series(high_values, index=dates)
    low = pd.Series(low_values, index=dates)
    close = pd.Series(close_values, index=dates)

    # Calculate Ichimoku
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku = calculate_ichimoku_cloud(high,
                                        low,
                                        close,
                                        tenkan_period=9,
                                        kijun_period=26,
                                        senkou_span_b_period=52,
                                        displacement=26,
                                        high_hash=high_hash,
                                        low_hash=low_hash,
                                        close_hash=close_hash)

    # In high volatility, we expect:
    # 1. Wider cloud (larger difference between Senkou Span A and B)
    # 2. Larger distance between Tenkan-sen and Kijun-sen

    # Skip initial NaN values
    # We need to use an index that's past all initialization periods including displacement
    # senkou_span_b has the longest period (52) and is displaced by 26, so we need at least 78 periods
    valid_idx = 80  # Ensure we're past all initialization periods

    # Calculate the average cloud width in the latter part of the data
    cloud_width = abs(ichimoku['senkou_span_a'].iloc[valid_idx:] - ichimoku['senkou_span_b'].iloc[valid_idx:]).mean()

    # The cloud width should be significant due to high volatility
    assert cloud_width > 10


def test_calculate_ichimoku_caching():
    """Test that the Ichimoku calculation uses caching correctly"""
    # Create sample price data
    dates = pd.date_range(start='2020-01-01', periods=100)
    high = pd.Series(np.random.randint(100, 150, size=100), index=dates)
    low = pd.Series(np.random.randint(50, 100, size=100), index=dates)
    close = pd.Series(np.random.randint(75, 125, size=100), index=dates)

    # Calculate Ichimoku twice with the same inputs
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku1 = calculate_ichimoku_cloud(high,
                                         low,
                                         close,
                                         tenkan_period=9,
                                         kijun_period=26,
                                         senkou_span_b_period=52,
                                         displacement=26,
                                         high_hash=high_hash,
                                         low_hash=low_hash,
                                         close_hash=close_hash)
    high_hash = hash_series(high)
    low_hash = hash_series(low)
    close_hash = hash_series(close)
    ichimoku2 = calculate_ichimoku_cloud(high,
                                         low,
                                         close,
                                         tenkan_period=9,
                                         kijun_period=26,
                                         senkou_span_b_period=52,
                                         displacement=26,
                                         high_hash=high_hash,
                                         low_hash=low_hash,
                                         close_hash=close_hash)

    # The results should be identical (and the second call should use the cache)
    assert ichimoku1['tenkan_sen'].equals(ichimoku2['tenkan_sen'])
    assert ichimoku1['kijun_sen'].equals(ichimoku2['kijun_sen'])
    assert ichimoku1['senkou_span_a'].equals(ichimoku2['senkou_span_a'])
    assert ichimoku1['senkou_span_b'].equals(ichimoku2['senkou_span_b'])
    assert ichimoku1['chikou_span'].equals(ichimoku2['chikou_span'])
