"""
Fixtures for strategy tests.

Provides realistic test data that matches actual historical data format.
"""
import numpy as np

# Import fixtures from parent fixtures directory to make them available
# This ensures PyCharm's test runner can find fixtures like zs_1h_data, load_real_data, etc.
from tests.backtesting.fixtures.data_fixtures import *  # noqa: F401, F403
from tests.backtesting.fixtures.strategy_fixtures import *  # noqa: F401, F403


# ==================== Strategy Factory Fixtures ====================

@pytest.fixture
def standard_ema_strategy():
    """Standard EMA crossover strategy with default parameters."""
    return EMACrossoverStrategy(
        short_ema_period=9,
        long_ema_period=21,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def standard_ichimoku_strategy():
    """Standard Ichimoku Cloud strategy with default parameters."""
    return IchimokuCloudStrategy(
        tenkan_period=9,
        kijun_period=26,
        senkou_span_b_period=52,
        displacement=26,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def standard_macd_strategy():
    """Standard MACD strategy with default parameters."""
    return MACDStrategy(
        fast_period=12,
        slow_period=26,
        signal_period=9,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def standard_rsi_strategy():
    """Standard RSI strategy with default parameters."""
    return RSIStrategy(
        rsi_period=14,
        lower_threshold=30,
        upper_threshold=70,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def standard_bollinger_strategy():
    """Standard Bollinger Bands strategy with default parameters."""
    return BollingerBandsStrategy(
        period=20,
        number_of_standard_deviations=2.0,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


# ==================== Test Data Generation ====================


# OLD below this point
def create_test_df(length=150, base_price=100.0, symbol='CME:ES2!', trend='neutral'):
    """
    Create test dataframe matching actual historical data format.

    Args:
        length: Number of periods to generate
        base_price: Starting price
        symbol: Symbol string (e.g., 'CME:ES2!', 'CBOT:ZC2!')
        trend: 'up', 'down', or 'neutral' for price pattern

    Returns:
        DataFrame with realistic OHLCV data in actual historical format
    """
    dates = pd.date_range('2023-01-01', periods=length, freq='h')

    # Generate price data based on trend
    if trend == 'up':
        close_prices = [base_price + i * 0.5 for i in range(length)]
    elif trend == 'down':
        close_prices = [base_price - i * 0.5 for i in range(length)]
    else:  # neutral with some variation
        close_prices = [base_price + np.random.randn() * 2 for _ in range(length)]

    df = pd.DataFrame({
        'symbol': [symbol] * length,
        'open': close_prices,
        'high': [p + abs(np.random.randn()) for p in close_prices],
        'low': [p - abs(np.random.randn()) for p in close_prices],
        'close': close_prices,
        'volume': [10000.0] * length
    }, index=pd.DatetimeIndex(dates, name='datetime'))

    return df


def create_bollinger_test_df(length=50):
    """
    Create test dataframe with price patterns suitable for Bollinger Bands testing.

    Returns DataFrame with expansion and contraction patterns.
    """
    dates = pd.date_range('2023-01-01', periods=length, freq='h')

    close_prices = []

    # Create a ranging market with increasing volatility
    for i in range(20):
        close_prices.append(100 + np.sin(i * 0.5) * 2)

    # Then a trending move with expansion
    for i in range(15):
        close_prices.append(100 + i * 1.5)

    # Then back to ranging
    for i in range(15):
        close_prices.append(122 + np.sin(i * 0.5) * 3)

    # Ensure the length matches the requested length
    while len(close_prices) < length:
        close_prices.append(close_prices[-1])

    df = pd.DataFrame({
        'symbol': ['CME:ES2!'] * length,
        'open': close_prices,
        'high': [p + abs(np.random.randn()) * 2 for p in close_prices],
        'low': [p - abs(np.random.randn()) * 2 for p in close_prices],
        'close': close_prices,
        'volume': [10000.0] * length
    }, index=pd.DatetimeIndex(dates, name='datetime'))

    return df


def create_ema_test_df(length=50):
    """
    Create test dataframe with price patterns suitable for EMA crossover testing.

    Returns DataFrame with trends that generate clear EMA crossovers.
    """
    dates = pd.date_range('2023-01-01', periods=length, freq='h')

    close_prices = []

    # Start with ranging/downtrend
    for i in range(15):
        close_prices.append(100 - i * 0.5)

    # Strong uptrend for bullish crossover
    for i in range(15):
        close_prices.append(92.5 + i * 2)

    # Ranging at top
    for i in range(10):
        close_prices.append(122.5 + np.sin(i) * 2)

    # Downtrend for bearish crossover
    for i in range(10):
        close_prices.append(122.5 - i * 2)

    # Ensure the length matches the requested length
    while len(close_prices) < length:
        close_prices.append(close_prices[-1])

    df = pd.DataFrame({
        'symbol': ['CME:ES2!'] * length,
        'open': close_prices,
        'high': [p + 1 for p in close_prices],
        'low': [p - 1 for p in close_prices],
        'close': close_prices,
        'volume': [10000.0] * length
    }, index=pd.DatetimeIndex(dates, name='datetime'))

    return df


def create_ichimoku_test_df(length=100):
    """
    Create test dataframe with price patterns suitable for Ichimoku Cloud testing.

    Returns DataFrame with clear trends for cloud analysis.
    """
    dates = pd.date_range('2023-01-01', periods=length, freq='h')

    close_prices = []

    # Long ranging period to establish cloud
    for i in range(40):
        close_prices.append(100 + np.sin(i * 0.2) * 5)

    # Strong uptrend above cloud
    for i in range(30):
        close_prices.append(100 + i * 2)

    # Ranging near top
    for i in range(15):
        close_prices.append(160 + np.sin(i * 0.3) * 3)

    # Downtrend
    for i in range(15):
        close_prices.append(160 - i * 1.5)

    # Ensure the length matches the requested length
    while len(close_prices) < length:
        close_prices.append(close_prices[-1])

    df = pd.DataFrame({
        'symbol': ['CME:ES2!'] * length,
        'open': close_prices,
        'high': [p + abs(np.random.randn()) for p in close_prices],
        'low': [p - abs(np.random.randn()) for p in close_prices],
        'close': close_prices,
        'volume': [10000.0] * length
    }, index=pd.DatetimeIndex(dates, name='datetime'))

    return df


def create_macd_test_df(length=60):
    """
    Create test dataframe with price patterns suitable for MACD testing.

    Returns DataFrame with alternating trends to generate MACD crossovers.
    """
    dates = pd.date_range('2023-01-01', periods=length, freq='h')

    close_prices = []

    # Start with a downtrend
    for i in range(15):
        close_prices.append(100 - i)

    # Then an uptrend to create a bullish crossover
    for i in range(15):
        close_prices.append(85 + i * 1.5)

    # Then a downtrend to create a bearish crossover
    for i in range(15):
        close_prices.append(107.5 - i * 1.5)

    # Then another uptrend
    for i in range(15):
        close_prices.append(85 + i)

    # Ensure the length matches the requested length
    while len(close_prices) < length:
        close_prices.append(close_prices[-1])

    df = pd.DataFrame({
        'symbol': ['CME:ES2!'] * length,
        'open': close_prices,
        'high': [p + 1 for p in close_prices],
        'low': [p - 1 for p in close_prices],
        'close': close_prices,
        'volume': [10000.0] * length
    }, index=pd.DatetimeIndex(dates, name='datetime'))

    return df


def create_rsi_test_df(length=50):
    """
    Create test dataframe with price patterns suitable for RSI testing.

    Returns DataFrame with downtrend (low RSI) then uptrend (high RSI).
    """
    dates = pd.date_range('2023-01-01', periods=length, freq='h')

    # Create a price series that will generate clear RSI signals
    close_prices = []

    # Downtrend for the first part (push RSI low)
    for i in range(20):
        close_prices.append(100 - i)

    # Uptrend for the second part (push RSI high)
    for i in range(20):
        close_prices.append(80 + i)

    # Downtrend again
    for i in range(10):
        close_prices.append(100 - i)

    # Ensure the length matches the requested length
    while len(close_prices) < length:
        close_prices.append(close_prices[-1])

    df = pd.DataFrame({
        'symbol': ['CME:ES2!'] * length,
        'open': close_prices,
        'high': [p + 1 for p in close_prices],
        'low': [p - 1 for p in close_prices],
        'close': close_prices,
        'volume': [10000.0] * length
    }, index=pd.DatetimeIndex(dates, name='datetime'))

    return df
