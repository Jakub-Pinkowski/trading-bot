"""
Fixtures for backtesting tests.

Provides realistic test data that matches an actual historical data format.
All backtesting fixtures are registered in tests/conftest.py via pytest_plugins.
"""
import pandas as pd
import pytest


@pytest.fixture
def realistic_ohlcv_data():
    """
    Generate test data matching actual historical data format.

    Actual format from data/historical_data/2!/{SYMBOL}/{SYMBOL}_{interval}.parquet:
    - Index: DatetimeIndex named 'datetime'
    - Columns: 'symbol', 'open', 'high', 'low', 'close', 'volume'
    - Symbol: Exchange-specific format like 'CME:ES2!' or 'CBOT:ZC2!'
    """
    dates = pd.date_range('2023-01-01', periods=200, freq='h')
    df = pd.DataFrame({
        'symbol': ['CME:ES2!'] * 200,
        'open': [4500 + i * 0.5 for i in range(200)],
        'high': [4505 + i * 0.5 for i in range(200)],
        'low': [4495 + i * 0.5 for i in range(200)],
        'close': [4502 + i * 0.5 for i in range(200)],
        'volume': [10000.0] * 200
    }, index=pd.DatetimeIndex(dates, name='datetime'))
    return df


@pytest.fixture
def realistic_ohlcv_data_short():
    """
    Generate minimal test data matching actual historical data format.

    Useful for tests that need little data (e.g., testing error handling).
    """
    dates = pd.date_range('2023-01-01', periods=50, freq='h')
    df = pd.DataFrame({
        'symbol': ['CME:ES2!'] * 50,
        'open': [4500 + i * 0.5 for i in range(50)],
        'high': [4505 + i * 0.5 for i in range(50)],
        'low': [4495 + i * 0.5 for i in range(50)],
        'close': [4502 + i * 0.5 for i in range(50)],
        'volume': [10000.0] * 50
    }, index=pd.DatetimeIndex(dates, name='datetime'))
    return df


@pytest.fixture
def realistic_ohlcv_data_small():
    """
    Generate very small test data for basic unit tests.

    Only 3 rows - useful for quick tests of basic functionality.
    """
    dates = pd.date_range('2023-01-01', periods=3, freq='h')
    df = pd.DataFrame({
        'symbol': ['CME:ES2!'] * 3,
        'open': [100.0, 101.0, 102.0],
        'high': [105.0, 106.0, 107.0],
        'low': [95.0, 96.0, 97.0],
        'close': [102.0, 103.0, 104.0],
        'volume': [1000.0, 1000.0, 1000.0]
    }, index=pd.DatetimeIndex(dates, name='datetime'))
    return df


def create_realistic_ohlcv_data(symbol='CME:ES2!', periods=200, start='2023-01-01', freq='h', base_price=4500):
    """
    Helper function to create realistic OHLCV data with custom parameters.

    Args:
        symbol: Symbol string (e.g., 'CME:ES2!', 'CBOT:ZC2!')
        periods: Number of periods to generate
        start: Start date string
        freq: Frequency string ('h', 'd', '5min', etc.)
        base_price: Starting price

    Returns:
        DataFrame with realistic OHLCV data in actual historical format
    """
    dates = pd.date_range(start, periods=periods, freq=freq)
    df = pd.DataFrame({
        'symbol': [symbol] * periods,
        'open': [base_price + i * 0.5 for i in range(periods)],
        'high': [base_price + 5 + i * 0.5 for i in range(periods)],
        'low': [base_price - 5 + i * 0.5 for i in range(periods)],
        'close': [base_price + 2 + i * 0.5 for i in range(periods)],
        'volume': [10000.0] * periods
    }, index=pd.DatetimeIndex(dates, name='datetime'))
    return df
