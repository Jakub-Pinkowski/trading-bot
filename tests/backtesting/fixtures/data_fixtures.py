"""
Real data loading fixtures for backtesting tests.

Provides fixtures for loading actual historical data from parquet files
and creating scenario-based datasets for testing.
"""

from pathlib import Path

import pandas as pd
import pytest

from config import HISTORICAL_DATA_DIR


@pytest.fixture(scope="session")
def load_real_data():
    """
    Factory fixture to load any symbol/interval combination from parquet files.

    Returns:
        Function that loads and returns DataFrame for given month, symbol, interval

    Example:
        df = load_real_data('1!', 'ZS', '1h')

    Raises:
        FileNotFoundError: If requested data file doesn't exist
        pytest.skip: If data not available (allows test to skip gracefully)
    """
    historical_data_path = Path(HISTORICAL_DATA_DIR)

    def _load_data(month, symbol, interval):
        file_path = historical_data_path / month / symbol / f"{symbol}_{interval}.parquet"

        if not file_path.exists():
            pytest.skip(f"Data not available: {file_path}")

        df = pd.read_parquet(file_path)
        return df

    return _load_data


# ==================== Pre-loaded ZS (Soybeans) Fixtures ====================

@pytest.fixture(scope="module")
def zs_1h_data(load_real_data):
    """
    ZS (soybeans) 1-hour data from 1! contract.

    Primary dataset for most tests - large enough for comprehensive testing
    (~9,554 records, ~2 years of data).

    Returns:
        DataFrame with OHLCV data and datetime index
    """
    return load_real_data('1!', 'ZS', '1h')


@pytest.fixture(scope="module")
def zs_1d_data(load_real_data):
    """
    ZS (soybeans) daily data from 1! contract.

    Smaller dataset for quick tests and daily timeframe validation.

    Returns:
        DataFrame with OHLCV data and datetime index
    """
    return load_real_data('1!', 'ZS', '1d')


# ==================== Pre-loaded CL (Crude Oil) Fixtures ====================

@pytest.fixture(scope="module")
def cl_15m_data(load_real_data):
    """
    CL (crude oil) 15-minute data from 1! contract.

    Used for multi-symbol tests to validate indicator calculations
    work across different instruments.

    Returns:
        DataFrame with OHLCV data and datetime index
    """
    return load_real_data('1!', 'CL', '15m')


# ==================== Scenario-Based Data Fixtures ====================

@pytest.fixture
def trending_market_data(zs_1h_data):
    """
    Extract period from real data showing clear uptrend.

    Finds a sustained upward trend in ZS data for testing trend-following
    strategies. Looks for periods where price consistently moves higher.

    Returns:
        DataFrame subset with 200+ bars showing uptrend

    Raises:
        pytest.skip: If no clear trend found in data
    """
    # Calculate simple trend: 50-period moving average slope
    df = zs_1h_data.copy()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['trend'] = df['ma50'].diff(10)  # Change over 10 periods

    # Find longest uptrend period
    df['is_uptrend'] = df['trend'] > 0

    # Find consecutive uptrend bars
    df['trend_group'] = (df['is_uptrend'] != df['is_uptrend'].shift()).cumsum()
    uptrend_groups = df[df['is_uptrend']].groupby('trend_group').size()

    if len(uptrend_groups) == 0 or uptrend_groups.max() < 100:
        pytest.skip("No clear uptrend period found in ZS data")

    # Get longest uptrend period
    longest_group = uptrend_groups.idxmax()
    trending_data = df[df['trend_group'] == longest_group]

    # Add some context before and after using iloc (integer-based indexing)
    first_idx = df.index.get_loc(trending_data.index[0])
    last_idx = df.index.get_loc(trending_data.index[-1])

    start_idx = max(0, first_idx - 50)
    end_idx = min(len(df), last_idx + 50)

    return df.iloc[start_idx:end_idx][['symbol', 'open', 'high', 'low', 'close', 'volume']]


@pytest.fixture
def ranging_market_data(zs_1h_data):
    """
    Extract period from real data showing sideways movement.

    Finds a period where price oscillates within a range without clear trend.
    Useful for testing mean-reversion strategies.

    Returns:
        DataFrame subset with 100+ bars showing range-bound movement

    Raises:
        pytest.skip: If no clear range found in data
    """
    df = zs_1h_data.copy()

    # Calculate volatility and trend - more lenient criteria
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['trend'] = df['ma50'].diff(10).abs()  # Absolute change
    df['volatility'] = df['close'].rolling(window=20).std()

    # Ranging market: low trend change (use 40th percentile for more results)
    # Also check price stays within narrow range
    df['price_range'] = (df['close'].rolling(window=50).max() - df['close'].rolling(window=50).min()) / df['close']

    df['is_ranging'] = (
            (df['trend'] < df['trend'].quantile(0.4)) &  # Low trend
            (df['volatility'] > 0) &  # Has some movement
            (df['price_range'] < 0.10)  # Within 10% range over 50 bars
    )

    # Find consecutive ranging bars
    df['range_group'] = (df['is_ranging'] != df['is_ranging'].shift()).cumsum()
    ranging_groups = df[df['is_ranging']].groupby('range_group').size()

    if len(ranging_groups) == 0 or ranging_groups.max() < 80:  # Lowered from 100
        pytest.skip("No clear ranging period found in ZS data")

    # Get longest ranging period
    longest_group = ranging_groups.idxmax()
    ranging_data = df[df['range_group'] == longest_group]

    # Add some context using iloc
    first_idx = df.index.get_loc(ranging_data.index[0])
    last_idx = df.index.get_loc(ranging_data.index[-1])

    start_idx = max(0, first_idx - 50)
    end_idx = min(len(df), last_idx + 50)

    return df.iloc[start_idx:end_idx][['symbol', 'open', 'high', 'low', 'close', 'volume']]


@pytest.fixture
def volatile_market_data(zs_1h_data):
    """
    Extract period from real data showing high volatility.

    Finds periods with extreme price swings, useful for testing strategy
    behavior during unusual market conditions.

    Returns:
        DataFrame subset with 200+ bars showing high volatility

    Raises:
        pytest.skip: If no volatile period found in data
    """
    df = zs_1h_data.copy()

    # Calculate rolling volatility
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()

    # Find high volatility periods (top 10%)
    volatility_threshold = df['volatility'].quantile(0.90)
    df['is_volatile'] = df['volatility'] > volatility_threshold

    # Find consecutive volatile bars
    df['vol_group'] = (df['is_volatile'] != df['is_volatile'].shift()).cumsum()
    volatile_groups = df[df['is_volatile']].groupby('vol_group').size()

    if len(volatile_groups) == 0 or volatile_groups.max() < 50:
        pytest.skip("No highly volatile period found in ZS data")

    # Get longest volatile period
    longest_group = volatile_groups.idxmax()
    volatile_data = df[df['vol_group'] == longest_group]

    # Add context using iloc
    first_idx = df.index.get_loc(volatile_data.index[0])
    last_idx = df.index.get_loc(volatile_data.index[-1])

    start_idx = max(0, first_idx - 50)
    end_idx = min(len(df), last_idx + 50)

    return df.iloc[start_idx:end_idx][['symbol', 'open', 'high', 'low', 'close', 'volume']]


# ==================== Synthetic OHLCV Data Generation ====================

@pytest.fixture(scope="session")
def ohlcv_factory():
    """
    Factory fixture for generating synthetic OHLCV test data.

    Creates realistic OHLCV data matching actual historical data format:
    - Index: DatetimeIndex named 'datetime'
    - Columns: 'symbol', 'open', 'high', 'low', 'close', 'volume'
    - Symbol: Exchange-specific format like 'CME:ES2!' or 'CBOT:ZC2!'

    Returns:
        Function that creates OHLCV DataFrame with specified parameters

    Example:
        data = ohlcv_factory(periods=100, symbol='ZS', trend='up')
        data = ohlcv_factory(periods=50, symbol='CL', base_price=75, freq='15min')
    """

    def _create_ohlcv_data(
        periods,
        symbol='CME:ES2!',
        base_price=4500,
        trend='neutral',
        freq='h',
        start='2023-01-01'
    ):
        """
        Generate synthetic OHLCV data for testing.

        Args:
            periods: Number of periods to generate
            symbol: Symbol string (e.g., 'CME:ES2!', 'CBOT:ZC2!')
            base_price: Starting price
            trend: 'up', 'down', or 'neutral' for price pattern
            freq: Frequency string ('h', 'd', '5min', etc.)
            start: Start date string

        Returns:
            DataFrame with realistic OHLCV data in actual historical format
        """
        dates = pd.date_range(start, periods=periods, freq=freq)

        # Generate price data based on trend
        if trend == 'up':
            close_prices = [base_price + i * 0.5 for i in range(periods)]
        elif trend == 'down':
            close_prices = [base_price - i * 0.5 for i in range(periods)]
        else:  # neutral with small upward drift
            close_prices = [base_price + i * 0.5 for i in range(periods)]

        df = pd.DataFrame({
            'symbol': [symbol] * periods,
            'open': close_prices,
            'high': [p + 5 for p in close_prices],
            'low': [p - 5 for p in close_prices],
            'close': [p + 2 for p in close_prices],
            'volume': [10000.0] * periods
        }, index=pd.DatetimeIndex(dates, name='datetime'))

        return df

    return _create_ohlcv_data


@pytest.fixture
def sample_ohlcv_data(ohlcv_factory):
    """
    Standard test OHLCV data (200 periods).

    Provides realistic test data matching actual historical data format.
    Primary test dataset for strategy and indicator tests.

    Returns:
        DataFrame with 200 periods of OHLCV data
    """
    return ohlcv_factory(periods=200, symbol='CME:ES2!', base_price=4500)
