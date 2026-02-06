"""
Real data loading fixtures for backtesting tests.

Provides fixtures for loading actual historical data from parquet files
and creating scenario-based datasets for testing.
"""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from config import HISTORICAL_DATA_DIR, SWITCH_DATES_FILE_PATH, TICK_SIZES, CONTRACT_MULTIPLIERS


# ==================== Path and Configuration Fixtures ====================

@pytest.fixture(scope="session")
def historical_data_path():
    """
    Base path to historical data directory.

    Returns:
        Path object pointing to data/historical_data/
    """
    return Path(HISTORICAL_DATA_DIR)


@pytest.fixture(scope="session")
def contract_switch_dates():
    """
    Load contract switch dates from YAML file.

    Returns:
        Dictionary mapping symbols to list of switch dates
    """
    with open(SWITCH_DATES_FILE_PATH, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def available_symbols(historical_data_path):
    """
    List of symbols with available data in the 1! (front month contract) directory.

    Returns:
        List of symbol strings (e.g., ['ZS', 'CL', 'ES', ...])
    """
    data_dir = historical_data_path / '1!'
    if not data_dir.exists():
        return []

    symbols = []
    for item in data_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            symbols.append(item.name)

    return sorted(symbols)


@pytest.fixture(scope="session")
def contract_info():
    """
    Factory fixture to get contract information for any symbol.

    Returns:
        Function that returns dict with tick_size, tick_value, contract_multiplier

    Example:
        info = contract_info('ZS')
        # {'symbol': 'ZS', 'tick_size': 0.25, 'tick_value': 12.50, 'contract_multiplier': 50}
    """

    def _get_contract_info(symbol):
        tick_size = TICK_SIZES.get(symbol, 0.01)
        multiplier = CONTRACT_MULTIPLIERS.get(symbol, 1)
        tick_value = tick_size * multiplier

        return {
            'symbol': symbol,
            'tick_size': tick_size,
            'tick_value': tick_value,
            'contract_multiplier': multiplier
        }

    return _get_contract_info


# ==================== Data Loading Factory Fixtures ====================

@pytest.fixture(scope="session")
def load_real_data(historical_data_path):
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
def zs_4h_data(load_real_data):
    """
    ZS (soybeans) 4-hour data from 1! contract.

    Medium-sized dataset for faster tests that still need substantial data.

    Returns:
        DataFrame with OHLCV data and datetime index
    """
    return load_real_data('1!', 'ZS', '4h')


@pytest.fixture(scope="module")
def zs_1d_data(load_real_data):
    """
    ZS (soybeans) daily data from 1! contract.

    Smaller dataset for quick tests and daily timeframe validation.

    Returns:
        DataFrame with OHLCV data and datetime index
    """
    return load_real_data('1!', 'ZS', '1d')


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


@pytest.fixture(scope="module")
def zs_15m_data(load_real_data):
    """
    ZS (soybeans) 15-minute data from 1! contract.

    High-frequency dataset for testing with more granular data.

    Returns:
        DataFrame with OHLCV data and datetime index
    """
    return load_real_data('1!', 'ZS', '15m')


@pytest.fixture(scope="module")
def zs_5m_data(load_real_data):
    """
    ZS (soybeans) 5-minute data from 1! contract.

    Very high-frequency dataset for intraday testing.

    Returns:
        DataFrame with OHLCV data and datetime index
    """
    return load_real_data('1!', 'ZS', '5m')


# ==================== Pre-loaded CL (Crude Oil) Fixtures ====================

@pytest.fixture(scope="module")
def cl_1h_data(load_real_data):
    """
    CL (crude oil) 1-hour data from 1! contract.

    Secondary symbol for multi-symbol tests. Energy sector with different
    characteristics than grains (ZS).

    Returns:
        DataFrame with OHLCV data and datetime index
    """
    return load_real_data('1!', 'CL', '1h')


@pytest.fixture(scope="module")
def cl_4h_data(load_real_data):
    """
    CL (crude oil) 4-hour data from 1! contract.

    Secondary symbol for multi-symbol tests on longer timeframe.

    Returns:
        DataFrame with OHLCV data and datetime index
    """
    return load_real_data('1!', 'CL', '4h')


@pytest.fixture(scope="module")
def cl_1d_data(load_real_data):
    """
    CL (crude oil) daily data from 1! contract.

    Secondary symbol for daily timeframe multi-symbol tests.

    Returns:
        DataFrame with OHLCV data and datetime index
    """
    return load_real_data('1!', 'CL', '1d')


# ==================== Multi-Symbol Fixtures ====================

@pytest.fixture(scope="module")
def multi_symbol_data(zs_1h_data, cl_1h_data):
    """
    Dictionary containing multiple symbols for portfolio/comparison tests.

    Returns:
        Dict with symbol keys and DataFrame values:
        {'ZS': zs_1h_df, 'CL': cl_1h_df}
    """
    return {
        'ZS': zs_1h_data,
        'CL': cl_1h_data
    }


@pytest.fixture(scope="module")
def multi_interval_zs_data(zs_5m_data, zs_15m_data, zs_1h_data, zs_4h_data, zs_1d_data):
    """
    Dictionary containing ZS data across multiple intervals.

    Useful for testing multi-timeframe strategies or interval comparisons.

    Returns:
        Dict with interval keys and DataFrame values:
        {'5m': df, '15m': df, '1h': df, '4h': df, '1d': df}
    """
    return {
        '5m': zs_5m_data,
        '15m': zs_15m_data,
        '1h': zs_1h_data,
        '4h': zs_4h_data,
        '1d': zs_1d_data
    }


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
