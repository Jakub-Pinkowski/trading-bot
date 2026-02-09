"""
Conftest for strategy tests.

Provides standard strategy fixtures with default parameters for common testing scenarios.
Imports shared fixtures from fixtures directory for PyCharm test runner compatibility.
"""

import pandas as pd
import pytest
import yaml

from app.backtesting.strategies import (
    BollingerBandsStrategy,
    EMACrossoverStrategy,
    MACDStrategy,
    IchimokuCloudStrategy,
    RSIStrategy
)
from config import SWITCH_DATES_FILE_PATH


# ==================== Standard Strategy Fixtures ====================
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


# ==================== Strategy-Specific Data Fixtures ====================

@pytest.fixture(scope="session")
def contract_switch_dates():
    """
    Load contract switch dates from YAML file.

    Returns:
        Dictionary mapping symbols to list of pandas Timestamps
    """
    with open(SWITCH_DATES_FILE_PATH, 'r') as f:
        dates_dict = yaml.safe_load(f)

    # Convert string dates to pandas Timestamps (skip non-list entries like symbol mappings)
    for symbol in list(dates_dict.keys()):
        if isinstance(dates_dict[symbol], list):
            dates_dict[symbol] = [pd.Timestamp(date) for date in dates_dict[symbol]]
        else:
            # Remove non-list entries (e.g., "MCL: CL" mappings)
            del dates_dict[symbol]

    return dates_dict


def _create_ohlcv_data(periods, symbol='CME:ES2!', base_price=4500, trend='neutral', freq='h', start='2023-01-01'):
    """
    Single source of truth for generating test OHLCV data.

    Creates realistic OHLCV data matching actual historical data format:
    - Index: DatetimeIndex named 'datetime'
    - Columns: 'symbol', 'open', 'high', 'low', 'close', 'volume'
    - Symbol: Exchange-specific format like 'CME:ES2!' or 'CBOT:ZC2!'

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


@pytest.fixture
def sample_ohlcv_data():
    """
    Standard test OHLCV data (200 periods).

    Provides realistic test data matching actual historical data format.
    Primary test dataset for strategy and indicator tests.

    Returns:
        DataFrame with 200 periods of OHLCV data
    """
    return _create_ohlcv_data(periods=200, symbol='CME:ES2!', base_price=4500)
