"""
Shared fixtures for integration tests.

These fixtures set up complete test environments with multiple components
working together. Integration tests validate that components integrate correctly
and data flows through the system as expected.
"""
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from app.backtesting.cache.dataframe_cache import dataframe_cache
from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.strategies import (
    RSIStrategy,
    EMACrossoverStrategy,
    MACDStrategy,
    BollingerBandsStrategy,
    IchimokuCloudStrategy
)
from app.backtesting.testing.mass_tester import SWITCH_DATES_FILE_PATH


# ==================== Cache Management ====================

@pytest.fixture
def clean_caches():
    """
    Clear all caches before and after test.

    Ensures tests start with clean cache state and don't pollute other tests.
    Resets both indicator cache and dataframe cache.
    """
    # Clear before test
    indicator_cache.cache_data.clear()
    dataframe_cache.cache_data.clear()
    indicator_cache.reset_stats()
    dataframe_cache.reset_stats()

    yield

    # Clear after test
    indicator_cache.cache_data.clear()
    dataframe_cache.cache_data.clear()


# ==================== Test Data ====================

@pytest.fixture
def integration_test_data(load_real_data):
    """
    Provide real data for integration tests.

    Returns smaller subset of real data to keep tests fast while still
    being large enough to generate meaningful trades and metrics.

    Returns:
        DataFrame with last 500 rows of ZS 1h data (enough for testing)
    """
    full_data = load_real_data('1!', 'ZS', '1h')
    return full_data.tail(500).copy()


@pytest.fixture
def small_test_data(load_real_data):
    """
    Provide small dataset for quick integration tests.

    Returns:
        DataFrame with last 200 rows of ZS 1h data
    """
    full_data = load_real_data('1!', 'ZS', '1h')
    return full_data.tail(200).copy()


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


# ==================== Strategy Instances ====================

@pytest.fixture
def all_strategy_instances():
    """
    Create instance of each strategy type for testing.

    Returns:
        Dictionary mapping strategy names to configured strategy instances
    """
    return {
        'RSI': RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        ),
        'EMA': EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        ),
        'MACD': MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        ),
        'Bollinger': BollingerBandsStrategy(
            period=20,
            number_of_standard_deviations=2.0,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        ),
        'Ichimoku': IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        ),
    }


# ==================== Sample Results ====================

@pytest.fixture
def temp_results_dir():
    """
    Create temporary directory for test results.

    Yields:
        Path object to temporary directory (cleaned up after test)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_backtest_results():
    """
    Create sample backtest results DataFrame for analysis tests.

    Returns realistic results data without actually running backtests.
    Used for testing analysis pipeline components.

    Returns:
        DataFrame with sample backtest results
    """
    return pd.DataFrame({
        'month': ['1!'] * 10,
        'symbol': ['ZS', 'CL'] * 5,
        'interval': ['1h'] * 10,
        'strategy': [
            'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage_ticks=1.0)',
            'RSI(period=14,lower=25,upper=75,rollover=False,trailing=None,slippage_ticks=1.0)',
            'RSI(period=20,lower=30,upper=70,rollover=False,trailing=None,slippage_ticks=1.0)',
            'EMA(short=9,long=21,rollover=False,trailing=None,slippage_ticks=1.0)',
            'EMA(short=12,long=26,rollover=False,trailing=None,slippage_ticks=1.0)',
            'MACD(fast=12,slow=26,signal=9,rollover=False,trailing=None,slippage_ticks=1.0)',
            'Bollinger(period=20,std=2.0,rollover=False,trailing=None,slippage_ticks=1.0)',
            'Bollinger(period=15,std=2.0,rollover=False,trailing=None,slippage_ticks=1.0)',
            'Ichimoku(tenkan=9,kijun=26,senkou_b=52,disp=26,rollover=False,trailing=None,slippage_ticks=1.0)',
            'Ichimoku(tenkan=7,kijun=22,senkou_b=44,disp=22,rollover=False,trailing=None,slippage_ticks=1.0)',
        ],
        'total_trades': [100, 120, 80, 95, 110, 88, 105, 92, 115, 98],
        'win_rate': [0.55, 0.60, 0.52, 0.58, 0.62, 0.50, 0.56, 0.54, 0.61, 0.57],
        'profit_factor': [1.5, 1.8, 1.2, 1.6, 2.0, 1.1, 1.4, 1.3, 1.9, 1.7],
        'sharpe_ratio': [1.2, 1.5, 0.9, 1.3, 1.7, 0.8, 1.1, 1.0, 1.6, 1.4],
        'total_return_percentage_of_contract': [5.2, 7.1, 3.8, 6.2, 8.5, 2.9, 4.8, 4.1, 7.8, 6.9],
        'maximum_drawdown_percentage': [8.5, 7.2, 10.1, 7.8, 6.5, 11.2, 9.0, 9.5, 6.8, 7.5],
    })
