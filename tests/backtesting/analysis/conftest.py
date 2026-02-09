"""
Shared fixtures for backtesting analysis tests.

Provides common fixtures for strategy results DataFrames used across
test_strategy_analyzer, test_formatters, and test_data_helpers.
"""
import os

from config import BACKTESTING_DIR
# Import backtesting fixtures to make them discoverable by PyCharm
from tests.backtesting.fixtures.data_fixtures import *  # noqa: F401, F403
from tests.backtesting.fixtures.strategy_fixtures import *  # noqa: F401, F403


# ==================== Core Strategy Results Fixtures ====================

@pytest.fixture
def base_strategy_results():
    """
    Comprehensive strategy results DataFrame for testing.

    12 rows with realistic data covering:
    - Multiple symbols (ZS, CL, ES)
    - Multiple intervals (1h, 4h, 1d)
    - Different performance levels (high, medium, low)
    - Various trade counts and metrics

    Used by: test_strategy_analyzer

    Returns:
        DataFrame with 12 rows representing different strategy/symbol/interval combinations
    """
    return pd.DataFrame([
        # High-performing strategy across multiple symbols
        {'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'TopStrategy',
         'total_trades': 100, 'win_rate': 65.0, 'average_trade_duration_hours': 4.5,
         'total_wins_percentage_of_contract': 150.0, 'total_losses_percentage_of_contract': 50.0,
         'total_return_percentage_of_contract': 100.0, 'average_trade_return_percentage_of_contract': 1.0,
         'average_win_percentage_of_contract': 2.5, 'average_loss_percentage_of_contract': -1.5,
         'profit_factor': 3.0, 'maximum_drawdown_percentage': 10.0, 'sharpe_ratio': 2.5,
         'sortino_ratio': 3.0, 'calmar_ratio': 10.0, 'value_at_risk': 5.0,
         'expected_shortfall': 7.0, 'ulcer_index': 3.0},

        {'month': '1!', 'symbol': 'CL', 'interval': '1h', 'strategy': 'TopStrategy',
         'total_trades': 80, 'win_rate': 60.0, 'average_trade_duration_hours': 4.0,
         'total_wins_percentage_of_contract': 120.0, 'total_losses_percentage_of_contract': 40.0,
         'total_return_percentage_of_contract': 80.0, 'average_trade_return_percentage_of_contract': 1.0,
         'average_win_percentage_of_contract': 2.5, 'average_loss_percentage_of_contract': -1.0,
         'profit_factor': 3.0, 'maximum_drawdown_percentage': 12.0, 'sharpe_ratio': 2.0,
         'sortino_ratio': 2.5, 'calmar_ratio': 6.67, 'value_at_risk': 6.0,
         'expected_shortfall': 8.0, 'ulcer_index': 4.0},

        {'month': '1!', 'symbol': 'ES', 'interval': '4h', 'strategy': 'TopStrategy',
         'total_trades': 50, 'win_rate': 70.0, 'average_trade_duration_hours': 8.0,
         'total_wins_percentage_of_contract': 105.0, 'total_losses_percentage_of_contract': 15.0,
         'total_return_percentage_of_contract': 90.0, 'average_trade_return_percentage_of_contract': 1.8,
         'average_win_percentage_of_contract': 3.0, 'average_loss_percentage_of_contract': -1.0,
         'profit_factor': 7.0, 'maximum_drawdown_percentage': 5.0, 'sharpe_ratio': 3.0,
         'sortino_ratio': 4.0, 'calmar_ratio': 18.0, 'value_at_risk': 3.0,
         'expected_shortfall': 4.0, 'ulcer_index': 2.0},

        # Medium-performing strategy
        {'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'MediumStrategy',
         'total_trades': 60, 'win_rate': 55.0, 'average_trade_duration_hours': 5.0,
         'total_wins_percentage_of_contract': 60.0, 'total_losses_percentage_of_contract': 30.0,
         'total_return_percentage_of_contract': 30.0, 'average_trade_return_percentage_of_contract': 0.5,
         'average_win_percentage_of_contract': 2.0, 'average_loss_percentage_of_contract': -1.2,
         'profit_factor': 2.0, 'maximum_drawdown_percentage': 15.0, 'sharpe_ratio': 1.5,
         'sortino_ratio': 2.0, 'calmar_ratio': 2.0, 'value_at_risk': 8.0,
         'expected_shortfall': 10.0, 'ulcer_index': 6.0},

        {'month': '1!', 'symbol': 'CL', 'interval': '4h', 'strategy': 'MediumStrategy',
         'total_trades': 40, 'win_rate': 50.0, 'average_trade_duration_hours': 6.0,
         'total_wins_percentage_of_contract': 40.0, 'total_losses_percentage_of_contract': 20.0,
         'total_return_percentage_of_contract': 20.0, 'average_trade_return_percentage_of_contract': 0.5,
         'average_win_percentage_of_contract': 2.0, 'average_loss_percentage_of_contract': -1.0,
         'profit_factor': 2.0, 'maximum_drawdown_percentage': 18.0, 'sharpe_ratio': 1.0,
         'sortino_ratio': 1.5, 'calmar_ratio': 1.11, 'value_at_risk': 10.0,
         'expected_shortfall': 12.0, 'ulcer_index': 8.0},

        # Low-performing strategy (should be filtered out with min trades)
        {'month': '1!', 'symbol': 'ZS', 'interval': '1d', 'strategy': 'LowTradeStrategy',
         'total_trades': 5, 'win_rate': 40.0, 'average_trade_duration_hours': 24.0,
         'total_wins_percentage_of_contract': 10.0, 'total_losses_percentage_of_contract': 15.0,
         'total_return_percentage_of_contract': -5.0, 'average_trade_return_percentage_of_contract': -1.0,
         'average_win_percentage_of_contract': 5.0, 'average_loss_percentage_of_contract': -5.0,
         'profit_factor': 0.67, 'maximum_drawdown_percentage': 25.0, 'sharpe_ratio': -0.5,
         'sortino_ratio': -0.5, 'calmar_ratio': -0.2, 'value_at_risk': 15.0,
         'expected_shortfall': 18.0, 'ulcer_index': 12.0},

        # Single symbol strategy (for symbol count filtering)
        {'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'SingleSymbolStrategy',
         'total_trades': 90, 'win_rate': 58.0, 'average_trade_duration_hours': 3.0,
         'total_wins_percentage_of_contract': 90.0, 'total_losses_percentage_of_contract': 40.0,
         'total_return_percentage_of_contract': 50.0, 'average_trade_return_percentage_of_contract': 0.56,
         'average_win_percentage_of_contract': 1.8, 'average_loss_percentage_of_contract': -1.1,
         'profit_factor': 2.25, 'maximum_drawdown_percentage': 12.0, 'sharpe_ratio': 1.8,
         'sortino_ratio': 2.2, 'calmar_ratio': 4.17, 'value_at_risk': 7.0,
         'expected_shortfall': 9.0, 'ulcer_index': 5.0},

        # Strategy with different slippage (for slippage filtering tests)
        {'month': '1!', 'symbol': 'ZS', 'interval': '1h', 'strategy': 'HighSlippageStrategy(slippage_ticks=5.0)',
         'total_trades': 70, 'win_rate': 52.0, 'average_trade_duration_hours': 4.0,
         'total_wins_percentage_of_contract': 50.0, 'total_losses_percentage_of_contract': 30.0,
         'total_return_percentage_of_contract': 20.0, 'average_trade_return_percentage_of_contract': 0.29,
         'average_win_percentage_of_contract': 1.4, 'average_loss_percentage_of_contract': -0.9,
         'profit_factor': 1.67, 'maximum_drawdown_percentage': 16.0, 'sharpe_ratio': 1.2,
         'sortino_ratio': 1.6, 'calmar_ratio': 1.25, 'value_at_risk': 9.0,
         'expected_shortfall': 11.0, 'ulcer_index': 7.0},

        {'month': '1!', 'symbol': 'CL', 'interval': '1h', 'strategy': 'HighSlippageStrategy(slippage_ticks=5.0)',
         'total_trades': 65, 'win_rate': 51.0, 'average_trade_duration_hours': 3.5,
         'total_wins_percentage_of_contract': 45.0, 'total_losses_percentage_of_contract': 28.0,
         'total_return_percentage_of_contract': 17.0, 'average_trade_return_percentage_of_contract': 0.26,
         'average_win_percentage_of_contract': 1.4, 'average_loss_percentage_of_contract': -0.9,
         'profit_factor': 1.61, 'maximum_drawdown_percentage': 17.0, 'sharpe_ratio': 1.1,
         'sortino_ratio': 1.5, 'calmar_ratio': 1.0, 'value_at_risk': 9.5,
         'expected_shortfall': 11.5, 'ulcer_index': 7.5},

        # Additional variations for comprehensive testing
        {'month': '1!', 'symbol': 'ES', 'interval': '1d', 'strategy': 'LongTermStrategy',
         'total_trades': 25, 'win_rate': 64.0, 'average_trade_duration_hours': 48.0,
         'total_wins_percentage_of_contract': 60.0, 'total_losses_percentage_of_contract': 20.0,
         'total_return_percentage_of_contract': 40.0, 'average_trade_return_percentage_of_contract': 1.6,
         'average_win_percentage_of_contract': 4.0, 'average_loss_percentage_of_contract': -2.5,
         'profit_factor': 3.0, 'maximum_drawdown_percentage': 8.0, 'sharpe_ratio': 2.2,
         'sortino_ratio': 2.8, 'calmar_ratio': 5.0, 'value_at_risk': 5.0,
         'expected_shortfall': 6.5, 'ulcer_index': 3.5},

        {'month': '1!', 'symbol': 'CL', 'interval': '1d', 'strategy': 'LongTermStrategy',
         'total_trades': 30, 'win_rate': 60.0, 'average_trade_duration_hours': 40.0,
         'total_wins_percentage_of_contract': 55.0, 'total_losses_percentage_of_contract': 25.0,
         'total_return_percentage_of_contract': 30.0, 'average_trade_return_percentage_of_contract': 1.0,
         'average_win_percentage_of_contract': 3.5, 'average_loss_percentage_of_contract': -2.3,
         'profit_factor': 2.2, 'maximum_drawdown_percentage': 10.0, 'sharpe_ratio': 1.8,
         'sortino_ratio': 2.3, 'calmar_ratio': 3.0, 'value_at_risk': 6.0,
         'expected_shortfall': 7.5, 'ulcer_index': 4.5},

        {'month': '1!', 'symbol': 'ZS', 'interval': '4h', 'strategy': 'LongTermStrategy',
         'total_trades': 35, 'win_rate': 62.0, 'average_trade_duration_hours': 36.0,
         'total_wins_percentage_of_contract': 58.0, 'total_losses_percentage_of_contract': 22.0,
         'total_return_percentage_of_contract': 36.0, 'average_trade_return_percentage_of_contract': 1.03,
         'average_win_percentage_of_contract': 3.7, 'average_loss_percentage_of_contract': -1.9,
         'profit_factor': 2.64, 'maximum_drawdown_percentage': 9.0, 'sharpe_ratio': 2.0,
         'sortino_ratio': 2.5, 'calmar_ratio': 4.0, 'value_at_risk': 5.5,
         'expected_shortfall': 7.0, 'ulcer_index': 4.0},
    ])


@pytest.fixture
def filtering_strategy_data():
    """
    Strategy results for filtering tests.

    7 rows designed for testing various filter criteria:
    - Min avg trades per combination
    - Symbol/interval filtering
    - Slippage filtering
    - Symbol count filtering

    Used by: test_data_helpers

    Returns:
        DataFrame with 7 rows for filter testing
    """
    return pd.DataFrame([
        # HighPerformer - multiple symbols/intervals
        {'strategy': 'HighPerformer(slippage_ticks=1.0)', 'symbol': 'ZS', 'interval': '1h',
         'total_trades': 100, 'win_rate': 65.0, 'profit_factor': 3.0,
         'sharpe_ratio': 2.5, 'maximum_drawdown_percentage': 10.0},
        {'strategy': 'HighPerformer(slippage_ticks=1.0)', 'symbol': 'CL', 'interval': '1h',
         'total_trades': 80, 'win_rate': 60.0, 'profit_factor': 2.8,
         'sharpe_ratio': 2.3, 'maximum_drawdown_percentage': 12.0},
        {'strategy': 'HighPerformer(slippage_ticks=1.0)', 'symbol': 'ES', 'interval': '4h',
         'total_trades': 50, 'win_rate': 70.0, 'profit_factor': 3.5,
         'sharpe_ratio': 3.0, 'maximum_drawdown_percentage': 8.0},

        # MediumPerformer - fewer trades
        {'strategy': 'MediumPerformer(slippage_ticks=2.0)', 'symbol': 'ZS', 'interval': '1h',
         'total_trades': 30, 'win_rate': 55.0, 'profit_factor': 2.0,
         'sharpe_ratio': 1.5, 'maximum_drawdown_percentage': 15.0},
        {'strategy': 'MediumPerformer(slippage_ticks=2.0)', 'symbol': 'CL', 'interval': '4h',
         'total_trades': 25, 'win_rate': 50.0, 'profit_factor': 1.8,
         'sharpe_ratio': 1.3, 'maximum_drawdown_percentage': 18.0},

        # LowTradeStrategy - insufficient trades
        {'strategy': 'LowTradeStrategy(slippage_ticks=0.5)', 'symbol': 'ZS', 'interval': '1d',
         'total_trades': 5, 'win_rate': 40.0, 'profit_factor': 0.8,
         'sharpe_ratio': 0.5, 'maximum_drawdown_percentage': 25.0},

        # SingleSymbolStrategy - only one symbol
        {'strategy': 'SingleSymbolStrategy(slippage_ticks=1.5)', 'symbol': 'ES', 'interval': '1h',
         'total_trades': 60, 'win_rate': 58.0, 'profit_factor': 2.2,
         'sharpe_ratio': 1.8, 'maximum_drawdown_percentage': 13.0},
    ])


@pytest.fixture
def formatting_strategy_data():
    """
    Strategy results for formatting tests.

    2 rows with strategy names containing parameters to extract:
    - rollover, trailing, slippage_ticks in strategy name
    - Various numeric precision for rounding tests

    Used by: test_formatters

    Returns:
        DataFrame with 2 rows for formatting tests
    """
    return pd.DataFrame([
        {
            'strategy': 'RSI(period=14,lower=30,upper=70,rollover=True,trailing=2.5,slippage_ticks=1.0)',
            'symbol': 'ZS',
            'interval': '1h',
            'total_trades': 100,
            'win_rate': 65.5555,
            'profit_factor': 3.141592653589793,
            'average_trade_return_percentage_of_contract': 1.23456789,
            'maximum_drawdown_percentage': 10.987654321,
            'sharpe_ratio': 2.5678901234,
            'sortino_ratio': 3.0123456789,
            'calmar_ratio': 10.5555555555,
            'value_at_risk': 5.12345,
            'expected_shortfall': 7.98765,
            'ulcer_index': 3.456789
        },
        {
            'strategy': 'MACD(fast=12,slow=26,signal=9,rollover=False,trailing=None,slippage_ticks=2.0)',
            'symbol': 'CL',
            'interval': '4h',
            'total_trades': 50,
            'win_rate': 55.0,
            'profit_factor': 2.0,
            'average_trade_return_percentage_of_contract': 0.5,
            'maximum_drawdown_percentage': 15.0,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'calmar_ratio': 3.33,
            'value_at_risk': 8.0,
            'expected_shortfall': 10.0,
            'ulcer_index': 6.0
        }
    ])


# ==================== Helper Fixtures ====================

@pytest.fixture
def real_results_file():
    """
    Path to real backtest results file (if it exists).

    Used by: test_strategy_analyzer (integration tests)

    Returns:
        Path to real results file or None if not found
    """
    file_path = f'{BACKTESTING_DIR}/mass_test_results_all.parquet'
    return file_path if os.path.exists(file_path) else None
