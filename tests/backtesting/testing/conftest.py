"""
Shared fixtures for testing module tests.

Provides common fixtures for MassTester, orchestrator, runner, and reporting tests.
Eliminates ~200 lines of duplicated mock setup code across test files.
"""
import concurrent.futures
from unittest.mock import MagicMock

import pandas as pd
import pytest

import app.backtesting.testing.orchestrator as orch_module
from app.backtesting.testing.mass_tester import MassTester


# ==================== MassTester Fixtures ====================

@pytest.fixture
def mock_strategy_factory(monkeypatch):
    """
    Mock strategy factory for testing strategy addition without real creation.

    Returns:
        Tuple of (mock_create, mock_get_name)

    Usage:
        def test_something(mock_strategy_factory):
            mock_create, mock_get_name = mock_strategy_factory
            tester.add_rsi_tests([14], [30], [70], [True], [None], [1])
            assert mock_create.call_count == 1
    """
    mock_create = MagicMock(return_value=MagicMock())
    mock_get_name = MagicMock(return_value="Test_Strategy")
    monkeypatch.setattr('app.backtesting.testing.mass_tester.create_strategy', mock_create)
    monkeypatch.setattr('app.backtesting.testing.mass_tester.get_strategy_name', mock_get_name)
    return (mock_create, mock_get_name)


@pytest.fixture
def simple_tester():
    """
    Create a basic MassTester instance for testing.

    Returns:
        MassTester instance with ['1!'], ['ZS'], ['1h']
    """
    return MassTester(['1!'], ['ZS'], ['1h'])


@pytest.fixture
def mock_tester():
    """
    Create a fully mocked MassTester instance.

    Returns:
        MagicMock configured with standard tester attributes

    Usage:
        def test_something(mock_tester):
            result = run_tests(mock_tester, verbose=False, max_workers=1, skip_existing=False)
    """
    tester = MagicMock()
    tester.strategies = [('RSI_14_30_70', MagicMock())]
    tester.tested_months = ['1!']
    tester.symbols = ['ZS']
    tester.intervals = ['1h']
    tester.switch_dates_dict = {'ZS': []}
    tester.results = []
    return tester


# ==================== Orchestrator Fixtures ====================

@pytest.fixture
def mock_orchestrator_environment(monkeypatch):
    """
    Mock all orchestrator dependencies in one fixture.

    Returns:
        Dict with all mocked components:
        - indicator_cache
        - dataframe_cache
        - load_existing_results
        - executor
        - as_completed

    Usage:
        def test_something(mock_orchestrator_environment):
            mocks = mock_orchestrator_environment
            # mocks['indicator_cache'], mocks['load_existing_results'], etc.
    """
    mock_ind = MagicMock()
    mock_df = MagicMock()
    mock_load = MagicMock(return_value=(pd.DataFrame(), set()))
    mock_executor = MagicMock()
    mock_as_completed = MagicMock(return_value=[])

    mock_ind.reset_stats = MagicMock()
    mock_df.reset_stats = MagicMock()

    monkeypatch.setattr(orch_module, 'indicator_cache', mock_ind)
    monkeypatch.setattr(orch_module, 'dataframe_cache', mock_df)
    monkeypatch.setattr(orch_module, 'load_existing_results', mock_load)
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', mock_executor)
    monkeypatch.setattr(concurrent.futures, 'as_completed', mock_as_completed)

    return {
        'indicator_cache': mock_ind,
        'dataframe_cache': mock_df,
        'load_existing_results': mock_load,
        'executor': mock_executor,
        'as_completed': mock_as_completed
    }


# ==================== Reporting Fixtures ====================

@pytest.fixture
def sample_test_result():
    """
    Create a sample test result dictionary with all required fields.

    Returns:
        Dict with complete test result structure
    """
    return {
        'month': '1!',
        'symbol': 'ZS',
        'interval': '1h',
        'strategy': 'RSI_14_30_70',
        'metrics': {
            'total_trades': 10,
            'win_rate': 60.0,
            'average_trade_duration_hours': 4.5,
            'total_wins_percentage_of_contract': 100.0,
            'total_losses_percentage_of_contract': 50.0,
            'total_return_percentage_of_contract': 50.0,
            'average_trade_return_percentage_of_contract': 5.0,
            'average_win_percentage_of_contract': 10.0,
            'average_loss_percentage_of_contract': -5.0,
            'profit_factor': 2.0,
            'maximum_drawdown_percentage': 15.0,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'calmar_ratio': 1.2,
            'value_at_risk': 10.0,
            'expected_shortfall': 12.0,
            'ulcer_index': 5.0
        }
    }


@pytest.fixture
def sample_test_results(sample_test_result):
    """
    Create a list of sample test results.

    Args:
        sample_test_result: Injected fixture

    Returns:
        List containing one test result
    """
    return [sample_test_result]


# ==================== Validator Fixtures ====================

@pytest.fixture
def valid_ohlcv_dataframe():
    """
    Create a valid OHLCV DataFrame for validator testing.

    Returns:
        DataFrame that passes all validation checks (3 rows)
    """
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0],
        'high': [102.0, 103.0, 104.0],
        'low': [99.0, 100.0, 101.0],
        'close': [101.0, 102.0, 103.0],
        'volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))


# ==================== Test Preparation Fixtures ====================

@pytest.fixture
def existing_results_factory():
    """
    Factory for creating existing results data structure.

    Returns:
        Function that creates (DataFrame, set) tuple

    Usage:
        def test_something(existing_results_factory):
            existing_data = existing_results_factory([
                ('1!', 'ZS', '1h', 'RSI_14_30_70'),
                ('2!', 'CL', '15m', 'EMA_9_21')
            ])
            result = check_test_exists(existing_data, '1!', 'ZS', '1h', 'RSI_14_30_70')
    """

    def _create_existing_data(combinations):
        """
        Create existing results data structure.

        Args:
            combinations: List of (month, symbol, interval, strategy) tuples

        Returns:
            Tuple of (DataFrame, set) matching load_existing_results format
        """
        if not combinations:
            return (pd.DataFrame(), set())

        df = pd.DataFrame(combinations, columns=['month', 'symbol', 'interval', 'strategy'])
        combinations_set = set(combinations)
        return (df, combinations_set)

    return _create_existing_data
