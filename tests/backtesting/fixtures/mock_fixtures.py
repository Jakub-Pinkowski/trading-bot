"""
Common mock and patch fixtures for backtesting tests.

Provides reusable mocks for:
- Logger suppression
- File operations
- Cache directories
- External dependencies
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


# ==================== Logger Mocks ====================

@pytest.fixture
def mock_logger():
    """
    Mock logger to suppress log output during tests.

    Useful for tests that generate lots of log messages that clutter test output.
    All logging methods (info, debug, warning, error) are mocked.

    Returns:
        MagicMock with all logger methods available

    Example:
        def test_something(mock_logger, monkeypatch):
            monkeypatch.setattr('app.backtesting.strategies.base_strategy.logger', mock_logger)
            # Now logger calls won't print anything
            strategy.run()  # Would normally log, but now silent
            assert mock_logger.info.called
    """
    logger_mock = MagicMock()
    logger_mock.info = MagicMock()
    logger_mock.debug = MagicMock()
    logger_mock.warning = MagicMock()
    logger_mock.error = MagicMock()
    logger_mock.critical = MagicMock()
    return logger_mock


@pytest.fixture
def suppress_all_logs(monkeypatch):
    """
    Automatically suppress all logger output across common modules.

    This is a convenience fixture that patches multiple logger instances
    at once. Use when you want to silence all logging without manually
    patching each module.

    Example:
        def test_quiet_execution(suppress_all_logs):
            # All logging from cache, indicators, testing, etc. is suppressed
            run_mass_backtest()  # Runs silently
    """
    mock = MagicMock()

    # Patch common logger locations that actually exist in the app
    logger_paths = [
        'app.utils.backtesting_utils.indicators_utils.logger',
        'app.backtesting.cache.cache_base.logger',
        'app.backtesting.cache.indicators_cache.logger',
        'app.backtesting.cache.dataframe_cache.logger',
        'app.backtesting.testing.orchestrator.logger',
        'app.backtesting.testing.mass_tester.logger',
        'app.utils.logger.logger',
        'app.utils.file_utils.logger',
    ]

    for path in logger_paths:
        try:
            monkeypatch.setattr(path, mock)
        except (AttributeError, ImportError):
            # Skip if module doesn't exist or logger not present
            pass

    return mock


# ==================== File Operation Mocks ====================

@pytest.fixture
def mock_file_save():
    """
    Mock file save operations without actually writing to disk.

    Returns:
        MagicMock that tracks all save attempts

    Example:
        def test_save_results(mock_file_save, monkeypatch):
            monkeypatch.setattr('app.utils.file_utils.save_file', mock_file_save)

            strategy.save_results('results.csv')

            assert mock_file_save.called
            assert mock_file_save.call_args[0][0] == 'results.csv'
    """
    return MagicMock()


@pytest.fixture
def mock_file_load():
    """
    Mock file load operations with configurable return values.

    Returns:
        MagicMock that can be configured to return test data

    Example:
        def test_load_results(mock_file_load, monkeypatch):
            mock_file_load.return_value = {'test': 'data'}
            monkeypatch.setattr('app.utils.file_utils.load_file', mock_file_load)

            data = load_strategy_results('results.json')
            assert data == {'test': 'data'}
    """
    return MagicMock()


@pytest.fixture
def mock_parquet_operations():
    """
    Mock pandas parquet read/write operations.

    Returns:
        Dict with 'read' and 'write' mock functions

    Example:
        def test_parquet_io(mock_parquet_operations, monkeypatch, zs_1h_data):
            mock_parquet_operations['read'].return_value = zs_1h_data
            monkeypatch.setattr('pandas.read_parquet', mock_parquet_operations['read'])

            df = pd.read_parquet('fake_path.parquet')
            assert len(df) == len(zs_1h_data)
    """
    return {
        'read': MagicMock(),
        'write': MagicMock()
    }


# ==================== Temporary Directory Fixtures ====================

@pytest.fixture
def temp_cache_dir(tmp_path):
    """
    Provide temporary directory for cache operations.

    Automatically cleaned up after test completes. Useful for testing
    cache functionality without polluting real cache directories.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path object pointing to temporary cache directory

    Example:
        def test_cache_storage(temp_cache_dir):
            cache = IndicatorCache(cache_dir=temp_cache_dir)
            cache.set('key', 'value')

            # Verify file was created
            assert (temp_cache_dir / 'cache.pkl').exists()
            # Automatically cleaned up after test
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


@pytest.fixture
def temp_results_dir(tmp_path):
    """
    Provide temporary directory for test results/output files.

    Automatically cleaned up after test completes.

    Returns:
        Path object pointing to temporary results directory

    Example:
        def test_save_results(temp_results_dir):
            save_backtest_results(temp_results_dir / 'results.csv', data)
            assert (temp_results_dir / 'results.csv').exists()
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """
    Provide temporary directory for test data files.

    Useful when tests need to create temporary data structures or
    simulate data directory organization.

    Returns:
        Path object pointing to temporary data directory

    Example:
        def test_data_loading(temp_data_dir):
            # Create test structure
            symbol_dir = temp_data_dir / 'ZS'
            symbol_dir.mkdir()
            test_df.to_parquet(symbol_dir / 'ZS_1h.parquet')

            # Test loading
            df = load_data(temp_data_dir, 'ZS', '1h')
            assert len(df) > 0
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def isolated_temp_dirs(tmp_path):
    """
    Provide complete set of isolated temporary directories.

    Returns:
        Dict with 'cache', 'results', and 'data' directory paths

    Example:
        def test_full_workflow(isolated_temp_dirs):
            dirs = isolated_temp_dirs

            # Use separate dirs for different purposes
            cache = Cache(dirs['cache'])
            save_results(dirs['results'] / 'output.csv')
            load_data_from(dirs['data'])
    """
    dirs = {
        'cache': tmp_path / "cache",
        'results': tmp_path / "results",
        'data': tmp_path / "data"
    }

    for directory in dirs.values():
        directory.mkdir(exist_ok=True)

    return dirs


# ==================== Cache Mocks ====================

@pytest.fixture
def mock_indicator_cache():
    """
    Mock indicator cache to control cache behavior in tests.

    Returns:
        MagicMock configured with cache interface (get, set, contains, clear)

    Example:
        def test_with_cache_hit(mock_indicator_cache, monkeypatch):
            # Configure mock to simulate cache hit
            mock_indicator_cache.contains.return_value = True
            mock_indicator_cache.get.return_value = pd.Series([70, 65, 60])

            monkeypatch.setattr('app.backtesting.cache.indicator_cache', mock_indicator_cache)

            rsi = calculate_rsi(prices, period=14)
            assert mock_indicator_cache.get.called  # Verify cache was checked
    """
    cache_mock = MagicMock()
    cache_mock.get = MagicMock(return_value=None)
    cache_mock.set = MagicMock()
    cache_mock.contains = MagicMock(return_value=False)
    cache_mock.clear = MagicMock()
    cache_mock.hits = 0
    cache_mock.misses = 0
    return cache_mock


@pytest.fixture
def mock_dataframe_cache():
    """
    Mock dataframe cache to control cache behavior in tests.

    Returns:
        MagicMock configured with cache interface

    Example:
        def test_dataframe_caching(mock_dataframe_cache, monkeypatch):
            mock_dataframe_cache.contains.return_value = True
            mock_dataframe_cache.get.return_value = test_df

            monkeypatch.setattr('app.backtesting.cache.dataframe_cache', mock_dataframe_cache)

            df = load_with_cache('ZS', '1h')
            assert mock_dataframe_cache.get.called
    """
    cache_mock = MagicMock()
    cache_mock.get = MagicMock(return_value=None)
    cache_mock.set = MagicMock()
    cache_mock.contains = MagicMock(return_value=False)
    cache_mock.clear = MagicMock()
    cache_mock.hits = 0
    cache_mock.misses = 0
    return cache_mock


@pytest.fixture
def clear_all_caches():
    """
    Fixture that clears all caches before and after test execution.

    Use this for tests that need clean cache state without mock interference.

    Example:
        def test_fresh_calculation(clear_all_caches, zs_1h_data):
            # Guaranteed no cached values will interfere
            rsi = calculate_rsi(zs_1h_data['close'], period=14)
            # This is a fresh calculation, not from cache
    """
    # Import here to avoid circular dependencies
    try:
        from app.backtesting.cache import indicator_cache, dataframe_cache

        # Clear before test
        indicator_cache.clear()
        dataframe_cache.clear()
        indicator_cache.reset_stats()
        dataframe_cache.reset_stats()

        yield

        # Clear after test
        indicator_cache.clear()
        dataframe_cache.clear()
        indicator_cache.reset_stats()
        dataframe_cache.reset_stats()
    except ImportError:
        # Cache modules not available, skip clearing
        yield


# ==================== Time and Randomness Control ====================

@pytest.fixture
def freeze_time():
    """
    Fixture factory to freeze time at a specific moment.

    Returns:
        Function that patches datetime.now() to return fixed time

    Example:
        def test_time_dependent(freeze_time):
            from datetime import datetime
            fixed_time = datetime(2024, 1, 1, 12, 0, 0)

            with freeze_time(fixed_time):
                trade = create_trade()
                assert trade.timestamp == fixed_time
    """

    @contextmanager
    def _freeze(frozen_datetime):
        with patch('datetime.datetime') as mock_dt:
            mock_dt.now.return_value = frozen_datetime
            mock_dt.utcnow.return_value = frozen_datetime
            yield mock_dt

    return _freeze


@pytest.fixture
def deterministic_random(monkeypatch):
    """
    Make random operations deterministic by fixing the seed.

    Useful for tests involving random elements (sampling, noise, etc.)
    to ensure reproducible results.

    Example:
        def test_random_sampling(deterministic_random):
            # Random operations will be consistent across test runs
            sample = random.sample(data, 10)
            # This sample will always be the same
    """
    import random
    import numpy as np

    # Fix seeds
    random.seed(42)
    np.random.seed(42)

    yield

    # Reset to non-deterministic after test
    random.seed(None)
    np.random.seed(None)


# ==================== Strategy Component Mocks ====================

@pytest.fixture
def mock_contract_info():
    """
    Mock contract information without loading from config.

    Returns:
        Function that returns mock contract info for any symbol

    Example:
        def test_strategy_init(mock_contract_info):
            info = mock_contract_info('ZS')
            strategy = Strategy(symbol='ZS', contract_info=info)
            assert strategy.tick_size == 0.25
    """

    def _get_info(symbol):
        # Default contract info for common symbols
        defaults = {
            'ZS': {'symbol': 'ZS', 'tick_size': 0.25, 'tick_value': 12.50, 'contract_multiplier': 50},
            'CL': {'symbol': 'CL', 'tick_size': 0.01, 'tick_value': 10.00, 'contract_multiplier': 1000},
            'ES': {'symbol': 'ES', 'tick_size': 0.25, 'tick_value': 12.50, 'contract_multiplier': 50},
            'NQ': {'symbol': 'NQ', 'tick_size': 0.25, 'tick_value': 5.00, 'contract_multiplier': 20},
        }

        return defaults.get(symbol, {
            'symbol': symbol,
            'tick_size': 0.01,
            'tick_value': 1.00,
            'contract_multiplier': 1
        })

    return _get_info


@pytest.fixture
def mock_position_manager():
    """
    Mock position manager for strategy testing.

    Returns:
        MagicMock with position manager interface

    Example:
        def test_strategy_positions(mock_position_manager, monkeypatch):
            mock_position_manager.has_position.return_value = False
            mock_position_manager.open_position.return_value = True

            strategy.position_manager = mock_position_manager
            strategy.execute()

            assert mock_position_manager.open_position.called
    """
    pm_mock = MagicMock()
    pm_mock.has_position = MagicMock(return_value=False)
    pm_mock.open_position = MagicMock()
    pm_mock.close_position = MagicMock()
    pm_mock.get_current_position = MagicMock(return_value=None)
    pm_mock.update_trailing_stop = MagicMock()
    return pm_mock


@pytest.fixture
def mock_trailing_stop_manager():
    """
    Mock trailing stop manager for strategy testing.

    Returns:
        MagicMock with trailing stop manager interface

    Example:
        def test_trailing_stops(mock_trailing_stop_manager):
            mock_trailing_stop_manager.should_exit.return_value = True

            strategy.trailing_stop_manager = mock_trailing_stop_manager
            exit_signal = strategy.check_exit()

            assert exit_signal is True
    """
    ts_mock = MagicMock()
    ts_mock.initialize = MagicMock()
    ts_mock.update = MagicMock()
    ts_mock.should_exit = MagicMock(return_value=False)
    ts_mock.get_stop_price = MagicMock(return_value=None)
    return ts_mock


# ==================== File Writing Mocks ====================

@pytest.fixture
def mock_csv_writer():
    """
    Mock CSV writing operations.

    Returns:
        MagicMock that tracks CSV write attempts

    Example:
        def test_export_csv(mock_csv_writer, monkeypatch):
            monkeypatch.setattr('pandas.DataFrame.to_csv', mock_csv_writer)

            df.to_csv('results.csv')

            assert mock_csv_writer.called
    """
    return MagicMock()


# ==================== Multiprocessing Mocks ====================

@pytest.fixture
def mock_process_pool():
    """
    Mock ProcessPoolExecutor for testing parallel execution without actual processes.

    Useful for testing multiprocessing logic without the overhead of spawning
    real processes. Executes tasks synchronously in same process.

    Returns:
        MagicMock configured to mimic ProcessPoolExecutor interface

    Example:
        def test_parallel_execution(mock_process_pool, monkeypatch):
            monkeypatch.setattr('concurrent.futures.ProcessPoolExecutor',
                              lambda *args, **kwargs: mock_process_pool)

            results = run_parallel_backtests(strategies)
            assert len(results) == len(strategies)
    """

    class MockFuture:
        def __init__(self, result):
            self._result = result

        def result(self, timeout=None):
            return self._result

    class MockExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, fn, *args, **kwargs):
            # Execute synchronously
            result = fn(*args, **kwargs)
            return MockFuture(result)

        def map(self, fn, *iterables, **kwargs):
            return map(fn, *iterables)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return MockExecutor


# ==================== Validator Mocks ====================

@pytest.fixture
def mock_validator():
    """
    Mock parameter validator for testing strategies without validation overhead.

    Returns:
        MagicMock that always passes validation

    Example:
        def test_strategy_logic_only(mock_validator, monkeypatch):
            # Skip validation to focus on strategy logic
            monkeypatch.setattr('app.backtesting.validators.rsi_validator.RSIValidator',
                              lambda *args: mock_validator)

            strategy = RSIStrategy(period=5, lower=10, upper=90, ...)
            # Would normally fail validation, but mock allows it
    """
    validator_mock = MagicMock()
    validator_mock.validate.return_value = (True, [])  # (is_valid, warnings)
    validator_mock.warnings = []
    return validator_mock


# ==================== Composite Mocks ====================

@pytest.fixture
def fully_mocked_environment(
    mock_indicator_cache,
    mock_dataframe_cache,
    temp_cache_dir,
    monkeypatch
):
    """
    Comprehensive mock environment for isolated testing.

    Patches cache dependencies to provide complete isolation.
    Use for integration tests that should run without side effects.

    Returns:
        Dict with all mocked components

    Example:
        def test_isolated_execution(fully_mocked_environment):
            env = fully_mocked_environment

            # Caches are mocked - no real I/O or caching
            result = run_full_backtest(strategy, data)

            # Verify behavior through mocks
            assert env['indicator_cache'].set.called
    """
    # Patch caches
    try:
        monkeypatch.setattr('app.backtesting.cache.indicator_cache', mock_indicator_cache)
        monkeypatch.setattr('app.backtesting.cache.dataframe_cache', mock_dataframe_cache)
    except (AttributeError, ImportError):
        # If caches don't exist, skip patching
        pass

    return {
        'indicator_cache': mock_indicator_cache,
        'dataframe_cache': mock_dataframe_cache,
        'cache_dir': temp_cache_dir
    }
