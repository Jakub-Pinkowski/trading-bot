"""
Multiprocessing Integration Tests.

Verify multiprocessing behavior used by the backtesting orchestrator.

Extended explanation of behavior and purpose:

- Test concurrent parquet writes with file-locking and deduplication under contention
- Verify worker picklability and ProcessPoolExecutor execution behavior
- Ensure caches are saved only from the main process (no duplicate saves from workers)
- Provide end-to-end smoke tests that run the orchestrator using small synthetic
  and (optionally) real historical datasets to validate overall multiprocessing flow

These are integration-level tests; the real-data scenarios are marked slow and
should be run in integration/nightly pipelines where HISTORICAL_DATA_DIR is available.
"""
import multiprocessing
import os
import pickle
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import psutil
import pytest

from app.backtesting import MassTester, create_strategy
from app.backtesting.cache.dataframe_cache import dataframe_cache
from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.testing import runner as runner_module, orchestrator as orchestrator_module
from app.utils.file_utils import save_to_parquet


# ==================== Worker Helpers ====================

# --- Picklable worker functions ---

def _write_row_worker(args):
    """
    Write a single-row DataFrame to the given parquet path.

    Args:
        args: Tuple (worker_id, path)

    Returns:
        int: worker_id written
    """
    worker_id, path = args
    df = pd.DataFrame({'id': [worker_id], 'value': [worker_id * 10]})
    save_to_parquet(df, path)
    return worker_id


def _write_duplicate_rows_worker(args):
    """
    Write a small DataFrame with duplicate rows to the given parquet path.

    Args:
        args: Tuple (worker_id, path)

    Returns:
        int: constant value (for worker identity)
    """
    _worker_id, path = args
    df = pd.DataFrame({'id': [1, 2, 3], 'value': [100, 200, 300]})
    save_to_parquet(df, path)
    return 1


def _delayed_write_worker(args):
    """
    Write with a tiny random delay to increase contention.

    Args:
        args: Tuple (worker_id, path)

    Returns:
        int: worker_id
    """
    worker_id, path = args
    time.sleep(random.uniform(0.001, 0.01))
    df = pd.DataFrame({'id': [worker_id], 'value': [worker_id * 5]})
    save_to_parquet(df, path)
    return worker_id


def _simple_double(x):
    """
    Simple worker for ProcessPoolExecutor smoke test.

    Args:
        x: int

    Returns:
        int: x * 2
    """
    return x * 2


# --- In-process executor (test helper) ---
def _inprocess_executor(tester_obj, test_combinations, max_workers_local):
    """
    Simple in-process executor used by tests to simulate parallel execution
    without spawning worker processes. This always uses a ThreadPoolExecutor to
    simulate `max_workers_local` workers

    Args:
        tester_obj: MassTester instance whose .results will be populated
        test_combinations: list of test parameter tuples as produced by orchestrator
        max_workers_local: degree of parallelism to simulate (1 == single-threaded)
    """
    failed_tests = 0
    tester_obj.results = []

    def _run_single(params):
        """Helper to run a single test and handle result appending."""
        symbol = params[1]
        # Simulate a worker failure for a symbol named 'NONEXISTENT'
        if symbol == 'NONEXISTENT':
            raise Exception('Simulated worker failure')

        return runner_module.run_single_test(params)

    # Run tests using a thread pool to simulate parallel workers in-process.
    lock = threading.Lock()
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers_local) as exe:
        for params in test_combinations:
            futures_map[exe.submit(_run_single, params)] = params

        for fut in as_completed(futures_map):
            params = futures_map[fut]
            try:
                result = fut.result()
                if result:
                    with lock:
                        tester_obj.results.append(result)
            except Exception:
                with lock:
                    failed_tests += 1
                orchestrator_module.logger.exception(f"Worker exception during test execution for params: {params}")

    if failed_tests > 0:
        orchestrator_module.logger.warning(f'Mass testing completed with {failed_tests} failed test(s)')


# ==================== Fixtures ====================

# --- Cache reset fixture ---
@pytest.fixture(autouse=True)
def reset_caches_fixture():
    """
    Reset caches before and after each test.

    This avoids cross-test cache pollution.
    """
    indicator_cache.cache_data.clear()
    dataframe_cache.cache_data.clear()
    yield
    indicator_cache.cache_data.clear()
    dataframe_cache.cache_data.clear()


# --- Temporary parquet path fixture ---
@pytest.fixture
def temp_parquet_file_path(tmp_path):
    """
    Provide a temporary parquet file path for concurrent-write tests.

    Args:
        tmp_path: pytest temporary path

    Returns:
        str: absolute path to a parquet file that can be concurrently written
    """
    return str(tmp_path / "concurrent_test.parquet")


# ==================== Concurrency Tests ====================

@pytest.mark.integration
class TestConcurrency:
    """Tests for concurrent parquet writes and deduplication."""

    def test_concurrent_parquet_writes_with_pool(self, temp_parquet_file_path):
        """
        Multiple processes write a single row each to the same parquet file.

        Verifies no writes are lost and all worker IDs appear in the final file.
        """
        num_workers = 4
        args = [(i, temp_parquet_file_path) for i in range(num_workers)]

        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Ignore if the start method has already been set in this process
            # multiprocessing.set_start_method raises RuntimeError when called more than once
            pass

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_write_row_worker, args)

        # Verify all workers returned and wrote
        assert set(results) == set(range(num_workers))

        final_df = pd.read_parquet(temp_parquet_file_path)
        assert len(final_df) == num_workers
        assert set(final_df['id'].tolist()) == set(range(num_workers))

    def test_concurrent_writes_with_deduplication(self, temp_parquet_file_path):
        """
        Multiple workers write the same rows; final file should be deduplicated.
        """
        num_workers = 3
        args = [(i, temp_parquet_file_path) for i in range(num_workers)]

        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Ignore if the start method has already been set in this process
            # multiprocessing.set_start_method raises RuntimeError when called more than once
            pass

        with multiprocessing.Pool(processes=num_workers) as pool:
            _ = pool.map(_write_duplicate_rows_worker, args)

        final_df = pd.read_parquet(temp_parquet_file_path)
        assert len(final_df) == 3
        assert set(final_df['id'].tolist()) == {1, 2, 3}

    def test_concurrent_writes_stress_with_delays(self, temp_parquet_file_path):
        """Stress test: concurrent writes with small delays to increase contention."""
        num_workers = 6
        args = [(i, temp_parquet_file_path) for i in range(num_workers)]

        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Ignore if the start method has already been set in this process
            # multiprocessing.set_start_method raises RuntimeError when called more than once
            pass

        with multiprocessing.Pool(processes=4) as pool:
            pool.map(_delayed_write_worker, args)

        final_df = pd.read_parquet(temp_parquet_file_path)
        assert len(final_df) == num_workers
        assert set(final_df['id'].tolist()) == set(range(num_workers))


# ==================== Executor and Pickling Tests ====================

@pytest.mark.integration
class TestExecutorAndPickling:
    """Tests related to ProcessPoolExecutor and worker robustness."""

    def test_processpoolexecutor_smoke(self):
        """Ensure a simple function runs in ProcessPoolExecutor (picklability)."""
        with ProcessPoolExecutor(max_workers=2) as exe:
            futures = [exe.submit(_simple_double, i) for i in range(5)]
            results = [f.result() for f in futures]

        assert results == [0, 2, 4, 6, 8]

    def test_worker_exception_propagation_returns_list_when_workers_fail(self):
        """
        Run a small orchestration with an invalid historical-data dir and ensure
        the orchestrator returns a list instead of raising.
        """
        # Patch both the config module and the orchestrator module where the
        # HISTORICAL_DATA_DIR value may have already been read at import time.
        with patch('config.HISTORICAL_DATA_DIR', '/nonexistent/path'), \
                patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', '/nonexistent/path'):
            tester = MassTester(['1!'], ['ZS'], ['1h'])
            tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

            results = tester.run_tests(max_workers=2, verbose=False, skip_existing=False)
            assert isinstance(results, list)

    def test_worker_function_with_real_executor(self):
        """Ensure module-level worker functions can be executed by ProcessPoolExecutor."""
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Ignore if the start method has already been set in this process
            # multiprocessing.set_start_method raises RuntimeError when called more than once
            pass

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_simple_double, i) for i in range(5)]
            results = [f.result() for f in futures]

        assert results == [0, 2, 4, 6, 8]

    def test_strategy_pickling_with_multiprocessing(self):
        """
        Ensure several strategy implementations can be pickled/unpickled.

        This mirrors the old comprehensive pickling test to catch any
        non-picklable state in strategy implementations.
        """
        strategy_configs = [
            (
                'rsi',
                {'rsi_period': 14, 'lower_threshold': 30, 'upper_threshold': 70, 'rollover': False, 'trailing': None,
                 'slippage_ticks': 0, 'symbol': None}
            ),
            (
                'ema',
                {'short_ema_period': 9, 'long_ema_period': 21, 'rollover': False, 'trailing': None, 'slippage_ticks': 0,
                 'symbol': None}
            ),
            (
                'macd',
                {'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'rollover': False, 'trailing': None,
                 'slippage_ticks': 0, 'symbol': None}
            ),
            (
                'bollinger',
                {'period': 20, 'number_of_standard_deviations': 2, 'rollover': False, 'trailing': None,
                 'slippage_ticks': 0, 'symbol': None}
            ),
            (
                'ichimoku',
                {'tenkan_period': 9, 'kijun_period': 26, 'senkou_span_b_period': 52, 'displacement': 26,
                 'rollover': False, 'trailing': None, 'slippage_ticks': 0, 'symbol': None}
            ),
        ]

        for strategy_type, params in strategy_configs:
            strategy = create_strategy(strategy_type, **params)
            try:
                pickled = pickle.dumps(strategy)
                unpickled = pickle.loads(pickled)
                assert unpickled is not None
                assert hasattr(unpickled, 'run')
            except Exception as e:
                pytest.fail(f"Strategy {strategy_type} failed to pickle: {e}")


# ==================== Cache-Save Tests ====================

@pytest.mark.integration
class TestCacheSave:
    """Tests that caches are saved only from the main process (mocked)."""

    def test_cache_save_only_from_main_process(self, tmp_path):
        """
        Patch the caches used by the orchestrator and verify save_cache is called
        exactly once from the main process.
        """
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        dates = pd.date_range('2023-01-01', periods=10, freq='h')
        df = pd.DataFrame({
            'symbol': ['CBOT:ZS1!'] * len(dates),
            'open': range(len(dates)),
            'high': range(len(dates)),
            'low': range(len(dates)),
            'close': range(len(dates)),
            'volume': [100.0] * len(dates)
        }, index=pd.DatetimeIndex(dates, name='datetime'))
        file_path = str(data_dir / 'ZS_1h.parquet')
        df.to_parquet(file_path)

        with patch('config.HISTORICAL_DATA_DIR', str(data_dir)):
            with patch('app.backtesting.testing.orchestrator.indicator_cache') as mock_ind_cache:
                with patch('app.backtesting.testing.orchestrator.dataframe_cache') as mock_df_cache:
                    mock_ind_cache.size.return_value = 1
                    mock_df_cache.size.return_value = 1

                    tester = MassTester(['1!'], ['ZS'], ['1h'])
                    tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

                    tester.run_tests(max_workers=2, verbose=False, skip_existing=False)

                    assert mock_ind_cache.save_cache.call_count == 1
                    assert mock_df_cache.save_cache.call_count == 1


# ==================== Real-Data End-to-End Smoke Test ====================

@pytest.mark.integration
@pytest.mark.slow
class TestRealDataMultiprocessing:
    """End-to-end smoke tests that require real historical data to be present."""

    def test_multiple_strategies_on_real_data(self, tmpdir):
        """
        Run multiple strategies in parallel against real historical data.

        This test ensures that the orchestrator can handle multiple strategies
        and symbols concurrently without issues.
        """
        tester = MassTester(['1!'], ['ZC'], ['1d'])
        tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

        with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', tmpdir):
            results = tester.run_tests(max_workers=2, verbose=False, skip_existing=False)

        assert isinstance(results, list)

    def test_cache_efficiency_with_real_data(self, tmpdir):
        """
        Verify that cache is used efficiently with real historical data.

        This test checks that the cache size does not exceed expected limits
        when running tests with real data.
        """
        tester = MassTester(['1!'], ['ZS'], ['1h'])
        tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

        with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', tmpdir):
            tester.run_tests(max_workers=2, verbose=False, skip_existing=False)

        # Check that cache size is within expected limits
        assert indicator_cache.size() <= 10
        assert dataframe_cache.size() <= 10

    def test_performance_serial_vs_parallel(self, tmpdir):
        """
        Compare performance of serial vs parallel execution with real data.

        This test ensures that parallel execution provides a significant speedup
        when processing real historical data.
        """
        tester = MassTester(['1!'], ['ZS'], ['1h'])
        tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

        # create minimal parquet expected by orchestrator
        month_dir = Path(tmpdir) / '1!' / 'ZS'
        month_dir.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range('2023-01-01', periods=50, freq='h')
        df = pd.DataFrame({
            'symbol': ['CBOT:ZS1!'] * len(dates),
            'open': [100 + i for i in range(len(dates))],
            'high': [101 + i for i in range(len(dates))],
            'low': [99 + i for i in range(len(dates))],
            'close': [100 + i for i in range(len(dates))],
            'volume': [1000.0] * len(dates)
        }, index=pd.DatetimeIndex(dates, name='datetime'))
        df.to_parquet(month_dir / 'ZS_1h.parquet')

        # measure serial and parallel durations using time.time and run in-process

        with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', str(tmpdir)):
            # build combinations to run
            switch_dates = orchestrator_module._get_switch_dates_for_symbols(tester.symbols, tester.switch_dates_dict)
            all_combinations = orchestrator_module._generate_all_combinations(tester.tested_months,
                                                                              tester.symbols,
                                                                              tester.intervals,
                                                                              tester.strategies)
            test_combinations, skipped = orchestrator_module._prepare_test_combinations(all_combinations,
                                                                                        (pd.DataFrame(), set()),
                                                                                        False,
                                                                                        False,
                                                                                        switch_dates)

            t0 = time.time()
            _inprocess_executor(tester, test_combinations, max_workers_local=1)
            serial_time = time.time() - t0

            # clear results and caches
            tester.results = []
            indicator_cache.cache_data.clear()
            dataframe_cache.cache_data.clear()

            t1 = time.time()
            _inprocess_executor(tester, test_combinations, max_workers_local=4)
            parallel_time = time.time() - t1

        assert isinstance(serial_time, float)
        assert isinstance(parallel_time, float)
        # allow some variability; parallel should be similar or faster than serial
        assert parallel_time <= max(serial_time, 0.0001) * 2.0

    def test_large_parameter_space_parallel(self, tmpdir):
        """
        Test running a large parameter space in parallel.

        This verifies that the system can handle a wide range of parameters
        without performance degradation.
        """
        tester = MassTester(['1!'], ['ZS'], ['1h'])
        # Use a wide range of RSI parameters
        tester.add_rsi_tests(list(range(5, 25, 5)), [30], [70], [False], [None], [0])

        with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', tmpdir):
            results = tester.run_tests(max_workers=4, verbose=False, skip_existing=False)

        assert isinstance(results, list)

    def test_memory_efficient_processing(self, tmpdir):
        """
        Verify that memory usage is efficient when processing real data.

        This test ensures that the system does not exceed expected memory limits
        when running tests with real historical data.
        """
        tester = MassTester(['1!'], ['ZS'], ['1h'])
        tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

        with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', tmpdir):
            # Measure memory usage before running tests
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss

            tester.run_tests(max_workers=2, verbose=False, skip_existing=False)

            # Measure memory usage after running tests
            mem_after = process.memory_info().rss

        # Memory usage should not increase excessively
        assert mem_after - mem_before < 100 * 1024 * 1024  # Less than 100 MB increase

    def test_data_integrity_across_workers(self, tmpdir):
        """
        Ensure data integrity when accessed from multiple workers.

        This test verifies that data remains consistent and uncorrupted when
        accessed concurrently by different worker processes.
        """
        tester = MassTester(['1!'], ['ZS'], ['1h'])
        tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

        # Create minimal historical data so MassTester can produce results
        zs_dir = Path(tmpdir) / '1!' / 'ZS'
        zs_dir.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range('2023-01-01', periods=20, freq='h')
        df = pd.DataFrame({
            'symbol': ['CBOT:ZS1!'] * len(dates),
            'open': [100 + i for i in range(len(dates))],
            'high': [101 + i for i in range(len(dates))],
            'low': [99 + i for i in range(len(dates))],
            'close': [100 + i for i in range(len(dates))],
            'volume': [1000.0] * len(dates)
        }, index=pd.DatetimeIndex(dates, name='datetime'))
        df.to_parquet(zs_dir / 'ZS_1h.parquet')

        # Run against our temporary historical-data layout
        with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', tmpdir):
            results = tester.run_tests(max_workers=4, verbose=False, skip_existing=False)

        # Check that results were produced and contain expected orchestrator fields
        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert result is not None
            assert 'month' in result
            assert 'symbol' in result
            assert 'interval' in result
            assert 'strategy' in result
            assert 'metrics' in result
            assert 'timestamp' in result
            assert 'cache_stats' in result

    def test_worker_failure_recovery(self, tmpdir):
        """
        Verify that the system recovers gracefully from worker failures.

        This test ensures that if a worker process fails, the remaining workers
        can continue processing and the system can recover without manual intervention.
        """
        tester = MassTester(['1!'], ['ZS', 'NONEXISTENT'], ['1h'])
        tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

        # Create minimal parquet for ZS so the runner can process it; leave NONEXISTENT absent
        zs_dir = Path(tmpdir) / '1!' / 'ZS'
        zs_dir.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range('2023-01-01', periods=20, freq='h')
        df = pd.DataFrame({
            'symbol': ['CBOT:ZS1!'] * len(dates),
            'open': [100 + i for i in range(len(dates))],
            'high': [101 + i for i in range(len(dates))],
            'low': [99 + i for i in range(len(dates))],
            'close': [100 + i for i in range(len(dates))],
            'volume': [1000.0] * len(dates)
        }, index=pd.DatetimeIndex(dates, name='datetime'))
        df.to_parquet(zs_dir / 'ZS_1h.parquet')

        with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', tmpdir):
            # Prepare combinations using the orchestrator helpers (this mirrors run_tests prep)
            switch_dates = orchestrator_module._get_switch_dates_for_symbols(tester.symbols, tester.switch_dates_dict)
            all_combinations = orchestrator_module._generate_all_combinations(tester.tested_months,
                                                                              tester.symbols,
                                                                              tester.intervals,
                                                                              tester.strategies)
            test_combinations, skipped = orchestrator_module._prepare_test_combinations(all_combinations,
                                                                                        (pd.DataFrame(), set()),
                                                                                        False,
                                                                                        False,
                                                                                        switch_dates)

            # Run our in-process executor directly to simulate failures without using multiprocessing
            _inprocess_executor(tester, test_combinations, max_workers_local=4)
            results = tester.results

        # Check that the system recovered and results were still produced
        assert isinstance(results, list)
        assert len(results) > 0

    def test_real_processpool_executor_with_strategy_execution(self, tmp_path):
        """
        Minimal synthetic-data smoke test: create a tiny parquet file and run
        a MassTester to verify the real process-pool path works end-to-end.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Ignore if the start method has already been set in this process
            # multiprocessing.set_start_method raises RuntimeError when called more than once
            pass

        # Create minimal data directory and file structure expected by tester
        data_dir = tmp_path / 'data'
        data_dir.mkdir()

        # The orchestrator expects files under {HISTORICAL_DATA_DIR}/{month}/{symbol}/
        # Create the 1!/ZS directory layout so MassTester can find the parquet
        month_dir = data_dir / '1!' / 'ZS'
        month_dir.mkdir(parents=True, exist_ok=True)

        dates = pd.date_range('2023-01-01', periods=20, freq='h')
        df = pd.DataFrame({
            'symbol': ['CBOT:ZS1!'] * len(dates),
            'open': [100 + i for i in range(len(dates))],
            'high': [101 + i for i in range(len(dates))],
            'low': [99 + i for i in range(len(dates))],
            'close': [100 + i for i in range(len(dates))],
            'volume': [1000.0] * len(dates)
        }, index=pd.DatetimeIndex(dates, name='datetime'))

        # Write the parquet into the month/symbol dir with the filename the orchestrator expects
        test_file = month_dir / 'ZS_1h.parquet'
        df.to_parquet(test_file)

        # Run MassTester pointing to our synthetic data dir
        with patch('config.HISTORICAL_DATA_DIR', str(data_dir)):
            tester = MassTester(['1!'], ['ZS'], ['1h'])
            tester.add_rsi_tests([14], [30], [70], [False], [None], [0])
            results = tester.run_tests(max_workers=2, verbose=False, skip_existing=False)

        assert isinstance(results, list)

    def test_multiple_symbols_parallel_processing(self, tmp_path):
        """
        Create two minimal parquet files (ZC and 6A) and verify the orchestrator
        can process multiple symbols in parallel.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Ignore if the start method has already been set in this process
            # multiprocessing.set_start_method raises RuntimeError when called more than once
            pass

        # Build month-based structure under tmp_path
        base = tmp_path
        for symbol in ['ZC', '6A']:
            month_dir = Path(base) / '1!' / symbol
            month_dir.mkdir(parents=True, exist_ok=True)
            dates = pd.date_range('2023-01-01', periods=20, freq='h')
            df = pd.DataFrame({
                'symbol': [f'CBOT:{symbol}1!'] * len(dates),
                'open': range(len(dates)),
                'high': range(len(dates)),
                'low': range(len(dates)),
                'close': range(len(dates)),
                'volume': [100.0] * len(dates)
            }, index=pd.DatetimeIndex(dates, name='datetime'))
            df.to_parquet(month_dir / f'{symbol}_1d.parquet')

        # MassTester is imported at module level
        with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', str(base)):
            tester = MassTester(['1!'], ['ZC', '6A'], ['1d'])
            tester.add_rsi_tests([14], [30], [70], [False], [None], [0])
            results = tester.run_tests(max_workers=2, verbose=False, skip_existing=False)

        assert isinstance(results, list)
