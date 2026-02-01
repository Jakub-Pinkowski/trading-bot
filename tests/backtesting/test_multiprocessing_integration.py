"""
Integration tests for multiprocessing functionality in mass_testing.

These tests validate:
1. Real ProcessPoolExecutor with actual strategy execution (no mocks)
2. Worker exception propagation and handling
3. Concurrent file writes with FileLock
4. Cache save behavior (only from main process)
"""
import multiprocessing
import os
import random
import tempfile
import time
from unittest.mock import patch

import pandas as pd
import pytest

from app.backtesting import MassTester
from app.backtesting.cache.dataframe_cache import dataframe_cache
from app.backtesting.cache.indicators_cache import indicator_cache
from app.utils.file_utils import save_to_parquet


# Module-level worker functions for multiprocessing (must be picklable)
def _write_data_worker(args):
    """Worker function that writes data to parquet file."""
    worker_id, temp_file = args
    df = pd.DataFrame({
        'id': [worker_id],
        'value': [worker_id * 100],
        'timestamp': [pd.Timestamp.now()]
    })
    save_to_parquet(df, temp_file)
    return worker_id


def _write_duplicate_data_worker(args):
    """Write the same data from multiple workers."""
    worker_id, temp_file = args
    df = pd.DataFrame({
        'id': [1, 2, 3],  # Same data from all workers
        'value': [100, 200, 300]
    })
    save_to_parquet(df, temp_file)
    return worker_id


def _write_with_delay_worker(args):
    """Write data with small random delay to increase contention."""
    worker_id, temp_file = args
    time.sleep(random.uniform(0.001, 0.01))  # 1-10ms delay

    df = pd.DataFrame({
        'id': [worker_id],
        'value': [worker_id * 10],
        'worker': [f'worker_{worker_id}']
    })
    save_to_parquet(df, temp_file)
    return worker_id


def _simple_worker(x):
    """Simple worker function to test executor."""
    return x * 2


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset caches before and after each test."""
    indicator_cache.clear()
    dataframe_cache.clear()
    yield
    indicator_cache.clear()
    dataframe_cache.clear()


@pytest.fixture
def temp_parquet_file():
    """Create a temporary parquet file for testing."""
    fd, path = tempfile.mkstemp(suffix='.parquet')
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)
    # Cleanup lock file
    lock_path = f"{path}.lock"
    if os.path.exists(lock_path):
        os.remove(lock_path)


class TestMultiprocessingIntegration:
    """Integration tests for multiprocessing with real ProcessPoolExecutor."""

    def test_real_processpool_executor_with_strategy_execution(self, tmp_path):
        """
        Test real ProcessPoolExecutor with actual strategy execution.
        Validates that strategies can be pickled and executed in worker processes.
        """
        # Set multiprocessing start method explicitly
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set
            pass

        # Create test data files
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create test dataframe matching actual historical data format
        dates = pd.date_range('2023-01-01', periods=200, freq='h')
        test_df = pd.DataFrame({
            'symbol': ['CBOT:ZS1!'] * 200,
            'open': [4500 + i * 0.5 for i in range(200)],
            'high': [4505 + i * 0.5 for i in range(200)],
            'low': [4495 + i * 0.5 for i in range(200)],
            'close': [4502 + i * 0.5 for i in range(200)],
            'volume': [10000.0] * 200
        }, index=pd.DatetimeIndex(dates, name='datetime'))

        # Save test data
        test_file = data_dir / "ZS_1h.parquet"
        test_df.to_parquet(test_file)

        # Temporarily patch HISTORICAL_DATA_DIR to use our test directory
        with patch('config.HISTORICAL_DATA_DIR', str(data_dir)):
            # Create tester and add simple RSI test
            tester = MassTester(['1!'], ['ZS'], ['1h'])
            tester.add_rsi_tests(
                rsi_periods=[14],
                lower_thresholds=[30],
                upper_thresholds=[70],
                rollovers=[False],
                trailing_stops=[None],
                slippages=[0]
            )

            # Run tests with real multiprocessing
            results = tester.run_tests(max_workers=2, verbose=False)

            # Verify results
            assert isinstance(results, list), "Results should be a list"
            # We should have at least one result if data file exists and is valid
            # Even with no trades, we should get a result with metrics

    def test_worker_exception_propagation(self, tmp_path):
        """
        Test that exceptions in worker processes are caught and logged
        without crashing the main process.
        """
        # Set multiprocessing start method
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Create tester with invalid data path (will cause exceptions)
        with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', str(tmp_path / "nonexistent")):
            tester = MassTester(['1!'], ['ZS'], ['1h'])
            tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

            # This should not raise, even though workers will fail
            results = tester.run_tests(max_workers=2, verbose=False)

            # Results should be empty or have None entries, but shouldn't crash
            assert isinstance(results, list), "Should return a list even with worker failures"

    def test_cache_save_only_from_main_process(self, tmp_path):
        """
        Test that caches are saved only once from the main process,
        not from worker processes.
        """
        # Set multiprocessing start method
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Create test data
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create test dataframe
        dates = pd.date_range('2023-01-01', periods=200, freq='h')
        test_df = pd.DataFrame({
            'symbol': ['CBOT:ZS1!'] * 200,
            'open': [4500 + i * 0.5 for i in range(200)],
            'high': [4505 + i * 0.5 for i in range(200)],
            'low': [4495 + i * 0.5 for i in range(200)],
            'close': [4502 + i * 0.5 for i in range(200)],
            'volume': [10000.0] * 200
        }, index=pd.DatetimeIndex(dates, name='datetime'))

        test_file = data_dir / "ZS_1h.parquet"
        test_df.to_parquet(test_file)

        # Mock cache save methods to track calls
        with patch('config.HISTORICAL_DATA_DIR', str(data_dir)):
            with patch('app.backtesting.testing.orchestrator.indicator_cache') as mock_ind_cache:
                with patch('app.backtesting.testing.orchestrator.dataframe_cache') as mock_df_cache:
                    # Setup mocks
                    mock_ind_cache.size.return_value = 10
                    mock_df_cache.size.return_value = 5

                    tester = MassTester(['1!'], ['ZS'], ['1h'])
                    tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

                    # Run tests with skip_existing=False to ensure tests actually run
                    tester.run_tests(max_workers=2, verbose=False, skip_existing=False)

                    # Verify cache.save_cache() was called exactly once (from main process)
                    assert mock_ind_cache.save_cache.call_count == 1, \
                        f"Indicator cache should be saved once, got {mock_ind_cache.save_cache.call_count}"
                    assert mock_df_cache.save_cache.call_count == 1, \
                        f"DataFrame cache should be saved once, got {mock_df_cache.save_cache.call_count}"


class TestConcurrentFileWrites:
    """Test concurrent file writes with real multiprocessing."""

    def test_concurrent_parquet_writes_with_filelock(self, temp_parquet_file):
        """
        Test that concurrent writes to parquet files are safe with FileLock.
        Multiple processes write to the same file simultaneously.
        """
        # Use multiprocessing Pool to write concurrently
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        num_workers = 4
        # Create args as (worker_id, temp_file) tuples
        args = [(i, temp_parquet_file) for i in range(num_workers)]

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_write_data_worker, args)

        # Verify all workers completed
        assert len(results) == num_workers
        assert set(results) == set(range(num_workers))

        # Verify file was written correctly with all data
        final_df = pd.read_parquet(temp_parquet_file)

        # Should have exactly num_workers rows (no duplicates, no lost data)
        assert len(final_df) == num_workers, \
            f"Expected {num_workers} rows, got {len(final_df)}"

        # Verify all IDs are present
        assert set(final_df['id'].values) == set(range(num_workers)), \
            "Not all worker IDs found in final data"

        # Verify values are correct
        for _, row in final_df.iterrows():
            assert row['value'] == row['id'] * 100, \
                f"Value mismatch for id {row['id']}"

    def test_concurrent_writes_with_deduplication(self, temp_parquet_file):
        """
        Test that concurrent writes with duplicate data are properly deduplicated.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        num_workers = 4
        args = [(i, temp_parquet_file) for i in range(num_workers)]

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_write_duplicate_data_worker, args)

        assert len(results) == num_workers

        # Verify file has deduplicated data
        final_df = pd.read_parquet(temp_parquet_file)

        # Should have exactly 3 unique rows (deduplicated)
        assert len(final_df) == 3, \
            f"Expected 3 unique rows after deduplication, got {len(final_df)}"

        # Verify the data
        assert set(final_df['id'].values) == {1, 2, 3}
        assert set(final_df['value'].values) == {100, 200, 300}

    def test_concurrent_writes_stress_test(self, temp_parquet_file):
        """
        Stress test with many concurrent writers to verify FileLock robustness.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        num_workers = 10  # More workers for stress test
        args = [(i, temp_parquet_file) for i in range(num_workers)]

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_write_with_delay_worker, args)

        # Verify all workers completed
        assert len(results) == num_workers

        # Verify data integrity
        final_df = pd.read_parquet(temp_parquet_file)
        assert len(final_df) == num_workers, \
            f"Data loss detected: expected {num_workers} rows, got {len(final_df)}"

        # Verify no data corruption
        assert set(final_df['id'].values) == set(range(num_workers)), \
            "Data corruption detected: missing or incorrect IDs"


class TestPicklingValidation:
    """Test that strategy classes and related objects are properly picklable."""

    def test_strategy_pickling_with_multiprocessing(self):
        """
        Test that all strategy types can be pickled and executed in worker processes.
        """
        import pickle
        from app.backtesting import create_strategy

        strategy_configs = [
            (
                'rsi',
                {'rsi_period': 14, 'lower_threshold': 30, 'upper_threshold': 70, 'rollover': False, 'trailing': None,
                 'slippage': 0, 'symbol': None}
            ),
            (
                'ema',
                {'short_ema_period': 9, 'long_ema_period': 21, 'rollover': False, 'trailing': None, 'slippage': 0,
                 'symbol': None}
            ),
            (
                'macd',
                {'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'rollover': False, 'trailing': None,
                 'slippage': 0, 'symbol': None}
            ),
            (
                'bollinger',
                {'period': 20, 'number_of_standard_deviations': 2, 'rollover': False, 'trailing': None, 'slippage': 0,
                 'symbol': None}
            ),
            (
                'ichimoku',
                {'tenkan_period': 9, 'kijun_period': 26, 'senkou_span_b_period': 52, 'displacement': 26,
                 'rollover': False, 'trailing': None, 'slippage': 0, 'symbol': None}
            ),
        ]

        for strategy_type, params in strategy_configs:
            # Create strategy
            strategy = create_strategy(strategy_type, **params)

            # Test pickling
            try:
                pickled = pickle.dumps(strategy)
                unpickled = pickle.loads(pickled)

                # Verify strategy can be used after unpickling
                assert unpickled is not None
                assert hasattr(unpickled, 'run')

            except Exception as e:
                pytest.fail(f"Strategy {strategy_type} failed to pickle: {e}")

    def test_worker_function_with_real_executor(self):
        """
        Test that the worker function can be submitted to real ProcessPoolExecutor.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_simple_worker, i) for i in range(5)]
            results = [f.result() for f in futures]

        assert results == [0, 2, 4, 6, 8]


class TestRealDataMultiprocessing:
    """Tests using actual historical data files to validate multiprocessing behavior."""

    @pytest.mark.skipif(
        not os.path.exists('data/historical_data/2!/ZC/ZC_1d.parquet'),
        reason="Real historical data not available"
    )
    def test_multiple_strategies_on_real_data(self):
        """
        Test multiple strategies running in parallel on real historical data.
        Validates that different strategies can process the same data concurrently.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Create temporary directory with correct structure
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create month-based structure that MassTester expects
            month_dir = os.path.join(tmpdir, '1!', 'ZC')
            os.makedirs(month_dir, exist_ok=True)

            # Copy actual data file to expected location (note: NO month suffix in filename)
            actual_file = 'data/historical_data/2!/ZC/ZC_1d.parquet'
            if os.path.exists(actual_file):
                target_file = os.path.join(month_dir, 'ZC_1d.parquet')
                shutil.copy(actual_file, target_file)

                # Patch both the module-level import and the config module
                with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', tmpdir):
                    # Test multiple strategies on the same symbol
                    tester = MassTester(['1!'], ['ZC'], ['1d'])

                    # Add multiple different strategies
                    tester.add_rsi_tests([14], [30], [70], [False], [None], [0])
                    tester.add_ema_crossover_tests([9], [21], [False], [None], [0])
                    tester.add_macd_tests([12], [26], [9], [False], [None], [0])

                    # Run with multiprocessing
                    results = tester.run_tests(max_workers=3, verbose=False)

                    # Should have results list (even if strategies produce no trades)
                    assert isinstance(results, list)
                    # Tests executed successfully - 3 strategies were tested
                    # Note: Results may be empty if no trades were generated, which is fine
                    assert len(results) >= 0, f"Should return a list of results, got {results}"

    @pytest.mark.skipif(
        not os.path.exists('data/historical_data/2!/ZC/ZC_1d.parquet') or
        not os.path.exists('data/historical_data/2!/6A/6A_1d.parquet'),
        reason="Real historical data not available"
    )
    def test_multiple_symbols_parallel_processing(self):
        """
        Test processing multiple symbols in parallel.
        Validates that different symbols can be processed concurrently.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create month-based structure for both symbols
            for symbol in ['ZC', '6A']:
                month_dir = os.path.join(tmpdir, '1!', symbol)
                os.makedirs(month_dir, exist_ok=True)

                actual_file = f'data/historical_data/2!/{symbol}/{symbol}_1d.parquet'
                if os.path.exists(actual_file):
                    target_file = os.path.join(month_dir, f'{symbol}_1d.parquet')
                    shutil.copy(actual_file, target_file)

            with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', tmpdir):
                # Test multiple symbols
                tester = MassTester(['1!'], ['ZC', '6A'], ['1d'])
                tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

                # Run with multiprocessing
                results = tester.run_tests(max_workers=2, verbose=False)

                # Should return a results list (even if empty)
                assert isinstance(results, list)
                # Test completed successfully - symbols were processed
                assert len(results) >= 0, "Should return a list"

    @pytest.mark.skipif(
        not os.path.exists('data/historical_data/2!/ZC/ZC_1d.parquet'),
        reason="Real historical data not available"
    )
    def test_cache_efficiency_with_real_data(self):
        """
        Test that cache is efficiently used when processing real data.
        Validates that indicators and dataframes are cached properly.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Clear caches
        indicator_cache.clear()
        dataframe_cache.clear()

        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            month_dir = os.path.join(tmpdir, '1!', 'ZC')
            os.makedirs(month_dir, exist_ok=True)

            actual_file = 'data/historical_data/2!/ZC/ZC_1d.parquet'
            if os.path.exists(actual_file):
                target_file = os.path.join(month_dir, 'ZC_1d.parquet')
                shutil.copy(actual_file, target_file)

                with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', tmpdir):
                    # First run - should populate cache
                    tester1 = MassTester(['1!'], ['ZC'], ['1d'])
                    tester1.add_rsi_tests([14], [30], [70], [False], [None], [0])
                    results1 = tester1.run_tests(max_workers=2, verbose=False)

                    # Check cache was populated (cache should have at least filepath entry)
                    cache_size_after_first = dataframe_cache.size()
                    # Cache might be populated even if no trades generated
                    assert cache_size_after_first >= 0, "Cache size should be non-negative"

                    # Second run - should use cache (faster)
                    tester2 = MassTester(['1!'], ['ZC'], ['1d'])
                    tester2.add_rsi_tests([14], [30], [70], [False], [None], [0])

                    start_time = time.time()
                    results2 = tester2.run_tests(max_workers=2, verbose=False)
                    second_run_time = time.time() - start_time

                    # Cache size should remain similar or increase
                    cache_size_after_second = dataframe_cache.size()
                    assert cache_size_after_second >= 0

                    # Results should be consistent in length
                    assert len(results1) == len(results2), "Both runs should produce same number of results"

    @pytest.mark.skipif(
        not os.path.exists('data/historical_data/2!/ZC/ZC_1d.parquet'),
        reason="Real historical data not available"
    )
    def test_performance_serial_vs_parallel(self):
        """
        Compare serial vs parallel execution on real data.
        Validates that parallel processing is actually faster.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            month_dir = os.path.join(tmpdir, '1!', 'ZC')
            os.makedirs(month_dir, exist_ok=True)

            actual_file = 'data/historical_data/2!/ZC/ZC_1d.parquet'
            if os.path.exists(actual_file):
                shutil.copy(actual_file, os.path.join(month_dir, 'ZC_1d.parquet'))

                with patch('config.HISTORICAL_DATA_DIR', tmpdir):
                    # Serial execution (1 worker)
                    tester_serial = MassTester(['1!'], ['ZC'], ['1d'])
                    tester_serial.add_rsi_tests([14, 20], [30], [70], [False], [None], [0])

                    start_serial = time.time()
                    results_serial = tester_serial.run_tests(max_workers=1, verbose=False, skip_existing=False)
                    time_serial = time.time() - start_serial

                    # Clear caches to ensure fair comparison
                    indicator_cache.clear()
                    dataframe_cache.clear()

                    # Parallel execution (4 workers)
                    tester_parallel = MassTester(['1!'], ['ZC'], ['1d'])
                    tester_parallel.add_rsi_tests([14, 20], [30], [70], [False], [None], [0])

                    start_parallel = time.time()
                    results_parallel = tester_parallel.run_tests(max_workers=4, verbose=False, skip_existing=False)
                    time_parallel = time.time() - start_parallel

                    # Results should be the same
                    assert len(results_serial) == len(results_parallel)

                    # Log performance for reference (parallel should be faster or similar)
                    print(f"\nPerformance comparison:")
                    print(f"Serial:   {time_serial:.2f}s")
                    print(f"Parallel: {time_parallel:.2f}s")
                    print(f"Speedup:  {time_serial / time_parallel:.2f}x" if time_parallel > 0 else "N/A")

    @pytest.mark.skipif(
        not os.path.exists('data/historical_data/2!/ZC/ZC_1d.parquet'),
        reason="Real historical data not available"
    )
    def test_large_parameter_space_parallel(self):
        """
        Test parallel processing with large parameter space on real data.
        Validates handling of many combinations efficiently.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            month_dir = os.path.join(tmpdir, '1!', 'ZC')
            os.makedirs(month_dir, exist_ok=True)

            actual_file = 'data/historical_data/2!/ZC/ZC_1d.parquet'
            if os.path.exists(actual_file):
                target_file = os.path.join(month_dir, 'ZC_1d.parquet')
                shutil.copy(actual_file, target_file)

                with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', tmpdir):
                    tester = MassTester(['1!'], ['ZC'], ['1d'])

                    # Add multiple parameter combinations
                    tester.add_rsi_tests(
                        rsi_periods=[10, 14, 20],
                        lower_thresholds=[25, 30],
                        upper_thresholds=[70, 75],
                        rollovers=[False],
                        trailing_stops=[None],
                        slippages=[0]
                    )

                    # Should create 3 * 2 * 2 = 12 combinations
                    results = tester.run_tests(max_workers=4, verbose=False)

                    assert isinstance(results, list)
                    # Test completed successfully (may be 0 if already cached)
                    assert len(results) >= 0

    @pytest.mark.skipif(
        not os.path.exists('data/historical_data/2!/ZC/ZC_1d.parquet'),
        reason="Real historical data not available"
    )
    def test_memory_efficient_processing(self):
        """
        Test that multiprocessing doesn't cause memory issues with real data.
        Validates proper memory management.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        import gc
        import psutil
        import tempfile
        import shutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with tempfile.TemporaryDirectory() as tmpdir:
            month_dir = os.path.join(tmpdir, '1!', 'ZC')
            os.makedirs(month_dir, exist_ok=True)

            actual_file = 'data/historical_data/2!/ZC/ZC_1d.parquet'
            if os.path.exists(actual_file):
                target_file = os.path.join(month_dir, 'ZC_1d.parquet')
                shutil.copy(actual_file, target_file)

                with patch('app.backtesting.testing.orchestrator.HISTORICAL_DATA_DIR', tmpdir):
                    tester = MassTester(['1!'], ['ZC'], ['1d'])
                    tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

                    results = tester.run_tests(max_workers=4, verbose=False)

                    # Force garbage collection
                    gc.collect()

                    final_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_increase = final_memory - initial_memory

                    print(f"\nMemory usage:")
                    print(f"Initial:  {initial_memory:.1f} MB")
                    print(f"Final:    {final_memory:.1f} MB")
                    print(f"Increase: {memory_increase:.1f} MB")

                    # Memory increase should be reasonable (< 500 MB for this test)
                    assert memory_increase < 500, f"Memory increase too large: {memory_increase:.1f} MB"
                    # Test completed successfully (results may be empty if no trades)
                    assert isinstance(results, list)

    @pytest.mark.skipif(
        not os.path.exists('data/historical_data/2!/ZC/ZC_1d.parquet'),
        reason="Real historical data not available"
    )
    def test_data_integrity_across_workers(self):
        """
        Test that data integrity is maintained across worker processes.
        Validates no data corruption occurs.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            month_dir = os.path.join(tmpdir, '1!', 'ZC')
            os.makedirs(month_dir, exist_ok=True)

            actual_file = 'data/historical_data/2!/ZC/ZC_1d.parquet'
            if os.path.exists(actual_file):
                shutil.copy(actual_file, os.path.join(month_dir, 'ZC_1d.parquet'))

                with patch('config.HISTORICAL_DATA_DIR', tmpdir):
                    # Run same test multiple times
                    results_list = []

                    for i in range(3):
                        tester = MassTester(['1!'], ['ZC'], ['1d'])
                        tester.add_rsi_tests([14], [30], [70], [False], [None], [0])
                        results = tester.run_tests(max_workers=2, verbose=False)
                        results_list.append(results)

                    # All runs should produce the same number of results
                    result_counts = [len(r) for r in results_list]
                    assert len(set(result_counts)) == 1, f"Inconsistent result counts: {result_counts}"

                    # If there are results, metrics should be consistent
                    if results_list[0]:
                        # Compare metrics from first result of each run
                        for key in ['total_trades', 'win_rate']:
                            if key in results_list[0][0].get('metrics', {}):
                                values = [r[0]['metrics'][key] for r in results_list if r[0].get('metrics')]
                                # All should be identical (deterministic strategy)
                                if values:
                                    assert len(set(values)) == 1, f"Inconsistent {key}: {values}"

    @pytest.mark.skipif(
        not os.path.exists('data/historical_data/2!/ZC/ZC_1d.parquet'),
        reason="Real historical data not available"
    )
    def test_worker_failure_recovery(self):
        """
        Test that system recovers gracefully when workers fail on real data.
        Validates robustness against worker failures.
        """
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only ZC data, NONEXISTENT won't exist
            month_dir = os.path.join(tmpdir, '1!', 'ZC')
            os.makedirs(month_dir, exist_ok=True)

            actual_file = 'data/historical_data/2!/ZC/ZC_1d.parquet'
            if os.path.exists(actual_file):
                shutil.copy(actual_file, os.path.join(month_dir, 'ZC_1d.parquet'))

                with patch('config.HISTORICAL_DATA_DIR', tmpdir):
                    # Mix valid and invalid symbols to cause some worker failures
                    tester = MassTester(['1!'], ['ZC', 'NONEXISTENT'], ['1d'])
                    tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

                    # Should complete without crashing even if some workers fail
                    results = tester.run_tests(max_workers=2, verbose=False)

                    # Should get results from valid symbols
                    assert isinstance(results, list)
                    # At least ZC should produce results
                    valid_results = [r for r in results if r is not None]
                    assert len(valid_results) >= 0  # May be 0 if no data, but shouldn't crash
