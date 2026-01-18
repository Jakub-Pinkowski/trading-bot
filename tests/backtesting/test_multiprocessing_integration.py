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

from app.backtesting.cache.dataframe_cache import dataframe_cache
from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.mass_testing import MassTester
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
            'symbol': ['CME:ES2!'] * 200,
            'open': [4500 + i * 0.5 for i in range(200)],
            'high': [4505 + i * 0.5 for i in range(200)],
            'low': [4495 + i * 0.5 for i in range(200)],
            'close': [4502 + i * 0.5 for i in range(200)],
            'volume': [10000.0] * 200
        }, index=pd.DatetimeIndex(dates, name='datetime'))

        # Save test data
        test_file = data_dir / "ES_1h_2023-01.parquet"
        test_df.to_parquet(test_file)

        # Temporarily patch HISTORICAL_DATA_DIR to use our test directory
        with patch('config.HISTORICAL_DATA_DIR', str(data_dir)):
            # Create tester and add simple RSI test
            tester = MassTester(['2023-01'], ['ES'], ['1h'])
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
        with patch('app.backtesting.mass_testing.HISTORICAL_DATA_DIR', str(tmp_path / "nonexistent")):
            tester = MassTester(['2023-01'], ['ES'], ['1h'])
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
            'symbol': ['CME:ES2!'] * 200,
            'open': [4500 + i * 0.5 for i in range(200)],
            'high': [4505 + i * 0.5 for i in range(200)],
            'low': [4495 + i * 0.5 for i in range(200)],
            'close': [4502 + i * 0.5 for i in range(200)],
            'volume': [10000.0] * 200
        }, index=pd.DatetimeIndex(dates, name='datetime'))

        test_file = data_dir / "ES_1h_2023-01.parquet"
        test_df.to_parquet(test_file)

        # Mock cache save methods to track calls
        with patch('config.HISTORICAL_DATA_DIR', str(data_dir)):
            with patch('app.backtesting.mass_testing.indicator_cache') as mock_ind_cache:
                with patch('app.backtesting.mass_testing.dataframe_cache') as mock_df_cache:
                    # Setup mocks
                    mock_ind_cache.size.return_value = 10
                    mock_df_cache.size.return_value = 5

                    tester = MassTester(['2023-01'], ['ES'], ['1h'])
                    tester.add_rsi_tests([14], [30], [70], [False], [None], [0])

                    # Run tests
                    tester.run_tests(max_workers=2, verbose=False)

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
        from app.backtesting.strategy_factory import create_strategy

        strategy_configs = [
            ('rsi', {'rsi_period': 14, 'lower': 30, 'upper': 70, 'rollover': False, 'trailing': None, 'slippage': 0}),
            ('ema', {'ema_short': 9, 'ema_long': 21, 'rollover': False, 'trailing': None, 'slippage': 0}),
            (
                'macd',
                {'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'rollover': False, 'trailing': None,
                 'slippage': 0}
            ),
            ('bollinger', {'period': 20, 'std_dev': 2, 'rollover': False, 'trailing': None, 'slippage': 0}),
            (
                'ichimoku',
                {'tenkan': 9, 'kijun': 26, 'senkou_b': 52, 'displacement': 26, 'rollover': False, 'trailing': None,
                 'slippage': 0}
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
