import concurrent.futures
import gc
import itertools
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from app.backtesting.cache.dataframe_cache import dataframe_cache, get_cached_dataframe
from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.metrics.per_trade_metrics import calculate_trade_metrics
from app.backtesting.metrics.summary_metrics import SummaryMetrics
from app.backtesting.strategy_factory import create_strategy, get_strategy_name
from app.utils.file_utils import save_to_parquet
from app.utils.logger import get_logger
from config import HISTORICAL_DATA_DIR, SWITCH_DATES_FILE_PATH, BACKTESTING_DIR

logger = get_logger('backtesting/mass_testing')

# DataFrame Validation Constants
MIN_ROWS_FOR_BACKTEST = 150  # Minimum rows required for reliable backtesting (100 warm-up + 50 for indicators)


def _load_existing_results():
    """Load existing results from the parquet file."""
    parquet_filename = f'{BACKTESTING_DIR}/mass_test_results_all.parquet'
    if os.path.exists(parquet_filename):
        try:
            df = pd.read_parquet(parquet_filename)
            # Create tuples directly from DataFrame columns - O(1) operation with vectorization
            existing_combinations = set(zip(
                df['month'].values,
                df['symbol'].values,
                df['interval'].values,
                df['strategy'].values
            ))
            return df, existing_combinations
        except Exception as error:
            logger.error(f'Failed to load existing results: {error}')
    return pd.DataFrame(), set()


def _test_already_exists(existing_data, month, symbol, interval, strategy):
    """Check if a test with the given parameters already exists in the results."""
    existing_results, existing_combinations = existing_data

    if existing_results.empty:
        return False

    # Check if the combination exists in the set (O(1) operation)
    return (month, symbol, interval, strategy) in existing_combinations


class MassTester:
    """A framework for mass-testing trading strategies with different parameter combinations."""

    def __init__(self, tested_months, symbols, intervals):
        """Initialize the mass tester with lists of months, symbols, and intervals to test."""
        self.strategies = []
        self.tested_months = tested_months
        self.symbols = symbols
        self.intervals = intervals

        # Load switch dates
        with open(SWITCH_DATES_FILE_PATH) as switch_dates_file:
            self.switch_dates_dict = yaml.safe_load(switch_dates_file)

        # Initialize results storage
        self.results = []

    # ==================== Public API - Strategy Configuration ====================

    def add_rsi_tests(self, rsi_periods, lower_thresholds, upper_thresholds, rollovers, trailing_stops, slippages=None):
        self._add_strategy_tests(
            strategy_type='rsi',
            param_grid={
                'rsi_period': rsi_periods,
                'lower': lower_thresholds,
                'upper': upper_thresholds,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage': slippages
            }
        )

    def add_ema_crossover_tests(self, ema_shorts, ema_longs, rollovers, trailing_stops, slippages=None):
        self._add_strategy_tests(
            strategy_type='ema',
            param_grid={
                'ema_short': ema_shorts,
                'ema_long': ema_longs,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage': slippages
            }
        )

    def add_macd_tests(self, fast_periods, slow_periods, signal_periods, rollovers, trailing_stops, slippages=None):
        self._add_strategy_tests(
            strategy_type='macd',
            param_grid={
                'fast_period': fast_periods,
                'slow_period': slow_periods,
                'signal_period': signal_periods,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage': slippages
            }
        )

    def add_bollinger_bands_tests(self, periods, num_stds, rollovers, trailing_stops, slippages=None):
        self._add_strategy_tests(
            strategy_type='bollinger',
            param_grid={
                'period': periods,
                'num_std': num_stds,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage': slippages
            }
        )

    def add_ichimoku_cloud_tests(
        self,
        tenkan_periods,
        kijun_periods,
        senkou_span_b_periods,
        displacements,
        rollovers,
        trailing_stops,
        slippages=None
    ):
        self._add_strategy_tests(
            strategy_type='ichimoku',
            param_grid={
                'tenkan_period': tenkan_periods,
                'kijun_period': kijun_periods,
                'senkou_span_b_period': senkou_span_b_periods,
                'displacement': displacements,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage': slippages
            }
        )

    # ==================== Public API - Execution ====================

    def run_tests(self, verbose=True, max_workers=None, skip_existing=True):
        """  Run all tests with the configured parameters in parallel. """
        start_time = time.time()  # Track the start time of the entire process

        # Reset cache statistics at the start of the run
        indicator_cache.reset_stats()
        dataframe_cache.reset_stats()

        if not hasattr(self, 'strategies') or not self.strategies:
            logger.error('No strategies added for testing. Use add_*_tests methods first.')
            raise ValueError('No strategies added for testing. Use add_*_tests methods first.')

        # Load existing results to check for already run tests
        existing_data = _load_existing_results() if skip_existing else (pd.DataFrame(), set())

        total_combinations = len(self.tested_months) * len(self.symbols) * len(self.intervals) * len(self.strategies)
        print(f'Found {total_combinations} potential test combinations...')

        # Clear previous results
        self.results = []

        # Preprocess switch dates for all symbols at once
        preprocessed_switch_dates = {}
        for symbol in self.symbols:
            # Check if the symbol has direct switch dates
            if symbol in self.switch_dates_dict:
                switch_dates = self.switch_dates_dict[symbol]
            # Check if the symbol is a mini/micro that maps to a main symbol
            elif '_symbol_mappings' in self.switch_dates_dict and symbol in self.switch_dates_dict['_symbol_mappings']:
                main_symbol = self.switch_dates_dict['_symbol_mappings'][symbol]
                switch_dates = self.switch_dates_dict.get(main_symbol, [])
            else:
                switch_dates = []

            preprocessed_switch_dates[symbol] = [pd.to_datetime(switch_date) for switch_date in switch_dates]

        # Cache filepath patterns for faster construction
        filepath_patterns = {}
        for tested_month in self.tested_months:
            for symbol in self.symbols:
                for interval in self.intervals:
                    filepath_patterns[(
                        tested_month, symbol, interval
                    )] = f'{HISTORICAL_DATA_DIR}/{tested_month}/{symbol}/{symbol}_{interval}.parquet'

        # Generate all combinations more efficiently
        all_combinations = [(tested_month, symbol, interval, strategy_name, strategy_instance)
                            for tested_month in self.tested_months
                            for symbol in self.symbols
                            for interval in self.intervals
                            for strategy_name, strategy_instance in self.strategies]

        # Prepare all test combinations
        test_combinations = []
        skipped_combinations = 0
        last_verbose_combo = None

        # Filter out already run tests
        for combo in all_combinations:
            tested_month, symbol, interval, strategy_name, strategy_instance = combo

            # Print verbose output only when a month/symbol/interval combination changes
            if verbose and (tested_month, symbol, interval) != last_verbose_combo:
                print(f'Preparing: Month={tested_month}, Symbol={symbol}, Interval={interval}')
                last_verbose_combo = (tested_month, symbol, interval)

            # Check if this test has already been run
            if skip_existing and _test_already_exists(existing_data,
                                                      tested_month,
                                                      symbol,
                                                      interval,
                                                      strategy_name):
                if verbose:
                    print(f'Skipping already run test: Month={tested_month}, Symbol={symbol}, Interval={interval}, Strategy={strategy_name}')
                skipped_combinations += 1
                continue

            # Add preprocessed switch dates and filepath to the test parameters
            test_combinations.append((
                tested_month,
                symbol,
                interval,
                strategy_name,
                strategy_instance,
                verbose,
                preprocessed_switch_dates[symbol],  # Pass preprocessed switch dates
                filepath_patterns[(tested_month, symbol, interval)]  # Pass cached filepath
            ))

        print(f'Skipped {skipped_combinations} already run test combinations.')
        print(f'Running {len(test_combinations)} new test combinations...')

        if not test_combinations:
            print('All tests have already been run. No new tests to execute.')

            # Calculate and print the total time even when no tests are run
            end_time = time.time()
            total_time = end_time - start_time
            print(f'Total execution time: {total_time:.2f} seconds')

            return self.results

        # Run tests in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {executor.submit(_run_single_test, test_params): test_params for test_params in
                              test_combinations}

            total_tests = len(test_combinations)
            completed_tests = 0
            failed_tests = 0
            batch_start_time = time.time()
            overall_start_time = time.time()

            for future in concurrent.futures.as_completed(future_to_test):
                completed_tests += 1
                if completed_tests % 100 == 0 or completed_tests == total_tests:
                    current_time = time.time()
                    batch_elapsed_time = current_time - batch_start_time
                    total_elapsed_time = current_time - overall_start_time
                    avg_time_per_100_tests = total_elapsed_time / completed_tests * 100

                    print(f'Progress: {completed_tests}/{total_tests} tests completed '
                          f'({(completed_tests / total_tests * 100):.1f}%) - '
                          f'Batch: {batch_elapsed_time:.2f}s - '
                          f'Total: {total_elapsed_time:.2f}s - '
                          f'Avg: {avg_time_per_100_tests:.2f}s/100tests')
                    batch_start_time = current_time

                    # Periodic cleanup
                    gc.collect()

                    # Save intermediate results and clear memory
                    if completed_tests % 1000 == 0 and self.results:
                        self._save_results()
                        self.results.clear()

                # Handle worker exceptions gracefully to prevent crashing the entire mass testing run
                try:
                    result = future.result()
                except Exception:
                    # Get test details for logging
                    test_params = future_to_test[future]
                    tested_month, symbol, interval, strategy_name = test_params[:4]

                    # Log the exception with a full traceback
                    logger.exception(
                        f'Worker exception during test execution: '
                        f'Month={tested_month}, Symbol={symbol}, Interval={interval}, Strategy={strategy_name}'
                    )
                    failed_tests += 1
                    continue

                if result:
                    self.results.append(result)

        # Save caches after all tests complete (only from main process)
        # Note: save_cache() already handles file locking internally, no need to wrap it
        logger.info('All tests completed, saving caches...')

        # Aggregate cache statistics from all test results
        total_ind_hits = 0
        total_ind_misses = 0
        total_df_hits = 0
        total_df_misses = 0

        for result in self.results:
            if 'cache_stats' in result:
                total_ind_hits += result['cache_stats']['ind_hits']
                total_ind_misses += result['cache_stats']['ind_misses']
                total_df_hits += result['cache_stats']['df_hits']
                total_df_misses += result['cache_stats']['df_misses']

        # Calculate aggregated statistics
        ind_total = total_ind_hits + total_ind_misses
        ind_hit_rate = (total_ind_hits / ind_total * 100) if ind_total > 0 else 0
        df_total = total_df_hits + total_df_misses
        df_hit_rate = (total_df_hits / df_total * 100) if df_total > 0 else 0

        print('\n' + '=' * 80)
        print('CACHE PERFORMANCE STATISTICS')
        print('=' * 80)
        print(f'\nIndicator Cache:')
        print(f'  - Entries: {indicator_cache.size()}')
        print(f'  - Hits: {total_ind_hits:,}')
        print(f'  - Misses: {total_ind_misses:,}')
        print(f'  - Total queries: {ind_total:,}')
        print(f'  - Hit rate: {ind_hit_rate:.2f}%')

        print(f'\nDataFrame Cache:')
        print(f'  - Entries: {dataframe_cache.size()}')
        print(f'  - Hits: {total_df_hits:,}')
        print(f'  - Misses: {total_df_misses:,}')
        print(f'  - Total queries: {df_total:,}')
        print(f'  - Hit rate: {df_hit_rate:.2f}%')
        print('=' * 80 + '\n')

        try:
            indicator_cache_size = indicator_cache.size()
            indicator_cache.save_cache()
            logger.info(f"Saved indicator cache with {indicator_cache_size} entries")

            dataframe_cache_size = dataframe_cache.size()
            dataframe_cache.save_cache()
            logger.info(f"Saved dataframe cache with {dataframe_cache_size} entries")
        except Exception as e:
            logger.error(f"Failed to save caches after test completion: {e}")

        if self.results:
            self._save_results()

        # Calculate and print the total and average time
        end_time = time.time()
        total_time = end_time - start_time

        if len(test_combinations) > 0:
            average_time_per_100_tests = total_time / len(test_combinations) * 100
            print(f'Average time per 100 tests: {average_time_per_100_tests:.4f} seconds')

        print(f'Total execution time: {total_time:.2f} seconds')

        # Report failed tests summary
        if failed_tests > 0:
            logger.warning(f'Mass testing completed with {failed_tests} failed test(s) out of {total_tests} total tests')
            print(f'Warning: {failed_tests} test(s) failed during execution. Check logs for details.')

        return self.results

    # ==================== Private Methods ====================

    def _add_strategy_tests(self, strategy_type, param_grid):
        """ Generic method for adding strategy tests with all combinations of given parameters. """

        # Ensure every parameter has at least one value. Use [None] if empty
        for param_name, values_list in param_grid.items():
            if not values_list:
                param_grid[param_name] = [None]

        # Extract the names of the parameters to preserve order.
        param_names = list(param_grid.keys())

        # Generate all possible combinations of parameter values using Cartesian product.
        param_value_combinations = itertools.product(*(param_grid[param_name] for param_name in param_names))

        # Loop over parameter values to create strategy instances and store them in self.strategies.
        for param_values in param_value_combinations:
            strategy_params = dict(zip(param_names, param_values))
            strategy_name = get_strategy_name(strategy_type, **strategy_params)
            strategy_instance = create_strategy(strategy_type, **strategy_params)

            self.strategies.append((strategy_name, strategy_instance))

    def _results_to_dataframe(self):
        """Convert results to a pandas DataFrame."""

        if not self.results:
            logger.warning('No results available to convert to DataFrame.')
            return pd.DataFrame()

        # Define column names
        columns = [
            # --- Basic Trade Statistics ---
            'month',
            'symbol',
            'interval',
            'strategy',
            'total_trades',
            'win_rate',
            'average_trade_duration_hours',

            # --- Return Metrics --- (contract-based)
            'total_wins_percentage_of_contract',
            'total_losses_percentage_of_contract',
            'total_return_percentage_of_contract',
            'average_trade_return_percentage_of_contract',
            'average_win_percentage_of_contract',
            'average_loss_percentage_of_contract',
            'profit_factor',

            # --- Risk Metrics ---
            'maximum_drawdown_percentage',
            'sharpe_ratio',
            'sortino_ratio',
            'calmar_ratio',
            'value_at_risk',
            'expected_shortfall',
            'ulcer_index'
        ]

        # Pre-allocate arrays for each column
        n_results = len(self.results)

        # Create arrays for numeric columns
        numeric_columns = columns[4:]  # All columns except month, symbol, interval, strategy

        # Pre-allocate arrays for all columns including string columns
        data = {
            'month': [None] * n_results,
            'symbol': [None] * n_results,
            'interval': [None] * n_results,
            'strategy': [None] * n_results,
        }

        for col in numeric_columns:
            data[col] = [0.0] * n_results

        # Track validation issues
        missing_metrics_count = 0
        type_mismatch_count = 0

        # Fill the arrays directly
        for i, result in enumerate(self.results):
            metrics = result['metrics']
            data['month'][i] = result['month']
            data['symbol'][i] = result['symbol']
            data['interval'][i] = result['interval']
            data['strategy'][i] = result['strategy']

            # Fill numeric columns with validation
            for col in numeric_columns:
                if col not in metrics:
                    # Log warning for missing critical metrics
                    if col in ['total_trades', 'win_rate', 'total_return_percentage_of_contract']:
                        missing_metrics_count += 1
                        if missing_metrics_count <= 5:  # Only log first 5 to avoid spam
                            logger.warning(
                                f"Critical metric '{col}' missing for {result.get('strategy', 'unknown')} "
                                f"({result.get('symbol', 'unknown')}, {result.get('interval', 'unknown')}, "
                                f"{result.get('month', 'unknown')}). Using 0."
                            )
                    data[col][i] = 0
                else:
                    value = metrics[col]
                    # Validate numeric type
                    if not isinstance(value, (int, float, np.number)):
                        type_mismatch_count += 1
                        if type_mismatch_count <= 5:  # Only log the first 5 to avoid spam
                            logger.warning(
                                f"Type mismatch for metric '{col}': expected numeric, got {type(value).__name__} "
                                f"(value: {value}) for {result.get('strategy', 'unknown')}. Using 0."
                            )
                        data[col][i] = 0
                    # Check for inf/NaN values
                    elif np.isnan(value) or np.isinf(value):
                        logger.warning(
                            f"Invalid value ({value}) for metric '{col}' in {result.get('strategy', 'unknown')}. Using 0."
                        )
                        data[col][i] = 0
                    else:
                        data[col][i] = value

        # Log summary if there were validation issues
        if missing_metrics_count > 0:
            logger.warning(f"Total missing critical metrics: {missing_metrics_count}")
        if type_mismatch_count > 0:
            logger.warning(f"Total type mismatches: {type_mismatch_count}")

        # Create DataFrame from pre-filled arrays
        return pd.DataFrame(data)

    def _save_results(self):
        """Save results to one big parquet file."""
        try:
            # Convert results to DataFrame
            results_df = self._results_to_dataframe()
            if not results_df.empty:
                # Save all results to one big parquet file with unique entries
                parquet_filename = f'{BACKTESTING_DIR}/mass_test_results_all.parquet'
                save_to_parquet(results_df, parquet_filename)
                print(f'Results saved to {parquet_filename}')
            else:
                print('No results to save.')
        except Exception as error:
            logger.error(f'Failed to save results: {error}')


def _validate_dataframe(df, filepath):
    """Comprehensive DataFrame validation.

    Args:
        df: DataFrame to validate
        filepath: Path to the source file (for logging)

    Returns:
        bool: True if DataFrame is valid, False otherwise
    """
    # Check if DataFrame exists and is not empty
    if df is None or df.empty:
        logger.error(f'Empty or None DataFrame: {filepath}')
        return False

    # Check required columns
    required_columns = ['open', 'high', 'low', 'close']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f'DataFrame missing required columns {missing}: {filepath}')
        return False

    # Check data types - all OHLC columns must be numeric
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f'Non-numeric column "{col}" (type: {df[col].dtype}): {filepath}')
            return False

    # Check for excessive NaN values
    for col in required_columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_pct = (nan_count / len(df)) * 100
            if nan_pct > 10:  # More than 10% NaN is concerning
                logger.warning(f'Column "{col}" has {nan_pct:.1f}% NaN values ({nan_count}/{len(df)} rows): {filepath}')

    # Check index is DatetimeIndex (required for time-series operations)
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error(f'DataFrame index is not a DatetimeIndex (type: {type(df.index).__name__}): {filepath}')
        return False

    # Check index is sorted (critical for time-series data)
    if not df.index.is_monotonic_increasing:
        logger.error(f'DataFrame index is not sorted in ascending order: {filepath}')
        return False

    # Check for duplicate timestamps
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        logger.warning(f'DataFrame has {dup_count} duplicate timestamp(s): {filepath}')
        # Don't fail validation, just warn - duplicates might be intentional

    return True


def _run_single_test(test_params):
    """Run a single test with the given parameters."""
    # Unpack parameters
    tested_month, symbol, interval, strategy_name, strategy_instance, verbose, switch_dates, filepath = test_params

    # Track cache stats before the test
    ind_hits_before = indicator_cache.hits
    ind_misses_before = indicator_cache.misses
    df_hits_before = dataframe_cache.hits
    df_misses_before = dataframe_cache.misses

    try:
        df = get_cached_dataframe(filepath)
    except Exception as error:
        logger.error(f'Failed to read file: {filepath}\nReason: {error}')
        return None

    # Comprehensive DataFrame validation
    if not _validate_dataframe(df, filepath):
        return None

    # Check for the minimum row count required for reliable backtesting
    if len(df) < MIN_ROWS_FOR_BACKTEST:
        logger.warning(
            f'DataFrame has only {len(df)} rows, need at least {MIN_ROWS_FOR_BACKTEST} '
            f'for reliable backtesting in {filepath}'
        )
        # Continue anyway but log warning - some strategies may still work with fewer rows

    if verbose:
        print(f'\nRunning strategy: {strategy_name} for {symbol} {interval} {tested_month}', flush=True)

    trades_list = strategy_instance.run(df, switch_dates)

    trades_with_metrics_list = [calculate_trade_metrics(trade, symbol) for trade in trades_list]

    if trades_with_metrics_list:
        metrics = SummaryMetrics(trades_with_metrics_list)
        summary_metrics = metrics.calculate_all_metrics()
        # Note: Verbose printing removed - use logger or return metrics for display

        # Calculate cache stats for this test
        ind_hits_delta = indicator_cache.hits - ind_hits_before
        ind_misses_delta = indicator_cache.misses - ind_misses_before
        df_hits_delta = dataframe_cache.hits - df_hits_before
        df_misses_delta = dataframe_cache.misses - df_misses_before

        result = {
            'month': tested_month,
            'symbol': symbol,
            'interval': interval,
            'strategy': strategy_name,
            'metrics': summary_metrics,
            'timestamp': datetime.now().isoformat(),
            'cache_stats': {
                'ind_hits': ind_hits_delta,
                'ind_misses': ind_misses_delta,
                'df_hits': df_hits_delta,
                'df_misses': df_misses_delta,
            }
        }
        return result
    else:
        if verbose:
            print(f'No trades generated by strategy {strategy_name} for {symbol} {interval} {tested_month}', flush=True)

        logger.info(f'No trades generated by strategy {strategy_name} for {symbol} {interval} {tested_month}')

        # Calculate cache stats for this test
        ind_hits_delta = indicator_cache.hits - ind_hits_before
        ind_misses_delta = indicator_cache.misses - ind_misses_before
        df_hits_delta = dataframe_cache.hits - df_hits_before
        df_misses_delta = dataframe_cache.misses - df_misses_before

        # Return a complete result dictionary even when no trades are generated
        return {
            'month': tested_month,
            'symbol': symbol,
            'interval': interval,
            'strategy': strategy_name,
            'metrics': {},  # Empty metrics
            'timestamp': datetime.now().isoformat(),
            'cache_stats': {
                'ind_hits': ind_hits_delta,
                'ind_misses': ind_misses_delta,
                'df_hits': df_hits_delta,
                'df_misses': df_misses_delta,
            }
        }
