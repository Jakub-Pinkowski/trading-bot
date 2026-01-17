import concurrent.futures
import gc
import io
import itertools
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from filelock import FileLock

from app.backtesting.cache.dataframe_cache import dataframe_cache, get_cached_dataframe
from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.per_trade_metrics import calculate_trade_metrics
from app.backtesting.strategy_factory import create_strategy, get_strategy_name
from app.backtesting.summary_metrics import SummaryMetrics
from app.utils.file_utils import save_to_parquet
from app.utils.logger import get_logger
from config import (HISTORICAL_DATA_DIR, SWITCH_DATES_FILE_PATH, BACKTESTING_DIR,
                    INDICATOR_CACHE_LOCK_FILE, DATAFRAME_CACHE_LOCK_FILE)

logger = get_logger('backtesting/mass_testing')


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

    def run_tests(self, verbose=True, max_workers=None, skip_existing=True):
        """  Run all tests with the configured parameters in parallel. """
        start_time = time.time()  # Track the start time of the entire process

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
            future_to_test = {executor.submit(self._run_single_test, test_params): test_params for test_params in
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
                    
                    # Log the exception with full traceback
                    logger.exception(
                        f'Worker exception during test execution: '
                        f'Month={tested_month}, Symbol={symbol}, Interval={interval}, Strategy={strategy_name}'
                    )
                    failed_tests += 1
                    continue

                if result:
                    # Print verbose output if available
                    if verbose and 'verbose_output' in result and result['verbose_output']:
                        print(result['verbose_output'])

                    # Remove verbose_output from a result before adding to a result list
                    if 'verbose_output' in result:
                        result_copy = result.copy()
                        del result_copy['verbose_output']
                        self.results.append(result_copy)
                    else:
                        self.results.append(result)

        # Save caches after all tests complete (only from main process)
        print('All tests completed, saving caches...')
        try:
            with FileLock(INDICATOR_CACHE_LOCK_FILE, timeout=60):
                indicator_cache.save_cache()
            logger.info(f"Saved indicator cache with {indicator_cache.size()} entries")
            
            with FileLock(DATAFRAME_CACHE_LOCK_FILE, timeout=60):
                dataframe_cache.save_cache()
            logger.info(f"Saved dataframe cache with {dataframe_cache.size()} entries")
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

    # --- Private methods ---

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

    def _run_single_test(self, test_params):
        """Run a single test with the given parameters."""
        # Unpack parameters
        tested_month, symbol, interval, strategy_name, strategy_instance, verbose, switch_dates, filepath = test_params

        output_buffer = []  # Buffer to collect verbose output
        try:
            df = get_cached_dataframe(filepath)
        except Exception as error:
            logger.error(f'Failed to read file: {filepath}\nReason: {error}')
            return None

        if verbose:
            output_buffer.append(f'\nRunning strategy: {strategy_name} for {symbol} {interval} {tested_month}')

        trades_list = strategy_instance.run(df, switch_dates)

        trades_with_metrics_list = [calculate_trade_metrics(trade, symbol) for trade in trades_list]

        if trades_with_metrics_list:
            metrics = SummaryMetrics(trades_with_metrics_list)
            summary_metrics = metrics.calculate_all_metrics()
            if verbose:
                original_stdout = sys.stdout
                string_io = io.StringIO()
                sys.stdout = string_io
                SummaryMetrics.print_summary_metrics(summary_metrics)
                sys.stdout = original_stdout
                output_buffer.append(string_io.getvalue())

            result = {
                'month': tested_month,
                'symbol': symbol,
                'interval': interval,
                'strategy': strategy_name,
                'metrics': summary_metrics,
                'timestamp': datetime.now().isoformat(),
                'verbose_output': '\n'.join(output_buffer) if verbose else None
            }
            return result
        else:
            if verbose:
                output_buffer.append(f'No trades generated by strategy {strategy_name} for {symbol} {interval} {tested_month}')

            logger.info(f'No trades generated by strategy {strategy_name} for {symbol} {interval} {tested_month}')
            # Return a complete result dictionary even when no trades are generated
            return {
                'month': tested_month,
                'symbol': symbol,
                'interval': interval,
                'strategy': strategy_name,
                'metrics': {},  # Empty metrics
                'timestamp': datetime.now().isoformat(),
                'verbose_output': '\n'.join(output_buffer) if verbose else None
            }

    def _results_to_dataframe(self):
        """Convert results to a pandas DataFrame."""

        if not self.results:
            logger.warning('No results available to convert to DataFrame.')
            return pd.DataFrame()

        # Define column names
        columns = [
            # Basic info
            'month',
            'symbol',
            'interval',
            'strategy',
            'total_trades',
            'win_rate',
            # Contract-based metrics
            'total_return_percentage_of_contract',
            'average_trade_return_percentage_of_contract',
            # Percentage-based metrics
            'total_return_percentage_of_margin',
            'average_trade_return_percentage_of_margin',
            'average_win_percentage_of_margin',
            'average_loss_percentage_of_margin',
            'commission_percentage_of_margin',
            'total_wins_percentage_of_margin',
            'total_losses_percentage_of_margin',
            # Risk metrics
            'profit_factor',
            'maximum_drawdown_percentage',
            'sharpe_ratio',
            'sortino_ratio',
            'calmar_ratio',
            'value_at_risk',
            'expected_shortfall',
            'ulcer_index'
        ]

        # Pre-allocate numpy arrays for each column
        n_results = len(self.results)

        # Create arrays for numeric columns
        numeric_columns = columns[4:]  # All columns except month, symbol, interval, strategy
        data = {
            'month': [],
            'symbol': [],
            'interval': [],
            'strategy': [],
        }

        for col in numeric_columns:
            data[col] = np.zeros(n_results)

        # Fill the arrays directly
        for i, result in enumerate(self.results):
            metrics = result['metrics']
            data['month'].append(result['month'])
            data['symbol'].append(result['symbol'])
            data['interval'].append(result['interval'])
            data['strategy'].append(result['strategy'])

            # Fill numeric columns
            for col in numeric_columns:
                data[col][i] = metrics.get(col, 0)

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
