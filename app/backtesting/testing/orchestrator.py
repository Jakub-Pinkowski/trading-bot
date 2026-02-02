import concurrent.futures
import gc
import time

import pandas as pd

from app.backtesting.cache.dataframe_cache import dataframe_cache
from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.testing.reporting import save_results
from app.backtesting.testing.runner import run_single_test
from app.backtesting.testing.utils.test_preparation import load_existing_results, check_test_exists
from app.utils.logger import get_logger
from config import HISTORICAL_DATA_DIR

logger = get_logger('backtesting/testing/orchestrator')


# ==================== Test Orchestration ====================

def run_tests(
    tester,
    verbose=True,
    max_workers=None,
    skip_existing=True
):
    """Run all tests with the configured parameters in parallel."""
    start_time = time.time()  # Track the start time of the entire process

    # Reset cache statistics at the start of the run
    indicator_cache.reset_stats()
    dataframe_cache.reset_stats()

    if not hasattr(tester, 'strategies') or not tester.strategies:
        logger.error('No strategies added for testing. Use add_*_tests methods first.')
        raise ValueError('No strategies added for testing. Use add_*_tests methods first.')

    # Load existing results to check for already run tests
    existing_data = load_existing_results() if skip_existing else (pd.DataFrame(), set())

    total_combinations = len(tester.tested_months) * len(tester.symbols) * len(tester.intervals) * len(tester.strategies)
    print(f'Found {total_combinations} potential test combinations...')

    # Clear previous results
    tester.results = []

    # --- Test Preparation ---

    # Get switch dates for all symbols (handles symbol mappings for mini/micro contracts)
    switch_dates_by_symbol = _get_switch_dates_for_symbols(tester.symbols, tester.switch_dates_dict)

    # Generate all combinations
    all_combinations = _generate_all_combinations(
        tester.tested_months,
        tester.symbols,
        tester.intervals,
        tester.strategies
    )

    # Prepare all test combinations
    test_combinations, skipped_combinations = _prepare_test_combinations(
        all_combinations,
        existing_data,
        skip_existing,
        verbose,
        switch_dates_by_symbol
    )

    print(f'Skipped {skipped_combinations} already run test combinations.')
    print(f'Running {len(test_combinations)} new test combinations...')

    if not test_combinations:
        print('All tests have already been run. No new tests to execute.')

        # Calculate and print the total time even when no tests are run
        end_time = time.time()
        total_time = end_time - start_time
        print(f'Total execution time: {total_time:.2f} seconds')

        return tester.results

    # --- Parallel Execution ---

    _execute_tests_in_parallel(
        tester,
        test_combinations,
        max_workers
    )

    # --- Cache Statistics Reporting ---

    _report_cache_statistics(tester.results)

    # --- Save Caches ---

    _save_caches()

    # --- Save Results ---

    if tester.results:
        save_results(tester.results)

    # --- Final Reporting ---

    _report_execution_summary(test_combinations, start_time)

    return tester.results


# ==================== Helper Functions ====================

# --- Switch Dates Mapping ---

def _get_switch_dates_for_symbols(symbols, switch_dates_dict):
    """
    Get switch dates for all symbols, handling mini/micro contract mappings.

    Maps mini/micro symbols (e.g., MCL, MGC) to their main contract symbols (CL, GC)
    and converts date strings to datetime objects for strategy consumption.

    Args:
        symbols: List of symbol strings
        switch_dates_dict: Dict with switch dates and optional _symbol_mappings

    Returns:
        Dict mapping each symbol to list of datetime objects
    """
    switch_dates_by_symbol = {}
    for symbol in symbols:
        # Check if the symbol has direct switch dates
        if symbol in switch_dates_dict:
            switch_dates = switch_dates_dict[symbol]
        # Check if the symbol is a mini/micro that maps to a main symbol
        elif '_symbol_mappings' in switch_dates_dict and symbol in switch_dates_dict['_symbol_mappings']:
            main_symbol = switch_dates_dict['_symbol_mappings'][symbol]
            switch_dates = switch_dates_dict.get(main_symbol, [])
        else:
            switch_dates = []

        switch_dates_by_symbol[symbol] = [pd.to_datetime(switch_date) for switch_date in switch_dates]

    return switch_dates_by_symbol



# --- Combination Generation ---

def _generate_all_combinations(tested_months, symbols, intervals, strategies):
    """Generate all combinations of months, symbols, intervals, and strategies."""
    return [(tested_month, symbol, interval, strategy_name, strategy_instance)
            for tested_month in tested_months
            for symbol in symbols
            for interval in intervals
            for strategy_name, strategy_instance in strategies]


# --- Test Preparation ---

def _prepare_test_combinations(
    all_combinations,
    existing_data,
    skip_existing,
    verbose,
    switch_dates_by_symbol
):
    """Prepare all test combinations, filtering out already run tests."""
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
        if skip_existing and check_test_exists(existing_data,
                                               tested_month,
                                               symbol,
                                               interval,
                                               strategy_name):
            if verbose:
                print(f'Skipping already run test: Month={tested_month}, Symbol={symbol}, Interval={interval}, Strategy={strategy_name}')
            skipped_combinations += 1
            continue

        # Build filepath on-demand (string interpolation is negligible overhead)
        filepath = f'{HISTORICAL_DATA_DIR}/{tested_month}/{symbol}/{symbol}_{interval}.parquet'

        # Add switch dates and filepath to the test parameters
        test_combinations.append((
            tested_month,
            symbol,
            interval,
            strategy_name,
            strategy_instance,
            verbose,
            switch_dates_by_symbol[symbol],
            filepath
        ))

    return test_combinations, skipped_combinations


# --- Parallel Execution ---

def _execute_tests_in_parallel(tester, test_combinations, max_workers):
    """Run tests in parallel using ProcessPoolExecutor."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_test = {executor.submit(run_single_test, test_params): test_params for test_params in
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
                if completed_tests % 1000 == 0 and tester.results:
                    save_results(tester.results)
                    tester.results.clear()

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
                tester.results.append(result)

        # Report failed tests summary
        if failed_tests > 0:
            logger.warning(f'Mass testing completed with {failed_tests} failed test(s) out of {total_tests} total tests')
            print(f'Warning: {failed_tests} test(s) failed during execution. Check logs for details.')


# --- Cache Statistics ---

def _report_cache_statistics(results):
    """Aggregate and report cache statistics from all test results."""
    # Aggregate cache statistics from all test results
    total_ind_hits = 0
    total_ind_misses = 0
    total_df_hits = 0
    total_df_misses = 0

    for result in results:
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


# --- Cache Saving ---

def _save_caches():
    """Save indicator and dataframe caches."""
    try:
        indicator_cache.save_cache()

        dataframe_cache.save_cache()
    except Exception as e:
        logger.error(f"Failed to save caches after test completion: {e}")


# --- Execution Summary ---

def _report_execution_summary(test_combinations, start_time):
    """Calculate and print execution summary statistics."""
    end_time = time.time()
    total_time = end_time - start_time

    if len(test_combinations) > 0:
        average_time_per_100_tests = total_time / len(test_combinations) * 100
        print(f'Average time per 100 tests: {average_time_per_100_tests:.4f} seconds')

    print(f'Total execution time: {total_time:.2f} seconds')
