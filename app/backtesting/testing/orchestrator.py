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
from config import DATA_DIR

# Module path
HISTORICAL_DATA_DIR = DATA_DIR / "historical_data"

logger = get_logger('backtesting/testing/orchestrator')


# ==================== Test Orchestration ====================

def run_tests(
    tester,
    verbose,
    max_workers,
    skip_existing,
    segment_filter=None
):
    """
    Core orchestration function that executes all configured backtests in parallel.

    This is the main entry point for running mass backtests. It coordinates test preparation,
    parallel execution across multiple worker processes, result aggregation, cache management,
    and result persistence. Handles thousands of test combinations efficiently through
    intelligent caching and multiprocessing.

    Args:
        tester: MassTester instance containing configured strategies, symbols, intervals,
               and months to test
        verbose: If True, print detailed progress for each test combination.
                If False, only show summary statistics and progress every 100 tests
        max_workers: Maximum number of parallel worker processes for test execution.
                    None = uses all available CPU cores.
                    Lower values reduce memory usage but increase execution time
        skip_existing: If True, check a database for existing results and skip already-run tests.
                      If False, re-run all tests regardless of existing results.
                      Useful when parameters or logic have changed
        segment_filter: Optional list of segment IDs to run (e.g., [1, 2, 3]).
                       Only applies when tester.segments is non-empty.
                       None = run all segments

    Returns:
        List of test result dictionaries stored in tester.results. Each dict contains:
        - month: Tested month identifier (e.g., '1!', '2!')
        - symbol: Tested futures symbol (e.g., 'ZS', 'CL', 'GC')
        - interval: Tested timeframe (e.g., '15m', '1h', '4h', '1d')
        - strategy: Strategy name with full parameter specification
        - metrics: Dict of performance metrics (profit_factor, win_rate, sharpe_ratio, etc.)
        - timestamp: ISO format timestamp of when test was executed
        - cache_stats: Dict with indicator and dataframe cache hit/miss statistics

    Raises:
        ValueError: If no strategies have been added to the tester
                   (strategies list is empty or not configured)
    """
    start_time = time.time()  # Track the start time of the entire process

    # Reset cache statistics at the start of the run
    indicator_cache.reset_stats()
    dataframe_cache.reset_stats()

    if not hasattr(tester, 'strategies') or not tester.strategies:
        logger.error('No strategies added for testing. Use add_*_tests methods first.')
        raise ValueError('No strategies added for testing. Use add_*_tests methods first.')

    # Load existing results to check for already run tests
    existing_data = load_existing_results() if skip_existing else (pd.DataFrame(), set())

    # Determine active segments based on filter
    active_segments = []
    if tester.segments:
        active_segments = [
            segment for segment in tester.segments
            if segment_filter is None or segment['segment_id'] in segment_filter
        ]

    # Handle case where segments is empty list vs None
    if not tester.segments:
        # No segments configured - run normal (non-segmented) tests
        segment_count = 1
    elif active_segments:
        # Segments configured and filter matched some
        segment_count = len(active_segments)
    else:
        # Segments configured but filter matched none
        segment_count = 0
        logger.warning(
            f"segment_filter resulted in 0 active segments. "
            f"No tests will run. Available segment IDs: "
            f"{[s['segment_id'] for s in tester.segments]}"
        )

    total_combinations = len(tester.tested_months) * len(tester.symbols) * len(tester.intervals) * len(tester.strategies) * segment_count
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
        tester.strategies,
        active_segments
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
        Dict mapping each symbol to a list of datetime objects
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

def _generate_all_combinations(tested_months, symbols, intervals, strategies, segments):
    """Generate all combinations of months, symbols, intervals, segments, and strategies."""
    if segments:
        return [
            (tested_month, symbol, interval, segment, strategy_name, strategy_instance)
            for tested_month in tested_months
            for symbol in symbols
            for interval in intervals
            for segment in segments
            for strategy_name, strategy_instance in strategies
        ]
    return [
        (tested_month, symbol, interval, None, strategy_name, strategy_instance)
        for tested_month in tested_months
        for symbol in symbols
        for interval in intervals
        for strategy_name, strategy_instance in strategies
    ]


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
        tested_month, symbol, interval, segment, strategy_name, strategy_instance = combo

        # Extract segment metadata
        segment_id = segment['segment_id'] if segment is not None else None
        period_id = segment['period_id'] if segment is not None else None
        start_date = segment['start_date'] if segment is not None else None
        end_date = segment['end_date'] if segment is not None else None

        # Print verbose output for each unique combination (including segment)
        if verbose and (tested_month, symbol, interval, segment_id) != last_verbose_combo:
            segment_info = f", Segment {segment_id}" if segment_id is not None else ""
            print(f'Preparing: Month={tested_month}, Symbol={symbol}, Interval={interval}{segment_info}')
            last_verbose_combo = (tested_month, symbol, interval, segment_id)

        # Check if this test has already been run
        if skip_existing and check_test_exists(existing_data,
                                               tested_month,
                                               symbol,
                                               interval,
                                               strategy_name,
                                               segment_id):
            if verbose:
                print(f'Skipping already run test: Month={tested_month}, Symbol={symbol}, Interval={interval}, Strategy={strategy_name}')
            skipped_combinations += 1
            continue

        # Build filepath on-demand (string interpolation is negligible overhead)
        filepath = f'{HISTORICAL_DATA_DIR}/{tested_month}/{symbol}/{symbol}_{interval}.parquet'

        # Add switch dates, filepath, and segment metadata to the test parameters
        test_combinations.append((
            tested_month,
            symbol,
            interval,
            strategy_name,
            strategy_instance,
            verbose,
            switch_dates_by_symbol[symbol],
            filepath,
            segment_id,
            period_id,
            start_date,
            end_date
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
