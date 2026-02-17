from datetime import datetime

from app.backtesting.cache.dataframe_cache import dataframe_cache, get_cached_dataframe
from app.backtesting.cache.indicators_cache import indicator_cache
from app.backtesting.metrics.per_trade_metrics import calculate_trade_metrics
from app.backtesting.metrics.summary_metrics import SummaryMetrics
from app.backtesting.testing.utils.dataframe_validators import validate_dataframe, MIN_ROWS_FOR_BACKTEST
from app.utils.logger import get_logger

logger = get_logger('backtesting/testing/runner')


# ==================== Single Test Execution ====================

def run_single_test(test_params):
    """
    Execute a single backtest for one strategy on one symbol/interval/month combination.

    This is the core worker function called by each parallel process. It handles the complete
    test lifecycle: loading historical data (with caching), running the strategy to generate
    trades, calculating performance metrics, and tracking cache statistics.

    Args:
        test_params: Tuple containing 12 elements in this order:
            - tested_month: Month identifier (e.g., '1!', '2!')
            - symbol: Futures symbol (e.g., 'ZS', 'CL', 'GC')
            - interval: Timeframe (e.g., '15m', '1h', '4h', '1d')
            - strategy_name: Full strategy name with parameters
            - strategy_instance: Instantiated strategy object ready to run
            - verbose: If True, print progress messages
            - switch_dates: List of datetime objects for contract rollover dates
            - filepath: Path to parquet file with historical price data
            - segment_id: Segment identifier, or None for non-segmented runs
            - period_id: Parent period identifier, or None for non-segmented runs
            - start_date: Segment start date for DataFrame slicing, or None
            - end_date: Segment end date for DataFrame slicing, or None

    Returns:
        Dictionary with test results, or None if the test failed.
        Success dict contains:
        - month: Tested month identifier
        - symbol: Tested symbol
        - interval: Tested timeframe
        - strategy: Strategy name with parameters
        - metrics: Dict of performance metrics (profit_factor, win_rate, etc.)
                  Empty dict if no trades were generated
        - timestamp: ISO format timestamp of test execution
        - cache_stats: Dict with cache hit/miss counts for this specific test
                      (ind_hits, ind_misses, df_hits, df_misses)

        Returns None if:
        - Failed to load historical data file
        - DataFrame validation failed (missing columns, too many NaN values, etc.)
    """
    # Unpack parameters
    tested_month, symbol, interval, strategy_name, strategy_instance, verbose, switch_dates, filepath, segment_id, period_id, start_date, end_date = test_params

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

    # Slice DataFrame to segment range if specified
    if start_date is not None and end_date is not None:
        df = df.loc[start_date:end_date]
        # Only include switch dates that actually exist in the sliced DataFrame
        switch_dates = [d for d in switch_dates if d in df.index]

    # Comprehensive DataFrame validation
    if not validate_dataframe(df, filepath):
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

        # Calculate cache stats for this test
        ind_hits_delta = indicator_cache.hits - ind_hits_before
        ind_misses_delta = indicator_cache.misses - ind_misses_before
        df_hits_delta = dataframe_cache.hits - df_hits_before
        df_misses_delta = dataframe_cache.misses - df_misses_before

        result = {
            'month': tested_month,
            'symbol': symbol,
            'interval': interval,
            'segment_id': segment_id,
            'period_id': period_id,
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
            'segment_id': segment_id,
            'period_id': period_id,
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
