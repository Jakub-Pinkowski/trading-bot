import pandas as pd

from app.utils.logger import get_logger

logger = get_logger('backtesting/testing/selection')


# ==================== Strategy Ranking ====================

def rank_strategies(
    results_df,
    metric,
    min_trades=0,
    ascending=False
):
    """
    Rank strategies by a performance metric.

    Args:
        results_df: DataFrame with backtest results
        metric: Column name to rank by (e.g., 'profit_factor', 'sharpe_ratio')
        min_trades: Minimum trades required to be included
        ascending: Sort order (False = best first)

    Returns:
        DataFrame with aggregated and ranked strategies
    """
    if results_df.empty:
        logger.warning('Empty results DataFrame provided')
        return pd.DataFrame()

    # Filter by minimum trades
    filtered = results_df[results_df['total_trades'] >= min_trades].copy()

    if filtered.empty:
        logger.warning(f'No strategies with >={min_trades} trades')
        return pd.DataFrame()

    # Aggregates it across symbols/intervals/months
    # Same strategy tested on multiple combinations
    # Build agg dict dynamically to avoid key collision when metric is one of the fixed columns
    agg_spec = {'total_trades': 'sum', 'win_rate': 'mean', 'symbol': 'nunique', 'interval': 'nunique', metric: 'mean'}
    aggregated = filtered.groupby('strategy').agg(agg_spec).reset_index()

    # Rename it for clarity
    aggregated.rename(columns={
        'symbol': 'symbol_count',
        'interval': 'interval_count'
    }, inplace=True)

    # Sort by metric
    ranked = aggregated.sort_values(metric, ascending=ascending)

    logger.info(
        f'Ranked {len(ranked):,} strategies by {metric} '
        f'(min_trades={min_trades})'
    )

    return ranked


# ==================== Strategy Selection ====================

def select_top_strategies(
    ranked_df,
    top_n=None,
    top_percentage=None
):
    """
    Select top N or top percentage of strategies.

    Args:
        ranked_df: Ranked DataFrame from rank_strategies()
        top_n: Number of top strategies to select
        top_percentage: Percentage of top strategies (e.g., 0.01 for top 1%)

    Returns:
        DataFrame with selected top strategies

    Raises:
        ValueError: If neither top_n nor top_percentage specified
    """
    if top_n is None and top_percentage is None:
        raise ValueError('Must specify either top_n or top_percentage')

    if top_n is not None and top_percentage is not None:
        raise ValueError('Specify only one of top_n or top_percentage, not both')

    if ranked_df.empty:
        logger.warning('Empty ranked DataFrame')
        return pd.DataFrame()

    if top_n is not None:
        selected = ranked_df.head(top_n)
        logger.info(f'Selected top {top_n} strategies')
    else:
        count = int(len(ranked_df) * top_percentage)
        if count == 0:
            logger.warning(
                f'top_percentage={top_percentage} yielded 0 strategies '
                f'from {len(ranked_df):,} ranked'
            )
            return pd.DataFrame()
        selected = ranked_df.head(count)
        logger.info(f'Selected top {top_percentage * 100:.1f}% ({count:,} strategies)')

    return selected
