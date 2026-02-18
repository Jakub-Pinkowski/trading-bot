import pandas as pd

from app.utils.logger import get_logger

logger = get_logger('backtesting/testing/selection')


# ==================== Helper Functions ====================

def _get_strategy_list(results_df):
    """
    Extract unique strategy names from results.

    Args:
        results_df: DataFrame with backtest results

    Returns:
        List of unique strategy names
    """
    return results_df['strategy'].unique().tolist()


def _compare_strategy_sets(set_a, set_b, label_a='Set A', label_b='Set B'):
    """
    Compare two sets of strategies and show overlap.

    Args:
        set_a: List or set of strategy names
        set_b: List or set of strategy names
        label_a: Label for the first set
        label_b: Label for the second set

    Returns:
        Dict with overlap statistics
    """
    set_a = set(set_a)
    set_b = set(set_b)

    overlap = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    stats = {
        'total_a': len(set_a),
        'total_b': len(set_b),
        'overlap': len(overlap),
        'only_a': len(only_a),
        'only_b': len(only_b),
        'overlap_percentage_a': len(overlap) / len(set_a) * 100 if set_a else 0,
        'overlap_percentage_b': len(overlap) / len(set_b) * 100 if set_b else 0
    }

    logger.info(
        f'{label_a}: {stats["total_a"]}, '
        f'{label_b}: {stats["total_b"]}, '
        f'Overlap: {stats["overlap"]} '
        f'({stats["overlap_percentage_a"]:.1f}%/{stats["overlap_percentage_b"]:.1f}%)'
    )

    return stats


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
