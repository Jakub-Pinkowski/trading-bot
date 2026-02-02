"""
Data Processing Helpers

This module contains helper functions for filtering DataFrames and calculating
metrics for strategy analysis.
"""

from app.utils.logger import get_logger

logger = get_logger('backtesting/analysis/helpers')

# Import constants from the main module
from app.backtesting.analysis.strategy_analyzer import (
    REQUIRED_COLUMNS,
    AGG_FUNCTIONS,
    DECIMAL_PLACES
)


# ==================== DataFrame Operations ====================

def filter_dataframe(
    df,
    min_avg_trades_per_combination=0,
    interval=None,
    symbol=None,
    min_slippage=None,
    min_symbol_count=None
):
    """
    Filter DataFrame based on common criteria.

    Args:
        df: DataFrame containing strategy results
        min_avg_trades_per_combination: Minimum average trades per symbol/interval combo
        interval: Filter by specific interval (e.g., '1h', '4h')
        symbol: Filter by specific symbol (e.g., 'ES', 'NQ')
        min_slippage: Minimum slippage value to filter by
        min_symbol_count: Minimum number of unique symbols per strategy

    Returns:
        Filtered DataFrame

    Raises:
        ValueError: If required columns are missing
    """
    # Validate required columns exist
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Handle empty DataFrame
    if df.empty:
        logger.warning("Empty DataFrame passed to filter_dataframe")
        return df.copy()

    # Filter by minimum average trades per combination or minimum symbol count
    if min_avg_trades_per_combination > 0 or min_symbol_count is not None:
        strategy_stats = df.groupby('strategy').agg(AGG_FUNCTIONS).reset_index()

        # Calculate combinations (symbol × interval)
        strategy_stats['combination_count'] = strategy_stats['symbol'] * strategy_stats['interval']
        strategy_stats['avg_trades_per_combination'] = (
                strategy_stats['total_trades'] / strategy_stats['combination_count']
        )

        # Apply filters
        filter_conditions = []

        if min_avg_trades_per_combination > 0:
            filter_conditions.append(
                strategy_stats['avg_trades_per_combination'] >= min_avg_trades_per_combination
            )

        if min_symbol_count is not None:
            filter_conditions.append(
                strategy_stats['symbol'] >= min_symbol_count
            )

        # Combine all filter conditions with AND logic
        if filter_conditions:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition

            valid_strategies = strategy_stats[combined_filter]['strategy'].tolist()
        else:
            valid_strategies = strategy_stats['strategy'].tolist()

        filtered_df = df[df['strategy'].isin(valid_strategies)]
    else:
        filtered_df = df.copy()

    # Filter by interval if provided
    if interval:
        filtered_df = filtered_df[filtered_df['interval'] == interval]

    # Filter by symbol if provided
    if symbol:
        filtered_df = filtered_df[filtered_df['symbol'] == symbol]

    # Filter by minimum slippage if provided
    if min_slippage is not None:
        # Extract slippage from the strategy name
        filtered_df = filtered_df[
            filtered_df['strategy'].str.extract(r'slippage=([^,\)]+)')[0].astype(float) >= min_slippage]

    return filtered_df


# ==================== Calculation Helpers ====================

def calculate_weighted_win_rate(filtered_df, grouped):
    """Calculate win rate weighted by total trades."""
    total_trades_by_strategy = grouped['total_trades'].sum()
    winning_trades_by_strategy = (filtered_df['win_rate'] * filtered_df['total_trades'] / 100).groupby(
        filtered_df['strategy']).sum()
    return (winning_trades_by_strategy / total_trades_by_strategy * 100).round(DECIMAL_PLACES)


def calculate_average_trade_return(total_return, total_trades):
    """Calculate average trade return from total return and total trades."""
    return (total_return / total_trades).round(DECIMAL_PLACES)


def calculate_profit_ratio(total_wins_percentage, total_losses_percentage):
    """Calculate a profit factor from total wins and losses percentages."""
    return abs(
        total_wins_percentage / total_losses_percentage
    ).replace([float('inf'), float('-inf')], float('inf')).round(DECIMAL_PLACES)


def calculate_trade_weighted_average(filtered_df, metric_name, total_trades_by_strategy):
    """Calculate trade-weighted average for a given metric."""
    weighted_sum = (filtered_df[metric_name] * filtered_df['total_trades']).groupby(
        filtered_df['strategy']).sum()
    return (weighted_sum / total_trades_by_strategy).round(DECIMAL_PLACES)
