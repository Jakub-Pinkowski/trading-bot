"""
Formatting and Export Helpers

This module contains helper functions for formatting strategy data and
preparing it for CSV export.
"""

import re
from datetime import datetime

from app.backtesting.analysis.constants import DECIMAL_PLACES
from app.utils.logger import get_logger

logger = get_logger('backtesting/analysis/formatters')


# ==================== Parsing Helpers ====================

def parse_strategy_name(strategy_name):
    """
    Parse strategy name to extract common parameters and clean the strategy name.
    Strategy names are created by strategy_factory.get_strategy_name()
    Format: "StrategyType(param1=val1,param2=val2,...,rollover=X,trailing=Y,slippage=Z)"
    Args:
        strategy_name: Full strategy name with parameters
    Returns:
        Tuple of (clean_name, rollover, trailing, slippage)
    """
    params = {'rollover': False, 'trailing': None, 'slippage': 0.0}
    # Extract all param=value pairs with single regex iteration
    for match in re.finditer(r'(\w+)=([^,)]+)', strategy_name):
        key, value = match.groups()
        if key in params:
            if key == 'rollover':
                params[key] = value == 'True'
            elif key == 'trailing':
                params[key] = None if value == 'None' else float(value)
            elif key == 'slippage':
                params[key] = float(value)
    # Remove common parameters from name with single regex
    clean_name = re.sub(r',?(rollover|trailing|slippage)=[^,)]+,?', '', strategy_name)
    clean_name = re.sub(r',+', ',', clean_name).strip(',')
    clean_name = re.sub(r',\)', ')', clean_name)
    clean_name = re.sub(r'\(,', '(', clean_name)
    return clean_name, params['rollover'], params['trailing'], params['slippage']


# ==================== Column Formatting ====================

def format_column_name(column_name):
    """Convert snake_case column names to Title Case with spaces."""
    column_name_mapping = {
        'average_trade_return_percentage_of_contract': 'avg_return_%',
        'average_win_percentage_of_contract': 'avg_win_%',
        'total_return_percentage_of_contract': 'total_return_%',
        'average_loss_percentage_of_contract': 'avg_loss_%',
        'maximum_drawdown_percentage': 'max_drawdown_%',
        'average_trade_duration_hours': 'avg_duration_h',
        'win_rate': 'win_rate_%',
        'max_consecutive_wins': 'max_cons_wins',
        'max_consecutive_losses': 'max_cons_losses',
        'sharpe_ratio': 'sharpe',
        'sortino_ratio': 'sortino',
        'calmar_ratio': 'calmar',
        'value_at_risk': 'var_95%',
        'expected_shortfall': 'cvar_95%',
        'ulcer_index': 'ulcer_idx'
    }
    if column_name in column_name_mapping:
        shortened_name = column_name_mapping[column_name]
        return ' '.join(word.capitalize() for word in shortened_name.split('_'))
    return ' '.join(word.capitalize() for word in column_name.split('_'))


def reorder_columns(df):
    """Reorder DataFrame columns to place common parameters after strategy."""
    cols = list(df.columns)
    if 'strategy' not in cols:
        return df
    try:
        strategy_idx = cols.index('strategy')
        cols = [col for col in cols if col not in ['rollover', 'trailing', 'slippage']]
        cols.insert(strategy_idx + 1, 'rollover')
        cols.insert(strategy_idx + 2, 'trailing')
        cols.insert(strategy_idx + 3, 'slippage')
        return df[cols]
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not reorder columns: {e}. Using original column order.")
        return df


def format_dataframe_for_export(df):
    """Format DataFrame for CSV export by parsing strategy names and formatting columns."""
    formatted_df = df.copy()
    if 'strategy' in formatted_df.columns:
        strategy_data = formatted_df['strategy'].apply(parse_strategy_name)
        formatted_df['strategy'] = [data[0] for data in strategy_data]
        formatted_df['rollover'] = [data[1] for data in strategy_data]
        formatted_df['trailing'] = [data[2] for data in strategy_data]
        formatted_df['slippage'] = [data[3] for data in strategy_data]
        formatted_df = reorder_columns(formatted_df)
    numeric_cols = formatted_df.select_dtypes(include='number').columns
    formatted_df[numeric_cols] = formatted_df[numeric_cols].round(DECIMAL_PLACES)
    formatted_df.columns = [format_column_name(col) for col in formatted_df.columns]
    return formatted_df


def build_filename(metric, aggregate, interval=None, symbol=None, weighted=True):
    """Build CSV filename with appropriate suffixes."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    agg_suffix = "_aggregated" if aggregate else ""
    weighted_suffix = "_weighted" if aggregate and weighted else "_simple" if aggregate else ""
    interval_suffix = f"_{interval}" if interval else ""
    symbol_suffix = f"_{symbol}" if symbol else ""
    return f"{timestamp} top_strategies_by_{metric}{interval_suffix}{symbol_suffix}{agg_suffix}{weighted_suffix}.csv"
