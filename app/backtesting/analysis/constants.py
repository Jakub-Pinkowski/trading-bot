"""
Analysis Constants

This module contains all constants used throughout the strategy analysis system.
Separated to avoid circular imports.
"""

# ==================== Default Values ====================

DEFAULT_LIMIT = 30
DECIMAL_PLACES = 2

# Minimum per-run trade count required for a row to be included in the
# trade-weighted average of ratio metrics (Sharpe, Sortino, Calmar).
# Rows below this threshold are too noisy and skew the aggregate.
MIN_TRADES_FOR_RATIO = 30

# Used when ratio calculations would result in infinity (e.g. zero losses)
# Rationale: Large but finite number that won't break aggregations
INFINITY_REPLACEMENT = 9999.99

# ==================== DataFrame Requirements ====================

# Required columns that must be present in strategy results DataFrames
REQUIRED_COLUMNS = ['strategy', 'total_trades', 'symbol', 'interval']

# Column aggregation mappings for group by operations
AGG_FUNCTIONS = {
    'total_trades': 'sum',
    'symbol': 'nunique',
    'interval': 'nunique'
}
