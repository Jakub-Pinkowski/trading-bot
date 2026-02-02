"""
Analysis Constants

This module contains all constants used throughout the strategy analysis system.
Separated to avoid circular imports.
"""

# ==================== Default Values ====================

DEFAULT_LIMIT = 30
DECIMAL_PLACES = 2

# ==================== DataFrame Requirements ====================

# Required columns that must be present in strategy results DataFrames
REQUIRED_COLUMNS = ['strategy', 'total_trades', 'symbol', 'interval']

# Column aggregation mappings for groupby operations
AGG_FUNCTIONS = {
    'total_trades': 'sum',
    'symbol': 'nunique',
    'interval': 'nunique'
}
