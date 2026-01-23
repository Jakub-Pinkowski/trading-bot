"""
Trading Bot Application Package

This module initializes global configurations for the entire application.
"""

import pandas as pd

# ============================================================================
# GLOBAL PANDAS CONFIGURATION
# ============================================================================

# Enable copy-on-write mode for ALL DataFrames in the application
# This significantly improves performance for DataFrame copy operations:
#   - Copies are 1.8x-3x faster
#   - Memory is shared between copies until modification
#   - Reduces memory overhead from ~100% to <5% for read-only access
#
# Why this matters:
#   - Backtesting creates many DataFrame copies (caching, strategy execution)
#   - Without CoW, each copy duplicates all data immediately
#   - With CoW, data is only copied when actually modified
#
# Impact:
#   - ALL .copy() operations throughout the application are optimized
#   - No code changes needed in individual modules
#   - See: docs/COPY_ON_WRITE_OPTIMIZATION.md for details
#
# Pandas version requirement: 1.5.0+
# Reference: https://pandas.pydata.org/docs/user_guide/copy_on_write.html
pd.options.mode.copy_on_write = True
