"""
Segmentation module for period-aware data splitting.

This module provides functionality for detecting continuous periods in gapped data
(e.g., 5m, 15m intervals) and splitting data into segments for backtesting.
"""

from .gap_detector import detect_periods, parse_interval_to_minutes
from .period_splitter import (
    split_period_equal_rows,
    split_all_periods,
    split_equal_segments_across_periods
)

__all__ = [
    'detect_periods',
    'parse_interval_to_minutes',
    'split_period_equal_rows',
    'split_all_periods',
    'split_equal_segments_across_periods',
]
