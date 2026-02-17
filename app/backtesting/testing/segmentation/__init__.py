"""
Segmentation module for period-aware data splitting.

This module provides functionality for detecting continuous periods in gapped data
(e.g., 5m, 15m intervals) and splitting data into segments for backtesting.
"""

from app.backtesting.testing.segmentation.gap_detector import detect_periods
from app.backtesting.testing.segmentation.period_splitter import split_all_periods

__all__ = [
    'detect_periods',
    'split_all_periods',
]
