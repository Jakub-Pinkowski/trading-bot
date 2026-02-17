"""
Gap detection for identifying continuous periods in gapped data.

This module detects large time gaps in historical data (e.g., 5m, 15m intervals)
and splits the data into continuous periods for clean backtesting.
"""

import pandas as pd

from app.utils.logger import get_logger

logger = get_logger('backtesting/testing/segmentation/gap_detector')

# ==================== Constants ====================

# Smarter default thresholds based on interval
# For 5m/15m: Medium-large threshold to handle week-long gaps but split on month-long gaps
# For other intraday: Moderate threshold for weekends
# For daily+: Small threshold
DEFAULT_GAP_THRESHOLD_5M_15M = 1000  # ~3.5 days for 5m, ~10.4 days for 15m (detects week-long gaps)
DEFAULT_GAP_THRESHOLD_SUBDAILY = 500  # ~2 days for 30m/1h (covers weekends)
DEFAULT_GAP_THRESHOLD_DAILY = 10  # ~10 days for daily data

MIN_PERIOD_ROWS = 1000


# ==================== Helper Functions ====================

def _create_period_dict(period_id, period_df):
    """
    Create a standardized period dictionary.

    Args:
        period_id: Unique period identifier
        period_df: DataFrame slice for this period

    Returns:
        Dict with period metadata
    """
    return {
        'period_id': period_id,
        'df': period_df,
        'start_date': period_df.index[0],
        'end_date': period_df.index[-1],
        'row_count': len(period_df),
        'is_continuous': True
    }


def parse_interval_to_minutes(interval):
    """
    Parse interval string to minutes.

    Args:
        interval: Interval string (e.g., '5m', '15m', '1h', '4h', '1d')

    Returns:
        Number of minutes as integer
    """
    interval = interval.lower().strip()

    if interval.endswith('m'):
        return int(interval[:-1])
    elif interval.endswith('h'):
        return int(interval[:-1]) * 60
    elif interval.endswith('d'):
        return int(interval[:-1]) * 60 * 24
    else:
        raise ValueError(f"Unsupported interval format: {interval}")


def _get_smart_gap_threshold(interval):
    """
    Get an intelligent gap threshold based on an interval.

    For 5m/15m data: Use a very large threshold (month-long gaps are normal in your data)
    For other intraday: Use a moderate threshold to avoid splitting on weekends
    For daily+ data: Use a smaller threshold to detect actual data gaps

    Args:
        interval: Interval string (e.g., '5m', '1h', '1d')

    Returns:
        Threshold multiplier appropriate for the interval
    """
    interval_minutes = parse_interval_to_minutes(interval)

    # Special handling for 5m and 15m (very gappy data)
    if interval_minutes == 5 or interval_minutes == 15:
        # For 5m: 1000 * 5 = 5000 min = ~3.5 days
        # For 15m: 1000 * 15 = 15,000 min = ~10.4 days
        return DEFAULT_GAP_THRESHOLD_5M_15M
    elif interval_minutes < 1440:
        # For other intraday (30m, 1h, 2h, 4h): 500x covers weekends
        return DEFAULT_GAP_THRESHOLD_SUBDAILY
    else:
        # For daily data, a smaller threshold is fine
        return DEFAULT_GAP_THRESHOLD_DAILY


def _calculate_gap_threshold(interval, threshold_multiplier):
    """
    Calculate a time threshold for gap detection.

    Args:
        interval: Interval string (e.g., '5m')
        threshold_multiplier: Multiplier for expected interval (e.g., 100)

    Returns:
        pd.Timedelta representing a gap threshold
    """
    interval_minutes = parse_interval_to_minutes(interval)
    expected_delta = pd.Timedelta(minutes=interval_minutes)
    return expected_delta * threshold_multiplier


# ==================== Main Detection ====================

def detect_periods(df, interval, gap_threshold=None, min_rows=MIN_PERIOD_ROWS):
    """
    Detect continuous periods in DataFrame by identifying large time gaps.

    Example for ZS1! 5m data with 3 periods:
    - Period 1: 2025-04-14 to 2025-05-20 (5443 rows)
    - Period 2: 2025-05-27 to 2025-07-03 (5618 rows)
    - Period 3: 2025-12-01 to 2026-02-13 (10,540 rows)

    Args:
        df: DataFrame with DatetimeIndex
        interval: Time interval (e.g., '5m', '15m')
        gap_threshold: Multiplier for an interval to trigger a new period
                      If None, uses smart default based on interval
        min_rows: Minimum rows required per period (default: 1000)

    Returns:
        List of dicts, each containing:
        {
            'period_id': 1,
            'df': DataFrame slice for this period,
            'start_date': pd.Timestamp,
            'end_date': pd.Timestamp,
            'row_count': int,
            'is_continuous': True
        }
    """
    # Validate input
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return []

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame must have a DatetimeIndex")
        raise ValueError("DataFrame must have a DatetimeIndex")

    # Use smart default if no threshold provided
    if gap_threshold is None:
        gap_threshold = _get_smart_gap_threshold(interval)

    # Calculate time differences between consecutive rows
    time_diffs = df.index.to_series().diff()

    # Get a gap threshold
    threshold = _calculate_gap_threshold(interval, gap_threshold)

    logger.info(f"Gap detection for {interval}: threshold = {gap_threshold}x interval = {threshold}")

    # Find gap indices
    gap_indices = time_diffs[time_diffs > threshold].index

    if len(gap_indices) == 0:
        # No gaps - single continuous period
        logger.info(f"No gaps detected in {interval} data - single continuous period")
        return [_create_period_dict(1, df)]

    # Split at gaps
    periods = []
    start_idx = 0
    period_id = 1

    for gap_start_time in gap_indices:
        # Get the row location where the gap starts
        gap_location = df.index.get_loc(gap_start_time)
        period_df = df.iloc[start_idx:gap_location].copy()

        if len(period_df) >= min_rows:
            periods.append(_create_period_dict(period_id, period_df))
            period_id += 1
        else:
            logger.warning(
                f"Skipping small period: {len(period_df)} rows "
                f"(min: {min_rows})"
            )

        start_idx = gap_location

    # Add a final period
    final_period_df = df.iloc[start_idx:].copy()
    if len(final_period_df) >= min_rows:
        periods.append(_create_period_dict(period_id, final_period_df))
    else:
        logger.warning(
            f"Skipping final small period: {len(final_period_df)} rows "
            f"(min: {min_rows})"
        )

    # Check if we found any valid periods
    if not periods:
        logger.warning(
            f"No valid periods found with min_rows={min_rows}. "
            f"All periods were too small."
        )
        return []

    logger.info(f"Detected {len(periods)} continuous period(s) in {interval} data")

    for period in periods:
        logger.info(
            f"  Period {period['period_id']}: "
            f"{period['start_date']} to {period['end_date']} "
            f"({period['row_count']} rows)"
        )

    return periods
