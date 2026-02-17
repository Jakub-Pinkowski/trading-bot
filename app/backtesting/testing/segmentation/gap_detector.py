"""
Gap detection for identifying continuous periods in gapped data.

This module detects large time gaps in historical data (e.g., 5m, 15m intervals)
and splits the data into continuous periods for clean backtesting.
"""

import pandas as pd

from app.utils.logger import get_logger

logger = get_logger('backtesting/testing/segmentation/gap_detector')

# ==================== Constants ====================

# Interval-specific gap thresholds (in number of intervals)
# All intervals use a consistent 4-day threshold to detect significant data gaps
# while avoiding normal market closures (weekends)
GAP_THRESHOLDS = {
    '3m': 1920,  # 1920 * 3min = 5760min = 4.0 days
    '5m': 1152,  # 1152 * 5min = 5760min = 4.0 days
    '15m': 384,  # 384 * 15min = 5760min = 4.0 days
    '30m': 192,  # 192 * 30min = 5760min = 4.0 days
    '45m': 128,  # 128 * 45min = 5760min = 4.0 days
    '1h': 96,  # 96 * 60min = 5760min = 4.0 days
    '2h': 48,  # 48 * 120min = 5760min = 4.0 days
    '3h': 32,  # 32 * 180min = 5760min = 4.0 days
    '4h': 24,  # 24 * 240min = 5760min = 4.0 days
    '1d': 4,  # 4 * 1440min = 5760min = 4.0 days
}

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
        'row_count': len(period_df)
    }


def _parse_interval_to_minutes(interval):
    """
    Parse interval string to minutes.

    Args:
        interval: Interval string (e.g., '5m', '15m', '1h', '4h', '1d')

    Returns:
        Number of minutes as integer

    Raises:
        ValueError: If an interval format is invalid or value is non-positive
    """
    if not interval or not isinstance(interval, str):
        raise ValueError(f"Interval must be a non-empty string, got: {interval}")

    interval = interval.lower().strip()

    if not interval:
        raise ValueError("Interval cannot be empty after stripping whitespace")

    try:
        if interval.endswith('m'):
            minutes = int(interval[:-1])
        elif interval.endswith('h'):
            minutes = int(interval[:-1]) * 60
        elif interval.endswith('d'):
            minutes = int(interval[:-1]) * 60 * 24
        else:
            raise ValueError(f"Unsupported interval format: '{interval}'. Must end with 'm', 'h', or 'd'")

        if minutes <= 0:
            raise ValueError(f"Interval must be positive, got: {interval}")

        return minutes

    except (ValueError, AttributeError) as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid interval format: '{interval}'. Expected format like '5m', '1h', '1d'")
        raise


def _get_smart_gap_threshold(interval):
    """
    Get interval-specific gap threshold.

    All intervals use consistent 4-day threshold to detect significant data gaps
    while avoiding normal market closures (weekends):

    - 3m: 1920x = 4.0 days
    - 5m: 1152x = 4.0 days
    - 15m: 384x = 4.0 days
    - 30m: 192x = 4.0 days
    - 45m: 128x = 4.0 days
    - 1h: 96x = 4.0 days
    - 2h: 48x = 4.0 days
    - 3h: 32x = 4.0 days
    - 4h: 24x = 4.0 days
    - 1d: 4x = 4.0 days

    Supported intervals: 3m, 5m, 15m, 30m, 45m, 1h, 2h, 3h, 4h, 1d

    Args:
        interval: Interval string (e.g., '3m', '5m', '1h', '1d')

    Returns:
        Threshold multiplier appropriate for the interval

    Raises:
        ValueError: If interval is not supported
    """
    if interval not in GAP_THRESHOLDS:
        supported = ', '.join(sorted(GAP_THRESHOLDS.keys()))
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Supported intervals: {supported}"
        )

    return GAP_THRESHOLDS[interval]


def _calculate_gap_threshold(interval, threshold_multiplier):
    """
    Calculate a time threshold for gap detection.

    Args:
        interval: Interval string (e.g., '5m')
        threshold_multiplier: Multiplier for expected interval (e.g., 100)

    Returns:
        pd.Timedelta representing a gap threshold
    """
    interval_minutes = _parse_interval_to_minutes(interval)
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
            'row_count': int
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

    # Find gap indices where the time difference exceeds a threshold
    gaps_mask = time_diffs > threshold
    gap_indices = time_diffs[gaps_mask].index

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
