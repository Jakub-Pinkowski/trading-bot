"""
Period splitting for equal-row segmentation within continuous periods.

This module splits continuous data periods into equal-row segments for backtesting.

# ==================== Period vs Segment ====================

PERIOD:
    A continuous block of data WITHOUT gaps (missing timestamps).
    Periods are detected automatically by gap_detector.py based on the data's time intervals.

    Example - ZC 5m data:
        Period 1: 2025-04-14 to 2025-05-20 (5,177 rows)  ← Continuous data
        [GAP: 6 days of missing data]
        Period 2: 2025-05-27 to 2025-07-03 (5,413 rows)  ← Continuous data
        [GAP: 5 months of missing data]
        Period 3: 2025-12-01 to 2026-02-13 (10,178 rows) ← Continuous data

    Why gaps exist:
    - Market closures (weekends, holidays)
    - Data collection issues
    - Contract rollovers in futures
    - Low-frequency intervals naturally have more continuity (30m, 1h, 2h)

SEGMENT:
    A subdivision of a period used for train/test splitting in backtesting.
    Segments are created by this module (period_splitter.py) based on user preferences.
    Segments NEVER cross period boundaries (gaps).

    Example - Split Period 3 above into 2 segments:
        Segment 3: 2025-12-01 to 2026-01-08 (5,089 rows)  ← First half of Period 3
        Segment 4: 2026-01-08 to 2026-02-13 (5,089 rows)  ← Second half of Period 3

SEGMENTATION MODES:

1. segments_per_period (fixed segments per period):
   - Each period gets the same number of segments
   - Example: 3 periods × 4 segments = 12 total segments
   - Good for: Understanding patterns within each period
   - Bad for: Comparing across periods (segments have different sizes)

2. segments_per_period=1 (each period is one segment):
   - Each period becomes a single segment
   - Example: 3 periods × 1 segment = 3 total segments
   - Good for: Treating periods as distinct market regimes
   - Bad for: Balanced splits when periods have very different sizes

3. total_segments (proportional distribution):
   - Distributes segments across periods based on their size
   - Example: Period 1 (25%) gets 1 segment, Period 2 (26%) gets 1 segment, Period 3 (49%) gets 2 segments
   - Good for: Balanced train/test splits with equal-sized segments
   - Good for: Fair comparison across segments
   - RECOMMENDED for strategy testing workflows
"""

from app.utils.logger import get_logger

logger = get_logger('backtesting/testing/segmentation/period_splitter')

# ==================== Constants ====================

DEFAULT_SEGMENT_COUNT = 4
MIN_SEGMENT_ROWS = 1000


# ==================== Helper Functions ====================

def _create_segment_dict(segment_id, period_id, segment_df):
    """
    Create a standardized segment dictionary.

    Args:
        segment_id: Unique segment identifier
        period_id: Parent period identifier
        segment_df: DataFrame slice for this segment

    Returns:
        Dict with segment metadata
    """
    return {
        'segment_id': segment_id,
        'period_id': period_id,
        'df': segment_df,
        'start_date': segment_df.index[0],
        'end_date': segment_df.index[-1],
        'row_count': len(segment_df)
    }


def _validate_segment_count(segment_count, period_rows):
    """
    Validate that segment count is possible for period size.

    Args:
        segment_count: Desired number of segments
        period_rows: Total rows in a period

    Returns:
        Boolean indicating if segmentation is valid
    """
    rows_per_segment = period_rows // segment_count
    return rows_per_segment >= MIN_SEGMENT_ROWS


# ==================== Main Splitting ====================

def split_period_equal_rows(period_dict, segments_per_period=DEFAULT_SEGMENT_COUNT):
    """
    Split a period into equal-row segments.

    Args:
        period_dict: Dict from detect_periods() containing period info
        segments_per_period: Number of segments to create per period (default: 4)
                            Set to 1 to treat each period as a single segment

    Returns:
        List of segment dicts, each containing:
        {
            'segment_id': 1,
            'period_id': 1,
            'df': DataFrame slice,
            'start_date': pd.Timestamp,
            'end_date': pd.Timestamp,
            'row_count': int
        }
    """
    period_df = period_dict['df']
    period_id = period_dict['period_id']
    total_rows = len(period_df)

    # If segments_per_period is 1, return the entire period as one segment
    if segments_per_period == 1:
        return [_create_segment_dict(1, period_id, period_df)]

    rows_per_segment = total_rows // segments_per_period

    # Validate
    if not _validate_segment_count(segments_per_period, total_rows):
        logger.warning(
            f"Period {period_id} has only {total_rows} rows, "
            f"segments will be smaller than {MIN_SEGMENT_ROWS}"
        )

    segments = []

    for i in range(segments_per_period):
        row_start = i * rows_per_segment

        # Last segment gets remaining rows
        if i == segments_per_period - 1:
            row_end = total_rows
        else:
            row_end = (i + 1) * rows_per_segment

        segment_df = period_df.iloc[row_start:row_end]
        segments.append(_create_segment_dict(i + 1, period_id, segment_df))

    return segments


def split_all_periods(periods_list, segments_per_period=DEFAULT_SEGMENT_COUNT, total_segments=None):
    """
    Split all detected periods into segments.

    Args:
        periods_list: List of period dicts from detect_periods()
        segments_per_period: Number of segments per period (default: 4)
                           Set to 1 to treat each period as a single segment
                           Ignored if total_segments is specified
        total_segments: Total number of segments distributed proportionally across ALL periods
                       If specified, overrides segments_per_period
                       Segments respect period boundaries (never cross gaps)

    Returns:
        List of all segment dicts across all periods
    """
    # If total_segments is specified, use equal-row splitting across all periods
    if total_segments is not None:
        return split_equal_segments_across_periods(periods_list, total_segments)

    # Otherwise, split each period individually
    all_segments = []

    for period in periods_list:
        period_segments = split_period_equal_rows(period, segments_per_period)
        all_segments.extend(period_segments)

    logger.info(
        f"Split {len(periods_list)} period(s) into "
        f"{len(all_segments)} total segments "
        f"({segments_per_period} segments per period)"
    )

    return all_segments


def split_equal_segments_across_periods(periods_list, total_segments):
    """
    Distribute segments proportionally across periods without crossing gaps.

    Allocates segments to periods based on their relative size, then splits
    each period into its allocated number of segments. Segments NEVER span
    across period boundaries (gaps).

    Example:
        Period 1: 5,177 rows (25%)  → gets 1 segment
        Period 2: 5,413 rows (26%)  → gets 1 segment
        Period 3: 10,178 rows (49%) → gets 2 segments
        Total: 20,768 rows → 4 total segments

        Result:
        - Segment 1: Period 1 (5,177 rows)
        - Segment 2: Period 2 (5,413 rows)
        - Segment 3: Period 3 first half (5,089 rows)
        - Segment 4: Period 3 second half (5,089 rows)

    Args:
        periods_list: List of period dicts from detect_periods()
        total_segments: Total number of segments to create

    Returns:
        List of segment dicts
    """
    if not periods_list:
        logger.warning("No periods provided for segmentation")
        return []

    if total_segments <= 0:
        logger.warning(f"total_segments must be positive, got {total_segments}")
        return []

    # If only 1 period, no need for proportional allocation
    if len(periods_list) == 1:
        logger.info(
            f"Single period detected, creating {total_segments} equal segments "
            f"(no gaps to worry about)"
        )
        return split_period_equal_rows(periods_list[0], total_segments)

    if total_segments < len(periods_list):
        logger.warning(
            f"total_segments ({total_segments}) is less than number of periods ({len(periods_list)}). "
            f"Some periods will not have any segments."
        )

    # Calculate total rows and proportional allocation
    total_rows = sum(len(p['df']) for p in periods_list)

    # Allocate segments proportionally to period size
    allocations = []
    segments_allocated = 0

    for i, period in enumerate(periods_list):
        period_rows = len(period['df'])
        period_proportion = period_rows / total_rows

        # Allocate segments proportionally
        if i == len(periods_list) - 1:
            # Last period gets remaining segments
            segments_for_period = total_segments - segments_allocated
        else:
            # Round to nearest integer, minimum 1 if period is large enough
            segments_for_period = max(1, round(period_proportion * total_segments))
            segments_for_period = min(segments_for_period, total_segments - segments_allocated)

        allocations.append({
            'period': period,
            'segments': segments_for_period,
            'rows': period_rows,
            'proportion': period_proportion
        })
        segments_allocated += segments_for_period

    logger.info(
        f"Creating {total_segments} segments from {len(periods_list)} period(s), "
        f"{total_rows:,} total rows"
    )

    # Log allocation
    for allocation in allocations:
        logger.info(
            f"  Period {allocation['period']['period_id']}: {allocation['rows']:,} rows "
            f"({allocation['proportion'] * 100:.0f}%) → {allocation['segments']} segment(s)"
        )

    # Split each period into its allocated segments
    all_segments = []
    global_segment_id = 1

    for allocation in allocations:
        period = allocation['period']
        segments_for_period = allocation['segments']

        # Skip if no segments allocated
        if segments_for_period == 0:
            continue

        # Split this period into equal segments
        period_segments = split_period_equal_rows(period, segments_for_period)

        # Update segment IDs to be global
        for segment in period_segments:
            segment['segment_id'] = global_segment_id
            global_segment_id += 1
            all_segments.append(segment)

    logger.info(f"Created {len(all_segments)} segments")
    return all_segments
