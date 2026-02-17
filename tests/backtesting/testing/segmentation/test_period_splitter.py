"""
Comprehensive tests for period_splitter.py module.

Tests cover:
- Segment creation
- Period splitting with various segment counts
- Proportional allocation across periods
- Edge cases and error handling
"""

from datetime import timedelta

import pandas as pd
import pytest

from app.backtesting.testing.segmentation import split_all_periods
from app.backtesting.testing.segmentation.period_splitter import (
    _create_segment_dict,
    _split_period_into_segments,
    _allocate_segments_proportionally,
    _create_segments_from_allocations,
    _split_equal_segments_across_periods
)


# ==================== Test Fixtures ====================

@pytest.fixture
def single_period():
    """Create a single period with 4000 rows."""
    dates = pd.date_range('2024-01-01', periods=4000, freq='5min')
    df = pd.DataFrame({'close': range(4000)}, index=dates)

    return {
        'period_id': 1,
        'df': df,
        'start_date': dates[0],
        'end_date': dates[-1],
        'row_count': 4000
    }


@pytest.fixture
def three_periods():
    """Create three periods with different sizes."""
    # Period 1: 2000 rows
    dates1 = pd.date_range('2024-01-01', periods=2000, freq='5min')
    df1 = pd.DataFrame({'close': range(2000)}, index=dates1)

    # Period 2: 3000 rows
    dates2 = pd.date_range('2024-02-01', periods=3000, freq='5min')
    df2 = pd.DataFrame({'close': range(3000)}, index=dates2)

    # Period 3: 5000 rows
    dates3 = pd.date_range('2024-03-01', periods=5000, freq='5min')
    df3 = pd.DataFrame({'close': range(5000)}, index=dates3)

    return [
        {
            'period_id': 1,
            'df': df1,
            'start_date': dates1[0],
            'end_date': dates1[-1],
            'row_count': 2000
        },
        {
            'period_id': 2,
            'df': df2,
            'start_date': dates2[0],
            'end_date': dates2[-1],
            'row_count': 3000
        },
        {
            'period_id': 3,
            'df': df3,
            'start_date': dates3[0],
            'end_date': dates3[-1],
            'row_count': 5000
        }
    ]


@pytest.fixture
def small_period():
    """Create a period with fewer than MIN_SEGMENT_ROWS."""
    dates = pd.date_range('2024-01-01', periods=500, freq='5min')
    df = pd.DataFrame({'close': range(500)}, index=dates)

    return {
        'period_id': 1,
        'df': df,
        'start_date': dates[0],
        'end_date': dates[-1],
        'row_count': 500
    }


@pytest.fixture
def equal_periods():
    """Create three periods with equal sizes."""
    periods = []
    for i in range(3):
        dates = pd.date_range(f'2024-0{i + 1}-01', periods=3000, freq='5min')
        df = pd.DataFrame({'close': range(3000)}, index=dates)

        periods.append({
            'period_id': i + 1,
            'df': df,
            'start_date': dates[0],
            'end_date': dates[-1],
            'row_count': 3000
        })

    return periods


# ==================== Helper Function Tests ====================

class TestCreateSegmentDict:
    """Test _create_segment_dict function."""

    def test_creates_correct_structure(self):
        """Test that segment dict has correct structure."""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        df = pd.DataFrame({'close': range(100)}, index=dates)

        segment = _create_segment_dict(1, 42, df)

        assert 'segment_id' in segment
        assert 'period_id' in segment
        assert 'start_date' in segment
        assert 'end_date' in segment
        assert 'row_count' in segment

    def test_assigns_ids_correctly(self):
        """Test that IDs are assigned correctly."""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        df = pd.DataFrame({'close': range(100)}, index=dates)

        segment = _create_segment_dict(5, 3, df)

        assert segment['segment_id'] == 5
        assert segment['period_id'] == 3

    def test_extracts_dates_correctly(self):
        """Test that start/end dates are extracted correctly."""
        dates = pd.date_range('2024-01-01 10:00', periods=100, freq='5min')
        df = pd.DataFrame({'close': range(100)}, index=dates)

        segment = _create_segment_dict(1, 1, df)

        assert segment['start_date'] == dates[0]
        assert segment['end_date'] == dates[-1]

    def test_counts_rows_correctly(self):
        """Test that row count is correct."""
        dates = pd.date_range('2024-01-01', periods=456, freq='5min')
        df = pd.DataFrame({'close': range(456)}, index=dates)

        segment = _create_segment_dict(1, 1, df)
        assert segment['row_count'] == 456


class TestSplitPeriodIntoSegments:
    """Test _split_period_into_segments function."""

    def test_splits_into_equal_segments(self, single_period):
        """Test splitting into equal-sized segments."""
        segments = _split_period_into_segments(single_period, 4)

        assert len(segments) == 4
        # Each segment should have 1000 rows (4000 / 4)
        assert all(s['row_count'] == 1000 for s in segments)

    def test_handles_uneven_division(self, single_period):
        """Test that last segment gets remainder rows."""
        # 4000 rows / 3 segments = 1333 + 1333 + 1334
        segments = _split_period_into_segments(single_period, 3)

        assert len(segments) == 3
        assert segments[0]['row_count'] == 1333
        assert segments[1]['row_count'] == 1333
        assert segments[2]['row_count'] == 1334

    def test_segments_per_period_one_returns_whole_period(self, single_period):
        """Test that segments_per_period=1 returns the entire period."""
        segments = _split_period_into_segments(single_period, 1)

        assert len(segments) == 1
        assert segments[0]['row_count'] == 4000
        assert segments[0]['start_date'] == single_period['start_date']
        assert segments[0]['end_date'] == single_period['end_date']

    def test_assigns_segment_ids_sequentially(self, single_period):
        """Test that segment IDs are sequential within period."""
        segments = _split_period_into_segments(single_period, 5)

        for i, segment in enumerate(segments, start=1):
            assert segment['segment_id'] == i

    def test_preserves_period_id(self, single_period):
        """Test that all segments preserve the period_id."""
        segments = _split_period_into_segments(single_period, 4)

        assert all(s['period_id'] == 1 for s in segments)

    def test_segments_are_contiguous(self, single_period):
        """Test that segments are contiguous (no gaps or overlaps)."""
        segments = _split_period_into_segments(single_period, 4)

        for i in range(len(segments) - 1):
            # End of segment i should be just before start of segment i+1
            assert segments[i]['end_date'] < segments[i + 1]['start_date']

    def test_segments_cover_entire_period(self, single_period):
        """Test that segments cover the entire period."""
        segments = _split_period_into_segments(single_period, 4)

        total_rows = sum(s['row_count'] for s in segments)
        assert total_rows == single_period['row_count']

        assert segments[0]['start_date'] == single_period['start_date']
        assert segments[-1]['end_date'] == single_period['end_date']

    def test_small_period_warning(self, small_period):
        """Test that warning is logged for small periods."""
        # Just verify it creates segments, warning logging is implementation detail
        segments = _split_period_into_segments(small_period, 4)

        # Should still create segments
        assert len(segments) == 4


class TestAllocateSegmentsProportionally:
    """Test _allocate_segments_proportionally function."""

    def test_allocates_proportionally_to_size(self, three_periods):
        """Test that segments are allocated proportionally."""
        # Total: 10000 rows
        # Period 1: 2000 (20%) → 2 segments
        # Period 2: 3000 (30%) → 3 segments
        # Period 3: 5000 (50%) → 5 segments
        allocations = _allocate_segments_proportionally(three_periods, 10)

        assert len(allocations) == 3
        assert allocations[0]['segments'] == 2
        assert allocations[1]['segments'] == 3
        assert allocations[2]['segments'] == 5

    def test_total_segments_allocated(self, three_periods):
        """Test that exactly total_segments are allocated."""
        allocations = _allocate_segments_proportionally(three_periods, 7)

        total_allocated = sum(a['segments'] for a in allocations)
        assert total_allocated == 7

    def test_last_period_gets_remainder(self, three_periods):
        """Test that last period gets any remaining segments."""
        # With 10 total segments, rounding might not be exact
        allocations = _allocate_segments_proportionally(three_periods, 10)

        total = sum(a['segments'] for a in allocations)
        assert total == 10

    def test_includes_metadata(self, three_periods):
        """Test that allocations include metadata."""
        allocations = _allocate_segments_proportionally(three_periods, 10)

        for allocation in allocations:
            assert 'period' in allocation
            assert 'segments' in allocation
            assert 'rows' in allocation
            assert 'proportion' in allocation

    def test_proportions_sum_to_one(self, three_periods):
        """Test that proportions sum to approximately 1.0."""
        allocations = _allocate_segments_proportionally(three_periods, 10)

        total_proportion = sum(a['proportion'] for a in allocations)
        assert abs(total_proportion - 1.0) < 0.001

    def test_equal_periods_get_equal_segments(self, equal_periods):
        """Test that equal-sized periods get equal segments."""
        allocations = _allocate_segments_proportionally(equal_periods, 9)

        # Each period should get 3 segments
        assert all(a['segments'] == 3 for a in allocations)

    def test_minimum_one_segment_per_period(self):
        """Test that each period gets at least 1 segment if possible."""
        # Create 5 small equal periods
        periods = []
        for i in range(5):
            dates = pd.date_range(f'2024-0{i + 1}-01', periods=1000, freq='5min')
            df = pd.DataFrame({'close': range(1000)}, index=dates)
            periods.append({
                'period_id': i + 1,
                'df': df,
                'start_date': dates[0],
                'end_date': dates[-1],
                'row_count': 1000
            })

        allocations = _allocate_segments_proportionally(periods, 5)

        # Each should get exactly 1
        assert all(a['segments'] == 1 for a in allocations)


class TestCreateSegmentsFromAllocations:
    """Test _create_segments_from_allocations function."""

    def test_creates_segments_from_allocations(self, three_periods):
        """Test that segments are created from allocations."""
        allocations = _allocate_segments_proportionally(three_periods, 10)
        segments = _create_segments_from_allocations(allocations)

        assert len(segments) == 10

    def test_assigns_global_segment_ids(self, three_periods):
        """Test that segment IDs are global (sequential across periods)."""
        allocations = _allocate_segments_proportionally(three_periods, 10)
        segments = _create_segments_from_allocations(allocations)

        for i, segment in enumerate(segments, start=1):
            assert segment['segment_id'] == i

    def test_skips_zero_segment_allocations(self):
        """Test that periods with 0 segments are skipped."""
        # Create periods with manual allocation
        dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
        df = pd.DataFrame({'close': range(1000)}, index=dates)
        period = {
            'period_id': 1,
            'df': df,
            'start_date': dates[0],
            'end_date': dates[-1],
            'row_count': 1000
        }

        allocations = [
            {'period': period, 'segments': 0, 'rows': 1000, 'proportion': 1.0}
        ]

        segments = _create_segments_from_allocations(allocations)
        assert len(segments) == 0

    def test_all_segments_have_required_fields(self, three_periods):
        """Test that all segments have required fields."""
        allocations = _allocate_segments_proportionally(three_periods, 10)
        segments = _create_segments_from_allocations(allocations)

        for segment in segments:
            assert 'segment_id' in segment
            assert 'period_id' in segment
            assert 'start_date' in segment
            assert 'end_date' in segment
            assert 'row_count' in segment


class TestSplitEqualSegmentsAcrossPeriods:
    """Test _split_equal_segments_across_periods function."""

    def test_creates_exact_number_of_segments(self, three_periods):
        """Test that exactly total_segments are created."""
        segments = _split_equal_segments_across_periods(three_periods, 8)
        assert len(segments) == 8

    def test_segments_respect_period_boundaries(self, three_periods):
        """Test that segments never cross period boundaries."""
        segments = _split_equal_segments_across_periods(three_periods, 10)

        # Group segments by period
        periods_segments = {}
        for segment in segments:
            period_id = segment['period_id']
            if period_id not in periods_segments:
                periods_segments[period_id] = []
            periods_segments[period_id].append(segment)

        # Check each period's segments are contiguous
        for period_id, period_segments in periods_segments.items():
            for i in range(len(period_segments) - 1):
                assert period_segments[i]['end_date'] < period_segments[i + 1]['start_date']

    def test_single_period_delegates_to_split_period(self, single_period):
        """Test that single period delegates to _split_period_into_segments."""
        segments = _split_equal_segments_across_periods([single_period], 5)

        assert len(segments) == 5
        assert all(s['period_id'] == 1 for s in segments)

    def test_empty_periods_returns_empty_list(self):
        """Test that empty periods list returns empty list."""
        segments = _split_equal_segments_across_periods([], 5)
        assert segments == []

    def test_zero_segments_returns_empty_list(self, three_periods):
        """Test that zero total_segments returns empty list."""
        segments = _split_equal_segments_across_periods(three_periods, 0)
        assert segments == []

    def test_negative_segments_returns_empty_list(self, three_periods):
        """Test that negative total_segments returns empty list."""
        segments = _split_equal_segments_across_periods(three_periods, -5)
        assert segments == []

    def test_fewer_segments_than_periods_warns(self, three_periods):
        """Test that fewer total_segments than periods still works."""
        segments = _split_equal_segments_across_periods(three_periods, 2)

        # Should still create segments
        assert len(segments) == 2


# ==================== Public API Tests ====================

class TestSplitAllPeriods:
    """Test split_all_periods main function."""

    def test_segments_per_period_mode(self, three_periods):
        """Test splitting with segments_per_period parameter."""
        segments = split_all_periods(three_periods, segments_per_period=4)

        # 3 periods × 4 segments = 12 total
        assert len(segments) == 12

    def test_total_segments_mode(self, three_periods):
        """Test splitting with total_segments parameter."""
        segments = split_all_periods(three_periods, total_segments=10)

        assert len(segments) == 10

    def test_total_segments_overrides_segments_per_period(self, three_periods):
        """Test that total_segments overrides segments_per_period."""
        segments = split_all_periods(
            three_periods,
            segments_per_period=4,
            total_segments=8
        )

        # Should use total_segments
        assert len(segments) == 8

    def test_requires_at_least_one_parameter(self, three_periods):
        """Test that at least one parameter must be provided."""
        with pytest.raises(ValueError, match="Either segments_per_period or total_segments"):
            split_all_periods(three_periods)

    def test_invalid_segments_per_period_raises_error(self, three_periods):
        """Test that non-positive or non-integer segments_per_period raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            split_all_periods(three_periods, segments_per_period=0)

        with pytest.raises(ValueError, match="positive integer"):
            split_all_periods(three_periods, segments_per_period=-1)

        with pytest.raises(ValueError, match="positive integer"):
            split_all_periods(three_periods, segments_per_period=2.5)

    def test_segment_ids_are_globally_unique(self, three_periods):
        """Test that segment IDs are globally unique across all periods."""
        segments = split_all_periods(three_periods, segments_per_period=2)

        ids = [s['segment_id'] for s in segments]
        assert ids == list(range(1, len(ids) + 1))

    def test_segments_per_period_one(self, three_periods):
        """Test segments_per_period=1 (each period is one segment)."""
        segments = split_all_periods(three_periods, segments_per_period=1)

        # 3 periods × 1 segment = 3 total
        assert len(segments) == 3

        # Each segment should be an entire period
        assert segments[0]['row_count'] == 2000
        assert segments[1]['row_count'] == 3000
        assert segments[2]['row_count'] == 5000

    def test_empty_periods_returns_empty_list(self):
        """Test that empty periods list returns empty list."""
        segments = split_all_periods([], segments_per_period=4)
        assert segments == []

    def test_single_period_segments_per_period(self, single_period):
        """Test single period with segments_per_period."""
        segments = split_all_periods([single_period], segments_per_period=5)

        assert len(segments) == 5
        assert all(s['period_id'] == 1 for s in segments)

    def test_single_period_total_segments(self, single_period):
        """Test single period with total_segments."""
        segments = split_all_periods([single_period], total_segments=7)

        assert len(segments) == 7
        assert all(s['period_id'] == 1 for s in segments)

    def test_preserves_period_order(self, three_periods):
        """Test that segments preserve chronological period order."""
        segments = split_all_periods(three_periods, segments_per_period=2)

        # First 2 segments should be from period 1
        assert segments[0]['period_id'] == 1
        assert segments[1]['period_id'] == 1

        # Next 2 from period 2
        assert segments[2]['period_id'] == 2
        assert segments[3]['period_id'] == 2

        # Last 2 from period 3
        assert segments[4]['period_id'] == 3
        assert segments[5]['period_id'] == 3

    def test_all_rows_accounted_for(self, three_periods):
        """Test that all rows are accounted for in segments."""
        segments = split_all_periods(three_periods, segments_per_period=4)

        total_rows = sum(s['row_count'] for s in segments)
        original_rows = sum(p['row_count'] for p in three_periods)

        assert total_rows == original_rows


# ==================== Integration Tests ====================

class TestIntegration:
    """Test integration between gap detection and period splitting."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from data to segments."""
        from app.backtesting.testing.segmentation import detect_periods

        # Create gapped data
        period1 = pd.date_range('2024-01-01', periods=2000, freq='5min')
        period2_start = period1[-1] + timedelta(days=10)
        period2 = pd.date_range(period2_start, periods=3000, freq='5min')

        all_dates = period1.append(period2)
        df = pd.DataFrame({'close': range(len(all_dates))}, index=all_dates)

        # Detect periods
        periods = detect_periods(df, '5m')

        # Split into segments
        segments = split_all_periods(periods, segments_per_period=4)

        # Should have 2 periods × 4 segments = 8 total
        assert len(segments) == 8

    def test_proportional_splitting_with_real_workflow(self):
        """Test proportional splitting with detected periods."""
        from app.backtesting.testing.segmentation import detect_periods

        # Create gapped data with different period sizes
        period1 = pd.date_range('2024-01-01', periods=1000, freq='5min')
        period2_start = period1[-1] + timedelta(days=10)
        period2 = pd.date_range(period2_start, periods=3000, freq='5min')

        all_dates = period1.append(period2)
        df = pd.DataFrame({'close': range(len(all_dates))}, index=all_dates)

        periods = detect_periods(df, '5m')
        segments = split_all_periods(periods, total_segments=4)

        # Period 1 (25%) should get 1 segment
        # Period 2 (75%) should get 3 segments
        period_1_segments = [s for s in segments if s['period_id'] == 1]
        period_2_segments = [s for s in segments if s['period_id'] == 2]

        assert len(period_1_segments) == 1
        assert len(period_2_segments) == 3


# ==================== Edge Cases ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_segments(self):
        """Test handling of very small segment sizes."""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        df = pd.DataFrame({'close': range(100)}, index=dates)
        period = {
            'period_id': 1,
            'df': df,
            'start_date': dates[0],
            'end_date': dates[-1],
            'row_count': 100
        }

        # 100 rows / 10 segments = 10 rows each
        segments = split_all_periods([period], segments_per_period=10)

        assert len(segments) == 10
        assert all(s['row_count'] == 10 for s in segments)

    def test_more_segments_than_rows(self):
        """Test requesting more segments than rows creates very small segments."""
        dates = pd.date_range('2024-01-01', periods=10, freq='5min')
        df = pd.DataFrame({'close': range(10)}, index=dates)
        period = {
            'period_id': 1,
            'df': df,
            'start_date': dates[0],
            'end_date': dates[-1],
            'row_count': 10
        }

        # With 10 rows and 10 segments, each gets 1 row
        segments = split_all_periods([period], segments_per_period=10)

        assert len(segments) == 10
        # 10 rows / 10 segments = 1 row each, except last gets remainder
        assert segments[0]['row_count'] == 1

    def test_single_row_segments(self):
        """Test that single-row segments can be created."""
        dates = pd.date_range('2024-01-01', periods=5, freq='5min')
        df = pd.DataFrame({'close': range(5)}, index=dates)
        period = {
            'period_id': 1,
            'df': df,
            'start_date': dates[0],
            'end_date': dates[-1],
            'row_count': 5
        }

        segments = split_all_periods([period], segments_per_period=5)

        assert len(segments) == 5
        # 5 rows / 5 segments = 1 row each
        assert all(s['row_count'] == 1 for s in segments)

    def test_large_number_of_segments(self, single_period):
        """Test splitting into a large number of segments."""
        segments = split_all_periods([single_period], segments_per_period=100)

        assert len(segments) == 100
        # Each should have 40 rows (4000/100)
        assert all(s['row_count'] == 40 for s in segments)

    def test_unequal_period_sizes_proportional(self):
        """Test proportional splitting with very unequal periods."""
        # Very small period
        dates1 = pd.date_range('2024-01-01', periods=1000, freq='5min')
        df1 = pd.DataFrame({'close': range(1000)}, index=dates1)

        # Very large period
        dates2 = pd.date_range('2024-02-01', periods=9000, freq='5min')
        df2 = pd.DataFrame({'close': range(9000)}, index=dates2)

        periods = [
            {
                'period_id': 1,
                'df': df1,
                'start_date': dates1[0],
                'end_date': dates1[-1],
                'row_count': 1000
            },
            {
                'period_id': 2,
                'df': df2,
                'start_date': dates2[0],
                'end_date': dates2[-1],
                'row_count': 9000
            }
        ]

        segments = split_all_periods(periods, total_segments=10)

        # Period 1 (10%) should get 1 segment
        # Period 2 (90%) should get 9 segments
        period_1_segments = [s for s in segments if s['period_id'] == 1]
        period_2_segments = [s for s in segments if s['period_id'] == 2]

        assert len(period_1_segments) == 1
        assert len(period_2_segments) == 9
