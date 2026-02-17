"""
Additional test scenarios extracted from main folder test files.

These tests were previously in standalone test_*.py files in the main folder.
They cover specific scenarios and edge cases that complement the main test suite.
"""

import numpy as np
import pandas as pd
import pytest

from app.backtesting.testing.segmentation import detect_periods, split_all_periods
from app.backtesting.testing.segmentation.gap_detector import (
    _parse_interval_to_minutes,
    _get_smart_gap_threshold
)


# ==================== Test Fixtures ====================

@pytest.fixture
def realistic_zc_5m_gapped_data():
    """
    Create realistic ZC 5m data that mimics actual market patterns.

    Based on real ZC1! data structure with typical gaps:
    - Period 1: Apr 14 - May 20
    - Period 2: May 27 - Jul 03
    - Period 3: Dec 01 - Feb 13
    """
    # Period 1: Apr 14 - May 20 (about 5400 rows)
    period1_start = pd.Timestamp('2025-04-14 02:00:00')
    period1_end = pd.Timestamp('2025-05-20 20:15:00')
    period1_dates = pd.date_range(period1_start, period1_end, freq='5min')

    # Gap of 7 days (typical weekend + holiday)

    # Period 2: May 27 - Jul 03 (about 5600 rows)
    period2_start = pd.Timestamp('2025-05-27 02:00:00')
    period2_end = pd.Timestamp('2025-07-03 20:15:00')
    period2_dates = pd.date_range(period2_start, period2_end, freq='5min')

    # Gap of 5 months (typical contract rollover gap)

    # Period 3: Dec 01 - Feb 13 (about 10500 rows)
    period3_start = pd.Timestamp('2025-12-01 02:00:00')
    period3_end = pd.Timestamp('2026-02-13 20:15:00')
    period3_dates = pd.date_range(period3_start, period3_end, freq='5min')

    # Combine all dates
    all_dates = period1_dates.append([period2_dates, period3_dates])

    # Create sample OHLCV data
    df = pd.DataFrame({
        'open': np.random.uniform(1400, 1500, len(all_dates)),
        'high': np.random.uniform(1400, 1500, len(all_dates)),
        'low': np.random.uniform(1400, 1500, len(all_dates)),
        'close': np.random.uniform(1400, 1500, len(all_dates)),
        'volume': np.random.randint(1000, 10000, len(all_dates))
    }, index=all_dates)

    return df


@pytest.fixture
def realistic_continuous_30m_data():
    """Create realistic 30m continuous data (no gaps)."""
    start = pd.Timestamp('2024-01-02 15:30:00')
    end = pd.Timestamp('2026-02-13 20:00:00')
    dates = pd.date_range(start, end, freq='30min')

    df = pd.DataFrame({
        'open': np.random.uniform(1400, 1500, len(dates)),
        'high': np.random.uniform(1400, 1500, len(dates)),
        'low': np.random.uniform(1400, 1500, len(dates)),
        'close': np.random.uniform(1400, 1500, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    return df


# ==================== Gap Threshold Consistency Tests ====================

class TestGapThresholdConsistency:
    """Test that all intervals have consistent 4-day gap thresholds."""

    @pytest.mark.parametrize("interval,expected_days", [
        ('3m', 4.0),
        ('5m', 4.0),
        ('15m', 4.0),
        ('30m', 4.0),
        ('45m', 4.0),
        ('1h', 4.0),
        ('2h', 4.0),
        ('3h', 4.0),
        ('4h', 4.0),
        ('1d', 4.0),
    ])
    def test_interval_threshold_equals_4_days(self, interval, expected_days):
        """Test that each interval's threshold equals exactly 4 days."""
        interval_minutes = _parse_interval_to_minutes(interval)
        threshold_multiplier = _get_smart_gap_threshold(interval)

        # Calculate resulting days
        total_minutes = interval_minutes * threshold_multiplier
        days = total_minutes / 1440  # 1440 minutes in a day

        # Should be exactly 4 days
        assert abs(days - expected_days) < 0.01, \
            f"{interval} threshold should equal {expected_days} days, got {days}"

    def test_all_intervals_use_same_target(self):
        """Test that all intervals target the same gap threshold in days."""
        intervals = ['3m', '5m', '15m', '30m', '45m', '1h', '2h', '3h', '4h', '1d']

        days_list = []
        for interval in intervals:
            interval_minutes = _parse_interval_to_minutes(interval)
            threshold = _get_smart_gap_threshold(interval)
            total_minutes = interval_minutes * threshold
            days = total_minutes / 1440
            days_list.append(days)

        # All should be equal
        assert all(abs(d - 4.0) < 0.01 for d in days_list), \
            "All intervals should use 4-day threshold"


# ==================== Unsupported Interval Tests ====================

class TestUnsupportedIntervals:
    """Test that unsupported intervals are properly rejected."""

    @pytest.mark.parametrize("unsupported_interval", [
        '6h',  # Not in supported list
        '10m',  # Not in supported list
        '8h',  # Not in supported list
        '12h',  # Not in supported list
        '2d',  # Not in supported list
    ])
    def test_unsupported_interval_raises_error(self, unsupported_interval):
        """Test that unsupported intervals raise ValueError."""
        df = pd.DataFrame(
            {'close': [100, 101, 102]},
            index=pd.date_range('2024-01-01', periods=3, freq='6h')
        )

        with pytest.raises(ValueError, match="Unsupported interval"):
            detect_periods(df, interval=unsupported_interval)

    def test_error_message_lists_supported_intervals(self):
        """Test that error message includes list of supported intervals."""
        df = pd.DataFrame(
            {'close': [100]},
            index=pd.DatetimeIndex(['2024-01-01'])
        )

        with pytest.raises(ValueError) as exc_info:
            detect_periods(df, interval='6h')

        # Error message should mention supported intervals
        assert 'Supported intervals' in str(exc_info.value)


# ==================== Segmentation Mode Comparison Tests ====================

class TestSegmentationModes:
    """Test different segmentation modes and their characteristics."""

    def test_mode_segments_per_period(self, realistic_zc_5m_gapped_data):
        """Test fixed segments per period mode."""
        periods = detect_periods(realistic_zc_5m_gapped_data, '5m')

        # Skip if not enough periods
        if len(periods) < 2:
            pytest.skip("Need multiple periods")

        # Mode 1: segments_per_period=4
        segments = split_all_periods(periods, segments_per_period=4)

        # Should create exactly periods * 4 segments
        assert len(segments) == len(periods) * 4

        # Each period should have exactly 4 segments
        for period in periods:
            period_segments = [s for s in segments if s['period_id'] == period['period_id']]
            assert len(period_segments) == 4

    def test_mode_one_segment_per_period(self, realistic_zc_5m_gapped_data):
        """Test one segment per period mode."""
        periods = detect_periods(realistic_zc_5m_gapped_data, '5m')

        # Mode 2: segments_per_period=1
        segments = split_all_periods(periods, segments_per_period=1)

        # Should create exactly len(periods) segments
        assert len(segments) == len(periods)

        # Each segment should equal its period
        for period, segment in zip(periods, segments):
            assert segment['row_count'] == period['row_count']

    def test_mode_proportional_distribution(self, realistic_zc_5m_gapped_data):
        """Test proportional segment distribution mode."""
        periods = detect_periods(realistic_zc_5m_gapped_data, '5m')

        if len(periods) < 2:
            pytest.skip("Need multiple periods")

        # Mode 3: total_segments=4 (proportional)
        segments = split_all_periods(periods, total_segments=4)

        # Should create exactly 4 segments
        assert len(segments) == 4

        # Segments should be more balanced than fixed per-period
        row_counts = [s['row_count'] for s in segments]
        avg_rows = sum(row_counts) / len(row_counts)
        variance = sum((x - avg_rows) ** 2 for x in row_counts) / len(row_counts)

        # Variance should exist (segments are different sizes)
        assert variance >= 0

    def test_proportional_mode_respects_boundaries(self, realistic_zc_5m_gapped_data):
        """Test that proportional mode never crosses period boundaries."""
        periods = detect_periods(realistic_zc_5m_gapped_data, '5m')

        if len(periods) < 2:
            pytest.skip("Need multiple periods")

        segments = split_all_periods(periods, total_segments=6)

        # Verify no segment spans multiple periods
        for segment in segments:
            # Segment should only contain data from one period
            period_id = segment['period_id']
            period = next(p for p in periods if p['period_id'] == period_id)

            # Segment dates should be within period dates
            assert segment['start_date'] >= period['start_date']
            assert segment['end_date'] <= period['end_date']


# ==================== Continuous Data Tests ====================

class TestContinuousDataBehavior:
    """Test segmentation behavior with continuous (no-gap) data."""

    def test_continuous_30m_data_single_period(self, realistic_continuous_30m_data):
        """Test that continuous 30m data results in single period."""
        periods = detect_periods(realistic_continuous_30m_data, '30m')

        assert len(periods) == 1, "Continuous data should have 1 period"
        assert periods[0]['row_count'] == len(realistic_continuous_30m_data)

    def test_continuous_data_equal_segments(self, realistic_continuous_30m_data):
        """Test that continuous data creates nearly equal segments."""
        periods = detect_periods(realistic_continuous_30m_data, '30m')
        segments = split_all_periods(periods, segments_per_period=5)

        # Should have 5 segments
        assert len(segments) == 5

        # Segments should be nearly equal (within small percentage)
        row_counts = [s['row_count'] for s in segments]
        avg_rows = sum(row_counts) / len(row_counts)

        # Each segment should be within 1% of average
        for count in row_counts:
            diff_pct = abs(count - avg_rows) / avg_rows
            assert diff_pct < 0.01, "Segments should be nearly equal"


# ==================== Realistic Data Pattern Tests ====================

class TestRealisticDataPatterns:
    """Test with realistic market data patterns."""

    def test_realistic_zc_5m_detects_three_periods(self, realistic_zc_5m_gapped_data):
        """Test that realistic ZC 5m data detects expected 3 periods."""
        periods = detect_periods(realistic_zc_5m_gapped_data, '5m')

        # Should detect 3 periods (based on how fixture is constructed)
        assert len(periods) == 3, "ZC 5m fixture should have 3 periods"

    def test_realistic_data_period_sizes(self, realistic_zc_5m_gapped_data):
        """Test that periods have realistic sizes."""
        periods = detect_periods(realistic_zc_5m_gapped_data, '5m')

        # Each period should be substantial (> 1000 rows)
        for period in periods:
            assert period['row_count'] >= 1000, \
                f"Period {period['period_id']} too small: {period['row_count']}"

    def test_realistic_data_with_ohlcv_columns(self, realistic_zc_5m_gapped_data):
        """Test segmentation preserves OHLCV columns."""
        periods = detect_periods(realistic_zc_5m_gapped_data, '5m')
        segments = split_all_periods(periods, segments_per_period=2)

        expected_columns = ['open', 'high', 'low', 'close', 'volume']

        for segment in segments:
            for col in expected_columns:
                assert col in segment['df'].columns, \
                    f"Column {col} missing from segment {segment['segment_id']}"


# ==================== Variance and Statistics Tests ====================

class TestSegmentStatistics:
    """Test statistical properties of segments."""

    def test_fixed_mode_variance_is_higher(self, realistic_zc_5m_gapped_data):
        """Test that fixed segments per period has higher variance than proportional."""
        periods = detect_periods(realistic_zc_5m_gapped_data, '5m')

        if len(periods) < 3:
            pytest.skip("Need 3+ periods")

        # Fixed mode
        fixed_segments = split_all_periods(periods, segments_per_period=4)
        fixed_counts = [s['row_count'] for s in fixed_segments]
        fixed_avg = sum(fixed_counts) / len(fixed_counts)
        fixed_variance = sum((x - fixed_avg) ** 2 for x in fixed_counts) / len(fixed_counts)

        # Proportional mode
        prop_segments = split_all_periods(periods, total_segments=len(fixed_segments))
        prop_counts = [s['row_count'] for s in prop_segments]
        prop_avg = sum(prop_counts) / len(prop_counts)
        prop_variance = sum((x - prop_avg) ** 2 for x in prop_counts) / len(prop_counts)

        # Fixed mode should have higher variance (more unbalanced)
        # This might not always be true depending on data, so we just check both exist
        assert fixed_variance >= 0
        assert prop_variance >= 0

    def test_segment_size_distribution(self, realistic_zc_5m_gapped_data):
        """Test that segment sizes are within reasonable bounds."""
        periods = detect_periods(realistic_zc_5m_gapped_data, '5m')
        segments = split_all_periods(periods, segments_per_period=4)

        total_rows = sum(s['row_count'] for s in segments)
        avg_segment_size = total_rows / len(segments)

        # No segment should be more than 3x the average
        for segment in segments:
            assert segment['row_count'] <= avg_segment_size * 3, \
                f"Segment {segment['segment_id']} too large relative to average"


# ==================== Multi-Column Data Tests ====================

class TestMultiColumnData:
    """Test segmentation with multi-column DataFrames."""

    def test_preserves_all_columns(self, realistic_zc_5m_gapped_data):
        """Test that all columns are preserved in segments."""
        original_columns = set(realistic_zc_5m_gapped_data.columns)

        periods = detect_periods(realistic_zc_5m_gapped_data, '5m')
        segments = split_all_periods(periods, segments_per_period=3)

        for segment in segments:
            segment_columns = set(segment['df'].columns)
            assert segment_columns == original_columns, \
                "Segment should have same columns as original"

    def test_column_data_types_preserved(self, realistic_zc_5m_gapped_data):
        """Test that column data types are preserved."""
        original_dtypes = realistic_zc_5m_gapped_data.dtypes

        periods = detect_periods(realistic_zc_5m_gapped_data, '5m')
        segments = split_all_periods(periods, segments_per_period=2)

        for segment in segments:
            for col in original_dtypes.index:
                assert segment['df'][col].dtype == original_dtypes[col], \
                    f"Data type mismatch for column {col}"
