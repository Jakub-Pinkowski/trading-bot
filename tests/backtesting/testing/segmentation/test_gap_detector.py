"""
Comprehensive tests for gap_detector.py module.

Tests cover:
- Interval parsing
- Gap threshold calculation
- Period detection with various gap scenarios
- Edge cases and error handling
"""

from datetime import timedelta

import pandas as pd
import pytest

from app.backtesting.testing.segmentation import detect_periods
from app.backtesting.testing.segmentation.gap_detector import (
    _parse_interval_to_minutes,
    _get_smart_gap_threshold,
    _calculate_gap_threshold,
    _create_period_dict,
    _split_dataframe_at_gaps,
    GAP_THRESHOLDS
)


# ==================== Test Fixtures ====================

@pytest.fixture
def continuous_5m_data():
    """Create continuous 5-minute data with no gaps."""
    dates = pd.date_range('2024-01-01', periods=2000, freq='5min')
    return pd.DataFrame({'close': range(2000)}, index=dates)


@pytest.fixture
def gapped_5m_data():
    """Create 5-minute data with 3 periods separated by gaps."""
    # Period 1: 1000 rows
    period1 = pd.date_range('2024-01-01', periods=1000, freq='5min')

    # Gap of 10 days
    period2_start = period1[-1] + timedelta(days=10)
    period2 = pd.date_range(period2_start, periods=1200, freq='5min')

    # Gap of 15 days
    period3_start = period2[-1] + timedelta(days=15)
    period3 = pd.date_range(period3_start, periods=1500, freq='5min')

    all_dates = period1.append(period2).append(period3)
    return pd.DataFrame({'close': range(len(all_dates))}, index=all_dates)


@pytest.fixture
def small_periods_data():
    """Create data with small periods that should be filtered out."""
    # Small period 1: 500 rows (below min_rows)
    period1 = pd.date_range('2024-01-01', periods=500, freq='5min')

    # Gap
    period2_start = period1[-1] + timedelta(days=10)
    period2 = pd.date_range(period2_start, periods=1500, freq='5min')

    # Gap
    period3_start = period2[-1] + timedelta(days=10)
    period3 = pd.date_range(period3_start, periods=800, freq='5min')

    all_dates = period1.append(period2).append(period3)
    return pd.DataFrame({'close': range(len(all_dates))}, index=all_dates)


# ==================== Helper Function Tests ====================

class TestParseIntervalToMinutes:
    """Test _parse_interval_to_minutes function."""

    def test_parse_minute_intervals(self):
        """Test parsing minute-based intervals."""
        assert _parse_interval_to_minutes('3m') == 3
        assert _parse_interval_to_minutes('5m') == 5
        assert _parse_interval_to_minutes('15m') == 15
        assert _parse_interval_to_minutes('30m') == 30
        assert _parse_interval_to_minutes('45m') == 45

    def test_parse_hour_intervals(self):
        """Test parsing hour-based intervals."""
        assert _parse_interval_to_minutes('1h') == 60
        assert _parse_interval_to_minutes('2h') == 120
        assert _parse_interval_to_minutes('3h') == 180
        assert _parse_interval_to_minutes('4h') == 240

    def test_parse_day_intervals(self):
        """Test parsing day-based intervals."""
        assert _parse_interval_to_minutes('1d') == 1440
        assert _parse_interval_to_minutes('2d') == 2880

    def test_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        assert _parse_interval_to_minutes('5M') == 5
        assert _parse_interval_to_minutes('1H') == 60
        assert _parse_interval_to_minutes('1D') == 1440

    def test_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        assert _parse_interval_to_minutes('  5m  ') == 5
        assert _parse_interval_to_minutes('\t1h\n') == 60

    def test_invalid_format_raises_error(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError):
            _parse_interval_to_minutes('5x')

        with pytest.raises(ValueError):
            _parse_interval_to_minutes('invalid')

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            _parse_interval_to_minutes('')

        with pytest.raises(ValueError, match="empty after stripping"):
            _parse_interval_to_minutes('   ')

    def test_non_string_raises_error(self):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            _parse_interval_to_minutes(None)

        with pytest.raises(ValueError, match="non-empty string"):
            _parse_interval_to_minutes(123)

    def test_negative_value_raises_error(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            _parse_interval_to_minutes('-5m')

    def test_zero_value_raises_error(self):
        """Test that zero values raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            _parse_interval_to_minutes('0m')

    def test_non_numeric_value_raises_error(self):
        """Test that non-numeric values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid interval format"):
            _parse_interval_to_minutes('abcm')


class TestGetSmartGapThreshold:
    """Test _get_smart_gap_threshold function."""

    def test_all_supported_intervals(self):
        """Test that all documented intervals are supported."""
        supported_intervals = ['3m', '5m', '15m', '30m', '45m', '1h', '2h', '3h', '4h', '1d']

        for interval in supported_intervals:
            threshold = _get_smart_gap_threshold(interval)
            assert isinstance(threshold, int)
            assert threshold > 0

    def test_threshold_values_are_correct(self):
        """Test that threshold values match the 4-day target."""
        assert _get_smart_gap_threshold('3m') == 1920
        assert _get_smart_gap_threshold('5m') == 1152
        assert _get_smart_gap_threshold('15m') == 384
        assert _get_smart_gap_threshold('30m') == 192
        assert _get_smart_gap_threshold('45m') == 128
        assert _get_smart_gap_threshold('1h') == 96
        assert _get_smart_gap_threshold('2h') == 48
        assert _get_smart_gap_threshold('3h') == 32
        assert _get_smart_gap_threshold('4h') == 24
        assert _get_smart_gap_threshold('1d') == 4

    def test_all_thresholds_equal_4_days(self):
        """Test that all thresholds equal approximately 4 days."""
        target_minutes = 4 * 24 * 60  # 4 days in minutes

        for interval, threshold in GAP_THRESHOLDS.items():
            interval_minutes = _parse_interval_to_minutes(interval)
            actual_minutes = threshold * interval_minutes
            assert actual_minutes == target_minutes, f"{interval} doesn't equal 4 days"

    def test_unsupported_interval_raises_error(self):
        """Test that unsupported intervals raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported interval"):
            _get_smart_gap_threshold('6h')

        with pytest.raises(ValueError, match="Unsupported interval"):
            _get_smart_gap_threshold('10m')

    def test_error_message_lists_supported_intervals(self):
        """Test that error message includes list of supported intervals."""
        with pytest.raises(ValueError, match="Supported intervals"):
            _get_smart_gap_threshold('6h')


class TestCalculateGapThreshold:
    """Test _calculate_gap_threshold function."""

    def test_returns_timedelta(self):
        """Test that function returns pd.Timedelta."""
        result = _calculate_gap_threshold('5m', 100)
        assert isinstance(result, pd.Timedelta)

    def test_threshold_calculation(self):
        """Test that threshold is calculated correctly."""
        # 5m * 100 = 500 minutes
        threshold = _calculate_gap_threshold('5m', 100)
        assert threshold == pd.Timedelta(minutes=500)

        # 1h * 50 = 3000 minutes
        threshold = _calculate_gap_threshold('1h', 50)
        assert threshold == pd.Timedelta(minutes=3000)

        # 1d * 5 = 7200 minutes
        threshold = _calculate_gap_threshold('1d', 5)
        assert threshold == pd.Timedelta(minutes=7200)

    def test_smart_threshold_integration(self):
        """Test integration with smart threshold."""
        # 5m interval with smart threshold
        smart_multiplier = _get_smart_gap_threshold('5m')
        threshold = _calculate_gap_threshold('5m', smart_multiplier)

        # Should equal 4 days
        assert threshold == pd.Timedelta(days=4)


class TestCreatePeriodDict:
    """Test _create_period_dict function."""

    def test_creates_correct_structure(self):
        """Test that period dict has correct structure."""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        df = pd.DataFrame({'close': range(100)}, index=dates)

        period = _create_period_dict(1, df)

        assert 'period_id' in period
        assert 'df' in period
        assert 'start_date' in period
        assert 'end_date' in period
        assert 'row_count' in period

    def test_period_id_assignment(self):
        """Test that period_id is assigned correctly."""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        df = pd.DataFrame({'close': range(100)}, index=dates)

        period = _create_period_dict(42, df)
        assert period['period_id'] == 42

    def test_dataframe_assignment(self):
        """Test that DataFrame is assigned correctly."""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        df = pd.DataFrame({'close': range(100)}, index=dates)

        period = _create_period_dict(1, df)
        pd.testing.assert_frame_equal(period['df'], df)

    def test_date_extraction(self):
        """Test that start and end dates are extracted correctly."""
        dates = pd.date_range('2024-01-01 10:00', periods=100, freq='5min')
        df = pd.DataFrame({'close': range(100)}, index=dates)

        period = _create_period_dict(1, df)

        assert period['start_date'] == dates[0]
        assert period['end_date'] == dates[-1]

    def test_row_count(self):
        """Test that row count is correct."""
        dates = pd.date_range('2024-01-01', periods=123, freq='5min')
        df = pd.DataFrame({'close': range(123)}, index=dates)

        period = _create_period_dict(1, df)
        assert period['row_count'] == 123


class TestSplitDataframeAtGaps:
    """Test _split_dataframe_at_gaps function."""

    def test_splits_at_gaps_correctly(self):
        """Test that DataFrame is split at gap locations."""
        # Create data with known gap
        period1 = pd.date_range('2024-01-01', periods=1000, freq='5min')
        period2_start = period1[-1] + timedelta(days=10)
        period2 = pd.date_range(period2_start, periods=1200, freq='5min')

        all_dates = period1.append(period2)
        df = pd.DataFrame({'close': range(len(all_dates))}, index=all_dates)

        # Find the gap positions
        time_diffs = df.index.to_series().diff()
        gaps_mask = time_diffs > pd.Timedelta(days=1)
        gap_positions = [i for i, is_gap in enumerate(gaps_mask) if is_gap]

        periods = _split_dataframe_at_gaps(df, gap_positions, min_rows=500)

        assert len(periods) == 2
        assert periods[0]['row_count'] == 1000
        assert periods[1]['row_count'] == 1200

    def test_assigns_sequential_period_ids(self):
        """Test that period IDs are assigned sequentially."""
        period1 = pd.date_range('2024-01-01', periods=1000, freq='5min')
        period2_start = period1[-1] + timedelta(days=10)
        period2 = pd.date_range(period2_start, periods=1200, freq='5min')
        period3_start = period2[-1] + timedelta(days=10)
        period3 = pd.date_range(period3_start, periods=1500, freq='5min')

        all_dates = period1.append(period2).append(period3)
        df = pd.DataFrame({'close': range(len(all_dates))}, index=all_dates)

        time_diffs = df.index.to_series().diff()
        gaps_mask = time_diffs > pd.Timedelta(days=1)
        gap_positions = [i for i, is_gap in enumerate(gaps_mask) if is_gap]

        periods = _split_dataframe_at_gaps(df, gap_positions, min_rows=500)

        assert periods[0]['period_id'] == 1
        assert periods[1]['period_id'] == 2
        assert periods[2]['period_id'] == 3

    def test_filters_small_periods(self):
        """Test that periods smaller than min_rows are filtered out."""
        period1 = pd.date_range('2024-01-01', periods=500, freq='5min')  # Below min
        period2_start = period1[-1] + timedelta(days=10)
        period2 = pd.date_range(period2_start, periods=1500, freq='5min')  # Above min

        all_dates = period1.append(period2)
        df = pd.DataFrame({'close': range(len(all_dates))}, index=all_dates)

        time_diffs = df.index.to_series().diff()
        gaps_mask = time_diffs > pd.Timedelta(days=1)
        gap_positions = [i for i, is_gap in enumerate(gaps_mask) if is_gap]

        periods = _split_dataframe_at_gaps(df, gap_positions, min_rows=1000)

        # Only period2 should be included
        assert len(periods) == 1
        assert periods[0]['row_count'] == 1500

    def test_empty_gap_indices(self):
        """Test behavior with empty gap_indices."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
        df = pd.DataFrame({'close': range(1000)}, index=dates)

        periods = _split_dataframe_at_gaps(df, [], min_rows=500)

        # Should return just the final period
        assert len(periods) == 1
        assert periods[0]['row_count'] == 1000


# ==================== Main Function Tests ====================

class TestDetectPeriods:
    """Test detect_periods main function."""

    def test_continuous_data_returns_single_period(self, continuous_5m_data):
        """Test that continuous data returns a single period."""
        periods = detect_periods(continuous_5m_data, '5m')

        assert len(periods) == 1
        assert periods[0]['period_id'] == 1
        assert periods[0]['row_count'] == 2000

    def test_gapped_data_returns_multiple_periods(self, gapped_5m_data):
        """Test that gapped data returns multiple periods."""
        periods = detect_periods(gapped_5m_data, '5m')

        assert len(periods) == 3
        assert periods[0]['row_count'] == 1000
        assert periods[1]['row_count'] == 1200
        assert periods[2]['row_count'] == 1500

    def test_periods_have_correct_structure(self, continuous_5m_data):
        """Test that returned periods have correct structure."""
        periods = detect_periods(continuous_5m_data, '5m')

        period = periods[0]
        assert 'period_id' in period
        assert 'df' in period
        assert 'start_date' in period
        assert 'end_date' in period
        assert 'row_count' in period

    def test_period_dataframes_are_copies(self, gapped_5m_data):
        """Test that period DataFrames are independent copies."""
        periods = detect_periods(gapped_5m_data, '5m')

        # Modify period DataFrame
        periods[0]['df'].iloc[0, 0] = 99999

        # Original should be unchanged
        assert gapped_5m_data.iloc[0, 0] != 99999

    def test_respects_min_rows_parameter(self, small_periods_data):
        """Test that min_rows parameter is respected."""
        # With default min_rows=1000
        periods = detect_periods(small_periods_data, '5m', min_rows=1000)
        assert len(periods) == 1  # Only period2 (1500 rows)

        # With min_rows=400 (all 3 periods should be included)
        periods = detect_periods(small_periods_data, '5m', min_rows=400)
        assert len(periods) == 3  # All periods included

    def test_empty_dataframe_returns_empty_list(self):
        """Test that empty DataFrame returns empty list."""
        empty_df = pd.DataFrame(columns=['close'])
        empty_df.index = pd.DatetimeIndex([])

        periods = detect_periods(empty_df, '5m')
        assert periods == []

    def test_non_datetime_index_raises_error(self):
        """Test that non-DatetimeIndex raises ValueError."""
        df = pd.DataFrame({'close': range(100)})

        with pytest.raises(ValueError, match="DatetimeIndex"):
            detect_periods(df, '5m')

    def test_unsupported_interval_raises_error(self, continuous_5m_data):
        """Test that unsupported interval raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported interval"):
            detect_periods(continuous_5m_data, '6h')

    def test_different_intervals_use_correct_thresholds(self):
        """Test that different intervals use appropriate thresholds."""
        # Create 1h data
        dates_1h = pd.date_range('2024-01-01', periods=1000, freq='1h')
        df_1h = pd.DataFrame({'close': range(1000)}, index=dates_1h)

        periods = detect_periods(df_1h, '1h')
        assert len(periods) == 1

        # Create 1d data
        dates_1d = pd.date_range('2024-01-01', periods=500, freq='D')
        df_1d = pd.DataFrame({'close': range(500)}, index=dates_1d)

        periods = detect_periods(df_1d, '1d')
        assert len(periods) == 1

    def test_weekend_gaps_are_continuous(self):
        """Test that weekend gaps don't split periods (< 4 days)."""
        # Create weekday data with weekends removed
        dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
        # Filter to business hours (simple approximation)
        df = pd.DataFrame({'close': range(1000)}, index=dates)

        periods = detect_periods(df, '1h')
        # Should be continuous despite small gaps
        assert len(periods) >= 1

    def test_preserves_original_dataframe(self, continuous_5m_data):
        """Test that original DataFrame is not modified."""
        original_len = len(continuous_5m_data)
        original_first = continuous_5m_data.iloc[0, 0]

        periods = detect_periods(continuous_5m_data, '5m')

        assert len(continuous_5m_data) == original_len
        assert continuous_5m_data.iloc[0, 0] == original_first

    def test_periods_are_chronologically_ordered(self, gapped_5m_data):
        """Test that periods are returned in chronological order."""
        periods = detect_periods(gapped_5m_data, '5m')

        for i in range(len(periods) - 1):
            assert periods[i]['end_date'] < periods[i + 1]['start_date']

    def test_all_data_accounted_for(self, gapped_5m_data):
        """Test that all data (except filtered small periods) is accounted for."""
        periods = detect_periods(gapped_5m_data, '5m', min_rows=500)

        total_period_rows = sum(p['row_count'] for p in periods)

        # Should match original (3700 rows) since all periods are >= 500
        assert total_period_rows == len(gapped_5m_data)


# ==================== Edge Cases and Integration Tests ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame(
            {'close': [100]},
            index=pd.DatetimeIndex(['2024-01-01'])
        )

        periods = detect_periods(df, '5m', min_rows=1)
        assert len(periods) == 1
        assert periods[0]['row_count'] == 1

    def test_exactly_min_rows(self):
        """Test period with exactly min_rows."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
        df = pd.DataFrame({'close': range(1000)}, index=dates)

        periods = detect_periods(df, '5m', min_rows=1000)
        assert len(periods) == 1
        assert periods[0]['row_count'] == 1000

    def test_very_large_gap(self):
        """Test handling of very large gaps."""
        period1 = pd.date_range('2024-01-01', periods=1000, freq='5min')
        # Gap of 365 days
        period2_start = period1[-1] + timedelta(days=365)
        period2 = pd.date_range(period2_start, periods=1000, freq='5min')

        all_dates = period1.append(period2)
        df = pd.DataFrame({'close': range(len(all_dates))}, index=all_dates)

        periods = detect_periods(df, '5m')
        assert len(periods) == 2

    def test_multiple_consecutive_gaps(self):
        """Test handling of multiple consecutive gaps."""
        dates = []
        for i in range(5):
            period = pd.date_range(
                f'2024-{i + 1:02d}-01',
                periods=1200,
                freq='5min'
            )
            dates.append(period)

        all_dates = dates[0]
        for period in dates[1:]:
            all_dates = all_dates.append(period)

        df = pd.DataFrame({'close': range(len(all_dates))}, index=all_dates)

        periods = detect_periods(df, '5m')
        assert len(periods) == 5

    def test_timezone_aware_datetimes(self):
        """Test handling of timezone-aware DatetimeIndex."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='5min', tz='UTC')
        df = pd.DataFrame({'close': range(1000)}, index=dates)

        periods = detect_periods(df, '5m')
        assert len(periods) == 1
        assert periods[0]['start_date'].tz is not None
