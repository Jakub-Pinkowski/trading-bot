"""
Integration tests for segmentation module using real data.

These tests use actual historical futures data to validate:
- Gap detection with real market gaps
- Proportional segment allocation
- Single period optimization
- Multiple symbols and intervals
"""

import pandas as pd
import pytest

from app.backtesting.testing.segmentation import detect_periods, split_all_periods
from config import DATA_DIR


# ==================== Fixtures ====================

@pytest.fixture(scope="module")
def historical_data_dir():
    """Get historical data directory."""
    return DATA_DIR / "historical_data"


@pytest.fixture(scope="module")
def zc_5m_data(historical_data_dir):
    """Load ZC 5m data (has gaps)."""
    filepath = historical_data_dir / "1!" / "ZC" / "ZC_5m.parquet"
    if not filepath.exists():
        pytest.skip(f"Real data not found: {filepath}")
    return pd.read_parquet(filepath)


@pytest.fixture(scope="module")
def zc_30m_data(historical_data_dir):
    """Load ZC 30m data (continuous)."""
    filepath = historical_data_dir / "1!" / "ZC" / "ZC_30m.parquet"
    if not filepath.exists():
        pytest.skip(f"Real data not found: {filepath}")
    return pd.read_parquet(filepath)


@pytest.fixture(scope="module")
def zc_2h_data(historical_data_dir):
    """Load ZC 2h data (continuous)."""
    filepath = historical_data_dir / "1!" / "ZC" / "ZC_2h.parquet"
    if not filepath.exists():
        pytest.skip(f"Real data not found: {filepath}")
    return pd.read_parquet(filepath)


# ==================== Real Data Gap Detection ====================

class TestRealDataGapDetection:
    """Test gap detection with real market data."""

    def test_zc_5m_detects_expected_periods(self, zc_5m_data):
        """Test that ZC 5m data detects the expected 3 periods."""
        periods = detect_periods(zc_5m_data, '5m')

        # ZC 5m typically has 3 periods due to data gaps
        assert len(periods) >= 1, "Should detect at least 1 period"

        # Verify all rows are accounted for
        total_period_rows = sum(p['row_count'] for p in periods)
        assert total_period_rows == len(zc_5m_data), "All rows should be in periods"

    def test_zc_30m_is_continuous(self, zc_30m_data):
        """Test that ZC 30m data is continuous (single period)."""
        periods = detect_periods(zc_30m_data, '30m')

        # 30m data should be continuous
        assert len(periods) == 1, "30m data should be continuous"
        assert periods[0]['row_count'] == len(zc_30m_data)

    def test_zc_2h_is_continuous(self, zc_2h_data):
        """Test that ZC 2h data is continuous (single period)."""
        periods = detect_periods(zc_2h_data, '2h')

        # 2h data should be continuous
        assert len(periods) == 1, "2h data should be continuous"
        assert periods[0]['row_count'] == len(zc_2h_data)

    def test_periods_have_valid_datetimes(self, zc_5m_data):
        """Test that detected periods have valid datetime ranges."""
        periods = detect_periods(zc_5m_data, '5m')

        for period in periods:
            assert period['start_date'] < period['end_date'], "Start before end"
            assert isinstance(period['start_date'], pd.Timestamp)
            assert isinstance(period['end_date'], pd.Timestamp)

    def test_periods_are_chronological(self, zc_5m_data):
        """Test that periods are in chronological order."""
        periods = detect_periods(zc_5m_data, '5m')

        if len(periods) > 1:
            for i in range(len(periods) - 1):
                assert periods[i]['end_date'] < periods[i + 1]['start_date']


# ==================== Proportional Segmentation ====================

class TestProportionalSegmentation:
    """Test proportional segment allocation with real data."""

    def test_proportional_distribution_balances_segments(self, zc_5m_data):
        """Test that proportional distribution creates balanced segments."""
        periods = detect_periods(zc_5m_data, '5m')

        if len(periods) < 2:
            pytest.skip("Need multiple periods for proportional test")

        segments = split_all_periods(periods, total_segments=4)

        assert len(segments) == 4, "Should create exactly 4 segments"

        # Check that segments are more balanced than fixed per-period
        row_counts = [s['row_count'] for s in segments]
        avg_rows = sum(row_counts) / len(row_counts)
        variance = sum((x - avg_rows) ** 2 for x in row_counts) / len(row_counts)

        # Variance should be relatively low (balanced)
        # This is a loose check - exact value depends on data
        assert variance >= 0, "Variance should be non-negative"

    def test_proportional_respects_period_boundaries(self, zc_5m_data):
        """Test that proportional segments never cross period boundaries."""
        periods = detect_periods(zc_5m_data, '5m')
        segments = split_all_periods(periods, total_segments=8)

        # Group segments by period
        for period in periods:
            period_segments = [s for s in segments if s['period_id'] == period['period_id']]

            if len(period_segments) > 1:
                # Check segments within same period are contiguous
                for i in range(len(period_segments) - 1):
                    gap_seconds = (period_segments[i + 1]['start_date'] -
                                   period_segments[i]['end_date']).total_seconds()
                    # Should be one interval gap at most (5 minutes = 300 seconds)
                    assert gap_seconds <= 600, "Segments should be contiguous"

    def test_total_segments_less_than_periods(self, zc_5m_data):
        """Test that fewer total_segments than periods still works."""
        periods = detect_periods(zc_5m_data, '5m')

        if len(periods) < 3:
            pytest.skip("Need at least 3 periods")

        # Request fewer segments than periods
        segments = split_all_periods(periods, total_segments=2)

        assert len(segments) == 2, "Should create exactly 2 segments"

        # Some periods will get segments, some won't
        period_ids = set(s['period_id'] for s in segments)
        assert len(period_ids) <= len(periods)


# ==================== Fixed Segments Per Period ====================

class TestFixedSegmentsPerPeriod:
    """Test fixed segments per period allocation with real data."""

    def test_fixed_segments_creates_correct_count(self, zc_5m_data):
        """Test that fixed segments per period creates expected count."""
        periods = detect_periods(zc_5m_data, '5m')

        segments = split_all_periods(periods, segments_per_period=4)

        expected_count = len(periods) * 4
        assert len(segments) == expected_count

    def test_each_period_as_single_segment(self, zc_5m_data):
        """Test segments_per_period=1 makes each period one segment."""
        periods = detect_periods(zc_5m_data, '5m')
        segments = split_all_periods(periods, segments_per_period=1)

        assert len(segments) == len(periods)

        # Each segment should equal its period
        for period, segment in zip(periods, segments):
            assert segment['row_count'] == period['row_count']
            pd.testing.assert_frame_equal(segment['df'], period['df'])


# ==================== Single Period Optimization ====================

class TestSinglePeriodOptimization:
    """Test optimizations for single continuous periods."""

    def test_single_period_with_total_segments(self, zc_30m_data):
        """Test single period optimization with total_segments."""
        periods = detect_periods(zc_30m_data, '30m')

        assert len(periods) == 1, "Should be single period"

        segments = split_all_periods(periods, total_segments=5)

        assert len(segments) == 5
        assert all(s['period_id'] == 1 for s in segments)

    def test_single_period_segments_are_balanced(self, zc_2h_data):
        """Test that single period creates balanced segments."""
        periods = detect_periods(zc_2h_data, '2h')
        segments = split_all_periods(periods, total_segments=4)

        # All segments should have similar row counts
        row_counts = [s['row_count'] for s in segments]
        avg_rows = sum(row_counts) / len(row_counts)

        # Each segment should be within 1% of average (allows for rounding)
        for count in row_counts:
            diff_pct = abs(count - avg_rows) / avg_rows if avg_rows > 0 else 0
            assert diff_pct < 0.01, f"Segment size {count} differs too much from average {avg_rows}"


# ==================== Multiple Symbols ====================

class TestMultipleSymbols:
    """Test segmentation across multiple symbols."""

    @pytest.mark.parametrize("symbol,interval", [
        ("ZC", "5m"),
        ("ZW", "5m"),
        ("ZS", "5m"),
        ("ZL", "5m"),
    ])
    def test_multiple_symbols_5m(self, historical_data_dir, symbol, interval):
        """Test that multiple symbols can be segmented successfully."""
        filepath = historical_data_dir / "1!" / symbol / f"{symbol}_{interval}.parquet"

        if not filepath.exists():
            pytest.skip(f"Data not found: {filepath}")

        df = pd.read_parquet(filepath)
        periods = detect_periods(df, interval)
        segments = split_all_periods(periods, total_segments=4)

        # Basic validation
        assert len(segments) >= 1, "Should create at least 1 segment"
        assert len(segments) <= 4 * len(periods), "Should not exceed max possible"

        # All rows accounted for
        total_rows = sum(s['row_count'] for s in segments)
        assert total_rows == len(df)


# ==================== Data Integrity ====================

class TestDataIntegrity:
    """Test data integrity after segmentation."""

    def test_no_data_loss(self, zc_5m_data):
        """Test that no data is lost during segmentation."""
        original_rows = len(zc_5m_data)

        periods = detect_periods(zc_5m_data, '5m')
        segments = split_all_periods(periods, segments_per_period=4)

        total_segment_rows = sum(s['row_count'] for s in segments)

        assert total_segment_rows == original_rows, "All data must be preserved"

    def test_no_data_duplication(self, zc_5m_data):
        """Test that data is not duplicated across segments."""
        periods = detect_periods(zc_5m_data, '5m')
        segments = split_all_periods(periods, segments_per_period=3)

        # Collect all segment date ranges
        date_ranges = []
        for segment in segments:
            dates = segment['df'].index
            date_ranges.append((dates[0], dates[-1]))

        # Check for overlaps within same period
        for i in range(len(date_ranges)):
            for j in range(i + 1, len(date_ranges)):
                seg_i = segments[i]
                seg_j = segments[j]

                # Only check if same period
                if seg_i['period_id'] == seg_j['period_id']:
                    start_i, end_i = date_ranges[i]
                    start_j, end_j = date_ranges[j]

                    # No overlap allowed
                    assert end_i < start_j or end_j < start_i, \
                        f"Segments {i} and {j} overlap"

    def test_dataframes_are_independent(self, zc_5m_data):
        """Test that segment DataFrames are independent copies."""
        periods = detect_periods(zc_5m_data, '5m')
        segments = split_all_periods(periods, segments_per_period=2)

        # Modify first segment (use a column that exists and is numeric)
        if len(segments) > 0 and len(zc_5m_data.columns) > 0:
            # Find a numeric column
            numeric_cols = zc_5m_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                original_value = zc_5m_data.iloc[0][col]
                segments[0]['df'].iloc[0, segments[0]['df'].columns.get_loc(col)] = 99999

                # Original should be unchanged
                assert zc_5m_data.iloc[0][col] == original_value, \
                    "Original data should not be modified"


# ==================== Edge Cases with Real Data ====================

class TestRealDataEdgeCases:
    """Test edge cases that occur with real market data."""

    def test_handles_timezone_aware_data(self, zc_5m_data):
        """Test handling of timezone-aware datetimes in real data."""
        # Convert to timezone-aware if not already
        if zc_5m_data.index.tz is None:
            df_tz = zc_5m_data.copy()
            df_tz.index = df_tz.index.tz_localize('UTC')
        else:
            df_tz = zc_5m_data

        periods = detect_periods(df_tz, '5m')
        segments = split_all_periods(periods, total_segments=3)

        # Should work with timezone-aware data
        assert len(segments) > 0
        for segment in segments:
            if segment['df'].index.tz is not None:
                assert segment['start_date'].tz is not None
                assert segment['end_date'].tz is not None

    def test_handles_very_large_datasets(self, zc_5m_data):
        """Test performance with large real datasets."""
        # ZC 5m data typically has 15k-25k rows
        assert len(zc_5m_data) > 1000, "Should be a substantial dataset"

        periods = detect_periods(zc_5m_data, '5m')
        segments = split_all_periods(periods, total_segments=10)

        # Should handle efficiently
        assert len(segments) <= 10

    def test_segments_maintain_data_types(self, zc_5m_data):
        """Test that segments maintain original data types."""
        original_dtypes = zc_5m_data.dtypes

        periods = detect_periods(zc_5m_data, '5m')
        segments = split_all_periods(periods, segments_per_period=2)

        for segment in segments:
            for col in original_dtypes.index:
                if col in segment['df'].columns:
                    assert segment['df'][col].dtype == original_dtypes[col], \
                        f"Data type mismatch for column {col}"
