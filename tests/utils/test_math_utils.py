"""
Tests for Math Utils Module.

Tests cover:
- safe_divide: normal division, zero denominator with default and custom fallback
- safe_average: list averages, empty list, custom count
- calculate_percentage: normal percentages, zero total, rounding
"""
from app.utils.math_utils import safe_divide, safe_average, calculate_percentage


# ==================== Test Classes ====================

class TestSafeDivide:
    """Test zero-safe division."""

    def test_positive_and_negative_operands(self):
        """Test safe_divide handles various sign combinations correctly."""
        assert safe_divide(10, 2) == 5
        assert safe_divide(5, 2) == 2.5
        assert safe_divide(-10, 2) == -5
        assert safe_divide(10, -2) == -5

    def test_zero_denominator_returns_default(self):
        """Test zero denominator returns 0 by default."""
        assert safe_divide(10, 0) == 0

    def test_zero_denominator_returns_custom_default(self):
        """Test zero denominator returns the caller-supplied default."""
        assert safe_divide(10, 0, default=100) == 100


class TestSafeAverage:
    """Test zero-safe arithmetic mean."""

    def test_standard_lists(self):
        """Test safe_average computes the mean for typical inputs."""
        assert safe_average([1, 2, 3, 4, 5]) == 3
        assert safe_average([0, 0, 0]) == 0
        assert safe_average([-1, 0, 1]) == 0
        assert safe_average([10]) == 10

    def test_empty_list_returns_zero(self):
        """Test empty list returns 0 instead of raising ZeroDivisionError."""
        assert safe_average([]) == 0

    def test_custom_count_overrides_length(self):
        """Test caller-supplied count is used as the divisor instead of len(values)."""
        assert safe_average([1, 2, 3], count=2) == 3  # 6 / 2 = 3
        assert safe_average([1, 2, 3], count=6) == 1  # 6 / 6 = 1
        assert safe_average([1, 2, 3], count=0) == 0  # zero count returns 0


class TestCalculatePercentage:
    """Test percentage calculation with rounding."""

    def test_standard_percentages(self):
        """Test common percentage calculations round correctly."""
        assert calculate_percentage(25, 100) == 25.0
        assert calculate_percentage(1, 3) == 33.33
        assert calculate_percentage(1, 3, decimal_places=3) == 33.333

    def test_zero_total_returns_zero(self):
        """Test zero total returns 0 instead of raising ZeroDivisionError."""
        assert calculate_percentage(10, 0) == 0
        assert calculate_percentage(0, 0) == 0

    def test_decimal_places_rounding(self):
        """Test decimal_places parameter controls rounding precision."""
        assert calculate_percentage(1, 3, decimal_places=0) == 33.0
        assert calculate_percentage(1, 3, decimal_places=1) == 33.3
        assert calculate_percentage(1, 3, decimal_places=4) == 33.3333
