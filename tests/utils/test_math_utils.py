from app.utils.math_utils import safe_divide, safe_average, calculate_percentage


def test_safe_divide_normal_case():
    """Test that safe_divide correctly divides when denominator is not zero"""
    assert safe_divide(10, 2) == 5
    assert safe_divide(5, 2) == 2.5
    assert safe_divide(-10, 2) == -5
    assert safe_divide(10, -2) == -5


def test_safe_divide_zero_denominator():
    """Test that safe_divide returns default value when denominator is zero"""
    assert safe_divide(10, 0) == 0  # Default value is 0
    assert safe_divide(10, 0, default=100) == 100  # Custom default value


def test_safe_average_normal_case():
    """Test that safe_average correctly calculates average for non-empty lists"""
    assert safe_average([1, 2, 3, 4, 5]) == 3
    assert safe_average([0, 0, 0]) == 0
    assert safe_average([-1, 0, 1]) == 0
    assert safe_average([10]) == 10


def test_safe_average_empty_list():
    """Test that safe_average returns 0 for empty lists"""
    assert safe_average([]) == 0


def test_safe_average_with_custom_count():
    """Test that safe_average uses provided count instead of len(values)"""
    assert safe_average([1, 2, 3], count=2) == 3  # sum=6, count=2 -> 6/2=3
    assert safe_average([1, 2, 3], count=6) == 1  # sum=6, count=6 -> 6/6=1
    assert safe_average([1, 2, 3], count=0) == 0  # count=0 should return 0


def test_calculate_percentage_normal_case():
    """Test that calculate_percentage correctly calculates percentages"""
    assert calculate_percentage(25, 100) == 25.0
    assert calculate_percentage(1, 3) == 33.33  # Rounded to 2 decimal places
    assert calculate_percentage(1, 3, decimal_places=3) == 33.333  # Custom decimal places


def test_calculate_percentage_zero_total():
    """Test that calculate_percentage returns 0 when total is zero"""
    assert calculate_percentage(10, 0) == 0
    assert calculate_percentage(0, 0) == 0


def test_calculate_percentage_rounding():
    """Test that calculate_percentage rounds to the specified number of decimal places"""
    assert calculate_percentage(1, 3, decimal_places=0) == 33.0
    assert calculate_percentage(1, 3, decimal_places=1) == 33.3
    assert calculate_percentage(1, 3, decimal_places=4) == 33.3333
