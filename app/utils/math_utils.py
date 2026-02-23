from app.utils.logger import get_logger

logger = get_logger('utils/math_utils')


# ==================== Safe Math Operations ====================

def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two numbers, returning a default value if the denominator is zero.

    Args:
        numerator: The numerator in the division
        denominator: The denominator in the division
        default: The value to return if the denominator is zero (default: 0)

    Returns:
        The result of the division, or the default value if the denominator is zero
    """
    return numerator / denominator if denominator != 0 else default


def safe_average(values, count=None):
    """
    Safely calculate the average of a list of values, handling empty lists and zero counts.

    Args:
        values: List of values to average
        count: Optional count to use instead of len(values)

    Returns:
        The average of the values, or 0 if the list is empty or the count is 0
    """
    if not values:
        return 0
    if count is None:
        count = len(values)
    return sum(values) / count if count > 0 else 0


def calculate_percentage(value, total, decimal_places=2):
    """
    Calculate a percentage, safely handling division by zero.

    Args:
        value: The value to calculate percentage for
        total: The total value (denominator)
        decimal_places: Number of decimal places to round to (default: 2)

    Returns:
        The percentage value rounded to specified decimal places, or 0 if total is zero
    """
    percentage = safe_divide(value, total) * 100
    return round(percentage, decimal_places)
