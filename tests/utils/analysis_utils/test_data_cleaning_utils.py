import pandas as pd

from app.utils.analysis_utils.data_cleaning_utils import parse_description, fractional_to_decimal


def test_parse_description_valid_json():
    # Test with valid JSON string
    json_str = '{"key": "value", "number": 42}'
    expected = {"key": "value", "number": 42}
    assert parse_description(json_str) == expected


def test_parse_description_invalid_json():
    # Test with invalid JSON string
    invalid_json = '{key: value}'  # Missing quotes around keys
    assert parse_description(invalid_json) == {}


def test_parse_description_empty_string():
    # Test with empty string
    assert parse_description("") == {}


def test_parse_description_none():
    # Test with None value
    assert parse_description(None) == {}


def test_parse_description_nan():
    # Test with pandas NaN value
    assert parse_description(pd.NA) == {}
    assert parse_description(float('nan')) == {}


def test_fractional_to_decimal_whole_and_fraction():
    # Test with whole number and fraction (e.g., "1 1/2")
    assert fractional_to_decimal("1 1/2") == 1.5
    assert fractional_to_decimal("2 3/4") == 2.75
    assert fractional_to_decimal("0 1/4") == 0.25


def test_fractional_to_decimal_fraction_only():
    # Test with fraction only (e.g., "3/4")
    assert fractional_to_decimal("3/4") == 0.75
    assert fractional_to_decimal("1/2") == 0.5
    assert fractional_to_decimal("1/8") == 0.125


def test_fractional_to_decimal_whole_number():
    # Test with whole number as string
    assert fractional_to_decimal("42") == 42.0
    assert fractional_to_decimal("0") == 0.0


def test_fractional_to_decimal_float():
    # Test with float as string
    assert fractional_to_decimal("3.14") == 3.14
    assert fractional_to_decimal("0.5") == 0.5


def test_fractional_to_decimal_numeric_types():
    # Test with numeric types (not strings)
    assert fractional_to_decimal(42) == 42.0
    assert fractional_to_decimal(3.14) == 3.14
    assert fractional_to_decimal(0) == 0.0
