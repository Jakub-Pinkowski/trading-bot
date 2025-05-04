import pandas as pd

from app.utils.analysis_utils.analysis_utils import is_nonempty


def test_is_nonempty_with_list():
    # Test with non-empty list
    assert is_nonempty([1, 2, 3]) is True

    # Test with empty list
    assert is_nonempty([]) is False


def test_is_nonempty_with_dict():
    # Test with non-empty dict
    assert is_nonempty({"key": "value"}) is True

    # Test with empty dict
    assert is_nonempty({}) is False


def test_is_nonempty_with_string():
    # Test with non-empty string
    assert is_nonempty("test") is True

    # Test with empty string
    assert is_nonempty("") is False


def test_is_nonempty_with_dataframe():
    # Test with non-empty DataFrame
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    assert is_nonempty(df) is True

    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    assert is_nonempty(empty_df) is False


def test_is_nonempty_with_none():
    # Test with None
    assert is_nonempty(None) is False


def test_is_nonempty_with_scalar():
    # Test with scalar values (no len() method)
    assert is_nonempty(42) is True
    assert is_nonempty(0) is True  # Even 0 is considered non-empty
    assert is_nonempty(True) is True
    assert is_nonempty(False) is True
