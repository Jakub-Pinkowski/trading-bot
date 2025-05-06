from unittest.mock import patch

import pytest

from app.utils.generic_utils import parse_symbol


def test_parse_symbol_valid():
    """Test that parse_symbol correctly extracts the base symbol from various formats"""

    assert parse_symbol("AAPL") == "AAPL"
    assert parse_symbol("AAPL123") == "AAPL"
    assert parse_symbol("AAPL.US") == "AAPL"
    assert parse_symbol("AAPL-B") == "AAPL"


def test_parse_symbol_special_case():
    """Test that parse_symbol correctly handles the special case of MHG symbol conversion to MHNG"""

    assert parse_symbol("MHG") == "MHNG"
    assert parse_symbol("MHG123") == "MHNG"
    assert parse_symbol("MHG.US") == "MHNG"


@patch("app.utils.generic_utils.logger")
def test_parse_symbol_invalid(mock_logger):
    """Test that parse_symbol raises ValueError for invalid symbol formats and logs errors"""

    with pytest.raises(ValueError, match="Invalid symbol format: 123"):
        parse_symbol("123")

    with pytest.raises(ValueError, match="Invalid symbol format: !@#"):
        parse_symbol("!@#")

    # Verify that error is logged for each invalid symbol
    assert mock_logger.error.call_count == 2
