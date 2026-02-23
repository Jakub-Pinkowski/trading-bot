"""
Tests for Generic Utils Module.

Tests cover:
- Symbol parsing: valid formats, TradingView suffix stripping, empty string, invalid inputs with error logging
"""
from unittest.mock import MagicMock

import pytest

from app.utils.generic_utils import parse_symbol


# ==================== Test Classes ====================

class TestParseSymbol:
    """Test ticker symbol parsing and validation."""

    def test_valid_symbols_return_base_letters(self):
        """Test alphanumeric and delimited symbols return only the leading letters."""
        assert parse_symbol("AAPL") == "AAPL"
        assert parse_symbol("AAPL123") == "AAPL"
        assert parse_symbol("AAPL.US") == "AAPL"
        assert parse_symbol("AAPL-B") == "AAPL"

    def test_tradingview_suffix_stripped(self):
        """Test TradingView continuous contract format returns only the base symbol."""
        assert parse_symbol("ZC1!") == "ZC"
        assert parse_symbol("ES1!") == "ES"

    def test_empty_string_raises_value_error(self, monkeypatch):
        """Test ValueError raised for an empty string with no leading letters."""
        mock_logger = MagicMock()
        monkeypatch.setattr("app.utils.generic_utils.logger", mock_logger)

        with pytest.raises(ValueError, match="Invalid symbol format: "):
            parse_symbol("")

        mock_logger.error.assert_called_once()

    def test_invalid_symbols_raise_value_error_and_log_error(self, monkeypatch):
        """Test ValueError raised and error logged for symbols with no leading letters."""
        mock_logger = MagicMock()
        monkeypatch.setattr("app.utils.generic_utils.logger", mock_logger)

        with pytest.raises(ValueError, match="Invalid symbol format: 123"):
            parse_symbol("123")

        with pytest.raises(ValueError, match="Invalid symbol format: !@#"):
            parse_symbol("!@#")

        assert mock_logger.error.call_count == 2
