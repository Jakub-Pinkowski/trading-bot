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

    def test_numeric_only_symbol_raises_value_error_and_logs_error(self, monkeypatch):
        """Test ValueError raised and error logged for a symbol starting with digits."""
        mock_logger = MagicMock()
        monkeypatch.setattr("app.utils.generic_utils.logger", mock_logger)

        with pytest.raises(ValueError, match="Invalid symbol format: 123"):
            parse_symbol("123")

        mock_logger.error.assert_called_once()

    def test_special_chars_only_symbol_raises_value_error_and_logs_error(self, monkeypatch):
        """Test ValueError raised and error logged for a symbol containing only special characters."""
        mock_logger = MagicMock()
        monkeypatch.setattr("app.utils.generic_utils.logger", mock_logger)

        with pytest.raises(ValueError, match="Invalid symbol format: !@#"):
            parse_symbol("!@#")

        mock_logger.error.assert_called_once()
