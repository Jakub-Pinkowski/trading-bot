"""
Tests for File Utils Module.

Tests cover:
- JSON file loading: existing and nonexistent files
- JSON file saving with directory creation
"""
import json
import os
from unittest.mock import MagicMock, mock_open

from app.utils.file_utils import (
    load_file,
    save_file,
)


# ==================== Test Classes ====================

class TestLoadFile:
    """Test JSON file loading."""

    def test_existing_file_returns_json(self, monkeypatch, sample_json_data):
        """Test load_file returns parsed JSON data from an existing file."""
        monkeypatch.setattr(os.path, "exists", lambda path: True)
        monkeypatch.setattr("builtins.open", mock_open(read_data=json.dumps(sample_json_data)))
        monkeypatch.setattr(json, "load", lambda f: sample_json_data)

        result = load_file("test_file.json")

        assert result == sample_json_data

    def test_nonexistent_file_returns_empty_dict(self, monkeypatch):
        """Test load_file returns an empty dict when the file does not exist."""
        monkeypatch.setattr(os.path, "exists", lambda path: False)

        result = load_file("nonexistent_file.json")

        assert result == {}


class TestSaveFile:
    """Test JSON file saving."""

    def test_creates_directory_and_writes_json(self, monkeypatch, sample_json_data):
        """Test save_file creates the parent directory and writes JSON data."""
        mock_makedirs = MagicMock()
        mock_json_dump = MagicMock()
        m = mock_open()

        monkeypatch.setattr(os, "makedirs", mock_makedirs)
        monkeypatch.setattr("builtins.open", m)
        monkeypatch.setattr(json, "dump", mock_json_dump)

        save_file(sample_json_data, "test_dir/test_file.json")

        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        m.assert_called_once_with("test_dir/test_file.json", "w")
        mock_json_dump.assert_called_once()
