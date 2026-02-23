"""
Tests for API Utils Module.

Tests cover:
- HTTP header construction
- GET request: success, HTTP error, URL construction, SSL disabled
- POST request: success, HTTP error, URL construction, SSL disabled
"""
import pytest
import requests

from app.utils.api_utils import get_headers, api_get, api_post


# ==================== Test Classes ====================

class TestGetHeaders:
    """Test HTTP header construction."""

    def test_returns_expected_default_headers(self):
        """Test get_headers returns the expected default headers."""
        headers = get_headers()

        assert headers == {
            "Host": "api.ibkr.com",
            "User-Agent": "python-requests/IBKR-client",
            "Accept": "*/*",
            "Connection": "keep-alive",
        }


class TestApiGet:
    """Test GET request wrapper."""

    def test_success_returns_json(self, monkeypatch, mock_response_factory):
        """Test successful GET request returns parsed JSON."""
        monkeypatch.setattr(
            "app.utils.api_utils.requests.get",
            lambda url, verify, headers: mock_response_factory(),
        )
        monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")

        result = api_get("/test-endpoint")

        assert result == {"data": "test_data"}

    def test_http_error_propagates(self, monkeypatch, mock_response_factory):
        """Test HTTPError from response.raise_for_status propagates to caller."""
        monkeypatch.setattr(
            "app.utils.api_utils.requests.get",
            lambda url, verify, headers: mock_response_factory(
                raise_exc=requests.HTTPError("404 Client Error")
            ),
        )
        monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")

        with pytest.raises(requests.HTTPError, match="404 Client Error"):
            api_get("/test-endpoint")

    def test_constructs_url_from_base_url_and_disables_ssl(self, monkeypatch, mock_response_factory):
        """Test api_get concatenates BASE_URL with endpoint and passes verify=False."""
        captured = {}

        def capturing_get(url, verify, headers):
            captured["url"] = url
            captured["verify"] = verify
            return mock_response_factory()

        monkeypatch.setattr("app.utils.api_utils.requests.get", capturing_get)
        monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")

        api_get("/test-endpoint")

        assert captured["url"] == "https://api.example.com/test-endpoint"
        assert captured["verify"] is False


class TestApiPost:
    """Test POST request wrapper."""

    def test_success_returns_json(self, monkeypatch, mock_response_factory):
        """Test successful POST request returns parsed JSON."""
        monkeypatch.setattr(
            "app.utils.api_utils.requests.post",
            lambda url, json, verify, headers: mock_response_factory(),
        )
        monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")

        result = api_post("/test-endpoint", {"key": "value"})

        assert result == {"data": "test_data"}

    def test_http_error_propagates(self, monkeypatch, mock_response_factory):
        """Test HTTPError from response.raise_for_status propagates to caller."""
        monkeypatch.setattr(
            "app.utils.api_utils.requests.post",
            lambda url, json, verify, headers: mock_response_factory(
                raise_exc=requests.HTTPError("500 Server Error")
            ),
        )
        monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")

        with pytest.raises(requests.HTTPError, match="500 Server Error"):
            api_post("/test-endpoint", {"key": "value"})

    def test_constructs_url_from_base_url_and_disables_ssl(self, monkeypatch, mock_response_factory):
        """Test api_post concatenates BASE_URL with endpoint and passes verify=False."""
        captured = {}

        def capturing_post(url, json, verify, headers):
            captured["url"] = url
            captured["verify"] = verify
            return mock_response_factory()

        monkeypatch.setattr("app.utils.api_utils.requests.post", capturing_post)
        monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")

        api_post("/test-endpoint", {"key": "value"})

        assert captured["url"] == "https://api.example.com/test-endpoint"
        assert captured["verify"] is False
