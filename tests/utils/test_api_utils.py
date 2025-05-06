import json

import pytest
import requests

from app.utils.api_utils import get_headers, api_get, api_post


@pytest.fixture
def mock_response_factory():
    def _make_mock_response(data=None, raise_exc=None):
        def json_func():
            return data if data is not None else {"data": "test_data"}

        def raise_for_status_func():
            if raise_exc:
                raise raise_exc

        response = type("MockResponse", (), {})()
        response.json = json_func
        response.raise_for_status = raise_for_status_func
        return response

    return _make_mock_response


def test_get_headers_without_payload():
    """Test that get_headers returns the expected default headers when no payload is provided"""

    # Call get_headers function without any payload parameter
    headers = get_headers()

    # Verify the returned headers contain the expected default values
    assert headers == {
        'Host': 'api.ibkr.com',
        'User-Agent': 'python-requests/IBKR-client',
        'Accept': '*/*',
        'Connection': 'keep-alive',
    }


def test_get_headers_with_payload():
    """Test that get_headers includes Content-Length header when payload is provided"""

    # Create a sample payload dictionary to pass to the function
    payload = {"key": "value"}

    # Call get_headers function with the payload parameter
    headers = get_headers(payload)

    # Verify the returned headers include Content-Length based on the payload size
    expected_headers = {
        'Host': 'api.ibkr.com',
        'User-Agent': 'python-requests/IBKR-client',
        'Accept': '*/*',
        'Connection': 'keep-alive',
        'Content-Length': str(len(json.dumps(payload)))
    }
    assert headers == expected_headers


def test_api_get_success(monkeypatch, mock_response_factory):
    """Test that api_get successfully returns data from a successful API response"""

    # Mock the requests.get function to return a successful response and set a test BASE_URL
    def mock_get(url, verify, headers):
        return mock_response_factory()

    monkeypatch.setattr("app.utils.api_utils.requests.get", mock_get)
    monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")

    # Call api_get with a test endpoint path
    result = api_get('/test-endpoint')

    # Verify the function returns the expected data from the response
    assert result == {"data": "test_data"}


def test_api_get_http_error(monkeypatch, mock_response_factory):
    """Test that api_get raises HTTPError when the API request fails"""

    # Mock the requests.get function to raise an HTTP error and set a test BASE_URL
    def mock_get(url, verify, headers):
        return mock_response_factory(raise_exc=requests.HTTPError("404 Client Error"))

    monkeypatch.setattr("app.utils.api_utils.requests.get", mock_get)
    monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")

    # Verify api_get raises the expected HTTPError when the request fails
    with pytest.raises(requests.HTTPError, match="404 Client Error"):
        api_get('/test-endpoint')


def test_api_post_success(monkeypatch, mock_response_factory):
    """Test that api_post successfully returns data from a successful API response"""

    # Mock the requests.post function to return a successful response and set a test BASE_URL
    def mock_post(url, json, verify, headers):
        return mock_response_factory()

    monkeypatch.setattr("app.utils.api_utils.requests.post", mock_post)
    monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")
    payload = {"key": "value"}

    # Call api_post with a test endpoint path and payload
    result = api_post('/test-endpoint', payload)

    # Verify the function returns the expected data from the response
    assert result == {"data": "test_data"}


def test_api_post_http_error(monkeypatch, mock_response_factory):
    """Test that api_post raises HTTPError when the API request fails"""

    # Mock the requests.post function to raise an HTTP error and set a test BASE_URL
    def mock_post(url, json, verify, headers):
        return mock_response_factory(raise_exc=requests.HTTPError("500 Server Error"))

    monkeypatch.setattr("app.utils.api_utils.requests.post", mock_post)
    monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")
    payload = {"key": "value"}

    # Verify api_post raises the expected HTTPError when the request fails
    with pytest.raises(requests.HTTPError, match="500 Server Error"):
        api_post('/test-endpoint', payload)
