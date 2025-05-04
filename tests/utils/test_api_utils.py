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
    # Setup & Execute
    headers = get_headers()

    # Assert
    assert headers == {
        'Host': 'api.ibkr.com',
        'User-Agent': 'python-requests/IBKR-client',
        'Accept': '*/*',
        'Connection': 'keep-alive',
    }


def test_get_headers_with_payload():
    # Setup
    payload = {"key": "value"}

    # Execute
    headers = get_headers(payload)

    # Assert
    expected_headers = {
        'Host': 'api.ibkr.com',
        'User-Agent': 'python-requests/IBKR-client',
        'Accept': '*/*',
        'Connection': 'keep-alive',
        'Content-Length': str(len(json.dumps(payload)))
    }
    assert headers == expected_headers


def test_api_get_success(monkeypatch, mock_response_factory):
    # Setup
    def mock_get(url, verify, headers):
        return mock_response_factory()

    monkeypatch.setattr("app.utils.api_utils.requests.get", mock_get)
    monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")

    # Execute
    result = api_get('/test-endpoint')

    # Assert
    assert result == {"data": "test_data"}


def test_api_get_http_error(monkeypatch, mock_response_factory):
    # Setup
    def mock_get(url, verify, headers):
        return mock_response_factory(raise_exc=requests.HTTPError("404 Client Error"))

    monkeypatch.setattr("app.utils.api_utils.requests.get", mock_get)
    monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")

    # Execute & Assert
    with pytest.raises(requests.HTTPError, match="404 Client Error"):
        api_get('/test-endpoint')


def test_api_post_success(monkeypatch, mock_response_factory):
    # Setup
    def mock_post(url, json, verify, headers):
        return mock_response_factory()

    monkeypatch.setattr("app.utils.api_utils.requests.post", mock_post)
    monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")
    payload = {"key": "value"}

    # Execute
    result = api_post('/test-endpoint', payload)

    # Assert
    assert result == {"data": "test_data"}


def test_api_post_http_error(monkeypatch, mock_response_factory):
    # Setup
    def mock_post(url, json, verify, headers):
        return mock_response_factory(raise_exc=requests.HTTPError("500 Server Error"))

    monkeypatch.setattr("app.utils.api_utils.requests.post", mock_post)
    monkeypatch.setattr("app.utils.api_utils.BASE_URL", "https://api.example.com")
    payload = {"key": "value"}

    # Execute & Assert
    with pytest.raises(requests.HTTPError, match="500 Server Error"):
        api_post('/test-endpoint', payload)
