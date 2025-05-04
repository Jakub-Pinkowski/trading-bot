from unittest.mock import patch, Mock

import pytest
import requests

from app.utils.api_utils import api_get, api_post


@patch('app.utils.api_utils.requests.get')
def test_api_get_success(mock_get):
    # Setup
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"key": "value"}

    mock_get.return_value = mock_response

    endpoint = "https://fakeapi.com/data"

    # Execute
    response = api_get(endpoint)

    # Assert
    mock_get.assert_called_once_with(url=endpoint, verify=False)
    mock_response.raise_for_status.assert_called_once()
    assert response.status_code == 200
    assert response.json() == {"key": "value"}


@patch('app.utils.api_utils.requests.get')
def test_api_get_failure(mock_get):
    # Setup
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Error")
    mock_get.return_value = mock_response

    endpoint = "https://fakeapi.com/data"

    # Execute & Assert
    with pytest.raises(requests.exceptions.HTTPError, match="404 Error"):
        api_get(endpoint)

    mock_get.assert_called_once_with(url=endpoint, verify=False)
    mock_response.raise_for_status.assert_called_once()


@patch('app.utils.api_utils.requests.post')
def test_api_post_success(mock_post):
    # Setup
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"status": "created"}

    mock_post.return_value = mock_response

    endpoint = "https://fakeapi.com/data"
    payload = {"name": "test"}

    # Execute
    response = api_post(endpoint, payload)

    # Assert
    mock_post.assert_called_once_with(url=endpoint, json=payload, verify=False)
    mock_response.raise_for_status.assert_called_once()
    assert response.status_code == 201
    assert response.json() == {"status": "created"}


@patch('app.utils.api_utils.requests.post')
def test_api_post_failure(mock_post):
    # Setup
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
    mock_post.return_value = mock_response

    endpoint = "https://fakeapi.com/data"
    payload = {"name": "test"}

    # Execute & Assert
    with pytest.raises(requests.exceptions.HTTPError, match="500 Server Error"):
        api_post(endpoint, payload)

    mock_post.assert_called_once_with(url=endpoint, json=payload, verify=False)
    mock_response.raise_for_status.assert_called_once()
