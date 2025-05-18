from unittest.mock import patch

from app.utils.ibkr_utils.orders_utils import invalidate_cache, get_contract_position, suppress_messages
from config import ACCOUNT_ID


@patch('app.utils.ibkr_utils.orders_utils.api_post')
def test_invalidate_cache_success(mock_api_post):
    """Test that invalidate_cache successfully calls the API."""

    # Call the function
    invalidate_cache()

    # Verify the API was called with the correct endpoint and data
    mock_api_post.assert_called_once_with(f"portfolio/{ACCOUNT_ID}/positions/invalidate", {})


@patch('app.utils.ibkr_utils.orders_utils.api_post')
def test_invalidate_cache_error(mock_api_post):
    """Test that invalidate_cache handles errors gracefully."""

    # Setup mock to raise an exception
    mock_api_post.side_effect = Exception("API error")

    # Call the function (should not raise an exception)
    invalidate_cache()

    # Verify the API was called
    mock_api_post.assert_called_once()


@patch('app.utils.ibkr_utils.orders_utils.invalidate_cache')
@patch('app.utils.ibkr_utils.orders_utils.api_get')
def test_get_contract_position_found(mock_api_get, mock_invalidate_cache):
    """Test that get_contract_position returns the correct position when found."""

    # Setup mocks
    mock_api_get.return_value = [
        {"conid": "123456", "position": 5},
        {"conid": "789012", "position": -3}
    ]

    # Call the function for a contract that exists
    result = get_contract_position("123456")

    # Verify the result and that the cache was invalidated
    assert result == 5
    mock_invalidate_cache.assert_called_once()
    mock_api_get.assert_called_once_with(f"portfolio/{ACCOUNT_ID}/positions")


@patch('app.utils.ibkr_utils.orders_utils.invalidate_cache')
@patch('app.utils.ibkr_utils.orders_utils.api_get')
def test_get_contract_position_not_found(mock_api_get, mock_invalidate_cache):
    """Test that get_contract_position returns 0 when the contract is not found."""

    # Setup mocks
    mock_api_get.return_value = [
        {"conid": "789012", "position": -3}
    ]

    # Call the function for a contract that doesn't exist
    result = get_contract_position("123456")

    # Verify the result is 0 and that the cache was invalidated
    assert result == 0
    mock_invalidate_cache.assert_called_once()
    mock_api_get.assert_called_once_with(f"portfolio/{ACCOUNT_ID}/positions")


@patch('app.utils.ibkr_utils.orders_utils.invalidate_cache')
@patch('app.utils.ibkr_utils.orders_utils.api_get')
def test_get_contract_position_api_error(mock_api_get, mock_invalidate_cache):
    """Test that get_contract_position handles API errors gracefully."""

    # Setup mock to raise an exception
    mock_api_get.side_effect = Exception("API error")

    # Call the function
    result = get_contract_position("123456")

    # Verify the result is 0 and that the cache was invalidated
    assert result == 0
    mock_invalidate_cache.assert_called_once()
    mock_api_get.assert_called_once_with(f"portfolio/{ACCOUNT_ID}/positions")


@patch('app.utils.ibkr_utils.orders_utils.api_post')
def test_suppress_messages_success(mock_api_post):
    """Test that suppress_messages successfully calls the API."""

    # Setup mock
    mock_api_post.return_value = {"success": True}

    # Call the function
    suppress_messages(["1", "2"])

    # Verify the API was called with the correct endpoint and data
    mock_api_post.assert_called_once_with("iserver/questions/suppress", {"messageIds": ["1", "2"]})


@patch('app.utils.ibkr_utils.orders_utils.api_post')
def test_suppress_messages_error(mock_api_post):
    """Test that suppress_messages handles errors gracefully."""
    # Setup mock to raise an exception
    mock_api_post.side_effect = Exception("API error")

    # Call the function (should not raise an exception)
    suppress_messages(["1", "2"])

    # Verify the API was called
    mock_api_post.assert_called_once()
