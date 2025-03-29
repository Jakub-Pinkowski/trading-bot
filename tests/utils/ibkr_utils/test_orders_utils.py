import pytest
from unittest.mock import patch, Mock
from app.utils.ibkr_utils.orders_utils import suppress_messages
from config import BASE_URL
import requests


@patch('app.utils.ibkr_utils.orders_utils.api_post')
def test_suppress_messages_success(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'status': 'successful', 'details': 'Messages suppressed'}
    mock_post.return_value = mock_response

    message_ids = [101, 102, 103]
    suppress_messages(message_ids)

    mock_post.assert_called_once_with(
        BASE_URL + "iserver/questions/suppress",
        {"messageIds": message_ids}
    )


@patch('app.utils.ibkr_utils.orders_utils.api_post')
def test_suppress_messages_failure(mock_post):
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Error")

    mock_post.return_value = mock_response

    message_ids = [404, 500]

    with pytest.raises(requests.exceptions.HTTPError, match="400 Error"):
        suppress_messages(message_ids)

    mock_post.assert_called_once_with(
        BASE_URL + "iserver/questions/suppress",
        {"messageIds": message_ids}
    )
