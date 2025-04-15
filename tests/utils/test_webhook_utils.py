from unittest.mock import Mock, patch

import pytest
from flask import Flask
from werkzeug.exceptions import HTTPException

from app.utils.webhook_utils import validate_ip, parse_request_data


@pytest.fixture
def flask_app():
    app = Flask(__name__)
    with app.app_context():
        yield app


@pytest.mark.parametrize("allowed_ip", [
    '52.89.214.238',
    'localhost',
    '127.0.0.1',
    '54.218.53.128'
])
@patch('app.utils.webhook_utils.abort')
def test_validate_ip_allowed(mock_abort, allowed_ip):
    validate_ip(allowed_ip)
    mock_abort.assert_not_called()


@patch('app.utils.webhook_utils.abort')
def test_validate_ip_disallowed(mock_abort):
    disallowed_ip = '10.10.10.10'
    validate_ip(disallowed_ip)
    mock_abort.assert_called_once_with(403)


def test_parse_request_data_json(flask_app):
    mock_request = Mock(content_type='application/json', get_json=Mock(return_value={"key": "value"}))
    with flask_app.test_request_context(json={'key': 'value'}):
        data = parse_request_data(mock_request)
        assert data == {'key': 'value'}


@patch('app.utils.webhook_utils.abort')
def test_parse_request_data_unsupported_content_type(mock_abort):
    mock_abort.side_effect = HTTPException(description='Unsupported Content-Type', response=Mock(status=400))
    mock_request = Mock(content_type='application/x-www-form-urlencoded')

    with pytest.raises(HTTPException) as exc_info:
        parse_request_data(mock_request)

    mock_abort.assert_called_once_with(400, description='Unsupported Content-Type')
    assert exc_info.value.response.status == 400
