from unittest.mock import patch

import pytest
from werkzeug.exceptions import Forbidden

from app.routes.webhook import webhook_blueprint


@patch('app.routes.webhook.validate_ip')
@patch('app.routes.webhook.process_trading_data')
def test_webhook_successful(mock_process_trading_data, mock_validate_ip, client):
    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    mock_validate_ip.assert_called_once_with('127.0.0.1')
    mock_process_trading_data.assert_called_once_with({"data": "valid"})
    assert response.status_code == 200


@patch('app.routes.webhook.validate_ip')
def test_webhook_unallowed_ip(mock_validate_ip, client):
    mock_validate_ip.side_effect = Forbidden(description='Forbidden IP')

    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '10.10.10.10'})

    mock_validate_ip.assert_called_once_with('10.10.10.10')
    assert response.status_code == 403
    assert b'Forbidden IP' in response.data


@patch('app.routes.webhook.validate_ip')
def test_webhook_bad_request_no_json(mock_validate_ip, client):
    response = client.post('/webhook',
                           data="some data",
                           headers={'Content-Type': 'text/plain'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    mock_validate_ip.assert_called_once_with('127.0.0.1')
    assert response.status_code == 400
    assert b'Unsupported Content-Type' in response.data


@patch('app.routes.webhook.validate_ip')
@patch('app.routes.webhook.process_trading_data')
def test_webhook_internal_server_error(mock_process_trading_data, mock_validate_ip, client):
    mock_process_trading_data.side_effect = Exception('Internal processing error')

    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    mock_validate_ip.assert_called_once_with('127.0.0.1')
    mock_process_trading_data.assert_called_once_with({"data": "valid"})
    assert response.status_code == 500
    assert b'Internal processing error' in response.data
