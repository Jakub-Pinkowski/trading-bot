from werkzeug.exceptions import Forbidden


def test_webhook_successful(mock_process_trading_data, mock_validate_ip, client):
    """Test that webhook endpoint successfully processes valid requests"""
    # Setup
    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    # Assert
    mock_validate_ip.assert_called_once_with('127.0.0.1')
    mock_process_trading_data.assert_called_once_with({"data": "valid"})
    assert response.status_code == 200


def test_webhook_unallowed_ip(mock_validate_ip, client):
    """Test that webhook endpoint returns 403 Forbidden for unallowed IP addresses"""
    # Setup
    mock_validate_ip.side_effect = Forbidden(description='Forbidden IP')

    # Execute
    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '10.10.10.10'})

    # Assert
    mock_validate_ip.assert_called_once_with('10.10.10.10')
    assert response.status_code == 403
    assert b'Forbidden IP' in response.data


def test_webhook_bad_request_no_json(mock_validate_ip, client):
    """Test that webhook endpoint returns 400 Bad Request for non-JSON content"""
    # Setup & Execute
    response = client.post('/webhook',
                           data="some data",
                           headers={'Content-Type': 'text/plain'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    # Assert
    mock_validate_ip.assert_called_once_with('127.0.0.1')
    assert response.status_code == 400
    assert b'Unsupported Content-Type' in response.data


def test_webhook_internal_server_error(mock_process_trading_data, mock_validate_ip, client):
    """Test that webhook endpoint returns 500 Internal Server Error when processing fails"""
    # Setup
    mock_process_trading_data.side_effect = Exception('Internal processing error')

    # Execute
    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    # Assert
    mock_validate_ip.assert_called_once_with('127.0.0.1')
    mock_process_trading_data.assert_called_once_with({"data": "valid"})
    assert response.status_code == 500
    assert b'Internal Server Error' in response.data
