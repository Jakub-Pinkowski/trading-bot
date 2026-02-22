from unittest.mock import patch, MagicMock

from werkzeug.exceptions import Forbidden

from app.routes.webhook import validate_ip, parse_request_data, save_alert_data_to_file
from config import ALLOWED_IPS


# ==================== validate_ip Tests ====================

def test_validate_ip_allowed():
    """Test that validate_ip allows IPs in the ALLOWED_IPS list"""

    # Use any allowed IP from the config
    allowed_ip = next(iter(ALLOWED_IPS)) if ALLOWED_IPS else "127.0.0.1"

    # This should not raise an exception
    validate_ip(allowed_ip)


def test_validate_ip_not_allowed():
    """Test that validate_ip aborts with 403 for IPs not in the ALLOWED_IPS list"""

    # Use an IP that's not in the allowed list
    not_allowed_ip = "192.168.1.100"
    if not_allowed_ip in ALLOWED_IPS:
        not_allowed_ip = "192.168.1.101"

    # Mock abort to avoid actually aborting the test
    with patch('app.routes.webhook.abort') as mock_abort:
        validate_ip(not_allowed_ip)
        mock_abort.assert_called_once_with(403)


# ==================== parse_request_data Tests ====================

def test_parse_request_data_json():
    """Test that parse_request_data correctly parses JSON data"""

    # Create a mock request flagged as JSON
    mock_request = MagicMock()
    mock_request.is_json = True
    mock_request.get_json.return_value = {"key": "value"}

    # Parse the request data
    result = parse_request_data(mock_request)

    # Verify the result
    assert result == {"key": "value"}
    mock_request.get_json.assert_called_once()


def test_parse_request_data_unsupported():
    """Test that parse_request_data aborts with 400 for unsupported content types"""

    # Create a mock request not flagged as JSON
    mock_request = MagicMock()
    mock_request.is_json = False

    # Mock abort to avoid actually aborting the test
    with patch('app.routes.webhook.abort') as mock_abort:
        parse_request_data(mock_request)
        mock_abort.assert_called_once_with(400, description='Unsupported Content-Type')


# ==================== save_alert_data_to_file Tests ====================

@patch('app.routes.webhook.load_file')
@patch('app.routes.webhook.save_file')
@patch('app.routes.webhook.datetime')
def test_save_alert_data_to_file(mock_datetime, mock_save_file, mock_load_file):
    """Test that save_alert_data_to_file correctly saves alert data to a file"""

    # Mock datetime to return a fixed date
    mock_now = MagicMock()
    mock_now.strftime.side_effect = lambda fmt: "23-05-01 10:30:45" if fmt == "%y-%m-%d %H:%M:%S" else "2023-05-01"
    mock_datetime.now.return_value = mock_now
    mock_load_file.return_value = {}

    # Call the function with test data
    data = {"symbol": "AAPL", "side": "B", "price": 150.0}
    save_alert_data_to_file(data, "alerts_dir")

    # Verify the mocks were called correctly
    mock_load_file.assert_called_once()
    mock_save_file.assert_called_once()
    saved_data = mock_save_file.call_args[0][0]
    assert "23-05-01 10:30:45" in saved_data
    assert saved_data["23-05-01 10:30:45"] == data


@patch('app.routes.webhook.load_file')
@patch('app.routes.webhook.save_file')
def test_save_alert_data_to_file_skips_dummy_yes(mock_save_file, mock_load_file):
    """Test that save_alert_data_to_file doesn't save signals with dummy='YES'"""

    # Call the function with a dummy signal
    data = {"dummy": "YES", "symbol": "AAPL"}
    save_alert_data_to_file(data, "alerts_dir")

    # Verify that neither load_file nor save_file were called
    mock_load_file.assert_not_called()
    mock_save_file.assert_not_called()


@patch('app.routes.webhook.load_file')
@patch('app.routes.webhook.save_file')
@patch('app.routes.webhook.datetime')
def test_save_alert_data_to_file_saves_dummy_no(mock_datetime, mock_save_file, mock_load_file):
    """Test that save_alert_data_to_file saves signals with dummy='NO' (real trade)"""

    # Mock datetime to return a fixed date
    mock_now = MagicMock()
    mock_now.strftime.side_effect = lambda fmt: "23-05-01 10:30:45" if fmt == "%y-%m-%d %H:%M:%S" else "2023-05-01"
    mock_datetime.now.return_value = mock_now
    mock_load_file.return_value = {}

    # Call the function with dummy='NO' — this is a real live trade
    data = {"dummy": "NO", "symbol": "AAPL", "side": "B"}
    save_alert_data_to_file(data, "alerts_dir")

    # Verify that the trade was saved to file
    mock_load_file.assert_called_once()
    mock_save_file.assert_called_once()


# ==================== webhook_route Tests ====================

def test_webhook_successful(mock_process_trading_data, mock_validate_ip, client):
    """Test that webhook endpoint successfully processes valid requests"""

    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    mock_validate_ip.assert_called_once_with('127.0.0.1')
    mock_process_trading_data.assert_called_once_with({"data": "valid"})
    assert response.status_code == 200


def test_webhook_unallowed_ip(mock_validate_ip, client):
    """Test that webhook endpoint returns 403 Forbidden for unallowed IP addresses"""

    mock_validate_ip.side_effect = Forbidden(description='Forbidden IP')

    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '10.10.10.10'})

    mock_validate_ip.assert_called_once_with('10.10.10.10')
    assert response.status_code == 403
    assert b'Forbidden IP' in response.data


def test_webhook_bad_request_no_json(mock_validate_ip, client):
    """Test that webhook endpoint returns 400 Bad Request for non-JSON content"""

    response = client.post('/webhook',
                           data="some data",
                           headers={'Content-Type': 'text/plain'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    mock_validate_ip.assert_called_once_with('127.0.0.1')
    assert response.status_code == 400
    assert b'Unsupported Content-Type' in response.data


def test_webhook_processing_error_still_returns_200(mock_process_trading_data, mock_validate_ip, client):
    """Test that webhook endpoint returns 200 even when processing raises an exception.

    TradingView requires a 200 response or it will keep retrying the same signal,
    so exceptions from process_trading_data are caught and logged rather than propagated.
    """

    mock_process_trading_data.side_effect = Exception('Internal processing error')

    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    mock_validate_ip.assert_called_once_with('127.0.0.1')
    mock_process_trading_data.assert_called_once_with({"data": "valid"})
    assert response.status_code == 200
