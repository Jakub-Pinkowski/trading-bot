from unittest.mock import patch, MagicMock

from app.utils.routes_utils.webhook_utils import (
    validate_ip,
    parse_request_data,
    save_alert_data_to_file,
    safe_process_trading_data
)
from config import ALLOWED_IPS


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

    # Make sure it's actually not in the allowed list
    if not_allowed_ip in ALLOWED_IPS:
        not_allowed_ip = "192.168.1.101"

    # Mock abort to avoid actually aborting the test
    with patch('app.utils.routes_utils.webhook_utils.abort') as mock_abort:
        validate_ip(not_allowed_ip)
        mock_abort.assert_called_once_with(403)


def test_parse_request_data_json():
    """Test that parse_request_data correctly parses JSON data"""

    # Create a mock request with JSON content type
    mock_request = MagicMock()
    mock_request.content_type = 'application/json'
    mock_request.get_json.return_value = {"key": "value"}

    # Parse the request data
    result = parse_request_data(mock_request)

    # Verify the result
    assert result == {"key": "value"}
    mock_request.get_json.assert_called_once()


def test_parse_request_data_unsupported():
    """Test that parse_request_data aborts with 400 for unsupported content types"""

    # Create a mock request with unsupported content type
    mock_request = MagicMock()
    mock_request.content_type = 'text/plain'

    # Mock abort to avoid actually aborting the test
    with patch('app.utils.routes_utils.webhook_utils.abort') as mock_abort:
        parse_request_data(mock_request)
        mock_abort.assert_called_once_with(400, description='Unsupported Content-Type')


@patch('app.utils.routes_utils.webhook_utils.load_file')
@patch('app.utils.routes_utils.webhook_utils.save_file')
@patch('app.utils.routes_utils.webhook_utils.datetime')
def test_save_alert_data_to_file(mock_datetime, mock_save_file, mock_load_file):
    """Test that save_alert_data_to_file correctly saves alert data to a file"""

    # Mock datetime to return a fixed date
    mock_now = MagicMock()
    mock_now.strftime.side_effect = lambda fmt: "23-05-01 10:30:45" if fmt == "%y-%m-%d %H:%M:%S" else "2023-05-01"
    mock_datetime.now.return_value = mock_now

    # Mock load_file to return an empty dict
    mock_load_file.return_value = {}

    # Call the function with test data
    data = {"symbol": "AAPL", "side": "B", "price": 150.0}
    save_alert_data_to_file(data, "alerts_dir")

    # Verify the mocks were called correctly
    mock_load_file.assert_called_once()
    mock_save_file.assert_called_once()
    # Check that the data was saved with the timestamp as the key
    saved_data = mock_save_file.call_args[0][0]
    assert "23-05-01 10:30:45" in saved_data
    assert saved_data["23-05-01 10:30:45"] == data


@patch('app.utils.routes_utils.webhook_utils.load_file')
@patch('app.utils.routes_utils.webhook_utils.save_file')
def test_save_alert_data_to_file_dummy(mock_save_file, mock_load_file):
    """Test that save_alert_data_to_file doesn't save dummy data"""

    # Call the function with dummy data
    data = {"dummy": True, "symbol": "AAPL"}
    save_alert_data_to_file(data, "alerts_dir")

    # Verify that neither load_file nor save_file were called
    mock_load_file.assert_not_called()
    mock_save_file.assert_not_called()


@patch('app.utils.routes_utils.webhook_utils.process_trading_data')
def test_safe_process_trading_data_success(mock_process_trading_data):
    """Test that safe_process_trading_data successfully processes data"""

    # Call the function with test data
    data = {"symbol": "AAPL", "side": "B", "price": 150.0}
    safe_process_trading_data(data)

    # Verify that process_trading_data was called with the correct data
    mock_process_trading_data.assert_called_once_with(data)


@patch('app.utils.routes_utils.webhook_utils.process_trading_data')
@patch('app.utils.routes_utils.webhook_utils.logger')
def test_safe_process_trading_data_exception(mock_logger, mock_process_trading_data):
    """Test that safe_process_trading_data handles exceptions"""

    # Make process_trading_data raise an exception
    mock_process_trading_data.side_effect = Exception("Test error")

    # Call the function with test data
    data = {"symbol": "AAPL", "side": "B", "price": 150.0}
    safe_process_trading_data(data)

    # Verify that process_trading_data was called and the exception was logged
    mock_process_trading_data.assert_called_once_with(data)
    mock_logger.exception.assert_called_once()
