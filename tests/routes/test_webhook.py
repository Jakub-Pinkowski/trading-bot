from unittest.mock import patch, MagicMock

from app.routes.webhook import save_alert_data_to_file


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

    data = {"dummy": "YES", "symbol": "AAPL"}
    save_alert_data_to_file(data, "alerts_dir")

    mock_load_file.assert_not_called()
    mock_save_file.assert_not_called()


@patch('app.routes.webhook.load_file')
@patch('app.routes.webhook.save_file')
@patch('app.routes.webhook.datetime')
def test_save_alert_data_to_file_saves_dummy_no(mock_datetime, mock_save_file, mock_load_file):
    """Test that save_alert_data_to_file saves signals with dummy='NO' (real trade)"""

    mock_now = MagicMock()
    mock_now.strftime.side_effect = lambda fmt: "23-05-01 10:30:45" if fmt == "%y-%m-%d %H:%M:%S" else "2023-05-01"
    mock_datetime.now.return_value = mock_now
    mock_load_file.return_value = {}

    data = {"dummy": "NO", "symbol": "AAPL", "side": "B"}
    save_alert_data_to_file(data, "alerts_dir")

    mock_load_file.assert_called_once()
    mock_save_file.assert_called_once()


# ==================== webhook_route Tests ====================

def test_webhook_successful(mock_process_trading_data, client):
    """Test that webhook endpoint successfully processes valid requests"""

    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    mock_process_trading_data.assert_called_once_with({"data": "valid"})
    assert response.status_code == 200


def test_webhook_unallowed_ip(client):
    """Test that webhook endpoint returns 403 Forbidden for unallowed IP addresses"""

    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '10.10.10.10'})

    assert response.status_code == 403


def test_webhook_bad_request_no_json(client):
    """Test that webhook endpoint returns 400 Bad Request for non-JSON content"""

    response = client.post('/webhook',
                           data="some data",
                           headers={'Content-Type': 'text/plain'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    assert response.status_code == 400
    assert b'Unsupported Content-Type' in response.data


def test_webhook_processing_error_still_returns_200(mock_process_trading_data, client):
    """Test that webhook endpoint returns 200 even when processing raises an exception.

    TradingView requires a 200 response or it will keep retrying the same signal,
    so exceptions from process_trading_data are caught and logged rather than propagated.
    """

    mock_process_trading_data.side_effect = Exception('Internal processing error')

    response = client.post('/webhook',
                           json={"data": "valid"},
                           headers={'Content-Type': 'application/json'},
                           environ_base={'REMOTE_ADDR': '127.0.0.1'})

    mock_process_trading_data.assert_called_once_with({"data": "valid"})
    assert response.status_code == 200
