"""
Tests for the webhook route handler.

Tests cover:
- Alert data persistence to daily JSON files
- Dummy signal filtering
- IP allowlist enforcement
- JSON content type validation
- Order processing dispatch
- Exception handling to preserve 200 response for TradingView
"""
from unittest.mock import MagicMock

from app.routes.webhook import save_alert_data_to_file


# ==================== Test Classes ====================

class TestSaveAlertDataToFile:
    """Test alert data persistence to daily JSON files."""

    def test_saves_real_alert_with_timestamp(
        self, mock_datetime_webhook, mock_load_file_webhook, mock_save_file_webhook
    ):
        """Test real trade alert is timestamped and saved to the daily file."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = (
            lambda fmt: "23-05-01 10:30:45" if fmt == "%y-%m-%d %H:%M:%S" else "2023-05-01"
        )
        mock_datetime_webhook.now.return_value = mock_now
        mock_load_file_webhook.return_value = {}
        data = {"symbol": "AAPL", "side": "B", "price": 150.0}

        save_alert_data_to_file(data, "alerts_dir")

        mock_load_file_webhook.assert_called_once()
        mock_save_file_webhook.assert_called_once()
        saved_data = mock_save_file_webhook.call_args[0][0]
        assert "23-05-01 10:30:45" in saved_data
        assert saved_data["23-05-01 10:30:45"] == data

    def test_skips_dummy_yes_alerts(self, mock_load_file_webhook, mock_save_file_webhook):
        """Test dummy=YES alerts are not persisted."""
        data = {"dummy": "YES", "symbol": "AAPL"}

        save_alert_data_to_file(data, "alerts_dir")

        mock_load_file_webhook.assert_not_called()
        mock_save_file_webhook.assert_not_called()

    def test_saves_dummy_no_alerts(
        self, mock_datetime_webhook, mock_load_file_webhook, mock_save_file_webhook
    ):
        """Test dummy=NO alerts are treated as real trades and persisted."""
        mock_now = MagicMock()
        mock_now.strftime.side_effect = (
            lambda fmt: "23-05-01 10:30:45" if fmt == "%y-%m-%d %H:%M:%S" else "2023-05-01"
        )
        mock_datetime_webhook.now.return_value = mock_now
        mock_load_file_webhook.return_value = {}
        data = {"dummy": "NO", "symbol": "AAPL", "side": "B"}

        save_alert_data_to_file(data, "alerts_dir")

        mock_load_file_webhook.assert_called_once()
        mock_save_file_webhook.assert_called_once()


class TestWebhookRoute:
    """Test webhook HTTP endpoint."""

    def test_valid_request_returns_200(self, mock_process_trading_data, client):
        """Test valid JSON from an allowed IP is processed and returns 200."""
        response = client.post(
            '/webhook',
            json={"data": "valid"},
            headers={'Content-Type': 'application/json'},
            environ_base={'REMOTE_ADDR': '127.0.0.1'},
        )

        mock_process_trading_data.assert_called_once_with({"data": "valid"})
        assert response.status_code == 200

    def test_unallowed_ip_returns_403(self, client):
        """Test request from a non-allowlisted IP is rejected with 403."""
        response = client.post(
            '/webhook',
            json={"data": "valid"},
            headers={'Content-Type': 'application/json'},
            environ_base={'REMOTE_ADDR': '10.10.10.10'},
        )

        assert response.status_code == 403

    def test_non_json_content_type_returns_400(self, client):
        """Test request with non-JSON content type is rejected with 400."""
        response = client.post(
            '/webhook',
            data="some data",
            headers={'Content-Type': 'text/plain'},
            environ_base={'REMOTE_ADDR': '127.0.0.1'},
        )

        assert response.status_code == 400
        assert b'Unsupported Content-Type' in response.data

    def test_processing_error_still_returns_200(self, mock_process_trading_data, client):
        """Test processing exception is caught and 200 is still returned.

        TradingView requires a 200 response or it will keep retrying the same
        signal, causing duplicate orders. Exceptions must never propagate.
        """
        mock_process_trading_data.side_effect = Exception('Internal processing error')

        response = client.post(
            '/webhook',
            json={"data": "valid"},
            headers={'Content-Type': 'application/json'},
            environ_base={'REMOTE_ADDR': '127.0.0.1'},
        )

        mock_process_trading_data.assert_called_once_with({"data": "valid"})
        assert response.status_code == 200
