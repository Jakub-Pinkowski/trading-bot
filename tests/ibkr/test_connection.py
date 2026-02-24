"""
Tests for IBKR Connection Module.

Tests cover:
- API heartbeat (tickle) with successful, error, and unauthenticated responses
- Scheduler initialization and configuration
- Missed-job listener callback logging
"""
from unittest.mock import MagicMock

from app.ibkr.connection import _tickle_ibkr_api, start_ibkr_scheduler


# ==================== Test Classes ====================

class TestTickleIbkrApi:
    """Test IBKR API heartbeat (tickle) function."""

    def test_success_logs_response(self, mock_logger_connection, mock_api_post_connection):
        """Test successful tickle logs the API response."""
        mock_api_post_connection.return_value = {"success": True}

        _tickle_ibkr_api()

        mock_api_post_connection.assert_called_once_with("tickle", {})
        mock_logger_connection.info.assert_called_once_with(
            f"IBKR API tickle response: {{'success': True}}"
        )

    def test_no_session_error_logs_error(self, mock_logger_connection, mock_api_post_connection):
        """Test no-session response logs an error."""
        mock_api_post_connection.return_value = {"error": "no session"}

        _tickle_ibkr_api()

        mock_api_post_connection.assert_called_once_with("tickle", {})
        mock_logger_connection.error.assert_called_once()

    def test_not_authenticated_logs_error(self, mock_logger_connection, mock_api_post_connection):
        """Test unauthenticated status logs an error."""
        mock_api_post_connection.return_value = {
            "iserver": {
                "authStatus": {
                    "authenticated": False,
                    "connected": True,
                }
            }
        }

        _tickle_ibkr_api()

        mock_api_post_connection.assert_called_once_with("tickle", {})
        mock_logger_connection.error.assert_called_once()

    def test_not_connected_logs_error(self, mock_logger_connection, mock_api_post_connection):
        """Test connected=False status logs an error."""
        mock_api_post_connection.return_value = {
            "iserver": {
                "authStatus": {
                    "authenticated": True,
                    "connected": False,
                }
            }
        }

        _tickle_ibkr_api()

        mock_api_post_connection.assert_called_once_with("tickle", {})
        mock_logger_connection.error.assert_called_once()

    def test_unexpected_exception_logged(self, mock_logger_connection, mock_api_post_connection):
        """Test unexpected exception is caught and logged with the exception message."""
        mock_api_post_connection.side_effect = Exception("Test error")

        _tickle_ibkr_api()

        mock_api_post_connection.assert_called_once_with("tickle", {})
        mock_logger_connection.error.assert_called_once_with(
            "Unexpected error while tickling IBKR API: Test error"
        )


class TestStartIbkrScheduler:
    """Test IBKR scheduler initialization."""

    def test_scheduler_configured_and_started(self, mock_scheduler):
        """Test scheduler receives a job and listener, and is started."""
        start_ibkr_scheduler()

        mock_scheduler.add_job.assert_called_once()
        mock_scheduler.add_listener.assert_called_once()
        mock_scheduler.start.assert_called_once()

    def test_missed_job_listener_logs_warning(self, mock_logger_connection, mock_scheduler):
        """Test the missed-job listener callback logs a warning with job details."""
        start_ibkr_scheduler()

        # Extract the on_job_missed callback registered with add_listener
        on_job_missed = mock_scheduler.add_listener.call_args[0][0]

        mock_event = MagicMock()
        mock_event.job_id = "tickle_job"
        mock_event.scheduled_run_time = "2024-01-01 00:00:00"

        on_job_missed(mock_event)

        mock_logger_connection.warning.assert_called_once()
