from unittest.mock import MagicMock

from apscheduler.events import JobExecutionEvent

from app.services.ibkr.connection import tickle_ibkr_api, log_missed_job, start_ibkr_scheduler


def test_tickle_ibkr_api_success(mock_logger_connection, mock_api_post_connection):
    mock_api_post_connection.return_value = {"success": True}

    tickle_ibkr_api()

    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.info.assert_called_once_with("IBKR API tickle response: %s", {"success": True})


def test_tickle_ibkr_api_no_session_error(mock_logger_connection, mock_api_post_connection):
    mock_api_post_connection.return_value = {"error": "no session"}

    tickle_ibkr_api()

    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.error.assert_called_once()


def test_tickle_ibkr_api_not_authenticated(mock_logger_connection, mock_api_post_connection):
    mock_api_post_connection.return_value = {
        "iserver": {
            "authStatus": {
                "authenticated": False,
                "connected": True
            }
        }
    }

    tickle_ibkr_api()

    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.error.assert_called_once()


def test_tickle_ibkr_api_not_connected(mock_logger_connection, mock_api_post_connection):
    mock_api_post_connection.return_value = {
        "iserver": {
            "authStatus": {
                "authenticated": True,
                "connected": False
            }
        }
    }

    tickle_ibkr_api()

    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.error.assert_called_once()


def test_tickle_ibkr_api_value_error(mock_logger_connection, mock_api_post_connection):
    mock_api_post_connection.side_effect = ValueError("Test error")

    tickle_ibkr_api()

    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.error.assert_called_once_with("Tickle IBKR API Error: Test error")


def test_tickle_ibkr_api_unexpected_error(mock_logger_connection, mock_api_post_connection):
    mock_api_post_connection.side_effect = Exception("Test error")

    tickle_ibkr_api()

    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.error.assert_called_once_with("Unexpected error while tickling IBKR API: Test error")


def test_log_missed_job(mock_logger_connection):
    event = MagicMock(spec=JobExecutionEvent)
    event.scheduled_run_time = "2023-01-01 12:00:00"
    event.job_id = "test_job"

    log_missed_job(event)

    mock_logger_connection.warning.assert_called_once()


def test_start_ibkr_scheduler(mock_scheduler):
    start_ibkr_scheduler()

    mock_scheduler.add_job.assert_called_once()
    mock_scheduler.add_listener.assert_called_once()
    mock_scheduler.start.assert_called_once()
