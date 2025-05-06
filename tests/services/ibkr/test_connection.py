from unittest.mock import MagicMock

from apscheduler.events import JobExecutionEvent

from app.services.ibkr.connection import tickle_ibkr_api, log_missed_job, start_ibkr_scheduler


def test_tickle_ibkr_api_success(mock_logger_connection, mock_api_post_connection):
    # Mock API to return a successful response
    mock_api_post_connection.return_value = {"success": True}

    # Call the tickle_ibkr_api function
    tickle_ibkr_api()

    # Verify API was called correctly and success response was logged
    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.info.assert_called_once_with("IBKR API tickle response: %s", {"success": True})


def test_tickle_ibkr_api_no_session_error(mock_logger_connection, mock_api_post_connection):
    # Mock API to return a no session error response
    mock_api_post_connection.return_value = {"error": "no session"}

    # Call the tickle_ibkr_api function with error response
    tickle_ibkr_api()

    # Verify API was called correctly and error was logged
    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.error.assert_called_once()


def test_tickle_ibkr_api_not_authenticated(mock_logger_connection, mock_api_post_connection):
    # Mock API to return a response indicating user is not authenticated
    mock_api_post_connection.return_value = {
        "iserver": {
            "authStatus": {
                "authenticated": False,
                "connected": True
            }
        }
    }

    # Call the tickle_ibkr_api function with not authenticated status
    tickle_ibkr_api()

    # Verify API was called correctly and authentication error was logged
    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.error.assert_called_once()


def test_tickle_ibkr_api_not_connected(mock_logger_connection, mock_api_post_connection):
    # Mock API to return a response indicating user is authenticated but not connected
    mock_api_post_connection.return_value = {
        "iserver": {
            "authStatus": {
                "authenticated": True,
                "connected": False
            }
        }
    }

    # Call the tickle_ibkr_api function with not connected status
    tickle_ibkr_api()

    # Verify API was called correctly and connection error was logged
    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.error.assert_called_once()


def test_tickle_ibkr_api_value_error(mock_logger_connection, mock_api_post_connection):
    # Mock API to raise a ValueError when called
    mock_api_post_connection.side_effect = ValueError("Test error")

    # Call the tickle_ibkr_api function which should handle the ValueError
    tickle_ibkr_api()

    # Verify API was called correctly and specific error message was logged
    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.error.assert_called_once_with("Tickle IBKR API Error: Test error")


def test_tickle_ibkr_api_unexpected_error(mock_logger_connection, mock_api_post_connection):
    # Mock API to raise a generic Exception when called
    mock_api_post_connection.side_effect = Exception("Test error")

    # Call the tickle_ibkr_api function which should handle the unexpected exception
    tickle_ibkr_api()

    # Verify API was called correctly and generic error message was logged
    mock_api_post_connection.assert_called_once_with("tickle", {})
    mock_logger_connection.error.assert_called_once_with("Unexpected error while tickling IBKR API: Test error")


def test_log_missed_job(mock_logger_connection):
    # Create a mock JobExecutionEvent with scheduled time and job ID
    event = MagicMock(spec=JobExecutionEvent)
    event.scheduled_run_time = "2023-01-01 12:00:00"
    event.job_id = "test_job"

    # Call the log_missed_job function with the mock event
    log_missed_job(event)

    # Verify that a warning was logged about the missed job
    mock_logger_connection.warning.assert_called_once()


def test_start_ibkr_scheduler(mock_scheduler):
    # Call the start_ibkr_scheduler function to initialize the scheduler
    start_ibkr_scheduler()

    # Verify scheduler was configured correctly with job, listener, and started
    mock_scheduler.add_job.assert_called_once()
    mock_scheduler.add_listener.assert_called_once()
    mock_scheduler.start.assert_called_once()
