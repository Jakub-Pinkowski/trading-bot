from unittest.mock import patch, MagicMock

from apscheduler.events import JobExecutionEvent

from app.services.ibkr.connection import tickle_ibkr_api, log_missed_job, start_ibkr_scheduler


@patch('app.services.ibkr.connection.api_post')
@patch('app.services.ibkr.connection.logger')
def test_tickle_ibkr_api_success(mock_logger, mock_api_post):
    # Setup
    mock_api_post.return_value = {"success": True}

    # Execute
    tickle_ibkr_api()

    # Assert
    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.info.assert_called_once_with("IBKR API tickle response: %s", {"success": True})


@patch('app.services.ibkr.connection.api_post')
@patch('app.services.ibkr.connection.logger')
def test_tickle_ibkr_api_no_session_error(mock_logger, mock_api_post):
    # Setup
    mock_api_post.return_value = {"error": "no session"}

    # Execute
    tickle_ibkr_api()

    # Assert
    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.error.assert_called_once()


@patch('app.services.ibkr.connection.api_post')
@patch('app.services.ibkr.connection.logger')
def test_tickle_ibkr_api_not_authenticated(mock_logger, mock_api_post):
    # Setup
    mock_api_post.return_value = {
        "iserver": {
            "authStatus": {
                "authenticated": False,
                "connected": True
            }
        }
    }

    # Execute
    tickle_ibkr_api()

    # Assert
    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.error.assert_called_once()


@patch('app.services.ibkr.connection.api_post')
@patch('app.services.ibkr.connection.logger')
def test_tickle_ibkr_api_not_connected(mock_logger, mock_api_post):
    # Setup
    mock_api_post.return_value = {
        "iserver": {
            "authStatus": {
                "authenticated": True,
                "connected": False
            }
        }
    }

    # Execute
    tickle_ibkr_api()

    # Assert
    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.error.assert_called_once()


@patch('app.services.ibkr.connection.api_post')
@patch('app.services.ibkr.connection.logger')
def test_tickle_ibkr_api_value_error(mock_logger, mock_api_post):
    # Setup
    mock_api_post.side_effect = ValueError("Test error")

    # Execute
    tickle_ibkr_api()

    # Assert
    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.error.assert_called_once_with("Tickle IBKR API Error: Test error")


@patch('app.services.ibkr.connection.api_post')
@patch('app.services.ibkr.connection.logger')
def test_tickle_ibkr_api_unexpected_error(mock_logger, mock_api_post):
    # Setup
    mock_api_post.side_effect = Exception("Test error")

    # Execute
    tickle_ibkr_api()

    # Assert
    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.error.assert_called_once_with("Unexpected error while tickling IBKR API: Test error")


@patch('app.services.ibkr.connection.logger')
def test_log_missed_job(mock_logger):
    # Setup
    event = MagicMock(spec=JobExecutionEvent)
    event.scheduled_run_time = "2023-01-01 12:00:00"
    event.job_id = "test_job"

    # Execute
    log_missed_job(event)

    # Assert
    mock_logger.warning.assert_called_once()


@patch('app.services.ibkr.connection.scheduler')
def test_start_ibkr_scheduler(mock_scheduler):
    # Execute
    start_ibkr_scheduler()

    # Assert
    mock_scheduler.add_job.assert_called_once()
    mock_scheduler.add_listener.assert_called_once()
    mock_scheduler.start.assert_called_once()
