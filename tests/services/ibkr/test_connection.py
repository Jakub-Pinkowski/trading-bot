from unittest.mock import MagicMock

import pytest
from apscheduler.events import JobExecutionEvent

from app.services.ibkr.connection import tickle_ibkr_api, log_missed_job, start_ibkr_scheduler


@pytest.fixture
def mock_logger(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr("app.services.ibkr.connection.logger", logger)
    return logger


@pytest.fixture
def mock_api_post(monkeypatch):
    api_post = MagicMock()
    monkeypatch.setattr("app.services.ibkr.connection.api_post", api_post)
    return api_post


@pytest.fixture
def mock_scheduler(monkeypatch):
    scheduler = MagicMock()
    monkeypatch.setattr("app.services.ibkr.connection.scheduler", scheduler)
    return scheduler


def test_tickle_ibkr_api_success(mock_logger, mock_api_post):
    mock_api_post.return_value = {"success": True}

    tickle_ibkr_api()

    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.info.assert_called_once_with("IBKR API tickle response: %s", {"success": True})


def test_tickle_ibkr_api_no_session_error(mock_logger, mock_api_post):
    mock_api_post.return_value = {"error": "no session"}

    tickle_ibkr_api()

    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.error.assert_called_once()


def test_tickle_ibkr_api_not_authenticated(mock_logger, mock_api_post):
    mock_api_post.return_value = {
        "iserver": {
            "authStatus": {
                "authenticated": False,
                "connected": True
            }
        }
    }

    tickle_ibkr_api()

    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.error.assert_called_once()


def test_tickle_ibkr_api_not_connected(mock_logger, mock_api_post):
    mock_api_post.return_value = {
        "iserver": {
            "authStatus": {
                "authenticated": True,
                "connected": False
            }
        }
    }

    tickle_ibkr_api()

    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.error.assert_called_once()


def test_tickle_ibkr_api_value_error(mock_logger, mock_api_post):
    mock_api_post.side_effect = ValueError("Test error")

    tickle_ibkr_api()

    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.error.assert_called_once_with("Tickle IBKR API Error: Test error")


def test_tickle_ibkr_api_unexpected_error(mock_logger, mock_api_post):
    mock_api_post.side_effect = Exception("Test error")

    tickle_ibkr_api()

    mock_api_post.assert_called_once_with("tickle", {})
    mock_logger.error.assert_called_once_with("Unexpected error while tickling IBKR API: Test error")


def test_log_missed_job(mock_logger):
    event = MagicMock(spec=JobExecutionEvent)
    event.scheduled_run_time = "2023-01-01 12:00:00"
    event.job_id = "test_job"

    log_missed_job(event)

    mock_logger.warning.assert_called_once()


def test_start_ibkr_scheduler(mock_scheduler):
    start_ibkr_scheduler()

    mock_scheduler.add_job.assert_called_once()
    mock_scheduler.add_listener.assert_called_once()
    mock_scheduler.start.assert_called_once()
