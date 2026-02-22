from apscheduler.events import EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler

from app.utils.api_utils import api_post
from app.utils.logger import get_logger

# ==================== Module Initialization ====================

scheduler = BackgroundScheduler()
logger = get_logger('ibkr/ibkr/connection')


def tickle_ibkr_api():
    """Send a heartbeat request to keep the IBKR session alive.

    Sends a tickle request to the IBKR API and validates the response for
    common error states: missing session, unauthenticated user, or disconnected user.
    """
    endpoint = 'tickle'
    payload = {}

    try:
        response = api_post(endpoint, payload)
        logger.info('IBKR API tickle response: %s', response)

        # No session error
        if 'error' in response and response['error'] == 'no session':
            logger.error(f'IBKR API responded with error: {response}')
            return

        # User not authenticated or connected error
        if 'iserver' in response and 'authStatus' in response['iserver']:
            auth_status = response['iserver']['authStatus']
            if not auth_status.get('authenticated', False):
                logger.error('IBKR API responded with User is not authenticated. ', response)
                return
            if not auth_status.get('connected', False):
                logger.error('IBKR API responded with User is not connected. ', response)
                return

    except ValueError as ve:
        logger.error(f'Tickle IBKR API Error: {ve}')

    except Exception as err:
        logger.error(f'Unexpected error while tickling IBKR API: {err}')


def log_missed_job(event):
    """Log a warning when a scheduled APScheduler job is missed.

    Args:
        event: APScheduler event object containing job_id and scheduled_run_time
    """
    if event.scheduled_run_time:
        logger.warning(
            f'Run time of job \'{event.job_id}\' was missed. '
            f'Scheduled run time: {event.scheduled_run_time}.'
        )


def start_ibkr_scheduler():
    """Start the IBKR connection heartbeat scheduler.

    Configures and starts an APScheduler background job that sends a tickle
    request every 60 seconds to maintain the IBKR session. Registers a listener
    to log any missed job executions.
    """
    # Add the heartbeat job to the scheduler
    scheduler.add_job(tickle_ibkr_api, 'interval', seconds=60, coalesce=True, max_instances=5)

    # Listen for missed job events
    scheduler.add_listener(log_missed_job, EVENT_JOB_MISSED)

    scheduler.start()
