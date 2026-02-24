from apscheduler.events import EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler

from app.utils.api_utils import api_post
from app.utils.logger import get_logger

# ==================== Module Initialization ====================

scheduler = BackgroundScheduler()
logger = get_logger('ibkr/connection')


# ==================== Helper Functions ====================

def _tickle_ibkr_api():
    """
    Send a heartbeat request to keep the IBKR session alive.

    Sends a tickle request to the IBKR API and checks the response for common
    error states: missing session, unauthenticated user, or disconnected user.
    Any of these conditions is logged as an error but does not raise, so the
    scheduler continues running and retries on the next interval.
    """
    try:
        response = api_post('tickle', {})
        logger.info(f'IBKR API tickle response: {response}')

        # No session error
        if 'error' in response and response['error'] == 'no session':
            logger.error(f'IBKR API responded with no session error: {response}')
            return

        # User not authenticated or connected error
        if 'iserver' in response and 'authStatus' in response['iserver']:
            auth_status = response['iserver']['authStatus']
            if not auth_status.get('authenticated', False):
                logger.error(f'IBKR API responded with user not authenticated: {response}')
                return
            if not auth_status.get('connected', False):
                logger.error(f'IBKR API responded with user not connected: {response}')
                return

    except Exception as err:
        logger.error(f'Unexpected error while tickling IBKR API: {err}')


# ==================== Scheduler ====================

def start_ibkr_scheduler():
    """ Start the IBKR connection heartbeat scheduler."""
    # Send a heartbeat every 60 seconds to keep the IBKR session alive
    scheduler.add_job(_tickle_ibkr_api, 'interval', seconds=60, coalesce=True, max_instances=5)

    # Log a warning whenever a scheduled job is missed
    def on_job_missed(event):
        logger.warning(
            f"Run time of job '{event.job_id}' was missed. "
            f"Scheduled run time: {event.scheduled_run_time}."
        )

    scheduler.add_listener(on_job_missed, EVENT_JOB_MISSED)

    scheduler.start()
