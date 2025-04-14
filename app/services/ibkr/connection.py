from apscheduler.events import EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler

from app.utils.api_utils import api_get, api_post
from app.utils.logger import get_logger
from config import BASE_URL

scheduler = BackgroundScheduler()
logger = get_logger()


def tickle_ibkr_api():
    endpoint = "tickle"
    payload = {}

    try:
        api_post(BASE_URL + endpoint, payload)
    except Exception as e:
        logger.error(f"Error tickling IBKR API: {e}")


def log_missed_job(event):
    if event.scheduled_run_time:
        logger.warning(
            f"Run time of job '{event.job_id}' was missed. "
            f"Scheduled run time: {event.scheduled_run_time}."
        )


# Start the IBKR scheduler
def start_ibkr_scheduler():
    # Add the job to the scheduler
    scheduler.add_job(tickle_ibkr_api, 'interval', seconds=60, coalesce=True, max_instances=5)

    # Listen for missed events
    scheduler.add_listener(log_missed_job, EVENT_JOB_MISSED)

    scheduler.start()


def check_connection():
    endpoint = "iserver/auth/status"
    auth_response = api_get(BASE_URL + endpoint)

    if not (auth_response.get('authenticated') and auth_response.get('connected') and auth_response.get('fail') == ''):
        raise Exception(f"Invalid authentication response: {auth_response}")
