from apscheduler.events import EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler

from app.utils.api_utils import api_get, api_post
from app.utils.logger import get_logger
from config import BASE_URL

scheduler = BackgroundScheduler()
logger = get_logger()


# BUG: Didn't catch that I'm not authenticated
# {
#   "session": "6c8001a0cc585cd09f53d0dfe2e604b1",
#   "ssoExpires": 90381,
#   "collission": false,
#   "userId": 137713338,
#   "hmds": {
#     "error": "no bridge"
#   },
#   "iserver": {
#     "authStatus": {
#       "authenticated": false,
#       "competing": false,
#       "connected": false,
#       "message": "",
#       "MAC": "F4:03:43:DC:B4:60",
#       "serverInfo": {
#         "serverName": "JifZ27122",
#         "serverVersion": "Build 10.34.1f, Apr 9, 2025 11:51:04 AM"
#       }
#     }
#   }
# }
# Response file saved.
def tickle_ibkr_api():
    endpoint = "tickle"
    payload = {}

    try:
        response = api_post(BASE_URL + endpoint, payload)

        # No session error
        if "error" in response and response["error"] == "no session":
            logger.error(f"Tickle API responded with error: {response['error']}")
            return

        # Unauthenticated user
        if not response.get("authenticated", True):
            logger.error("User is not authenticated. Please log in.")
            return

        # Log a success message when everything runs as expected
        logger.info("Tickle IBKR API executed successfully.")




    except ValueError as ve:
        logger.error(f"Tickle IBKR API Error: {ve}")

    except Exception as err:
        logger.error(f"Unexpected error while tickling IBKR API: {err}")


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
