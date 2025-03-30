from apscheduler.schedulers.background import BackgroundScheduler

from app.utils.api_utils import api_get, api_post
from config import BASE_URL

scheduler = BackgroundScheduler()


def tickle_ibkr_api():
    endpoint = "/tickle"
    payload = {}
    try:
        api_post(BASE_URL + endpoint, payload)
    except Exception as e:
        print(f"Error tickling IBKR API: {e}")


def start_ibkr_scheduler():
    scheduler.add_job(tickle_ibkr_api, 'interval', seconds=60)
    scheduler.start()


def check_connection():
    endpoint = "iserver/auth/status"
    auth_response = api_get(BASE_URL + endpoint)

    if not (auth_response.get('authenticated') and auth_response.get('connected') and auth_response.get('fail') == ''):
        raise Exception(f"Invalid authentication response: {auth_response}")
