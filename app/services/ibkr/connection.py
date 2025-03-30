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

    if auth_response.status_code != 200:
        raise Exception(f"Error: Authentication request failed with status code {auth_response.status_code}")

    json_response = auth_response.json()

    if not (json_response.get('authenticated') and json_response.get('connected') and json_response.get('fail') == ''):
        raise Exception(f"Error: Authentication response invalid: {json_response}")
