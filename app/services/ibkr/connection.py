from app.utils.api_helpers import api_get
from config import BASE_URL


# TODO: Make sure we tickle the API every 1 minute
# NOTE: I can place an order even if tickle returns "no session"

def check_connection():
    endpoint = "iserver/auth/status"
    auth_response = api_get(BASE_URL + endpoint)

    if auth_response.status_code != 200:
        raise Exception(f"Error: Authentication request failed with status code {auth_response.status_code}")

    json_response = auth_response.json()

    if not (json_response.get('authenticated') and json_response.get('connected') and json_response.get('fail') == ''):
        raise Exception(f"Error: Authentication response invalid: {json_response}")
