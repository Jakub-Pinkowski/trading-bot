import requests
import urllib3

from config import BASE_URL
from app.utils.ibkr_helpers import api_get

# TODO: Make sure we tickle the API every 1 minute

# Disable SSL warnings only if needed for the specific context.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def check_connection():
    endpoint = "iserver/auth/status"
    auth_response = api_get(BASE_URL + endpoint)

    if auth_response.status_code != 200:
        raise Exception(f"Error: Authentication request failed with status code {auth_response.status_code}")

    json_response = auth_response.json()

    if not (json_response.get('authenticated') and json_response.get('connected') and json_response.get('fail') == ''):
        raise Exception(f"Error: Authentication response invalid: {json_response}")
