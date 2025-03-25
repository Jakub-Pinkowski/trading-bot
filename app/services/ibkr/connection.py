import requests
import urllib3

# TODO: Make sure we ticket the API every 1 minute

# Disable SSL warnings only if needed for the specific context.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def check_connection():
    base_url = "https://localhost:5001/v1/api/"
    endpoint = "iserver/auth/status"

    auth_req = requests.get(url=base_url + endpoint, verify=False)

    if auth_req.status_code != 200:
        raise Exception(f"Error: Authentication request failed with status code {auth_req.status_code}")

    json_response = auth_req.json()

    if not (json_response.get('authenticated') and json_response.get('connected') and json_response.get('fail') == ''):
        raise Exception(f"Error: Authentication response invalid: {json_response}")
