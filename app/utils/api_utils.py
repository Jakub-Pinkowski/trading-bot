import json

import requests
import urllib3

from app.utils.logger import get_logger
from config import BASE_URL

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = get_logger()


def get_headers(payload=None):
    headers = {
        'Host': 'api.ibkr.com',
        'User-Agent': 'python-requests/IBKR-client',
        'Accept': '*/*',
        'Connection': 'keep-alive',
    }
    if payload:
        headers['Content-Length'] = str(len(json.dumps(payload)))
    return headers


def api_get(endpoint):
    url = BASE_URL + endpoint
    response = requests.get(url=url, verify=False, headers=get_headers())
    try:
        response.raise_for_status()
    except requests.HTTPError as err:
        logger.error(f"HTTP GET Error: {response.status_code} - {response.text}")
        raise err
    return response.json()


def api_post(endpoint, payload):
    url = BASE_URL + endpoint
    response = requests.post(url=url, json=payload, verify=False, headers=get_headers(payload))
    try:
        response.raise_for_status()
    except requests.HTTPError as err:
        logger.error(f"HTTP POST Error: {response.status_code} - {response.text}")
        raise err
    return response.json()
