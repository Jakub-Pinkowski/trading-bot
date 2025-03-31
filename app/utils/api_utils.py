import json

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
    response = requests.get(url=endpoint, verify=False, headers=get_headers())
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP GET Error: {response.status_code} - {response.text}")
        raise e
    return response.json()


def api_post(endpoint, payload):
    response = requests.post(url=endpoint, json=payload, verify=False, headers=get_headers(payload))
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP POST Error: {response.status_code} - {response.text}")
        raise e
    return response.json()
