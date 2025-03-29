import requests
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def api_get(endpoint):
    url = endpoint
    response = requests.get(url=url, verify=False)
    response.raise_for_status()
    return response


def api_post(endpoint, payload):
    url = endpoint
    response = requests.post(url=url, json=payload, verify=False)
    response.raise_for_status()
    return response
