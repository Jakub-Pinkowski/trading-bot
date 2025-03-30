import requests
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def api_get(endpoint):
    url = endpoint
    response = requests.get(url=url, verify=False)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP GET Error: {response.status_code} - {response.text}")
        raise e
    return response.json()


def api_post(endpoint, payload):
    url = endpoint
    response = requests.post(url=url, json=payload, verify=False)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP POST Error: {response.status_code} - {response.text}")
        raise e
    return response.json()
