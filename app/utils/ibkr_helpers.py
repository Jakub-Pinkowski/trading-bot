import requests

def api_get(endpoint):
    url = endpoint
    response = requests.get(url=url, verify=False)
    response.raise_for_status()
    return response

def api_post(endpoint, payload):
    url = endpoint
    response = requests.post(url=url, json=payload, verify=False)
    response.raise_for_status()
    return response.json()
