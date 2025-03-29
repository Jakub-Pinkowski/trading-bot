import requests
import urllib3
import json
import os
import re
from config import BASE_DIR

CACHE_FILE_PATH = os.path.join(BASE_DIR, "data", "contracts.json")

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

def load_cache():
    if not os.path.exists(CACHE_FILE_PATH):
        return {}

    with open(CACHE_FILE_PATH, 'r') as cache_file:
        cache_data = json.load(cache_file)
    return cache_data

def save_cache(cache_data):
    with open(CACHE_FILE_PATH, 'w') as cache_file:
        json.dump(cache_data, cache_file, indent=4)

def parse_symbol(symbol):
    match = re.match(r'^([A-Za-z]+)', symbol)
    if not match:
        raise ValueError(f"Invalid symbol format: {symbol}")
    return match.group(1)