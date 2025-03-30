import json
import os

from config import BASE_DIR

CACHE_FILE_PATH = os.path.join(BASE_DIR, "data", "contracts.json")


def load_file():
    if not os.path.exists(CACHE_FILE_PATH):
        return {}

    with open(CACHE_FILE_PATH, 'r') as cache_file:
        cache_data = json.load(cache_file)
    return cache_data


def save_file(cache_data):
    with open(CACHE_FILE_PATH, 'w') as cache_file:
        json.dump(cache_data, cache_file, indent=4)  # type: ignore[arg-type]
