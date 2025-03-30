import json
import os


def load_file(file_path):
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  # type: ignore[arg-type]
