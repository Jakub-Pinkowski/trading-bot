import json
import os

import pandas as pd


def load_file(file_path):
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_file(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  # type: ignore[arg-type]


def save_to_csv(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check if the file exists to determine header inclusion
    if os.path.isfile(file_path):
        # Append data without headers
        data.to_csv(file_path, mode='a', index=False, header=False)
    else:
        # Write new CSV with headers
        data.to_csv(file_path, mode='w', index=False, header=True)


def json_to_dataframe(data, date_fields=None):
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data, orient='index').reset_index()

    # Rename 'index' to a more descriptive name if it's timestamp or similar
    if 'index' in df.columns:
        df.rename(columns={'index': 'timestamp'}, inplace=True)

    # Convert specified fields to datetime objects
    if date_fields:
        for field in date_fields:
            df[field] = pd.to_datetime(df[field])

    return df
