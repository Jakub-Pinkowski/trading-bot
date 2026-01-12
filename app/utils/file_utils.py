import json
import os
from glob import glob

import pandas as pd

from app.utils.logger import get_logger

logger = get_logger()


def save_to_parquet(data, file_path):
    """ Save data to a parquet file with deduplication. """
    # Create a directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Load existing data if a file exists
    if os.path.exists(file_path):
        try:
            existing = pd.read_parquet(file_path)
        except Exception as err:
            logger.error(f'Could not read existing parquet file for deduplication: {err}')
            existing = None
    else:
        existing = None

    if not isinstance(data, pd.DataFrame):
        raise ValueError('Data must be a Pandas DataFrame for parquet format.')

    # Concatenate and deduplicate if a file exists; else save data
    if existing is not None:
        concat = pd.concat([existing, data], ignore_index=True)
        deduped = concat.drop_duplicates()
    else:
        deduped = data

    # Save (overwrite) deduped data
    deduped.to_parquet(file_path, index=False)


def load_file(file_path):
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def load_data_from_json_files(directory, file_prefix, date_fields, datetime_format, index_name):
    # Retrieve a sorted list of all JSON files matching the pattern
    file_pattern = os.path.join(directory, f'{file_prefix}_*.json')
    files = sorted(glob(file_pattern))

    data_frames = []
    for file_path in files:
        file_json = load_file(file_path)

        # Convert the JSON content to a DataFrame
        if file_json:
            file_df = json_to_dataframe(
                file_json,
                date_fields=date_fields,
                datetime_format=datetime_format,
                orient='index',
                index_name=index_name
            )
            data_frames.append(file_df)

    if data_frames:
        # Combine all DataFrames into a single DataFrame and sort by index
        combined_df = pd.concat(data_frames).sort_index().reset_index(drop=True)
        return combined_df
    else:
        # Return an empty DataFrame if no valid JSON files were found
        return pd.DataFrame()


def save_file(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  # type: ignore[arg-type]


def save_to_csv(data, file_path, dictionary_columns=None):
    # Load existing data if a file exists
    if os.path.exists(file_path):
        try:
            existing = pd.read_csv(file_path)
        except Exception as err:
            logger.error(f'Could not read existing CSV for deduplication: {err}')
            existing = None
    else:
        existing = None

    if isinstance(data, pd.DataFrame):
        new_data = data.copy()
    elif isinstance(data, dict):
        columns = dictionary_columns if dictionary_columns else ['Key', 'Value']
        new_data = pd.DataFrame(list(data.items()), columns=columns)
    else:
        raise ValueError('Data must be either a Pandas DataFrame or a dictionary.')

    # Concatenate and deduplicate if a file exists; else just save data
    if existing is not None:
        concat = pd.concat([existing, new_data], ignore_index=True)
        deduped = concat.drop_duplicates()
    else:
        deduped = new_data

    # Save (overwrite) deduped data
    deduped.to_csv(file_path, index=False)


def json_to_dataframe(data, date_fields=None, datetime_format=None, orient='columns', index_name='timestamp'):
    if not data:
        return pd.DataFrame()

    # Check data type to choose a conversion method appropriately
    if isinstance(data, dict):
        df = pd.DataFrame.from_dict(data, orient=orient).reset_index()
        if orient == 'index' and 'index' in df.columns:
            df.rename(columns={'index': index_name}, inplace=True)
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError('Unsupported data format. Provide either dictionary or list.')

    # Convert specified fields to datetime objects using a given format
    if date_fields:
        for field in date_fields:
            df[field] = pd.to_datetime(df[field], format=datetime_format, errors='coerce')

    return df
