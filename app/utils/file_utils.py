import json
import os
from glob import glob

import pandas as pd


def load_file(file_path):
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def load_data_from_json_files(directory, file_prefix, date_fields, datetime_format, index_name):
    file_pattern = os.path.join(directory, f"{file_prefix}_*.json")
    files = sorted(glob(file_pattern))

    data_frames = []
    for file_path in files:
        file_json = load_file(file_path)

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
        # Combine and sort by index initially
        combined_df = pd.concat(data_frames).sort_index().reset_index(drop=True)
        return combined_df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no data found


def save_file(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  # type: ignore[arg-type]


def save_to_csv(data, file_path, dictionary_columns=None):
    try:
        if isinstance(data, pd.DataFrame):
            # Directly save the DataFrame
            data.to_csv(file_path, index=False)
        elif isinstance(data, dict):
            # Convert dictionary to DataFrame and save
            columns = dictionary_columns if dictionary_columns else ["Key", "Value"]
            df = pd.DataFrame(list(data.items()), columns=columns)
            df.to_csv(file_path, index=False)
        else:
            raise ValueError("Data must be either a Pandas DataFrame or a dictionary.")

        print(f"Data successfully saved to {file_path}")

    except Exception as e:
        print(f"An error occurred while saving to CSV: {e}")


def json_to_dataframe(data, date_fields=None, datetime_format=None, orient='columns', index_name='timestamp'):
    if not data:
        return pd.DataFrame()

    # Check data type to choose conversion method appropriately
    if isinstance(data, dict):
        df = pd.DataFrame.from_dict(data, orient=orient).reset_index()
        if orient == 'index' and 'index' in df.columns:
            df.rename(columns={'index': index_name}, inplace=True)
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported data format. Provide either dictionary or list.")

    # Convert specified fields to datetime objects using given format
    if date_fields:
        for field in date_fields:
            df[field] = pd.to_datetime(df[field], format=datetime_format, errors='coerce')

    return df
