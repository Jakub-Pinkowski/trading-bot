import os
from datetime import timedelta, datetime

import pandas as pd
from tvDatafeed import TvDatafeed, Interval

from app.utils.logger import get_logger

# Set up logging using app.utils logger
logger = get_logger('tw_fetching')

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def detect_datetime_gaps(data, interval_label, symbol):
    """ Detect gaps in the datetime index of the data. Only log gaps larger than 4 days. """
    if len(data) < 2:
        return []

    four_day_threshold = timedelta(days=4)
    gaps = []
    sorted_index = data.index.sort_values()

    for i in range(1, len(sorted_index)):
        current_time = sorted_index[i]
        previous_time = sorted_index[i - 1]
        actual_gap = current_time - previous_time

        # Only log gaps larger than 4 days
        if actual_gap > four_day_threshold:
            gaps.append((previous_time, current_time, actual_gap))
            msg = (f"Data gap detected in {symbol} {interval_label}: "
                   f"from {RED}{previous_time}{RESET} to {RED}{current_time}{RESET} "
                   f"(duration: {RED}{actual_gap}{RESET})")
            logger.warning(msg)

    return gaps


def filter_data_from_2020(data):
    """Filter out any data points with timestamps before 2020."""
    if data is None or len(data) == 0:
        return data

    # Create a 2020 threshold
    year_2020_threshold = datetime(2020, 1, 1)

    # Filter data to only include dates from 2020 onwards
    filtered_data = data[data.index >= year_2020_threshold]

    # Log filtering information if any data was removed
    original_count = len(data)
    filtered_count = len(filtered_data)
    if original_count > filtered_count:
        removed_count = original_count - filtered_count
        logger.info(f"Filtered out {removed_count} data points from before 2020 "
                    f"(kept {filtered_count} out of {original_count} total)")

    return filtered_data


def print_data_date_range(data, symbol, interval_label):
    """Print the first and last date of the dataset."""
    if data is None or len(data) == 0:
        print(f"No data available for {symbol} {interval_label}")
        return

    first_date = data.index.min()
    last_date = data.index.max()
    print(f"Data range for {symbol} {interval_label}: {CYAN}{first_date}{RESET} to {CYAN}{last_date}{RESET}")


# Initialize tvDatafeed
tv = TvDatafeed()

# Symbols to fetch (add "2!" suffix for 2nd continuous contract)
base_symbols = ['MZL', 'MET', 'RTY']

cont_suffix = "1!"
symbols = {sym: sym + cont_suffix for sym in base_symbols}
exchange = 'CBOT'

# Intervals to fetch
intervals = {
    '5m': Interval.in_5_minute,
    '15m': Interval.in_15_minute,
    '30m': Interval.in_30_minute,
    '1h': Interval.in_1_hour,
    '2h': Interval.in_2_hour,
    '4h': Interval.in_4_hour,
    '1d': Interval.in_daily,
}

# data
MAX_BARS = 100000


def main():
    """Main function to fetch and process trading data."""
    for sym_folder, symbol in symbols.items():
        output_dir = os.path.join('data/historical_data', cont_suffix, sym_folder)
        os.makedirs(output_dir, exist_ok=True)
        for label, interval in intervals.items():
            print(f"Downloading ({label}) data for {symbol}...")
            data = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                n_bars=MAX_BARS,
                fut_contract=None
            )

            # Filter out data before 2020
            data = filter_data_from_2020(data)

            file_path = os.path.join(output_dir, f"{sym_folder}_{label}.parquet")

            # Check if a file exists and append unique entries
            if os.path.exists(file_path):
                try:
                    # Load existing data
                    existing_data = pd.read_parquet(file_path)
                    print(f"Found existing data with {len(existing_data)} rows for {sym_folder} {label}")

                    # Filter existing data to remove anything before 2020
                    existing_data = filter_data_from_2020(existing_data)

                    # Combine new and existing data, removing duplicates
                    # Datetime is what makes entries unique
                    combined_data = pd.concat([existing_data, data])
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data = combined_data.sort_index()

                    # Final filter to ensure no pre-2020 data remains
                    combined_data = filter_data_from_2020(combined_data)

                    # Check for datetime gaps in the combined data
                    gaps = detect_datetime_gaps(combined_data, label, symbol)

                    # Save combined data
                    combined_data.to_parquet(file_path)
                    new_entries = len(combined_data) - len(existing_data)
                    print(f"Appended {GREEN}{new_entries}{RESET} unique entries for {sym_folder} {label} (total: {len(combined_data)} rows)")

                    # Print first and last date of the final dataset
                    print_data_date_range(combined_data, symbol, label)

                except Exception as e:
                    print(f"Error reading existing file {file_path}: {e}")
                    print(f"Overwriting with new data...")

                    # Check for datetime gaps in the new data before overwriting
                    gaps = detect_datetime_gaps(data, label, symbol)

                    data.to_parquet(file_path)
                    print(f"Saved {label} for {sym_folder} to {file_path}")

                    # Print first and last date of the final dataset
                    print_data_date_range(data, symbol, label)
            else:
                # File doesn't exist, create new
                # Check for datetime gaps in the new data
                gaps = detect_datetime_gaps(data, label, symbol)

                data.to_parquet(file_path)
                print(f"Created new file with {len(data)} rows for {sym_folder} {label}")

                # Print first and last date of the final dataset
                print_data_date_range(data, symbol, label)

    print("All data downloaded")


if __name__ == "__main__":
    main()
