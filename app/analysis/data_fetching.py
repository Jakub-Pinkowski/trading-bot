import os
import time
from datetime import datetime, timedelta
from glob import glob

import pandas as pd

from app.analysis.analysis_utils.data_fetching_utils import save_trades_data
from app.utils.api_utils import api_get
from app.utils.file_utils import load_file
from app.utils.logger import get_logger
from config import DATA_DIR

logger = get_logger('analysis/data_fetching')

# ==================== Module Configuration ====================

TIMEFRAME_TO_ANALYZE = 7

# ==================== Module Paths ====================

ALERTS_DIR = DATA_DIR / "alerts"
IBKR_ALERTS_DIR = ALERTS_DIR / "ibkr_alerts"
TV_ALERTS_DIR = ALERTS_DIR / "tv_alerts"
TRADES_DIR = ALERTS_DIR / "trades"


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


def get_ibkr_alerts_data():
    alerts_df = load_data_from_json_files(
        directory=IBKR_ALERTS_DIR,
        file_prefix='ibkr_alerts',
        date_fields=['timestamp'],
        datetime_format='%y-%m-%d %H:%M:%S',
        index_name='timestamp'
    )

    if not alerts_df.empty:
        return alerts_df
    else:
        return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'price'])


def get_tv_alerts_data():
    files = [f for f in os.listdir(TV_ALERTS_DIR) if f.startswith('TradingView_Alerts_Log_') and f.endswith('.csv')]

    if not files:
        logger.warning(f'No files found in \'{TV_ALERTS_DIR}\' with prefix \'TradingView_Alerts_Log_\'.')
        return pd.DataFrame()  # Return an empty DataFrame if missing

    try:
        files.sort(
            key=lambda x: datetime.strptime(x.replace('TradingView_Alerts_Log_', '').replace('.csv', ''), '%Y-%m-%d'),
            reverse=True
        )
    except ValueError as err:
        logger.error(f'Error parsing dates from filenames: {err}')
        return pd.DataFrame()

    latest_file = files[0]
    alerts_file_path = os.path.join(TV_ALERTS_DIR, latest_file)

    if not os.path.exists(alerts_file_path):
        logger.warning(f'The file \'{alerts_file_path}\' does not exist.')
        return pd.DataFrame()

    try:
        alerts_df = pd.read_csv(alerts_file_path)
        return alerts_df
    except Exception as err:
        logger.error(f'Error reading the alerts file: {err}')
        return pd.DataFrame()


def get_trades_data():
    trades_df = load_data_from_json_files(
        directory=TRADES_DIR,
        file_prefix='trades',
        date_fields=['trade_time'],
        datetime_format='%Y%m%d-%H:%M:%S',
        index_name='trade_time'
    )

    if not trades_df.empty:
        trades_df = trades_df.sort_values('trade_time').reset_index(drop=True)
        seven_days_ago = datetime.now() - timedelta(days=TIMEFRAME_TO_ANALYZE)
        trades_last_7_days = trades_df[trades_df['trade_time'] >= seven_days_ago]
        if not trades_last_7_days.empty:
            return trades_last_7_days

    fetch_result = fetch_trades_data()
    if fetch_result.get('success'):
        return get_trades_data()
    else:
        logger.error(f'Failed to fetch trades data: {fetch_result.get('error')}')
        return pd.DataFrame(columns=['conid', 'side', 'price', 'trade_time'])


def fetch_trades_data(max_retries=3, retry_delay=2):
    endpoint = 'iserver/account/trades?days=7'
    attempt = 0

    while attempt < max_retries:
        try:
            trades_json = api_get(endpoint)
            if trades_json:
                save_trades_data(trades_json, TRADES_DIR)
                logger.info('Trades fetched successfully.')
                return {'success': True, 'message': 'Trades fetched successfully'}
            attempt += 1
            if attempt < max_retries:
                logger.warning(f'No data returned from IBKR API, retrying ({attempt}/{max_retries})...')
                time.sleep(retry_delay)
        except Exception as err:
            logger.error(f'Unexpected error during trades fetch: {err}')
            return {'success': False, 'error': f'Unexpected error: {err}'}

    logger.error('No data returned from IBKR API after multiple retries.')
    return {'success': False, 'error': 'No data returned from IBKR API after multiple retries'}
