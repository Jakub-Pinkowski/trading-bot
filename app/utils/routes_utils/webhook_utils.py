import os
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import abort

from app.services.ibkr_service import process_trading_data
from app.utils.file_utils import load_file, save_file
from app.utils.logger import get_logger
from config import ALLOWED_IPS

logger = get_logger()


def validate_ip(remote_addr):
    if remote_addr not in ALLOWED_IPS:
        abort(403)


def parse_request_data(request):
    if request.content_type.startswith('application/json'):
        data = request.get_json()
        return data
    else:
        abort(400, description='Unsupported Content-Type')


def save_alert_data_to_file(data, alerts_dir, timezone="Europe/Berlin"):
    # Don't save if it's just the dummy data
    if 'dummy' in data:
        return

    # Current timestamp with timezone
    current_dt = datetime.now(ZoneInfo(timezone))
    timestamp = current_dt.strftime("%y-%m-%d %H:%M:%S")

    # Preparing the daily file path
    daily_file_name = f"alerts_{current_dt.strftime('%Y-%m-%d')}.json"
    daily_file_path = os.path.join(alerts_dir, daily_file_name)

    # Load existing data if the file already exists
    alerts_data = load_file(daily_file_path)

    # Add new record
    alerts_data[timestamp] = data

    # Save updated data to daily file
    save_file(alerts_data, daily_file_path)


def safe_process_trading_data(data):
    try:
        process_trading_data(data)
    except Exception as err:
        logger.exception(
            f"Error processing TradingView webhook with data {data}: {err}"
        )
