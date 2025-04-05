import os
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import abort

from app.utils.file_utils import load_file, save_file

ALLOWED_IPS = {
    '52.89.214.238',
    '34.212.75.30',
    '54.218.53.128',
    '52.32.178.7',
    '127.0.0.1',
    'localhost'
}


def validate_ip(remote_addr):
    if remote_addr not in ALLOWED_IPS:
        abort(403)


def parse_request_data(request):
    if request.content_type.startswith('application/json'):
        data = request.get_json()
    else:
        abort(400, description='Unsupported Content-Type')

    return data


def save_alert_data_to_file(data, alerts_dir, timezone="Europe/Berlin"):
    # Don't save if it's just a dummy data
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
