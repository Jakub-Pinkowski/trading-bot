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


def save_alert_data_to_file(data, file_path, timezone="Europe/Berlin"):
    # Don't save if it's just a dummy data
    if 'dummy' in data:
        return

    alerts_data = load_file(file_path)

    timestamp = datetime.now(ZoneInfo(timezone)).strftime("%y-%m-%d %H:%M:%S")
    alerts_data[timestamp] = data

    save_file(alerts_data, file_path)
