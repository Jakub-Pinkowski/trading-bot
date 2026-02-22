import os
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Blueprint, abort, request

from app.ibkr.ibkr_service import process_trading_data
from app.utils.file_utils import load_file, save_file
from app.utils.logger import get_logger
from config import ALLOWED_IPS, DATA_DIR

webhook_blueprint = Blueprint('webhook', __name__)
logger = get_logger('routes/webhook')

# ==================== Module Paths ====================

IBKR_ALERTS_DIR = DATA_DIR / "alerts" / "ibkr_alerts"


# ==================== Helper Functions ====================

def validate_ip(remote_addr):
    if remote_addr not in ALLOWED_IPS:
        abort(403)


def parse_request_data(req):
    if not req.is_json:
        abort(400, description='Unsupported Content-Type')
    return req.get_json()


def save_alert_data_to_file(data, alerts_dir, timezone='Europe/Berlin'):
    # Don't save dummy signals
    if data.get('dummy') == 'YES':
        return

    # Current timestamp with timezone
    current_dt = datetime.now(ZoneInfo(timezone))
    timestamp = current_dt.strftime('%y-%m-%d %H:%M:%S')

    # Preparing the daily file path
    daily_file_name = f'alerts_{current_dt.strftime("%Y-%m-%d")}.json'
    daily_file_path = os.path.join(alerts_dir, daily_file_name)

    # Load existing data if the file already exists
    alerts_data = load_file(daily_file_path)

    # Add new record
    alerts_data[timestamp] = data

    # Save updated data to daily file
    save_file(alerts_data, daily_file_path)


# ==================== Routes ====================

@webhook_blueprint.route('/webhook', methods=['POST'])
def webhook_route():
    validate_ip(request.remote_addr)
    data = parse_request_data(request)
    save_alert_data_to_file(data, IBKR_ALERTS_DIR)

    # Always returns 200 regardless of outcome — TradingView retries any non-200
    # response, which would cause duplicate orders
    try:
        process_trading_data(data)
    except Exception as err:
        logger.exception('Error processing webhook data %s: %s', data, err)

    return '', 200
