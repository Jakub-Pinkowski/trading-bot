from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests as req_lib
from flask import Blueprint, abort, request

from app.ibkr.rollover import process_rollover_data
from app.ibkr.trading import process_trading_data
from app.utils.file_utils import load_file, save_file
from app.utils.logger import get_logger
from config import ALLOWED_IPS, DATA_DIR

webhook_blueprint = Blueprint('webhook', __name__)
logger = get_logger('routes/webhook')

# ==================== Module Paths ====================

IBKR_ALERTS_DIR = DATA_DIR / "alerts" / "ibkr_alerts"


# ==================== Helper Functions ====================

def save_alert_data_to_file(data, alerts_dir, timezone='Europe/Berlin'):
    # Don't save dummy signals
    if data.get('dummy') == 'YES':
        return

    # Current timestamp with timezone
    current_dt = datetime.now(ZoneInfo(timezone))
    timestamp = current_dt.strftime('%y-%m-%d %H:%M:%S')

    # Preparing the daily file path
    daily_file_name = f'alerts_{current_dt.strftime("%Y-%m-%d")}.json'
    daily_file_path = Path(alerts_dir) / daily_file_name

    # Load existing data if the file already exists
    alerts_data = load_file(daily_file_path)

    # Add new record
    alerts_data[timestamp] = data

    # Save updated data to daily file
    save_file(alerts_data, daily_file_path)


# ==================== Routes ====================

@webhook_blueprint.before_request
def _validate_request():
    if request.remote_addr not in ALLOWED_IPS:
        abort(403)
    if not request.is_json:
        abort(400, description='Unsupported Content-Type')


@webhook_blueprint.route('/trading', methods=['POST'])
def trading_route():
    data = request.get_json()
    save_alert_data_to_file(data, IBKR_ALERTS_DIR)

    # Always return 200 — TradingView retries any non-200, causing duplicate orders
    try:
        process_trading_data(data)
    except req_lib.exceptions.ConnectionError:
        logger.error('IBKR gateway unreachable — trading data not processed: %s', data)
    except Exception as err:
        logger.exception('Error processing trading data %s: %s', data, err)

    return '', 200


@webhook_blueprint.route('/rollover', methods=['POST'])
def rollover_route():
    data = request.get_json()
    save_alert_data_to_file(data, IBKR_ALERTS_DIR)

    # Always return 200 — TradingView retries any non-200, causing duplicate orders
    try:
        process_rollover_data(data)
    except req_lib.exceptions.ConnectionError:
        logger.error('IBKR gateway unreachable — rollover data not processed: %s', data)
    except Exception as err:
        logger.exception('Error processing rollover data %s: %s', data, err)

    return '', 200
