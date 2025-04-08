from flask import Blueprint, request

from app.services.ibkr_service import process_trading_data
from app.utils.logger import get_logger
from app.utils.webhook_utils import validate_ip, parse_request_data, save_alert_data_to_file
from config import ALERTS_DIR

logger = get_logger()

webhook_blueprint = Blueprint('webhook', __name__)


def safe_process_trading_data(data):
    try:
        process_trading_data(data)
    except Exception as e:
        logger.error(
            f"Error processing TradingView webhook with data {data}: {e}",
            exc_info=True
        )


@webhook_blueprint.route('/webhook', methods=['POST'])
def webhook_route():
    validate_ip(request.remote_addr)
    data = parse_request_data(request)
    save_alert_data_to_file(data, ALERTS_DIR)

    # Call the safe version that handles exceptions internally
    safe_process_trading_data(data)

    # Always return 200 as long as the data was received
    return '', 200
