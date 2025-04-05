from flask import Blueprint, request

from app.services.ibkr_service import process_trading_data
from app.utils.webhook_utils import validate_ip, parse_request_data, save_alert_data_to_file
from config import ALERTS_DIR

webhook_blueprint = Blueprint('webhook', __name__)


def safe_process_trading_data(data):
    try:
        process_trading_data(data)
    except Exception as e:
        # Log the exception without aborting the request
        print(f"Error processing TradingView webhook: {e}")
        # TODO: store errors in a logging/monitoring system here.


@webhook_blueprint.route('/webhook', methods=['POST'])
def webhook_route():
    validate_ip(request.remote_addr)
    data = parse_request_data(request)
    save_alert_data_to_file(data, ALERTS_DIR)

    # Call the safe version that handles exceptions internally
    safe_process_trading_data(data)

    # Always return 200 as long as the data was received
    return '', 200
