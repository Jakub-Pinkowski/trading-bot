from flask import Blueprint, request, abort

from app.services.ibkr_service import process_trading_data
from app.utils.webhook_utils import validate_ip, parse_request_data, save_alert_data_to_file
from config import ALERTS_FILE_PATH

webhook_blueprint = Blueprint('webhook', __name__)


@webhook_blueprint.route('/webhook', methods=['POST'])
def webhook_route():
    validate_ip(request.remote_addr)
    data = parse_request_data(request)
    save_alert_data_to_file(data, ALERTS_FILE_PATH)

    try:
        process_trading_data(data)
    except Exception as e:
        print(f"Error processing TradingView webhook: {e}")
        abort(500, description=str(e))

    return '', 200
