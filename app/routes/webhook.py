from flask import Blueprint, request

from app.utils.routes_utils.webhook_utils import (validate_ip,
                                                  parse_request_data,
                                                  save_alert_data_to_file,
                                                  safe_process_trading_data)
from config import ALERTS_DIR

webhook_blueprint = Blueprint('webhook', __name__)


@webhook_blueprint.route('/webhook', methods=['POST'])
def webhook_route():
    validate_ip(request.remote_addr)
    data = parse_request_data(request)
    save_alert_data_to_file(data, ALERTS_DIR)

    # Call the safe version that handles exceptions internally
    safe_process_trading_data(data)

    # Always return 200 as long as the data was received
    return '', 200
