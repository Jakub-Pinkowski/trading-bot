from flask import Blueprint, request, abort

from app.utils.helpers import validate_ip, parse_request_data

webhook_blueprint = Blueprint('webhook', __name__)


@webhook_blueprint.route('/webhook', methods=['POST'])
def webhook_route():
    """Full webhook handling logic."""
    validate_ip(request.remote_addr)
    data = parse_request_data(request)

    # TODO: Further processing of data here

    # Call IBKR service directly:
    from app.services.ibkr_service import IBKRService
    ibkr_service = IBKRService()

    try:
        ibkr_service.process_tradingview_data(data)
    except Exception as e:
        print(f"Error processing TradingView webhook: {e}")
        abort(500, description=str(e))

    return '', 200
