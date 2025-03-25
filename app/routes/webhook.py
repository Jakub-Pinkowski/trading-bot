from flask import Blueprint, request

from services.tradingview_service import handle_tradingview_webhook

webhook_blueprint = Blueprint('webhook', __name__)


@webhook_blueprint.route('/webhook', methods=['POST'])
def webhook_route():
    return handle_tradingview_webhook(request)
