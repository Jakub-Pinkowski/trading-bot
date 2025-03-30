from app.utils.api_utils import api_get
from config import BASE_URL

def get_trade(order_id):
    endpoint = f"iserver/account/order/status/{order_id}"
    trade_response = api_get(BASE_URL + endpoint)

    return trade_response