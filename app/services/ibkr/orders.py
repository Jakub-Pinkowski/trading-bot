from app.utils.api_helpers import api_post
from app.utils.ibkr_helpers.orders_helpers import handle_suppression
from config import BASE_URL, ACCOUNT_ID


def place_order(conid, order):
    endpoint = f"iserver/account/{ACCOUNT_ID}/orders"

    order_details = {
        "orders": [
            {
                "conid": conid,
                "orderType": "MKT",
                "side": order,
                "tif": "DAY",
                "quantity": 1,
            }
        ]
    }

    response = api_post(BASE_URL + endpoint, order_details)

    if response.status_code == 200:
        order_response = response.json()

        # Handle suppression dynamically if required
        if isinstance(order_response, list) and 'messageIds' in order_response[0]:
            message_ids = order_response[0].get('messageIds', [])
            if message_ids:
                return handle_suppression(endpoint, order_details, message_ids)

        # TODO: Sometimes I get an error in the response so I can't assume that it always works,
        # NOTE: code 200 doesn't equal order placed
        print("Order successfully placed:", order_response)
        return order_response
    else:
        print(f"Order submission error: {response.status_code} - {response.text}")
        response.raise_for_status()
