import requests

from app.utils.api_helpers import api_post
from app.utils.ibkr_helpers.orders_helpers import suppress_messages
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

    try:
        response = api_post(BASE_URL + endpoint, order_details)
        response.raise_for_status()

        order_response = response.json()

        # Handle suppression dynamically if required
        if isinstance(order_response, list) and 'messageIds' in order_response[0]:
            message_ids = order_response[0].get('messageIds', [])
            if message_ids:
                suppress_messages(message_ids)
                place_order(conid, order)

        # Handle specific scenarios if "error" key exists
        if isinstance(order_response, dict) and 'error' in order_response:
            error_message = order_response['error'].lower()

            if "available funds are in sufficient" in error_message or "available funds are insufficient" in error_message:
                print("Order Error: Insufficient funds.", order_response)
                # TODO: Handle insufficient funds scenario

            elif "does not comply with our order handling rules for derivatives" in error_message:
                print("Order Error: Non-compliance with derivative rules.", order_response)
                # TODO: Handle derivatives rule compliance scenario

            else:
                print("Order Error: Unhandled error message received.", order_response)

        else:
            print("Order successfully placed:", order_response)

    except (requests.HTTPError, ValueError, Exception) as err:
        print(f"An error occurred: {err}")
