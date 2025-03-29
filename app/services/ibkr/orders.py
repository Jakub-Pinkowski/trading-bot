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

        # Handle specific scenario if "error" key exists
        if isinstance(order_response, dict) and 'error' in order_response:
            error_message = order_response['error'].lower()

            if "available funds are insufficient" in error_message:
                print("Order Error: Insufficient funds.")
                # TODO: Handle insufficient funds scenario here
                return {"status": "failed", "reason": "insufficient funds", "details": order_response}

            elif "does not comply with our order handling rules for derivatives" in error_message:
                print("Order Error: Non-compliance with derivative rules.")
                # TODO: Handle derivatives rule compliance scenario here
                return {"status": "failed", "reason": "derivatives compliance", "details": order_response}

            else:
                print("Order Error: Unhandled error message received.")
                return {"status": "failed", "reason": "unknown", "details": order_response}

        # If no error, order is placed
        print("Order successfully placed:", order_response)
        return {"status": "success", "details": order_response}

    else:
        print(f"Order submission error: {response.status_code} - {response.text}")
        response.raise_for_status()
