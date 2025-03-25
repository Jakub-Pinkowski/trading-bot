from app.utils.ibkr_helpers import api_post
from config import BASE_URL, ACCOUNT_ID


def suppress_messages(message_ids):
    endpoint = "iserver/questions/suppress"

    suppression_data = {
        "messageIds": message_ids
    }

    response = api_post(BASE_URL + endpoint, suppression_data)

    if response.status_code == 200:
        print("Suppression successful:", response.json())
        return response.json()
    else:
        print(f"Suppression error: {response.status_code} - {response.text}")
        response.raise_for_status()


def handle_suppression(endpoint, order_details, message_ids):
    print("Suppressing message IDs:", message_ids)
    suppress_messages(message_ids)

    response_retry = api_post(BASE_URL + endpoint, order_details)

    if response_retry.status_code == 200:
        print("Order successfully placed after suppression:", response_retry.json())
        return response_retry.json()
    else:
        print(f"Failed retry order: {response_retry.status_code} - {response_retry.text}")
        response_retry.raise_for_status()


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

        print("Order successfully placed:", order_response)
        return order_response
    else:
        print(f"Order submission error: {response.status_code} - {response.text}")
        response.raise_for_status()
