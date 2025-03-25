import requests
from config import BASE_URL, ACCOUNT_ID

def suppress_messages(message_ids):
    endpoint = "iserver/questions/suppress"

    suppression_data = {
        "messageIds": message_ids
    }

    response = requests.post(
        url=BASE_URL + endpoint,
        json=suppression_data,
        verify=False
    )

    if response.status_code == 200:
        print("Suppression successful:", response.json())
        return response.json()
    else:
        print(f"Suppression error: {response.status_code} - {response.text}")
        response.raise_for_status()


def place_order_and_handle_suppression(conid, order):
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

    response = requests.post(
        url=BASE_URL + endpoint,
        json=order_details,
        verify=False
    )

    if response.status_code == 200:
        order_response = response.json()

        # Handle suppression dynamically if required
        if isinstance(order_response, list) and 'messageIds' in order_response[0]:
            message_ids = order_response[0].get('messageIds', [])
            if message_ids:
                print("Suppressing message IDs:", message_ids)
                suppress_messages(message_ids)

                # Retry placing the order after suppression
                response_retry = requests.post(
                    url=BASE_URL + endpoint,
                    json=order_details,
                    verify=False
                )
                if response_retry.status_code == 200:
                    print("Order successfully placed after suppression:", response_retry.json())
                    return response_retry.json()
                else:
                    print(f"Failed retry order: {response_retry.status_code} - {response_retry.text}")
                    response_retry.raise_for_status()
        else:
            print("Order successfully placed:", order_response)
            return order_response
    else:
        print(f"Order submission error: {response.status_code} - {response.text}")
        response.raise_for_status()
