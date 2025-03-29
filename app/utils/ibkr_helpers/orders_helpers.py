from app.utils.api_helpers import api_post
from config import BASE_URL


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
