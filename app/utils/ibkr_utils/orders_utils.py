from app.utils.api_utils import api_post
from config import BASE_URL


def suppress_messages(message_ids):
    endpoint = "iserver/questions/suppress"
    suppression_data = {"messageIds": message_ids}

    try:
        suppression_response = api_post(BASE_URL + endpoint, suppression_data)
        print("Suppression successful:", suppression_response)

    except Exception as err:
        print(f"Unexpected error during suppression: {err}")
