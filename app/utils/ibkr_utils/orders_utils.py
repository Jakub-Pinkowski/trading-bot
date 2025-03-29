from app.utils.api_utils import api_post
from config import BASE_URL


def suppress_messages(message_ids):
    endpoint = "iserver/questions/suppress"

    suppression_data = {"messageIds": message_ids}

    response = api_post(BASE_URL + endpoint, suppression_data)

    if response.status_code == 200:
        print("Suppression successful:", response.json())
    else:
        print(f"Suppression error: {response.status_code} - {response.text}")
        response.raise_for_status()
