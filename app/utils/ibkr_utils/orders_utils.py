from app.utils.api_utils import api_get, api_post
from config import BASE_URL


def has_contract(conid):
    endpoint = f"portfolio/positions/{conid}"

    try:
        positions = api_get(BASE_URL + endpoint)
        # Iterate through each account's positions
        for position_list in positions.values():
            if isinstance(position_list, list):
                for position in position_list:
                    # Check if the position matches the conid and has a non-zero quantity
                    if position.get("conid") == conid and position.get("position", 0) != 0:
                        return True
        return False  # conid either does not exist or position is zero

    except Exception as err:
        print(f"Unexpected error checking contract: {err}")
        return False


def suppress_messages(message_ids):
    endpoint = "iserver/questions/suppress"
    suppression_data = {"messageIds": message_ids}

    try:
        suppression_response = api_post(BASE_URL + endpoint, suppression_data)
        print("Suppression successful:", suppression_response)

    except Exception as err:
        print(f"Unexpected error during suppression: {err}")
