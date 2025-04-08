from app.utils.api_utils import api_get, api_post
from config import BASE_URL, ACCOUNT_ID

def invalidate_cache():
    endpoint = f"portfolio/{ACCOUNT_ID}/positions/invalidate"

    try:
        api_post(BASE_URL + endpoint, {})
    except Exception as err:
        print(f"Unexpected error invalidating cache: {err}")


def get_contract_position(conid):
    # Invalidate cache to get the real contracts data
    invalidate_cache()

    endpoint = f"portfolio/positions/{conid}"

    try:
        positions = api_get(BASE_URL + endpoint)
        for position_list in positions.values():
            if isinstance(position_list, list):
                for position in position_list:
                    if position.get("conid") == conid:
                        pos_quantity = position.get("position", 0)
                        if pos_quantity != 0:
                            # Return the actual position (- for short, + for long)
                            return int(pos_quantity)
        return 0  # No position held (flat)

    except Exception as err:
        print(f"Unexpected error retrieving position: {err}")
        return 0


def suppress_messages(message_ids):
    endpoint = "iserver/questions/suppress"
    suppression_data = {"messageIds": message_ids}

    try:
        suppression_response = api_post(BASE_URL + endpoint, suppression_data)
        print("Suppression successful:", suppression_response)

    except Exception as err:
        print(f"Unexpected error during suppression: {err}")
