from app.utils.api_utils import api_get, api_post
from app.utils.logger import get_logger
from config import ACCOUNT_ID

logger = get_logger()


def invalidate_cache():
    endpoint = f"portfolio/{ACCOUNT_ID}/positions/invalidate"

    try:
        api_post(endpoint, {})
    except Exception as err:
        logger.error(f"Error invalidating cache: {err}")


def get_contract_position(conid):
    # Invalidate cache to get the real contracts data
    invalidate_cache()

    endpoint = f"portfolio/{ACCOUNT_ID}/positions"

    try:
        # Fetch all positions data
        positions = api_get(endpoint)

        # Iterate through the list of positions and find the one matching the conid
        for position in positions:
            if position.get("conid") == conid:
                pos_quantity = position.get("position", 0)
                return int(pos_quantity)  # Return found quantity (+/- values for positions)

        return 0  # No position found for the given conid


    except Exception as err:
        logger.error(f"Error fetching contract position: {err}")
        return 0


def suppress_messages(message_ids):
    endpoint = "iserver/questions/suppress"
    suppression_data = {"messageIds": message_ids}

    try:
        suppression_response = api_post(endpoint, suppression_data)
        print("Suppression successful:", suppression_response)

    except Exception as err:
        logger.error(f"Error suppressing messages: {err}")
