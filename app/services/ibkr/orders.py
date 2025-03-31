from app.utils.api_utils import api_get, api_post
from app.utils.ibkr_utils.orders_utils import suppress_messages
from config import BASE_URL, ACCOUNT_ID


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


def place_order(conid, order):
    contract_exists = has_contract(conid)
    print(f"Contract {conid}: {contract_exists}")

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
        order_response = api_post(BASE_URL + endpoint, order_details)

        # Handle suppression dynamically if required
        if isinstance(order_response, list) and 'messageIds' in order_response[0]:
            message_ids = order_response[0].get('messageIds', [])
            if message_ids:
                suppress_messages(message_ids)
                return place_order(conid, order)

        # Handle specific scenarios if "error" key exists
        if isinstance(order_response, dict) and 'error' in order_response:
            error_message = order_response['error'].lower()

            if "available funds are in sufficient" in error_message or "available funds are insufficient" in error_message:
                return {"success": False, "error": "Insufficient funds", "details": order_response}

            elif "does not comply with our order handling rules for derivatives" in error_message:
                return {"success": False, "error": "Non-compliance with derivative rules", "details": order_response}

            else:
                return {"success": False, "error": "Unhandled error", "details": order_response}

        # TODO: There are still more cases to handle besides error
        else:
            return order_response

    except Exception as err:
        return {"success": False, "error": f"Unexpected error: {err}"}
