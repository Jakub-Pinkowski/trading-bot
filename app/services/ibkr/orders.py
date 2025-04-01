from app.utils.api_utils import api_post
from app.utils.ibkr_utils.orders_utils import suppress_messages, get_contract_position
from config import BASE_URL, ACCOUNT_ID, AGGRESSIVE_TRADING, QUANTITY_TO_TRADE


def place_order(conid, order):
    contract_position = get_contract_position(conid)
    print(f"Contract {conid} current position: {contract_position}")

    quantity = QUANTITY_TO_TRADE

    # Existing position opposite to incoming signal; adjust quantity if aggressive trading
    if (contract_position > 0 and order == "SELL") or (contract_position < 0 and order == "BUY"):
        if AGGRESSIVE_TRADING:
            quantity *= 2

    # Existing position same as incoming signal; no action needed
    elif (contract_position > 0 and order == "BUY") or (contract_position < 0 and order == "SELL"):
        print(f"No action taken. Already holding position {contract_position} with matching signal '{order}'.")
        return {"success": True, "message": "No action needed: already in desired position"}

    # Buy the default quantity if no position is present
    endpoint = f"iserver/account/{ACCOUNT_ID}/orders"
    order_details = {
        "orders": [
            {
                "conid": conid,
                "orderType": "MKT",
                "side": order,
                "tif": "DAY",
                "quantity": quantity,
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
