from app.utils.api_utils import api_post
from app.utils.ibkr_utils.orders_utils import suppress_messages, get_contract_position
from app.utils.logger import get_logger
from config import ACCOUNT_ID, AGGRESSIVE_TRADING, QUANTITY_TO_TRADE

logger = get_logger()


def place_order(conid, side):
    contract_position = get_contract_position(conid)

    quantity = QUANTITY_TO_TRADE

    # Existing position opposite to incoming signal; adjust quantity if aggressive trading
    if (contract_position > 0 and side == 'S') or (contract_position < 0 and side == 'B'):
        if AGGRESSIVE_TRADING:
            quantity *= 2

    # Existing position same as incoming signal; no action needed
    elif (contract_position > 0 and side == 'B') or (contract_position < 0 and side == 'S'):
        return {'success': True, 'message': 'No action needed: already in desired position'}

    # Convert side: "B" -> "BUY", "S" -> "SELL"
    side = 'BUY' if side == 'B' else 'SELL'

    # Buy the default quantity if no position is present
    endpoint = f'iserver/account/{ACCOUNT_ID}/orders'
    order_details = {
        'orders': [
            {
                'conid': conid,
                'orderType': 'MKT',
                'side': side,
                'tif': 'DAY',
                'quantity': quantity,
            }
        ]
    }

    try:
        while True:
            order_response = api_post(endpoint, order_details)

            if isinstance(order_response, list) and 'messageIds' in order_response[0]:
                message_ids = order_response[0].get('messageIds', [])
                if message_ids:
                    suppress_messages(message_ids)
                    continue  # try again after suppression
            break  # exit loop if no suppression needed

        # Handle specific scenarios if the "error" key exists
        if isinstance(order_response, dict) and 'error' in order_response:
            error_message = order_response['error'].lower()

            if 'available funds are in sufficient' in error_message or 'available funds are insufficient' in error_message:
                logger.error(f'Insufficient funds: {order_response}')
                return {'success': False, 'error': 'Insufficient funds', 'details': order_response}

            elif 'does not comply with our order handling rules for derivatives' in error_message:
                logger.error(f'Non-compliance with derivative rules: {order_response}')
                return {'success': False, 'error': 'Non-compliance with derivative rules', 'details': order_response}

            else:
                logger.error(f'Unhandled API error: {order_response}')
                return {'success': False, 'error': 'Unhandled error', 'details': order_response}


        else:
            return order_response

    except Exception as err:
        logger.exception(f'Unexpected error while placing order: {err}')
        return {'success': False, 'error': 'An unexpected error occurred'}
