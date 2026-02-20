from app.utils.api_utils import api_get, api_post
from app.utils.logger import get_logger
from config import ACCOUNT_ID

logger = get_logger('services/ibkr/orders')

# ==================== Module Configuration ====================

QUANTITY_TO_TRADE = 1  # Default number of contracts per order
AGGRESSIVE_TRADING = True  # Double quantity when reversing an existing position to close and reverse in one trade


def invalidate_cache():
    """Invalidate the IBKR portfolio position cache to force fresh data on the next fetch.

    IBKR caches position data server-side; calling this before fetching positions
    ensures the response reflects the current real-time state.
    """
    endpoint = f'portfolio/{ACCOUNT_ID}/positions/invalidate'

    try:
        api_post(endpoint, {})
    except Exception as err:
        logger.error(f'Error invalidating cache: {err}')


def get_contract_position(conid):
    """Get the current open position quantity for a given contract.

    Invalidates the server-side cache before fetching to ensure the position
    reflects real-time state. Returns a positive value for long positions,
    negative for short, and 0 if no position is held.

    Args:
        conid: IBKR contract ID to look up

    Returns:
        Integer position quantity (positive = long, negative = short, 0 = no position)
    """
    # Invalidate cache to get the real contracts data
    invalidate_cache()

    endpoint = f'portfolio/{ACCOUNT_ID}/positions'

    try:
        # Fetch all positions data
        positions = api_get(endpoint)

        # Iterate through the list of positions and find the one matching the conid
        for position in positions:
            if position.get('conid') == conid:
                pos_quantity = position.get('position', 0)
                return int(pos_quantity)  # Return found quantity (+/- values for positions)

        return 0  # No position found for the given conid

    except Exception as err:
        logger.error(f'Error fetching contract position: {err}')
        return 0


def suppress_messages(message_ids):
    """Suppress IBKR confirmation dialogs that would otherwise block order submission.

    Some orders trigger server-side confirmation prompts that must be acknowledged
    before the order can proceed. Called automatically by place_order when the
    API returns messageIds in the response.

    Args:
        message_ids: List of message ID strings to suppress
    """
    endpoint = 'iserver/questions/suppress'
    suppression_data = {'messageIds': message_ids}

    try:
        suppression_response = api_post(endpoint, suppression_data)
        logger.info(f'Suppression response: {suppression_response}')

    except Exception as err:
        logger.error(f'Error suppressing messages: {err}')


# TODO [MEDIUM]: Remove Aggressive trading variable as this is now integrates into the strategies instead
def place_order(conid, side):
    """Place a market order for a futures contract, handling position reversals and message suppression.

    Checks the current position before placing the order. If a position in the
    opposite direction exists, doubles the quantity when AGGRESSIVE_TRADING is enabled
    to close and reverse in one trade. Skips the order if already in the desired
    direction. Automatically suppresses any IBKR confirmation prompts and retries.

    Args:
        conid: IBKR contract ID to trade
        side: Direction of the order ('B' for buy, 'S' for sell)

    Returns:
        API response dict on success, or a dict with 'success': False and 'error' on failure
    """
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
