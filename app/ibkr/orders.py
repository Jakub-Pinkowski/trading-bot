from app.utils.api_utils import api_get, api_post
from app.utils.logger import get_logger
from config import ACCOUNT_ID

logger = get_logger('ibkr/orders')

# ==================== Module Configuration ====================

QUANTITY_TO_TRADE = 1  # Default number of contracts per order
MAX_SUPPRESS_RETRIES = 3  # Maximum attempts to suppress IBKR confirmation dialogs before giving up


# ==================== Helper Functions ====================

def _invalidate_cache():
    """
    Invalidate the IBKR portfolio position cache to force fresh data on the next fetch.

    Raises:
        Exception: Propagates any API error to the caller so stale data is never
            silently used for position decisions
    """
    try:
        api_post(f'portfolio/{ACCOUNT_ID}/positions/invalidate', {})
    except Exception as err:
        logger.error(f'Error invalidating cache: {err}')
        raise


def _get_contract_position(conid):
    """
    Get the current open position quantity for a given contract.

    Invalidates the server-side cache before fetching to ensure the position
    reflects real-time state. Returns a positive value for long positions,
    negative for short, and 0 if no position is held.

    Args:
        conid: IBKR contract ID to look up (e.g. '265598')

    Returns:
        Integer position quantity:
            - Positive value for a long position
            - Negative value for a short position
            - 0 if no position is held or on API error
    """
    # Invalidate server-side position cache to ensure fresh data
    _invalidate_cache()

    try:
        positions = api_get(f'portfolio/{ACCOUNT_ID}/positions')

        for position in positions:
            if position.get('conid') == conid:
                return int(position.get('position', 0))

        return 0  # No position found for the given conid

    except Exception as err:
        logger.error(f'Error fetching contract position: {err}')
        return 0


def _suppress_messages(message_ids):
    """
    Suppress IBKR confirmation dialogs that would otherwise block order submission.

    Args:
        message_ids: List of message ID strings to suppress (e.g. ['msg1', 'msg2'])
    """
    try:
        suppression_response = api_post('iserver/questions/suppress', {'messageIds': message_ids})
        logger.info(f'Suppression response: {suppression_response}')

    except Exception as err:
        logger.error(f'Error suppressing messages: {err}')


# ==================== Order Placement ====================

def place_order(conid, side):
    """
    Place a market order for a futures contract, handling position reversals and message suppression.

    Checks the current position before placing the order. Returns early if already
    in the desired direction. Automatically suppresses any IBKR confirmation prompts
    and retries up to MAX_SUPPRESS_RETRIES times.

    Args:
        conid: IBKR contract ID to trade (e.g. '265598')
        side: Order direction — 'B' for buy, 'S' for sell

    Returns:
        Dict with the outcome:
            - API response dict on a successful order
            - {'success': True, 'message': '...'} if already in the desired position
            - {'success': False, 'error': '...', 'details': ...} on a known API error
            - {'success': False, 'error': 'An unexpected error occurred'} on an exception

    Raises:
        ValueError: If side is not 'B' or 'S'
    """
    contract_position = _get_contract_position(conid)

    # Existing position same as incoming signal; no action needed
    if (contract_position > 0 and side == 'B') or (contract_position < 0 and side == 'S'):
        return {'success': True, 'message': 'No action needed: already in desired position'}

    if side not in ('B', 'S'):
        raise ValueError(f"Invalid side '{side}': expected 'B' or 'S'")

    # Convert side: 'B' -> 'BUY', 'S' -> 'SELL'
    side = 'BUY' if side == 'B' else 'SELL'

    order_details = {
        'orders': [
            {
                'conid': conid,
                'orderType': 'MKT',
                'side': side,
                'tif': 'DAY',
                'quantity': QUANTITY_TO_TRADE,
            }
        ]
    }

    try:
        # Retry after suppressing confirmation dialogs, up to MAX_SUPPRESS_RETRIES times
        for attempt in range(MAX_SUPPRESS_RETRIES):
            order_response = api_post(f'iserver/account/{ACCOUNT_ID}/orders', order_details)

            if isinstance(order_response, list) and 'messageIds' in order_response[0]:
                message_ids = order_response[0].get('messageIds', [])
                if message_ids:
                    _suppress_messages(message_ids)
                    continue  # Retry after suppression
            break  # No suppression needed, exit loop
        else:
            logger.error('Order failed: exceeded maximum suppression retries')
            return {'success': False, 'error': 'Exceeded maximum suppression retries'}

        # Handle specific error scenarios if the "error" key exists in the response
        if isinstance(order_response, dict) and 'error' in order_response:
            error_message = order_response['error'].lower()

            # Note: 'in sufficient' with a space matches a known IBKR API typo in some error responses
            if 'available funds are in sufficient' in error_message or 'available funds are insufficient' in error_message:
                logger.error(f'Insufficient funds: {order_response}')
                return {'success': False, 'error': 'Insufficient funds', 'details': order_response}

            if 'does not comply with our order handling rules for derivatives' in error_message:
                logger.error(f'Non-compliance with derivative rules: {order_response}')
                return {'success': False, 'error': 'Non-compliance with derivative rules', 'details': order_response}

            logger.error(f'Unhandled API error: {order_response}')
            return {'success': False, 'error': 'Unhandled error', 'details': order_response}

        return order_response

    except Exception as err:
        logger.exception(f'Unexpected error while placing order: {err}')
        return {'success': False, 'error': 'An unexpected error occurred'}
