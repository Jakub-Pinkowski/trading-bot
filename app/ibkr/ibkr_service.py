from app.ibkr.contracts import get_contract_id
from app.ibkr.orders import place_order
from app.utils.logger import get_logger

logger = get_logger('ibkr/service')


# ==================== Public API ====================

def process_trading_data(trading_data):
    """
    Process an incoming TradingView webhook signal and place an order via IBKR.

    Validates required fields, skips execution for dummy signals, resolves the
    contract ID for the given symbol, and delegates order placement to place_order.

    Args:
        trading_data: Dict containing signal fields. Expected keys:
            - 'symbol': TradingView symbol string (e.g. 'ZC1!')
            - 'side': Order direction ('B' for buy, 'S' for sell)
            - 'dummy': Set to 'YES' to simulate without placing a real order

    Returns:
        Dict with a 'status' key indicating the outcome:
            - {'status': 'dummy_skip'} if the signal was a dummy
            - {'status': 'order_placed', 'order': <api_response>} on success

    Raises:
        ValueError: If 'symbol' or 'side' is missing from trading_data
    """
    logger.info(f'Trading data received: {trading_data}')

    symbol = trading_data.get('symbol')
    side = trading_data.get('side')
    dummy = trading_data.get('dummy')

    if not symbol:
        raise ValueError('Missing required field: symbol')
    if not side:
        raise ValueError('Missing required field: side')

    # Skip order placement for dummy signals
    if dummy == 'YES':
        return {'status': 'dummy_skip'}

    contract_id = get_contract_id(symbol)
    order = place_order(contract_id, side)

    # Return order details on failure
    if isinstance(order, dict) and order.get('success') is False:
        logger.error(f'Order failed: {order}')
        return {'status': 'order_failed', 'order': order}

    logger.info(f'Order placed: {order}')
    return {'status': 'order_placed', 'order': order}
