from app.ibkr.contracts import get_contract_id
from app.ibkr.orders import place_order
from app.utils.logger import get_logger

logger = get_logger('ibkr/service')


def process_trading_data(trading_data):
    logger.info(f'Trading data received: {trading_data}')

    symbol = trading_data.get('symbol')
    side = trading_data.get('side')
    dummy = trading_data.get('dummy')

    if not symbol:
        raise ValueError('Missing required field: symbol')
    if not side:
        raise ValueError('Missing required field: side')

    if dummy == 'YES':
        return {'status': 'dummy_skip'}

    contract_id = get_contract_id(symbol)
    order = place_order(contract_id, side)
    logger.info(f'Order placed: {order}')

    return {'status': 'order_placed', 'order': order}
