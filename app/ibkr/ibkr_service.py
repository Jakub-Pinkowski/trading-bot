from app.ibkr.contracts import get_contract_id
from app.ibkr.orders import place_order
from app.utils.logger import get_logger

logger = get_logger()


def process_trading_data(trading_data):
    logger.info(f'Trading data received: {trading_data}')

    dummy = trading_data.get('dummy')
    symbol = trading_data.get('symbol')
    side = trading_data.get('side')
    price = trading_data.get('price')

    contract = get_contract_id(symbol)

    if dummy == 'YES':
        return

    order = place_order(contract, side)
    logger.info(f'Order placed: {order}')
