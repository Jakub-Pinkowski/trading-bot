from app.services.ibkr.connection import check_connection
from app.services.ibkr.contracts import get_contract_id
from app.services.ibkr.orders import place_order


def process_trading_data(trading_data):
    print("trading_data", trading_data)

    check_connection()

    symbol = trading_data.get('symbol')
    order = trading_data.get('order')
    price = trading_data.get('price')

    contract = get_contract_id(symbol)

    order = place_order(contract, order)
    print("order", order)