from app.services.ibkr.contracts import get_contract_id
from app.services.ibkr.orders import place_order


def process_trading_data(trading_data):
    print("trading_data", trading_data)

    dummy = trading_data.get('dummy')
    symbol = trading_data.get('symbol')
    side = trading_data.get('side')
    price = trading_data.get('price')

    contract = get_contract_id(symbol)

    if dummy == "YES":
        return

    order = place_order(contract, side)
    print("order", order)
