from app.services.ibkr.connection import check_connection
from app.services.ibkr.contracts import get_contract
from app.services.ibkr.orders import place_order


class IBKRService:
    def __init__(self):
        pass

    def process_data(self, trading_data):
        print("trading_data", trading_data)

        check_connection()

        symbol = trading_data.get('symbol')
        order = trading_data.get('order')

        contract = get_contract(symbol)

        place_order(contract, order)
