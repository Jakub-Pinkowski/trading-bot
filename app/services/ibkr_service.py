from app.services.ibkr.connection import check_connection
from app.services.ibkr.contracts import search_contract
from app.services.ibkr.orders import place_order_and_handle_suppression

class IBKRService:
    def __init__(self):
        # Initialize resources or setup here.
        pass

    def process_data(self, trading_data):
        print("trading_data", trading_data)

        # Check connection and halt execution immediately if it fails
        check_connection()

        # Extract the required values from trading data
        symbol = trading_data.get('symbol')
        order = trading_data.get('order')

        contract = search_contract(symbol)

        order = place_order_and_handle_suppression(contract, order)
