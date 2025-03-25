import json

import requests

from app.services.ibkr.connection import check_connection


def contract_search(symbol, sec_type):
    base_url = "https://localhost:5001/v1/api/"
    endpoint = "iserver/secdef/search"

    json_body = {
        "symbol": symbol,
        "secType": sec_type,
        "name": False
    }

    contract_req = requests.post(url=base_url + endpoint, json=json_body, verify=False)

    contract_json = json.dumps(contract_req.json(), indent=2)

    print("contract_json", contract_json)


class IBKRService:
    def __init__(self):
        # Initialize resources or setup here.
        pass

    def process_data(self, trading_data):
        print("trading_data", trading_data)

        # Check connection and halt execution immediately if it fails:
        check_connection()

        # Execution continues only if authentication succeeded.
        print("authenticated")

        # Extract the required values from trading data safely.
        symbol = trading_data.get('symbol')
        exchange = trading_data.get('Exchange')
        sec_type = trading_data.get('secType')

        print(f"Symbol: {symbol}, Exchange: {exchange}, SecType: {sec_type}")

        contract_search(symbol=symbol, sec_type=sec_type)

        # You can now use these extracted values further in your code
        return symbol, exchange, sec_type
