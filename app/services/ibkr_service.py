import requests

from app.services.ibkr.connection import check_connection


def contract_search(symbol):
    base_url = "https://localhost:5001/v1/api/"
    endpoint = f"/trsrv/futures?symbols={symbol}"

    contract_req = requests.get(url=base_url + endpoint, verify=False)

    print("contract_req", contract_req)
    print("contract_req.json()", contract_req.json())


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

        contract_search(symbol)
