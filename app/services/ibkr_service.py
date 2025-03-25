import requests
from datetime import datetime

import json

from app.services.ibkr.connection import check_connection


def contract_search(symbol):
    base_url = "https://localhost:5001/v1/api/"
    endpoint = f"/trsrv/futures?symbols={symbol}"

    contract_req = requests.get(url=base_url + endpoint, verify=False)

    contracts = contract_req.json()
    print("contracts", contracts)

    if symbol in contracts and isinstance(contracts[symbol], list):
        # Select the contract with the nearest expiration date
        closest_contract = min(
            contracts[symbol],
            key=lambda x: datetime.strptime(str(x['expirationDate']), "%Y%m%d")
        )
        conid = closest_contract['conid']
        return conid
    else:
        raise ValueError(f"No contracts found for symbol: {symbol}")



class IBKRService:
    def __init__(self):
        # Initialize resources or setup here.
        pass

    def process_data(self, trading_data):
        print("trading_data", trading_data)

        # Check connection and halt execution immediately if it fails
        check_connection()

        # Extract the required values from trading data safely
        symbol = trading_data.get('symbol')
        exchange = trading_data.get('Exchange')
        sec_type = trading_data.get('secType')

        contract = contract_search(symbol)
        print("contract", contract)
