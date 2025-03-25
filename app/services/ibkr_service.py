import requests
from datetime import datetime

import json

from app.services.ibkr.connection import check_connection


def contract_search(symbol):
    base_url = "https://localhost:5001/v1/api/"
    endpoint = f"/trsrv/futures?symbols={symbol}"

    # Fetch the contract
    contract_req = requests.get(url=base_url + endpoint, verify=False)
    contracts = contract_req.json()

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

def place_order(conid, order):
    account_id = "DUE343675"

    base_url = "https://localhost:5001/v1/api/"
    endpoint = f"iserver/account/{account_id}/orders"

    # Define the order details
    order_details = {
        "orders": [
            {
                "conid": conid,
                "orderType": "MKT",
                "side": order,
                "tif": "DAY",
                "quantity": 1,
            }
        ]
    }

    print("order_details", order_details)

    response = requests.post(
        url=base_url + endpoint,
        json=order_details,
        verify=False
    )

    # Check for successful submission and extract the response data
    if response.status_code == 200:
        order_response = response.json()
        print("Order successfully placed:", order_response)
        return order_response
    else:
        print(f"Error placing order: {response.status_code} - {response.text}")
        response.raise_for_status()



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

        contract = contract_search(symbol)
        print("contract", contract)
        print("order", order)

        order = place_order(contract, order)
