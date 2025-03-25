from datetime import datetime

import requests

from config import BASE_URL


def search_contract(symbol):
    endpoint = f"/trsrv/futures?symbols={symbol}"

    # Fetch the contract
    contract_req = requests.get(url=BASE_URL + endpoint, verify=False)
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
