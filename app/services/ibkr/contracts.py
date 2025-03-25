from datetime import datetime

import requests


def search_contract(symbol):
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
