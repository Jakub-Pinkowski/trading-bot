from datetime import datetime

from app.utils.ibkr_helpers import api_get
from config import BASE_URL


def search_contract(symbol):
    endpoint = f"/trsrv/futures?symbols={symbol}"

    # Fetch the contract
    contract_req = api_get(BASE_URL + endpoint)
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
