from datetime import datetime

from app.utils.ibkr_helpers import api_get
from config import BASE_URL


# TODO: Handle near delivery cases

def get_closest_contract(contracts):
    if not contracts:
        raise ValueError("No contracts provided.")
    return min(
        contracts,
        key=lambda x: datetime.strptime(str(x['expirationDate']), "%Y%m%d")
    )


def search_contract(symbol):
    endpoint = f"/trsrv/futures?symbols={symbol}"

    # Fetch the contract
    contract_req = api_get(BASE_URL + endpoint)
    contracts_data = contract_req.json()

    if symbol in contracts_data and isinstance(contracts_data[symbol], list):
        closest_contract = get_closest_contract(contracts_data[symbol])
        return closest_contract['conid']
    else:
        raise ValueError(f"No contracts found for symbol: {symbol}")
