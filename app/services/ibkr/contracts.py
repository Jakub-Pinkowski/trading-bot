import re
from datetime import datetime

from app.utils.ibkr_helpers import api_get
from config import BASE_URL


# TODO: Handle near delivery cases

def parse_symbol(symbol):
    match = re.match(r'^([A-Za-z]+)', symbol)
    if not match:
        raise ValueError(f"Invalid symbol format: {symbol}")
    return match.group(1)


def get_closest_contract(contracts):
    if not contracts:
        raise ValueError("No contracts provided.")
    return min(
        contracts,
        key=lambda x: datetime.strptime(str(x['expirationDate']), "%Y%m%d")
    )


def search_contract(symbol):
    parsed_symbol = parse_symbol(symbol)
    endpoint = f"/trsrv/futures?symbols={parsed_symbol}"

    # Fetch the contract
    contract_req = api_get(BASE_URL + endpoint)
    contracts_data = contract_req.json()

    if parsed_symbol in contracts_data and isinstance(contracts_data[parsed_symbol], list):
        closest_contract = get_closest_contract(contracts_data[parsed_symbol])
        return closest_contract['conid']
    else:
        raise ValueError(f"No contracts found for symbol: {parsed_symbol}")
