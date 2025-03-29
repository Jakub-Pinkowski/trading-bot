import re

from app.utils.api_helpers import api_get
from config import BASE_URL


def parse_symbol(symbol):
    match = re.match(r'^([A-Za-z]+)', symbol)
    if not match:
        raise ValueError(f"Invalid symbol format: {symbol}")
    return match.group(1)


def fetch_contract(symbol):
    parsed_symbol = parse_symbol(symbol)
    endpoint = f"/trsrv/futures?symbols={parsed_symbol}"
    contract_req = api_get(BASE_URL + endpoint)
    contracts_data = contract_req.json()

    return contracts_data.get(parsed_symbol, [])
