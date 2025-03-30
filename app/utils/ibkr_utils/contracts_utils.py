import re
from datetime import datetime, timedelta

from app.utils.api_utils import api_get
from config import BASE_URL, MIN_DAYS_UNTIL_EXPIRY


def parse_symbol(symbol):
    match = re.match(r'^([A-Za-z]+)', symbol)
    if not match:
        raise ValueError(f"Invalid symbol format: {symbol}")
    return match.group(1)


def fetch_contract(symbol):
    parsed_symbol = parse_symbol(symbol)
    endpoint = f"/trsrv/futures?symbols={parsed_symbol}"

    try:
        contracts_data = api_get(BASE_URL + endpoint)
        return contracts_data.get(parsed_symbol, [])
    except Exception as err:
        print(f"Unexpected error fetching contract details for {symbol}: {err}")
        return []


def get_closest_contract(contracts, min_days_until_expiry=MIN_DAYS_UNTIL_EXPIRY):
    # Look for any valid contracts
    valid_contracts = [
        contract for contract in contracts
        if datetime.strptime(str(contract['expirationDate']), "%Y%m%d")
           > datetime.today() + timedelta(days=min_days_until_expiry)
    ]

    if not valid_contracts:
        raise ValueError("No valid (liquid, distant enough) contracts available.")

    # Sort by expiration and pick the earliest liquid enough contract
    chosen_contract = min(valid_contracts, key=lambda x: datetime.strptime(str(x['expirationDate']), "%Y%m%d"))

    return chosen_contract
