import json
import os
import re
from datetime import datetime, timedelta

from app.utils.ibkr_helpers import api_get
from config import BASE_URL, BASE_DIR

CACHE_FILE_PATH = os.path.join(BASE_DIR, "data", "contracts.json")


# TODO: Store all the conid locally as they never change
# TODO: Handle near delivery cases
# TODO: Put the expiry time in a config so that it can be different for daytrading vs swing trades
# TODO: If I hold a position for too long I'm getting too close to the mandatory selling date, handle that scenario

def load_cache():
    if not os.path.exists(CACHE_FILE_PATH):
        return {}

    with open(CACHE_FILE_PATH, 'r') as cache_file:
        cache_data = json.load(cache_file)
    return cache_data


def save_cache(cache_data):
    with open(CACHE_FILE_PATH, 'w') as cache_file:
        json.dump(cache_data, cache_file, indent=4)


def parse_symbol(symbol):
    match = re.match(r'^([A-Za-z]+)', symbol)
    if not match:
        raise ValueError(f"Invalid symbol format: {symbol}")
    return match.group(1)


def get_closest_contract(contracts, min_days_until_expiry=60):
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


def search_contract(symbol, min_days_until_expiry=60):
    parsed_symbol = parse_symbol(symbol)
    contracts_cache = load_cache()

    # Check cache first
    if parsed_symbol in contracts_cache and isinstance(contracts_cache[parsed_symbol], list):
        try:
            closest_contract = get_closest_contract(contracts_cache[parsed_symbol], min_days_until_expiry)
            print(f"Using cached contract: {closest_contract['conid']}")
            return closest_contract['conid']
        except ValueError:
            pass

    # Fetch the contract if cache is empty or invalid
    endpoint = f"/trsrv/futures?symbols={parsed_symbol}"
    contract_req = api_get(BASE_URL + endpoint)
    contracts_data = contract_req.json()

    # Update cache with fresh data
    save_cache({
        **contracts_cache,
        parsed_symbol: contracts_data.get(parsed_symbol, [])
    })

    if parsed_symbol in contracts_data and isinstance(contracts_data[parsed_symbol], list):
        closest_contract = get_closest_contract(contracts_data[parsed_symbol], min_days_until_expiry)
        return closest_contract['conid']
    else:
        raise ValueError(f"No contracts found for symbol: {parsed_symbol}")

