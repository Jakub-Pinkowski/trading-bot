from datetime import datetime, timedelta

from app.utils.ibkr_helpers import api_get, load_cache, save_cache, parse_symbol
from config import BASE_URL, MIN_DAYS_UNTIL_EXPIRY


# TODO: Handle near delivery cases
# TODO: Put the expiry time in a config so that it can be different for daytrading vs swing trades
# TODO: If I hold a position for too long I'm getting too close to the mandatory selling date, handle that scenario


def get_closest_contract(contracts, min_days_until_expiry=MIN_DAYS_UNTIL_EXPIRY):
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


def fetch_contract(symbol):
    parsed_symbol = parse_symbol(symbol)
    endpoint = f"/trsrv/futures?symbols={parsed_symbol}"
    contract_req = api_get(BASE_URL + endpoint)
    contracts_data = contract_req.json()

    return contracts_data.get(parsed_symbol, [])


def get_contract_id(symbol, min_days_until_expiry=MIN_DAYS_UNTIL_EXPIRY):
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

    # Cache miss - fetch new contracts
    fresh_contracts = fetch_contract(symbol)

    # Update cache with fresh data
    contracts_cache[parsed_symbol] = fresh_contracts
    save_cache(contracts_cache)

    if fresh_contracts:
        closest_contract = get_closest_contract(fresh_contracts, min_days_until_expiry)
        return closest_contract['conid']
    else:
        raise ValueError(f"No contracts found for symbol: {parsed_symbol}")
