from app.utils.file_utils import load_cache, save_cache
from app.utils.ibkr_utils.contracts_utils import parse_symbol, fetch_contract, get_closest_contract
from config import MIN_DAYS_UNTIL_EXPIRY


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
