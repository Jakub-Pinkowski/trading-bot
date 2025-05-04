from app.utils.file_utils import load_file, save_file
from app.utils.ibkr_utils.contracts_utils import parse_symbol, fetch_contract, get_closest_contract
from app.utils.logger import get_logger
from config import MIN_DAYS_UNTIL_EXPIRY, CONTRACTS_FILE_PATH

logger = get_logger()


# TODO: Sometimes I do something twice, check tests
def get_contract_id(symbol, min_days_until_expiry=MIN_DAYS_UNTIL_EXPIRY):
    parsed_symbol = parse_symbol(symbol)
    contracts_cache = load_file(CONTRACTS_FILE_PATH)
    contract_list = contracts_cache.get(parsed_symbol)

    # Check cache first
    if isinstance(contract_list, list):
        try:
            closest_contract = get_closest_contract(contract_list, min_days_until_expiry)
            return closest_contract['conid']
        except ValueError as err:
            logger.warning(f"Cache invalid for symbol '{parsed_symbol}': {err}")

    # Cache miss or invalid entry; fetch and update cache
    fresh_contracts = fetch_contract(symbol)
    if not fresh_contracts:
        logger.error(f"No contracts found for symbol: {parsed_symbol}")
        raise ValueError(f"No contracts found for symbol: {parsed_symbol}")


    # Update cache with fresh data
    contracts_cache[parsed_symbol] = fresh_contracts
    save_file(contracts_cache, CONTRACTS_FILE_PATH, )

    try:
        closest_contract = get_closest_contract(fresh_contracts, min_days_until_expiry)
        return closest_contract['conid']
    except ValueError as err:
        logger.error(f"No valid contract found in fresh data for symbol '{parsed_symbol}': {err}")
        raise

