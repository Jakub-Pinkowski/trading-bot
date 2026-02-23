from datetime import datetime, timedelta

from app.utils.api_utils import api_get
from app.utils.file_utils import load_file, save_file
from app.utils.generic_utils import parse_symbol
from app.utils.logger import get_logger
from config import DATA_DIR
from futures_config.symbol_mapping import map_tv_to_ibkr

logger = get_logger('ibkr/contracts')

# ==================== Module Configuration ====================

MIN_DAYS_UNTIL_EXPIRY = 60  # Minimum days before expiration for a contract to be considered valid

# ==================== Module Paths ====================

CONTRACTS_FILE_PATH = DATA_DIR / "contracts" / "contracts.json"


# ==================== API ====================

def fetch_contract(parsed_symbol):
    """
    Fetch contract data from the IBKR API for a given symbol.

    Args:
        parsed_symbol: Plain symbol string (e.g. 'ZC'), with any TradingView
            suffix already stripped by the caller

    Returns:
        List of contract dicts from the API, or an empty list on error
    """
    try:
        contracts_data = api_get(f'/trsrv/futures?symbols={parsed_symbol}')
        return contracts_data.get(parsed_symbol, [])
    except Exception as err:
        logger.error(f'Error fetching contract data: {err}')
        return []


# ==================== Contract Selection ====================

def get_closest_contract(contracts, min_days_until_expiry=MIN_DAYS_UNTIL_EXPIRY):
    """
    Select the nearest valid contract that is not close to expiration.

    Parses expiration dates once, filters to those beyond the cutoff, then
    returns the contract with the earliest remaining valid expiration date.

    Args:
        contracts: List of contract dicts each containing an 'expirationDate'
            field in YYYYMMDD format (e.g. [{'conid': '123', 'expirationDate': '20251215'}])
        min_days_until_expiry: Minimum days until expiration for a contract to
            be considered valid (default: MIN_DAYS_UNTIL_EXPIRY)

    Returns:
        Contract dict with the earliest valid expiration date

    Raises:
        ValueError: If no contracts with sufficient time until expiry are available
    """
    # Parse dates once, then filter and pick the minimum in a single pass
    cutoff = datetime.today() + timedelta(days=min_days_until_expiry)
    contracts_with_dates = [
        (datetime.strptime(str(contract['expirationDate']), '%Y%m%d'), contract)
        for contract in contracts
    ]
    valid_contracts = [
        (expiry_date, contract)
        for expiry_date, contract in contracts_with_dates
        if expiry_date > cutoff
    ]

    if not valid_contracts:
        raise ValueError(f'No valid contracts available for expiry cutoff of {min_days_until_expiry} days.')

    valid_contracts.sort(key=lambda item: item[0])
    return valid_contracts[0][1]


# ==================== Cache Management ====================

def get_contract_id(symbol, min_days_until_expiry=MIN_DAYS_UNTIL_EXPIRY):
    """
    Get the IBKR contract ID for a symbol, using a file cache to avoid redundant API calls.

    Looks up the cached contract list for the symbol and selects the closest valid
    contract. If the cache is missing, invalid, or contains no valid contracts,
    fetches fresh data from the IBKR API and updates the cache.

    Args:
        symbol: TradingView-formatted symbol string (e.g. 'ZC1!')
        min_days_until_expiry: Minimum days until expiry for a contract to be
            considered valid (default: MIN_DAYS_UNTIL_EXPIRY)

    Returns:
        Integer contract ID (conid) of the nearest valid contract

    Raises:
        ValueError: If no contracts are found for the symbol, or none meet the
            expiry requirement
    """
    parsed_symbol = parse_symbol(symbol)
    ibkr_symbol = map_tv_to_ibkr(parsed_symbol)
    contracts_cache = load_file(CONTRACTS_FILE_PATH)
    contract_list = contracts_cache.get(ibkr_symbol)

    # Return from cache if the entry is valid
    if isinstance(contract_list, list):
        try:
            closest_contract = get_closest_contract(contract_list, min_days_until_expiry)
            return closest_contract['conid']
        except ValueError as err:
            logger.warning(f"Cache invalid for symbol '{ibkr_symbol}': {err}")

    # Cache miss or stale entry; fetch fresh data and update cache
    fresh_contracts = fetch_contract(ibkr_symbol)
    if not fresh_contracts:
        logger.error(f'No contracts found for symbol: {ibkr_symbol}')
        raise ValueError(f'No contracts found for symbol: {ibkr_symbol}')

    contracts_cache[ibkr_symbol] = fresh_contracts
    save_file(contracts_cache, CONTRACTS_FILE_PATH)

    try:
        closest_contract = get_closest_contract(fresh_contracts, min_days_until_expiry)
        return closest_contract['conid']
    except ValueError as err:
        logger.error(f"No valid contract found in fresh data for symbol '{ibkr_symbol}': {err}")
        raise
