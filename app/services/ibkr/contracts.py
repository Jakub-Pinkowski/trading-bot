from datetime import datetime, timedelta

from app.utils.api_utils import api_get
from app.utils.file_utils import load_file, save_file
from app.utils.generic_utils import parse_symbol
from app.utils.logger import get_logger
from config import DATA_DIR

logger = get_logger('services/ibkr/contracts')

# ==================== Module Configuration ====================

MIN_DAYS_UNTIL_EXPIRY = 60  # Minimum days before expiration for a contract to be considered valid

# ==================== Module Paths ====================

CONTRACTS_FILE_PATH = DATA_DIR / "contracts" / "contracts.json"


def fetch_contract(symbol):
    """Fetch contract data from the IBKR API for a given symbol.

    Args:
        symbol: TradingView-formatted symbol string (e.g. 'ES1!')

    Returns:
        List of contract dictionaries, or an empty list on error
    """
    parsed_symbol = parse_symbol(symbol)
    endpoint = f'/trsrv/futures?symbols={parsed_symbol}'

    try:
        contracts_data = api_get(endpoint)
        return contracts_data.get(parsed_symbol, [])
    except Exception as err:
        logger.error(f'Error fetching contract data: {err}')
        return []


def get_closest_contract(contracts, min_days_until_expiry=MIN_DAYS_UNTIL_EXPIRY):
    """Select the nearest valid contract that is not close to expiration.

    Filters contracts to those expiring more than min_days_until_expiry days
    from today, then returns the one with the earliest expiration date.

    Args:
        contracts: List of contract dicts each containing an 'expirationDate' (format: YYYYMMDD)
        min_days_until_expiry: Minimum days until expiration for a contract to be valid

    Returns:
        Contract dictionary with the earliest valid expiration date

    Raises:
        ValueError: If no contracts with sufficient time until expiry are available
    """
    # Look for any valid contracts
    valid_contracts = [
        contract for contract in contracts
        if datetime.strptime(str(contract['expirationDate']), '%Y%m%d')
           > datetime.today() + timedelta(days=min_days_until_expiry)
    ]

    if not valid_contracts:
        raise ValueError('No valid (liquid, distant enough) contracts available.')

    # Sort by expiration and pick the earliest liquid enough contract
    chosen_contract = min(valid_contracts, key=lambda x: datetime.strptime(str(x['expirationDate']), '%Y%m%d'))

    return chosen_contract


def get_contract_id(symbol, min_days_until_expiry=MIN_DAYS_UNTIL_EXPIRY):
    """Get the IBKR contract ID for a symbol, using a file cache to avoid redundant API calls.

    Looks up the cached contract list for the symbol and selects the closest valid
    contract. If the cache is missing, invalid, or contains no valid contracts,
    fetches fresh data from the IBKR API and updates the cache.

    Args:
        symbol: TradingView-formatted symbol string (e.g. 'ES1!')
        min_days_until_expiry: Minimum days until expiry for a contract to be considered valid

    Returns:
        Integer contract ID (conid) of the nearest valid contract

    Raises:
        ValueError: If no contracts are found or none meet the expiry requirement
    """
    parsed_symbol = parse_symbol(symbol)
    contracts_cache = load_file(CONTRACTS_FILE_PATH)
    contract_list = contracts_cache.get(parsed_symbol)

    # Check cache first
    if isinstance(contract_list, list):
        try:
            closest_contract = get_closest_contract(contract_list, min_days_until_expiry)
            return closest_contract['conid']
        except ValueError as err:
            logger.warning(f'Cache invalid for symbol \'{parsed_symbol}\': {err}')

    # Cache miss or invalid entry; fetch and update cache
    fresh_contracts = fetch_contract(symbol)
    if not fresh_contracts:
        logger.error(f'No contracts found for symbol: {parsed_symbol}')
        raise ValueError(f'No contracts found for symbol: {parsed_symbol}')

    # Update cache with fresh data
    contracts_cache[parsed_symbol] = fresh_contracts
    save_file(contracts_cache, CONTRACTS_FILE_PATH)

    try:
        closest_contract = get_closest_contract(fresh_contracts, min_days_until_expiry)
        return closest_contract['conid']
    except ValueError as err:
        logger.error(f'No valid contract found in fresh data for symbol \'{parsed_symbol}\': {err}')
        raise
