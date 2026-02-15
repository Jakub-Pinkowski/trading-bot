from datetime import datetime, timedelta

from app.utils.api_utils import api_get
from app.utils.generic_utils import parse_symbol
from app.utils.logger import get_logger

logger = get_logger('ibkr_utils/contracts_utils')

# ==================== Module Configuration ====================

MIN_DAYS_UNTIL_EXPIRY = 60


def fetch_contract(symbol):
    parsed_symbol = parse_symbol(symbol)
    endpoint = f'/trsrv/futures?symbols={parsed_symbol}'

    try:
        contracts_data = api_get(endpoint)
        return contracts_data.get(parsed_symbol, [])
    except Exception as err:
        logger.error(f'Error fetching contract data: {err}')
        return []


def get_closest_contract(contracts, min_days_until_expiry=MIN_DAYS_UNTIL_EXPIRY):
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
