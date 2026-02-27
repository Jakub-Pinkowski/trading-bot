from datetime import datetime

from app.ibkr.contracts import ContractResolver
from app.ibkr.orders import _get_contract_position, place_order
from app.utils.logger import get_logger

logger = get_logger('ibkr/rollover')

# ==================== Configuration ====================

# Keep CLOSE_OUT_WARNING_DAYS in sync with warningDays in contract_switch_warning.pine
CLOSE_OUT_WARNING_DAYS = 1

# False → close old position only, do not reopen
# True  → close old position and reopen on new contract
REOPEN_ON_ROLLOVER = False


# ==================== Public API ====================

def process_rollover_data(data):
    """
    Handle a rollover alert from TradingView.

    Validates the payload and delegates to _check_and_rollover_position.

    Args:
        data: Dict from the /rollover webhook payload. Expected keys:
            - 'symbol': TradingView symbol string (e.g. 'ZC1!')

    Returns:
        Dict with a 'status' key indicating the outcome:
            - {'status': 'no_rollover_needed'} — outside the warning window
            - {'status': 'warning', 'days_until_switch': N} — near switch, no open position
            - {'status': 'rolled', ...} — position closed and reopened on new contract
            - {'status': 'closed', ...} — position closed, not reopened
            - {'status': 'error', ...} — close order failed
            - {'status': 'reopen_failed', ...} — close succeeded but reopen failed

    Raises:
        ValueError: If 'symbol' is missing from data, or if the symbol has no
            upcoming switch date in the YAML (propagated from _check_and_rollover_position).
    """
    logger.info(f'Rollover data received: {data}')

    symbol = data.get('symbol')
    if not symbol:
        raise ValueError('Missing required field: symbol')

    result = _check_and_rollover_position(symbol)
    return result


# ==================== Implementation ====================

def _check_and_rollover_position(symbol):
    """
    Close (and optionally reopen) an open position before the contract switch date.

    Always returns a dict — never None.

    Args:
        symbol: TradingView symbol string (e.g. 'ZC1!').

    Returns:
        Dict with a 'status' key indicating the outcome:
            - {'status': 'no_rollover_needed'} — outside the warning window
            - {'status': 'warning', 'days_until_switch': N} — near switch, no open position
            - {'status': 'rolled', ...} — position closed and reopened on new contract
            - {'status': 'closed', ...} — position closed, not reopened
            - {'status': 'error', ...} — close order failed
            - {'status': 'reopen_failed', ...} — close succeeded but reopen failed

    Raises:
        ValueError: If the symbol has no upcoming switch date in the YAML.
    """
    resolver = ContractResolver(symbol)
    today = datetime.today()
    days_until_switch = (resolver.next_switch_date - today).days

    if days_until_switch > CLOSE_OUT_WARNING_DAYS:
        logger.info(f'No rollover needed for {symbol}: {days_until_switch} day(s) until switch')
        return {'status': 'no_rollover_needed'}

    # Within the warning window — check for an open position and roll if found
    current_contract, new_contract = resolver.get_rollover_pair()
    logger.info(
        f'Rollover pair for {symbol}: {current_contract["conid"]} (expiry {current_contract["expirationDate"]})'
        f' → {new_contract["conid"]} (expiry {new_contract["expirationDate"]})'
    )
    current_position = _get_contract_position(current_contract['conid'])

    if current_position is None:
        logger.error(
            f'Cannot proceed with rollover for {symbol}: position check failed for conid {current_contract["conid"]}'
        )
        return {'status': 'error', 'message': 'Position check failed: cannot determine current position'}

    if current_position == 0:
        logger.warning(
            f'No open position on {resolver.ibkr_symbol} conid {current_contract["conid"]} '
            f'— {days_until_switch} day(s) until switch'
        )
        return {'status': 'warning', 'days_until_switch': days_until_switch}

    close_side = 'S' if current_position > 0 else 'B'
    close_result = place_order(current_contract['conid'], close_side)

    if isinstance(close_result, dict) and close_result.get('success') is False:
        logger.error(f'Failed to close position on {current_contract["conid"]}: {close_result}')
        return {'status': 'error', 'order': close_result}

    logger.info(f'Closed position on {current_contract["conid"]} ({close_side})')

    if not REOPEN_ON_ROLLOVER:
        return {
            'status': 'closed',
            'old_conid': current_contract['conid'],
            'new_conid': new_contract['conid'],
        }

    reopen_side = 'B' if current_position > 0 else 'S'
    reopen_result = place_order(new_contract['conid'], reopen_side)

    if isinstance(reopen_result, dict) and reopen_result.get('success') is False:
        logger.critical(
            f'Partial rollover for {resolver.ibkr_symbol}: position closed on {current_contract["conid"]} '
            f'but reopen on {new_contract["conid"]} failed — manual intervention required'
        )
        logger.error(f'Failed to reopen position on {new_contract["conid"]}: {reopen_result}')
        return {'status': 'reopen_failed', 'old_conid': current_contract['conid'], 'new_conid': new_contract['conid'],
                'order': reopen_result}

    logger.info(f'Reopened position on {new_contract["conid"]} ({reopen_side})')

    return {
        'status': 'rolled',
        'old_conid': current_contract['conid'],
        'new_conid': new_contract['conid'],
        'side': reopen_side,
    }
