from datetime import datetime, timedelta

import yaml

from app.utils.api_utils import api_get
from app.utils.file_utils import load_file, save_file
from app.utils.generic_utils import parse_symbol
from app.utils.logger import get_logger
from config import DATA_DIR
from futures_config.symbol_mapping import map_tv_to_ibkr

logger = get_logger('ibkr/contracts')

# ==================== Configuration ====================

MIN_BUFFER_DAYS = 5  # Minimum days before expiry to still trade a contract
CONTRACTS_FILE_PATH = DATA_DIR / "contracts" / "contracts.json"
SWITCH_DATES_FILE_PATH = DATA_DIR / "contracts" / "contract_switch_dates.yaml"


# ==================== Pure Functions ====================

def _parse_expiry(contract):
    """Parse the expirationDate field of a contract dict into a datetime.

    Accepts both string ('20260514') and integer (20260514) values as returned
    by the IBKR API.

    Args:
        contract: Contract dict containing an 'expirationDate' field.

    Returns:
        datetime representing the contract expiry.
    """
    return datetime.strptime(str(contract['expirationDate']), '%Y%m%d')


def _select_front_month(contracts, last_switch_date, next_switch_date):
    """
    Select the front-month contract based on whether a TradingView switch has occurred.

    If no switch has occurred yet (last_switch_date is None): returns the earliest
    contract still valid after MIN_BUFFER_DAYS.

    If a switch has occurred: skips contracts expiring at or before next_switch_date
    and returns the first remaining contract. This handles the case where TV has
    already rolled to a later contract even though an earlier one has not yet expired.

    Args:
        contracts: List of contract dicts each containing an 'expirationDate'
            field in YYYYMMDD format as string or integer (e.g. [{'conid': '123', 'expirationDate': 20260514}])
        last_switch_date: The most recently passed TV switch date, or None if no
            switch has occurred yet.
        next_switch_date: datetime of the next upcoming TradingView contract switch.

    Returns:
        Contract dict of the selected front-month contract.

    Raises:
        ValueError: If all contracts expire within MIN_BUFFER_DAYS, or no contract
            is found expiring after next_switch_date.
    """
    today = datetime.today()
    cutoff = today + timedelta(days=MIN_BUFFER_DAYS)

    valid_contracts = sorted(
        [contract for contract in contracts if _parse_expiry(contract) > cutoff],
        key=_parse_expiry,
    )

    if not valid_contracts:
        raise ValueError(f'No valid contracts available within buffer of {MIN_BUFFER_DAYS} days.')

    if last_switch_date is None:
        # No switch has occurred yet — return the earliest valid contract
        return valid_contracts[0]

    # A switch has occurred — skip contracts expiring at or before the next switch date,
    # matching the contract TV has already rolled into
    front_month_contract = next(
        (contract for contract in valid_contracts if _parse_expiry(contract) > next_switch_date),
        None,
    )
    if front_month_contract is None:
        raise ValueError(
            f'No contracts found expiring after next switch date {next_switch_date.date()}.'
        )
    return front_month_contract


# ==================== Contract Resolver ====================

class ContractResolver:
    """
    Resolves contracts and switch dates for a single TradingView symbol.

    Lazily loads the switch dates, contracts, and front-month contract on first
    access, caching each so subsequent lookups within the same instance are free.

    Attributes:
        ibkr_symbol: IBKR symbol string resolved from the TradingView symbol.
    """

    def __init__(self, symbol):
        """
        Args:
            symbol: TradingView symbol string (e.g. 'ZC1!').
        """
        self.ibkr_symbol = map_tv_to_ibkr(parse_symbol(symbol))
        self._last_switch_date = None
        self._next_switch_date = None  # None also used as "not yet loaded" sentinel
        self._contracts = None
        self._front_month = None

    # --- Public Interface ---

    @property
    def last_switch_date(self):
        """Most recently passed TradingView switch date, or None if no switch has occurred."""
        self._ensure_switch_dates_loaded()
        return self._last_switch_date

    @property
    def next_switch_date(self):
        """Next upcoming TradingView switch date for this symbol."""
        self._ensure_switch_dates_loaded()
        return self._next_switch_date

    @property
    def contracts(self):
        """Contract list guaranteed to contain a valid front-month."""
        self._ensure_loaded()
        return self._contracts

    @property
    def front_month(self):
        """Front-month contract dict for the current switch date."""
        self._ensure_loaded()
        return self._front_month

    def get_front_month_conid(self):
        """
        Return the IBKR contract ID of the front-month contract.

        Returns:
            Contract ID (conid) string.

        Raises:
            ValueError: If no qualifying contracts are found.
        """
        return self.front_month['conid']

    def get_rollover_pair(self):
        """
        Return the current and next front-month contracts for a rollover.

        The current contract is the cached front-month. The next contract is
        the first contract expiring strictly after the current one.

        Returns:
            Tuple of (current_contract, new_contract) — each a contract dict with
            at least 'conid' and 'expirationDate'.

        Raises:
            ValueError: If no next contract is found after the current one.
        """
        current_contract = self.front_month
        current_expiry = _parse_expiry(current_contract)

        sorted_contracts = sorted(self.contracts, key=_parse_expiry)
        new_contract = next(
            (contract for contract in sorted_contracts if _parse_expiry(contract) > current_expiry),
            None,
        )
        if new_contract is None:
            raise ValueError(f'No next contract found after {current_expiry.date()} for {self.ibkr_symbol}.')
        return current_contract, new_contract

    # --- Implementation ---

    def _ensure_switch_dates_loaded(self):
        """
        Load and cache last and next switch dates if not already done.

        Uses _next_switch_date is None as the "not yet loaded" sentinel since
        _load_switch_context always sets it to a datetime on success.
        """
        if self._next_switch_date is None:
            self._last_switch_date, self._next_switch_date = self._load_switch_context()

    def _load_switch_context(self):
        """
        Load switch dates from YAML and return the most recently passed and next upcoming dates.

        Resolves mini/micro symbols through the '_symbol_mappings' key in the YAML.

        Returns:
            Tuple of (last_switch_date or None, next_switch_date). last_switch_date is
            None if no switch has occurred yet (today is before all known switch dates).

        Raises:
            ValueError: If the symbol has no switch dates in the YAML, or all
                known switch dates are in the past (YAML needs updating).
        """
        with open(SWITCH_DATES_FILE_PATH, 'r') as f:
            switch_dates = yaml.safe_load(f)

        symbol_mappings = switch_dates.get('_symbol_mappings', {})
        resolved_symbol = symbol_mappings.get(self.ibkr_symbol, self.ibkr_symbol)

        dates = switch_dates.get(resolved_symbol)
        if not dates:
            raise ValueError(f"No switch dates found for symbol '{self.ibkr_symbol}' in YAML.")

        today = datetime.today()
        parsed_dates = [
            date if isinstance(date, datetime) else datetime.fromisoformat(date)
            for date in dates
            if date is not None
        ]
        past_dates = [date for date in parsed_dates if date <= today]
        future_dates = [date for date in parsed_dates if date > today]

        if not future_dates:
            raise ValueError(
                f"All switch dates for '{self.ibkr_symbol}' are in the past. YAML needs updating."
            )

        last_switch_date = max(past_dates) if past_dates else None
        return last_switch_date, min(future_dates)

    def _ensure_loaded(self):
        """
        Load and cache contracts and the front-month contract if not already done.

        Tries stored contracts first; fetches fresh from the API if no stored
        contract clears the MIN_BUFFER_DAYS window. Both `contracts` and
        `front_month` properties delegate here so loading always happens in one place.
        """
        if self._contracts is not None:
            return

        raw_contracts = self._load_contracts()
        try:
            front_month = _select_front_month(raw_contracts, self.last_switch_date, self.next_switch_date)
        except ValueError:
            raw_contracts = self._fetch_contracts()
            self._store_contracts(raw_contracts)
            front_month = _select_front_month(raw_contracts, self.last_switch_date, self.next_switch_date)
        self._front_month = front_month
        self._contracts = raw_contracts

    def _load_contracts(self):
        """
        Return the stored contract list, fetching and storing from the API if not yet stored.

        Conids are permanent once assigned by IBKR, so a stored list is always
        valid regardless of how old it is.

        Returns:
            List of contract dicts, each containing at least 'conid' and 'expirationDate'.

        Raises:
            ValueError: If the symbol is not stored and the API returns no contracts.
        """
        stored = load_file(CONTRACTS_FILE_PATH)
        contract_list = stored.get(self.ibkr_symbol)

        if isinstance(contract_list, list):
            return contract_list

        fresh_contracts = self._fetch_contracts()
        self._store_contracts(fresh_contracts)
        return fresh_contracts

    def _fetch_contracts(self):
        """
        Fetch and validate contracts from the IBKR API.

        Returns:
            List of contract dicts from the API.

        Raises:
            ValueError: If the API call fails or returns no contracts for the symbol.
        """
        try:
            contracts_data = api_get(f'/trsrv/futures?symbols={self.ibkr_symbol}')
            fresh_contracts = contracts_data.get(self.ibkr_symbol, [])
        except Exception as err:
            logger.error(f'Error fetching contract data for {self.ibkr_symbol}: {err}')
            raise ValueError(f'API error fetching contracts for symbol: {self.ibkr_symbol}') from err

        if not fresh_contracts:
            logger.error(f'No contracts found for symbol: {self.ibkr_symbol}')
            raise ValueError(f'No contracts found for symbol: {self.ibkr_symbol}')

        return fresh_contracts

    def _store_contracts(self, contracts):
        """
        Write a contract list to the local contracts file.

        Args:
            contracts: List of contract dicts to persist.
        """
        stored = load_file(CONTRACTS_FILE_PATH)
        stored[self.ibkr_symbol] = contracts
        save_file(stored, CONTRACTS_FILE_PATH)
