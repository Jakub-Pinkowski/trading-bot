"""
Tests for IBKR Contracts Module.

Tests cover:
- Expiry date parsing (_parse_expiry)
- Front-month contract selection logic (_select_front_month)
- ContractResolver initialisation
- API fetch with validation (_fetch_contracts)
- File store (_store_contracts)
- File-first load with API fallback (_load_contracts)
- Switch context loading (_load_switch_context)
- Lazy loading with retry on expired data (_ensure_loaded)
- Public properties and methods (last_switch_date, next_switch_date, contracts,
  front_month, get_front_month_conid, get_rollover_pair)
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from app.ibkr.contracts import (
    MIN_BUFFER_DAYS,
    CONTRACTS_FILE_PATH,
    ContractResolver,
    _parse_expiry,
    _select_front_month,
)


# ==================== Helpers ====================

def make_resolver(ibkr_symbol='ZC'):
    """Return a ContractResolver with __init__ bypassed."""
    resolver = ContractResolver.__new__(ContractResolver)
    resolver.ibkr_symbol = ibkr_symbol
    resolver._last_switch_date = None
    resolver._next_switch_date = None  # None = "not yet loaded" sentinel
    resolver._contracts = None
    resolver._front_month = None
    return resolver


def make_resolver_with_switch_dates(ibkr_symbol='ZC', last=None, next_=None):
    """Return a ContractResolver with switch dates pre-loaded."""
    resolver = make_resolver(ibkr_symbol)
    resolver._last_switch_date = last
    resolver._next_switch_date = next_ if next_ is not None else (datetime.today() + timedelta(days=30))
    return resolver


# ==================== Test Classes ====================

class TestParseExpiry:
    """Test _parse_expiry handles both string and integer expirationDate values."""

    def test_parses_string_expiration_date(self):
        """String expirationDate in YYYYMMDD format is parsed to datetime."""
        result = _parse_expiry({'expirationDate': '20261215'})
        assert result == datetime(2026, 12, 15)

    def test_parses_integer_expiration_date(self):
        """Integer expirationDate as returned by the IBKR API is parsed to datetime."""
        result = _parse_expiry({'expirationDate': 20261215})
        assert result == datetime(2026, 12, 15)


class TestSelectFrontMonth:
    """Test pure front-month selection logic."""

    def test_no_switch_occurred_returns_earliest_valid_contract(self):
        """When last_switch_date is None, the earliest contract after MIN_BUFFER_DAYS is returned."""
        today = datetime.today()
        next_switch_date = today + timedelta(days=60)
        earlier_exp = today + timedelta(days=MIN_BUFFER_DAYS + 10)
        later_exp = today + timedelta(days=MIN_BUFFER_DAYS + 40)

        contracts = [
            {'conid': 'A', 'expirationDate': earlier_exp.strftime('%Y%m%d')},
            {'conid': 'B', 'expirationDate': later_exp.strftime('%Y%m%d')},
        ]

        result = _select_front_month(contracts, last_switch_date=None, next_switch_date=next_switch_date)

        assert result['conid'] == 'A'

    def test_switch_occurred_skips_contracts_expiring_before_next_switch(self):
        """When a switch has occurred, contracts expiring before next_switch_date are skipped."""
        today = datetime.today()
        last_switch_date = today - timedelta(days=10)
        next_switch_date = today + timedelta(days=50)
        early_exp = today + timedelta(days=MIN_BUFFER_DAYS + 10)  # before next_switch: skipped
        later_exp = today + timedelta(days=MIN_BUFFER_DAYS + 60)  # after next_switch: returned

        contracts = [
            {'conid': 'old', 'expirationDate': early_exp.strftime('%Y%m%d')},
            {'conid': 'new', 'expirationDate': later_exp.strftime('%Y%m%d')},
        ]

        result = _select_front_month(contracts, last_switch_date, next_switch_date)

        assert result['conid'] == 'new'

    def test_switch_occurred_today_follows_post_switch_branch(self):
        """last_switch_date == today enters the post-switch branch."""
        today = datetime.today()
        last_switch_date = today
        next_switch_date = today + timedelta(days=30)
        exp = today + timedelta(days=MIN_BUFFER_DAYS + 35)  # after next_switch

        contracts = [{'conid': 'X', 'expirationDate': exp.strftime('%Y%m%d')}]

        result = _select_front_month(contracts, last_switch_date, next_switch_date)

        assert result['conid'] == 'X'

    def test_all_contracts_within_buffer_raises_value_error(self):
        """ValueError raised when all contracts expire within MIN_BUFFER_DAYS."""
        today = datetime.today()
        next_switch_date = today + timedelta(days=60)
        near_exp = today + timedelta(days=MIN_BUFFER_DAYS - 1)

        contracts = [{'conid': 'Z', 'expirationDate': near_exp.strftime('%Y%m%d')}]

        with pytest.raises(ValueError, match='No valid contracts available within buffer'):
            _select_front_month(contracts, last_switch_date=None, next_switch_date=next_switch_date)

    def test_no_contract_after_next_switch_raises_value_error(self):
        """ValueError raised when a switch occurred but all valid contracts expire before next_switch_date."""
        today = datetime.today()
        last_switch_date = today - timedelta(days=10)
        next_switch_date = today + timedelta(days=50)
        exp = today + timedelta(days=MIN_BUFFER_DAYS + 10)  # valid but before next_switch

        contracts = [{'conid': 'X', 'expirationDate': exp.strftime('%Y%m%d')}]

        with pytest.raises(ValueError, match='No contracts found expiring after next switch date'):
            _select_front_month(contracts, last_switch_date, next_switch_date)

    def test_integer_expiration_date_is_handled(self):
        """expirationDate stored as int (as returned by IBKR API) is parsed correctly."""
        today = datetime.today()
        next_switch_date = today + timedelta(days=60)
        exp = today + timedelta(days=MIN_BUFFER_DAYS + 10)

        contracts = [{'conid': 'A', 'expirationDate': int(exp.strftime('%Y%m%d'))}]

        result = _select_front_month(contracts, last_switch_date=None, next_switch_date=next_switch_date)

        assert result['conid'] == 'A'

    def test_reverse_ordered_input_still_returns_earliest_front_month(self):
        """Sort is applied so the earliest valid contract is selected regardless of input order."""
        today = datetime.today()
        next_switch_date = today + timedelta(days=60)
        earlier_exp = today + timedelta(days=MIN_BUFFER_DAYS + 10)
        later_exp = today + timedelta(days=MIN_BUFFER_DAYS + 40)

        contracts = [
            {'conid': 'B', 'expirationDate': later_exp.strftime('%Y%m%d')},
            {'conid': 'A', 'expirationDate': earlier_exp.strftime('%Y%m%d')},
        ]

        result = _select_front_month(contracts, last_switch_date=None, next_switch_date=next_switch_date)

        assert result['conid'] == 'A'


class TestContractResolverInit:
    """Test ContractResolver initialisation."""

    def test_ibkr_symbol_set_from_tv_symbol(self, monkeypatch):
        """ibkr_symbol is derived from the TradingView symbol via parse + map."""
        mock_parse = MagicMock(return_value='ZC')
        mock_map = MagicMock(return_value='ZC')
        monkeypatch.setattr('app.ibkr.contracts.parse_symbol', mock_parse)
        monkeypatch.setattr('app.ibkr.contracts.map_tv_to_ibkr', mock_map)

        resolver = ContractResolver('ZC1!')

        assert resolver.ibkr_symbol == 'ZC'
        mock_parse.assert_called_once_with('ZC1!')
        mock_map.assert_called_once_with('ZC')

    def test_cache_fields_start_as_none(self, monkeypatch):
        """All lazy-cache fields are None after init."""
        monkeypatch.setattr('app.ibkr.contracts.parse_symbol', MagicMock(return_value='ZC'))
        monkeypatch.setattr('app.ibkr.contracts.map_tv_to_ibkr', MagicMock(return_value='ZC'))

        resolver = ContractResolver('ZC1!')

        assert resolver._last_switch_date is None
        assert resolver._next_switch_date is None
        assert resolver._contracts is None
        assert resolver._front_month is None


class TestFetchContracts:
    """Test _fetch_contracts: API call, validation, and error handling."""

    def test_returns_contracts_from_api(self, mock_api_get_contracts):
        """Contracts returned by the API are passed back to the caller."""
        contracts = [{'conid': '333', 'expirationDate': '20261215'}]
        mock_api_get_contracts.return_value = {'ZC': contracts}
        resolver = make_resolver('ZC')

        result = resolver._fetch_contracts()

        assert result == contracts
        mock_api_get_contracts.assert_called_once_with('/trsrv/futures?symbols=ZC')

    def test_raises_and_logs_on_empty_api_response(
        self, mock_logger_contracts, mock_api_get_contracts
    ):
        """ValueError raised and error logged when API returns no contracts."""
        mock_api_get_contracts.return_value = {}
        resolver = make_resolver('ZC')

        with pytest.raises(ValueError, match='No contracts found for symbol: ZC'):
            resolver._fetch_contracts()

        mock_logger_contracts.error.assert_called_once()

    def test_raises_and_logs_on_api_exception(
        self, mock_logger_contracts, mock_api_get_contracts
    ):
        """ValueError raised and error logged when the API call itself raises."""
        mock_api_get_contracts.side_effect = Exception('connection refused')
        resolver = make_resolver('ZC')

        with pytest.raises(ValueError, match='API error fetching contracts for symbol: ZC'):
            resolver._fetch_contracts()

        mock_logger_contracts.error.assert_called_once()


class TestStoreContracts:
    """Test _store_contracts: merges with existing data and writes to file."""

    def test_merges_and_saves_contracts(self, mock_load_file, mock_save_file):
        """New contracts merged with existing stored data and written to file."""
        existing = {'ES': [{'conid': '999'}]}
        mock_load_file.return_value = existing
        contracts = [{'conid': '111', 'expirationDate': '20261215'}]
        resolver = make_resolver('ZC')

        resolver._store_contracts(contracts)

        mock_save_file.assert_called_once_with(
            {'ES': [{'conid': '999'}], 'ZC': contracts}, CONTRACTS_FILE_PATH
        )


class TestLoadContracts:
    """Test _load_contracts: file-first lookup with API fallback."""

    def test_returns_stored_list_without_fetching(self, mock_load_file, monkeypatch):
        """Stored contract list returned directly without calling the API."""
        stored = [{'conid': '111', 'expirationDate': '20261215'}]
        mock_load_file.return_value = {'ZC': stored}
        mock_fetch = MagicMock()
        resolver = make_resolver('ZC')
        monkeypatch.setattr(resolver, '_fetch_contracts', mock_fetch)

        result = resolver._load_contracts()

        assert result == stored
        mock_fetch.assert_not_called()

    def test_fetches_and_stores_when_symbol_absent(self, mock_load_file, monkeypatch):
        """API is called and result stored when symbol is absent from the file."""
        mock_load_file.return_value = {}
        fresh = [{'conid': '222', 'expirationDate': '20261215'}]
        mock_fetch = MagicMock(return_value=fresh)
        mock_store = MagicMock()
        resolver = make_resolver('ZC')
        monkeypatch.setattr(resolver, '_fetch_contracts', mock_fetch)
        monkeypatch.setattr(resolver, '_store_contracts', mock_store)

        result = resolver._load_contracts()

        assert result == fresh
        mock_fetch.assert_called_once()
        mock_store.assert_called_once_with(fresh)

    def test_propagates_when_fetch_raises(self, mock_load_file, monkeypatch):
        """ValueError from _fetch_contracts propagates out."""
        mock_load_file.return_value = {}
        mock_fetch = MagicMock(side_effect=ValueError('No contracts from API'))
        resolver = make_resolver('ZC')
        monkeypatch.setattr(resolver, '_fetch_contracts', mock_fetch)

        with pytest.raises(ValueError, match='No contracts from API'):
            resolver._load_contracts()


class TestEnsureLoaded:
    """Test _ensure_loaded: lazy loading and retry on expired stored data."""

    def test_returns_early_when_already_loaded(self, monkeypatch):
        """No loading happens if _contracts is already populated."""
        resolver = make_resolver_with_switch_dates()
        resolver._contracts = [{'conid': 'A'}]
        resolver._front_month = {'conid': 'A'}
        mock_load = MagicMock()
        monkeypatch.setattr(resolver, '_load_contracts', mock_load)

        resolver._ensure_loaded()

        mock_load.assert_not_called()

    def test_loads_and_caches_contracts_and_front_month(self, monkeypatch):
        """Contracts and front_month are cached after successful load."""
        resolver = make_resolver_with_switch_dates(last=datetime.today() - timedelta(days=10))
        contracts = [{'conid': 'A', 'expirationDate': '20261215'}]
        front_month = {'conid': 'A'}
        monkeypatch.setattr(resolver, '_load_contracts', MagicMock(return_value=contracts))
        monkeypatch.setattr('app.ibkr.contracts._select_front_month', MagicMock(return_value=front_month))

        resolver._ensure_loaded()

        assert resolver._contracts == contracts
        assert resolver._front_month == front_month

    def test_fetches_fresh_when_stored_contracts_expired(self, monkeypatch):
        """Fresh contracts fetched and cached when stored contracts yield no valid front month."""
        resolver = make_resolver_with_switch_dates()
        stale = [{'conid': 'old', 'expirationDate': '20240101'}]
        fresh = [{'conid': 'new', 'expirationDate': '20261215'}]
        front_month = {'conid': 'new'}

        mock_select = MagicMock(side_effect=[ValueError('expired'), front_month])
        monkeypatch.setattr(resolver, '_load_contracts', MagicMock(return_value=stale))
        monkeypatch.setattr(resolver, '_fetch_contracts', MagicMock(return_value=fresh))
        mock_store = MagicMock()
        monkeypatch.setattr(resolver, '_store_contracts', mock_store)
        monkeypatch.setattr('app.ibkr.contracts._select_front_month', mock_select)

        resolver._ensure_loaded()

        assert resolver._contracts == fresh
        assert resolver._front_month == front_month
        mock_store.assert_called_once_with(fresh)

    def test_propagates_when_fresh_fetch_also_fails(self, monkeypatch):
        """ValueError from _fetch_contracts propagates when retry also fails."""
        resolver = make_resolver_with_switch_dates()
        monkeypatch.setattr(resolver, '_load_contracts', MagicMock(return_value=[]))
        monkeypatch.setattr(
            'app.ibkr.contracts._select_front_month', MagicMock(side_effect=ValueError('No valid'))
        )
        monkeypatch.setattr(
            resolver, '_fetch_contracts', MagicMock(side_effect=ValueError('API down'))
        )

        with pytest.raises(ValueError, match='API down'):
            resolver._ensure_loaded()

    def test_propagates_when_second_select_fails_after_successful_fetch(self, monkeypatch):
        """ValueError from second _select_front_month propagates when fresh contracts are
        fetched successfully but still yield no valid front-month (e.g. all within buffer)."""
        resolver = make_resolver_with_switch_dates()
        fresh = [{'conid': 'new', 'expirationDate': '20261215'}]
        mock_store = MagicMock()

        monkeypatch.setattr(resolver, '_load_contracts', MagicMock(return_value=[]))
        monkeypatch.setattr(resolver, '_fetch_contracts', MagicMock(return_value=fresh))
        monkeypatch.setattr(resolver, '_store_contracts', mock_store)
        monkeypatch.setattr(
            'app.ibkr.contracts._select_front_month',
            MagicMock(side_effect=ValueError('all within buffer')),
        )

        with pytest.raises(ValueError, match='all within buffer'):
            resolver._ensure_loaded()

        # Contracts are stored even when selection fails, preserving the fresh data
        mock_store.assert_called_once_with(fresh)


class TestLoadSwitchContext:
    """Test _load_switch_context: reads YAML, returns (last_switch_date, next_switch_date)."""

    def test_returns_last_and_next_switch_dates(self, mock_yaml_load):
        """Most recently passed and next upcoming dates are returned."""
        today = datetime.today()
        past = today - timedelta(days=5)
        future = today + timedelta(days=10)
        mock_yaml_load.return_value = {'ZC': [past, future]}
        resolver = make_resolver('ZC')

        last, next_ = resolver._load_switch_context()

        assert last == past
        assert next_ == future

    def test_returns_none_for_last_when_no_past_dates(self, mock_yaml_load):
        """last_switch_date is None when all dates are in the future."""
        today = datetime.today()
        future1 = today + timedelta(days=10)
        future2 = today + timedelta(days=40)
        mock_yaml_load.return_value = {'ZC': [future1, future2]}
        resolver = make_resolver('ZC')

        last, next_ = resolver._load_switch_context()

        assert last is None
        assert next_ == future1

    def test_returns_most_recent_of_multiple_past_dates(self, mock_yaml_load):
        """max of past dates is returned as last_switch_date."""
        today = datetime.today()
        older_past = today - timedelta(days=20)
        recent_past = today - timedelta(days=5)
        future = today + timedelta(days=10)
        mock_yaml_load.return_value = {'ZC': [older_past, recent_past, future]}
        resolver = make_resolver('ZC')

        last, next_ = resolver._load_switch_context()

        assert last == recent_past
        assert next_ == future

    def test_returns_nearest_of_multiple_future_dates(self, mock_yaml_load):
        """min of future dates is returned as next_switch_date."""
        today = datetime.today()
        past = today - timedelta(days=5)
        near_future = today + timedelta(days=10)
        far_future = today + timedelta(days=40)
        mock_yaml_load.return_value = {'ZC': [past, near_future, far_future]}
        resolver = make_resolver('ZC')

        last, next_ = resolver._load_switch_context()

        assert next_ == near_future

    def test_raises_informative_error_when_yaml_file_missing(self, monkeypatch):
        """FileNotFoundError from a missing YAML is re-raised as a clear ValueError."""
        monkeypatch.setattr('builtins.open', MagicMock(side_effect=FileNotFoundError))
        resolver = make_resolver('ZC')

        with pytest.raises(ValueError, match='Switch dates file not found'):
            resolver._load_switch_context()

    def test_raises_when_symbol_not_in_yaml(self, mock_yaml_load):
        """ValueError raised when no switch dates exist for the symbol."""
        mock_yaml_load.return_value = {}
        resolver = make_resolver('ZC')

        with pytest.raises(ValueError, match="No switch dates found for symbol 'ZC'"):
            resolver._load_switch_context()

    def test_raises_when_all_dates_are_past(self, mock_yaml_load):
        """ValueError raised when all known switch dates are in the past."""
        today = datetime.today()
        mock_yaml_load.return_value = {'ZC': [today - timedelta(days=1)]}
        resolver = make_resolver('ZC')

        with pytest.raises(ValueError, match="All switch dates for 'ZC' are in the past"):
            resolver._load_switch_context()

    def test_parses_iso_string_dates_from_yaml(self, mock_yaml_load):
        """Dates stored as ISO strings in YAML are parsed to datetime before comparison."""
        today = datetime.today()
        future = today + timedelta(days=10)
        mock_yaml_load.return_value = {'ZC': [future.strftime('%Y-%m-%dT%H:%M:%S')]}
        resolver = make_resolver('ZC')

        last, next_ = resolver._load_switch_context()

        assert last is None
        assert next_.date() == future.date()

    def test_skips_none_entries_in_yaml(self, mock_yaml_load):
        """None entries in the YAML dates list are silently skipped."""
        today = datetime.today()
        future = today + timedelta(days=10)
        mock_yaml_load.return_value = {'ZC': [None, future.strftime('%Y-%m-%dT%H:%M:%S')]}
        resolver = make_resolver('ZC')

        last, next_ = resolver._load_switch_context()

        assert next_.date() == future.date()

    def test_resolves_mini_symbol_via_mappings(self, mock_yaml_load):
        """Mini/micro symbols are resolved through _symbol_mappings before lookup."""
        today = datetime.today()
        future = today + timedelta(days=10)
        mock_yaml_load.return_value = {
            '_symbol_mappings': {'MES': 'ES'},
            'ES': [future],
        }
        resolver = make_resolver('MES')

        last, next_ = resolver._load_switch_context()

        assert next_ == future


class TestEnsureSwitchDatesLoaded:
    """Test _ensure_switch_dates_loaded: lazy load and cache of switch context."""

    def test_loads_on_first_access(self, monkeypatch):
        """_load_switch_context called when _next_switch_date is None."""
        resolver = make_resolver('ZC')
        last = datetime.today() - timedelta(days=5)
        next_ = datetime.today() + timedelta(days=30)
        mock_load = MagicMock(return_value=(last, next_))
        monkeypatch.setattr(resolver, '_load_switch_context', mock_load)

        resolver._ensure_switch_dates_loaded()

        assert resolver._last_switch_date == last
        assert resolver._next_switch_date == next_
        mock_load.assert_called_once()

    def test_skips_load_when_already_loaded(self, monkeypatch):
        """_load_switch_context not called when _next_switch_date is already set."""
        resolver = make_resolver_with_switch_dates()
        mock_load = MagicMock()
        monkeypatch.setattr(resolver, '_load_switch_context', mock_load)

        resolver._ensure_switch_dates_loaded()

        mock_load.assert_not_called()


class TestSwitchDateProperties:
    """Test last_switch_date and next_switch_date properties."""

    def test_last_switch_date_triggers_load_and_returns_value(self, monkeypatch):
        """last_switch_date triggers _ensure_switch_dates_loaded and returns cached value."""
        resolver = make_resolver('ZC')
        last = datetime.today() - timedelta(days=5)
        next_ = datetime.today() + timedelta(days=30)
        monkeypatch.setattr(resolver, '_load_switch_context', MagicMock(return_value=(last, next_)))

        assert resolver.last_switch_date == last

    def test_next_switch_date_triggers_load_and_returns_value(self, monkeypatch):
        """next_switch_date triggers _ensure_switch_dates_loaded and returns cached value."""
        resolver = make_resolver('ZC')
        next_ = datetime.today() + timedelta(days=30)
        monkeypatch.setattr(resolver, '_load_switch_context', MagicMock(return_value=(None, next_)))

        assert resolver.next_switch_date == next_

    def test_properties_load_only_once(self, monkeypatch):
        """Both properties share the same load; _load_switch_context called once total."""
        resolver = make_resolver('ZC')
        next_ = datetime.today() + timedelta(days=30)
        mock_load = MagicMock(return_value=(None, next_))
        monkeypatch.setattr(resolver, '_load_switch_context', mock_load)

        _ = resolver.last_switch_date
        _ = resolver.next_switch_date

        mock_load.assert_called_once()


class TestContractsAndFrontMonthProperties:
    """Test the contracts and front_month properties delegate to _ensure_loaded."""

    def test_contracts_property_calls_ensure_loaded(self, monkeypatch):
        """contracts property triggers _ensure_loaded and returns _contracts."""
        resolver = make_resolver('ZC')
        contracts = [{'conid': 'A'}]
        resolver._contracts = contracts
        monkeypatch.setattr(resolver, '_ensure_loaded', lambda: None)

        assert resolver.contracts == contracts

    def test_front_month_property_calls_ensure_loaded(self, monkeypatch):
        """front_month property triggers _ensure_loaded and returns _front_month."""
        resolver = make_resolver('ZC')
        front = {'conid': 'A'}
        resolver._front_month = front
        monkeypatch.setattr(resolver, '_ensure_loaded', lambda: None)

        assert resolver.front_month == front


class TestGetFrontMonthConid:
    """Test get_front_month_conid delegates to the front_month property."""

    def test_returns_conid_from_front_month(self, monkeypatch):
        """conid extracted from front_month and returned."""
        resolver = make_resolver('ZC')
        monkeypatch.setattr(
            type(resolver), 'front_month', property(lambda self: {'conid': '123', 'expirationDate': '20261215'})
        )

        result = resolver.get_front_month_conid()

        assert result == '123'


class TestGetRolloverPair:
    """Test get_rollover_pair returns (current, next) contract tuple."""

    def test_returns_current_and_next_contract(self, monkeypatch):
        """Next contract is the first one expiring strictly after the current contract."""
        resolver = make_resolver('ZC')
        today = datetime.today()
        current_contract = {'conid': 'A', 'expirationDate': int((today + timedelta(days=30)).strftime('%Y%m%d'))}
        next_contract = {'conid': 'B', 'expirationDate': int((today + timedelta(days=90)).strftime('%Y%m%d'))}
        contracts = [next_contract, current_contract]  # intentionally unsorted

        monkeypatch.setattr(type(resolver), 'front_month', property(lambda self: current_contract))
        monkeypatch.setattr(type(resolver), 'contracts', property(lambda self: contracts))

        result = resolver.get_rollover_pair()

        assert result == (current_contract, next_contract)

    def test_next_contract_is_immediate_successor(self, monkeypatch):
        """Next contract is the immediate successor, not a further-out contract."""
        resolver = make_resolver('ZC')
        today = datetime.today()
        current_contract = {'conid': 'A', 'expirationDate': int((today + timedelta(days=10)).strftime('%Y%m%d'))}
        next_contract = {'conid': 'B', 'expirationDate': int((today + timedelta(days=70)).strftime('%Y%m%d'))}
        far_contract = {'conid': 'C', 'expirationDate': int((today + timedelta(days=130)).strftime('%Y%m%d'))}
        contracts = [current_contract, next_contract, far_contract]

        monkeypatch.setattr(type(resolver), 'front_month', property(lambda self: current_contract))
        monkeypatch.setattr(type(resolver), 'contracts', property(lambda self: contracts))

        result = resolver.get_rollover_pair()

        assert result == (current_contract, next_contract)

    def test_raises_when_no_contract_after_current(self, monkeypatch):
        """ValueError raised when no contract expires after the current one."""
        resolver = make_resolver('ZC')
        today = datetime.today()
        current_contract = {'conid': 'A', 'expirationDate': int((today + timedelta(days=90)).strftime('%Y%m%d'))}

        monkeypatch.setattr(type(resolver), 'front_month', property(lambda self: current_contract))
        monkeypatch.setattr(type(resolver), 'contracts', property(lambda self: [current_contract]))

        with pytest.raises(ValueError, match='No next contract found after'):
            resolver.get_rollover_pair()
