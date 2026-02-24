# Plan: Contracts Front-Month Rewrite + Near-Delivery Rollover

**Status:** Draft
**Priority:** HIGH
**Date:** 2026-02-24

---

## Problem

`MIN_DAYS_UNTIL_EXPIRY = 60` in `_get_closest_contract` filters out the front-month contract and
causes live trades to land on the second (or third) month instead. This is actively wrong —
it inflates the bid/ask spread and means we're not aligned with TradingView's `1!` signal.

Additionally, there is no handling for open positions approaching the IBKR close-out date. If
a position is held through the switch window, IBKR will force-close it at an unpredictable price.

---

## Source of Truth for Switch Dates

`data/historical_data/contract_switch_dates.yaml` already contains the rollover dates for all
symbols (used by the backtesting engine). These dates represent when TradingView's `1!`
continuous contract flips from one physical contract to the next — which is the exact signal
we need to determine which contract is "current" for live trading.

**Important:** The YAML currently has dates through **2025** only (last ZC entry:
`2025-08-28`). Today is 2026-02-24 — meaning the YAML must be extended before the new
contracts logic can work reliably. See Part 0 below.

Mini/micro symbol mappings are already handled via the `_symbol_mappings` key in the YAML
and match the mappings in `futures_config/symbol_mapping.py`.

---

## Part 0 — Extend `contract_switch_dates.yaml` (Prerequisite)

The YAML must be kept current. Two options:

### Option A: Manual extension (immediate fix)

Add 2026 dates for all symbols by extrapolating from the historical pattern in the file.
Each symbol has a predictable cycle (e.g., ZC switches ~Feb/Apr/Jun/Aug/Nov, CL monthly).

Steps:

1. For each symbol, inspect the last 2 years of switch dates to identify the pattern.
2. Append 2026 entries (at minimum) following the same cadence and time-of-day.
3. Commit the updated YAML.

### Option B: Automated extension script (longer term)

Create `scripts/update_switch_dates.py` that:

- Reads the TradingView `fetch_data.py` download logic.
- Parses the timestamps where `1!` changes underlying contract in the downloaded data.
- Writes new entries back to the YAML.

**Recommendation:** Do Option A now; Option B is a future improvement.

---

## Part 1 — Rewrite `app/ibkr/contracts.py`

### 1.1 Remove `MIN_DAYS_UNTIL_EXPIRY`

Delete the constant entirely. It is the root cause of the bug.

### 1.2 Add constants and switch-date helpers

```python
import yaml

SWITCH_DATES_FILE_PATH = DATA_DIR / "historical_data" / "contract_switch_dates.yaml"
MIN_BUFFER_DAYS = 5  # Minimum days before expiry to still trade a contract (guards
# against accidentally entering an expiring-today contract)
```

New private functions:

```python
def _load_switch_dates():
    """Load and return the contract switch dates dict from YAML."""


def _get_next_switch_date(ibkr_symbol, switch_dates):
    """
    Return the next upcoming switch date for ibkr_symbol.
    Resolves mini/micro symbols through _symbol_mappings automatically.

    Raises:
        ValueError: If the symbol has no switch dates in the YAML, or all
            known switch dates are in the past (YAML needs updating).
    """


def _resolve_next_switch_date(ibkr_symbol):
  """
  Load switch dates and return the next upcoming switch date for ibkr_symbol.

  Convenience wrapper — _load_switch_dates and _get_next_switch_date always
  travel together; this collapses them into a single call.
  """
  switch_dates = _load_switch_dates()
  return _get_next_switch_date(ibkr_symbol, switch_dates)
```

The following public wrapper is also added so callers outside `contracts.py` (specifically
`rollover.py`) can get the next switch date without importing parsing utils:

```python
def get_next_switch_date(symbol):
  """
  Return the next upcoming switch date for a TradingView symbol.

  Args:
      symbol: TradingView symbol string (e.g. 'ZC1!').

  Raises:
      ValueError: If the symbol has no upcoming switch date in the YAML.
  """
  ibkr_symbol = map_tv_to_ibkr(parse_symbol(symbol))
  return _resolve_next_switch_date(ibkr_symbol)
```

### 1.3 Add `_get_contracts`

Extracts the file-first lookup pattern shared by both public functions. Conids are
permanent once assigned by IBKR, so data in the contracts file is always valid — there
is no expiry. The file is only written on first use per symbol (or if all stored contracts
have expired and no qualifying one can be found, forcing a re-fetch).

```python
def _get_contracts(ibkr_symbol):
  """
  Return the contract list for ibkr_symbol, reading from the contracts file
  and falling back to the IBKR API if the symbol is not yet stored.

  Args:
      ibkr_symbol: IBKR symbol string (e.g. 'ZC').

  Returns:
      List of contract dicts, each containing at least 'conid' and 'expirationDate'.

  Raises:
      ValueError: If no contracts are found for the symbol from the API.
  """
  stored = load_file(CONTRACTS_FILE_PATH)
  contract_list = stored.get(ibkr_symbol)

  if isinstance(contract_list, list):
    return contract_list

  fresh_contracts = _fetch_contract(ibkr_symbol)
  if not fresh_contracts:
    logger.error(f'No contracts found for symbol: {ibkr_symbol}')
    raise ValueError(f'No contracts found for symbol: {ibkr_symbol}')

  stored[ibkr_symbol] = fresh_contracts
  save_file(stored, CONTRACTS_FILE_PATH)
  return fresh_contracts
```

### 1.4 Add `_refresh_contracts`

Extracts the identical fallback pattern shared by both public functions — fetch fresh from
the API, update the contracts file, return the list. Both public functions' `except`
blocks collapse to two lines.

```python
def _refresh_contracts(ibkr_symbol):
  """
  Fetch contracts fresh from the IBKR API, update the contracts file, and return the list.

  Called when stored contracts have all expired and a fresh fetch is required.

  Args:
      ibkr_symbol: IBKR symbol string (e.g. 'ZC').

  Returns:
      List of fresh contract dicts.

  Raises:
      ValueError: If the API returns no contracts for the symbol.
  """
  fresh_contracts = _fetch_contract(ibkr_symbol)
  if not fresh_contracts:
    logger.error(f'No contracts found for symbol: {ibkr_symbol}')
    raise ValueError(f'No contracts found for symbol: {ibkr_symbol}')

  stored = load_file(CONTRACTS_FILE_PATH)
  stored[ibkr_symbol] = fresh_contracts
  save_file(stored, CONTRACTS_FILE_PATH)
  return fresh_contracts
```

### 1.5 Rewrite `_get_closest_contract` → `_get_front_month_contract`

Rename to make intent explicit. New signature:

```python
def _get_front_month_contract(contracts, next_switch_date):
```

**Selection logic:**

```
cutoff = today + MIN_BUFFER_DAYS

Sort contracts ascending by expirationDate.
Filter to contracts expiring after cutoff.

If today < next_switch_date:
    → We are on the current front month.
    → Return earliest contract after cutoff.

If today >= next_switch_date:
    → The roll has already happened; the "old" front month should be avoided.
    → Skip contracts that expire before or too close to next_switch_date.
    → Return the first contract that expires AFTER next_switch_date.
```

If no qualifying contract is found → raise `ValueError`.
`_resolve_next_switch_date` always provides a date (raises if not found), so there is no
None branch.

### 1.6 Add `get_contracts_for_rollover`

Public function used exclusively by `rollover.py`. Both public functions now accept TV
symbols consistently — symbol parsing stays inside `contracts.py`, `rollover.py` imports
no utils for symbol handling. The sentinel `next_switch_date + timedelta(days=1)` lives
here rather than leaking into `rollover.py`.

```python
def get_contracts_for_rollover(symbol):
  """
  Return the current and next front-month contracts for a rollover.

  Conids are permanent once assigned, so the contracts file is always a valid
  source of truth. Only falls back to the API if the symbol is not yet stored.

  Args:
      symbol: TradingView symbol string (e.g. 'ZC1!').

  Returns:
      Tuple of (current_contract, next_contract) — each a contract dict with at
      least 'conid' and 'expirationDate'.

  Raises:
      ValueError: If the symbol has no upcoming switch date or no qualifying
          contracts are found.
  """
  ibkr_symbol = map_tv_to_ibkr(parse_symbol(symbol))
  next_switch_date = _resolve_next_switch_date(ibkr_symbol)
  contracts = _get_contracts(ibkr_symbol)

  try:
    current = _get_front_month_contract(contracts, next_switch_date)
    new = _get_front_month_contract(contracts, next_switch_date + timedelta(days=1))
    return current, new
  except ValueError:
    # All stored contracts have expired — fetch fresh and retry once
    contracts = _refresh_contracts(ibkr_symbol)
    current = _get_front_month_contract(contracts, next_switch_date)
    new = _get_front_month_contract(contracts, next_switch_date + timedelta(days=1))
    return current, new
```

### 1.7 Update `get_front_month_conid`

Simplified to use the shared helpers. Both public functions now have the same shape.

```python
def get_front_month_conid(symbol):
  ibkr_symbol = map_tv_to_ibkr(parse_symbol(symbol))
  next_switch_date = _resolve_next_switch_date(ibkr_symbol)
  contracts = _get_contracts(ibkr_symbol)

  try:
    contract = _get_front_month_contract(contracts, next_switch_date)
    return contract['conid']
  except ValueError:
    # All stored contracts have expired — fetch fresh and retry once
    contracts = _refresh_contracts(ibkr_symbol)
    contract = _get_front_month_contract(contracts, next_switch_date)
    return contract['conid']
```

**Note:** `min_days_until_expiry` parameter removed from the public signature — it was only
ever called with the default value in production code and was the source of the bug.

---

## Two Independent Flows

Two separate routes, each with a single responsibility. Neither triggers the other.
`place_order` and `_get_contract_position` from `orders.py` are shared by both flows.

```
Flow 1 — Trading Signal
───────────────────────
strategy indicator (e.g. indicator_rsi.pine)
  → POST /trading  {"symbol": "ZC1!", "side": "B", ...}
  → process_trading_data(data)            [trading.py]
  → get_front_month_conid(symbol)              [contracts.py]
  → place_order(conid, side)             [orders.py]

Flow 2 — Contract Rollover
──────────────────────────
contract_switch_warning.pine  (fires once per day during warning window)
  → POST /rollover  {"symbol": "ZC1!"}
  → process_rollover_data(data)           [rollover.py]
  → check_and_rollover_position(symbol)  [rollover.py]
  → get_contracts_for_rollover(symbol)       [contracts.py — returns (current, next) in one call]
  → _get_contract_position(conid)        [orders.py ← shared]
  → place_order(conid, side)             [orders.py ← shared]
```

---

## Part 2 — Flow 1: Trading Signal

`process_trading_data` in `app/ibkr/trading.py` handles all trading signals.
No rollover logic lives here.

Part 1 (contracts.py rewrite) is the only change to this flow — `get_front_month_conid` now
selects the correct front-month contract using switch dates instead of the 60-day filter.

---

## Part 3 — Flow 2: Contract Rollover

### 3.1 Pine Script — `strategies/indicators/contract_switch_warning.pine`

One universal script covering all traded symbols. Added to each symbol's chart in TradingView
independently of any strategy indicator.

```pine
//@version=6
indicator("Contract Switch Warning", overlay=true)

warningDays = input.int(5, title="Warning Days Before Switch")

// Switch dates per symbol — keep in sync with contract_switch_dates.yaml
// syminfo.root strips the contract suffix: "ZC1!" → "ZC", "CL1!" → "CL"
dates = switch syminfo.root
    "ZC" => array.from(
        timestamp("2026-02-28 01:00:00"),
        timestamp("2026-04-30 01:00:00"),
        timestamp("2026-06-29 01:00:00")
        // ...
    )
    "CL" => array.from(
        timestamp("2026-01-15 00:00:00"),
        timestamp("2026-02-13 00:00:00")
        // ...
    )
    // ... all symbols
    => array.new_int(0)  // unknown symbol — warning never fires

// Check if any switch date falls within the warning window
approachingSwitch = false
for i = 0 to array.size(dates) - 1
    daysUntil = (array.get(dates, i) - time) / 86400000
    if daysUntil >= 0 and daysUntil <= warningDays
        approachingSwitch := true

// Visual: orange background tint during warning window
bgcolor(approachingSwitch ? color.new(color.orange, 80) : na,
    title="Switch Warning Background")

// Alert — fires once per bar close during warning window
alertcondition(approachingSwitch,
    title="Contract Switch Warning",
    message='{"symbol": "{{ticker}}"}')
```

**TradingView alert setup (per chart):**

1. Add the indicator to the chart for each active symbol (`ZC1!`, `CL1!`, etc.).
2. Create an alert on the condition **"Contract Switch Warning"**.
3. Set frequency to **"Once Per Bar Close"** on the **1D** chart — fires once daily.
4. Set the webhook URL to `/rollover` (separate from the `/trading` route).
5. Leave the message field as-is (the Pine script provides the JSON payload).

**Maintenance:** When `contract_switch_dates.yaml` is extended with new dates, the Pine
script must be updated in parallel. These two files are the only places switch dates live.

### 3.2 Routes — `app/routes/webhook.py`

Two sibling routes, no branching logic. The existing `webhook_blueprint` gains a second route:

```python
from app.ibkr.trading import process_trading_data
from app.ibkr.rollover import process_rollover_data


@webhook_blueprint.route('/trading', methods=['POST'])
def trading_route():
    ...
    process_trading_data(data)
    return '', 200


@webhook_blueprint.route('/rollover', methods=['POST'])
def rollover_route():
    ...
    process_rollover_data(data)
    return '', 200
```

Both routes call `save_alert_data_to_file` — rollover alerts are saved to the daily file
alongside trading signal alerts.

### 3.3 Rollover logic and handler — `app/ibkr/rollover.py`

All rollover logic lives here. `trading.py` is untouched by Flow 2.

**Constants:**

```python
# Keep CLOSE_OUT_WARNING_DAYS in sync with warningDays in contract_switch_warning.pine
CLOSE_OUT_WARNING_DAYS = 5

REOPEN_ON_ROLLOVER = True  # True  → close old position and reopen on new contract
# False → close old position only, do not reopen
```

**Imports from `contracts.py`:** only `get_next_switch_date` and `get_contracts_for_rollover`
— both public, both accept TV symbols. No private symbols, no parsing utils, no file paths
imported into `rollover.py`.

**`process_rollover_data`** — entry point called by `rollover_route` in `webhook.py`:

```python
def process_rollover_data(data):
    """
    Handle a rollover alert from TradingView.

    Validates the payload and delegates to check_and_rollover_position.

    Args:
        data: Dict from the /rollover webhook payload. Expected keys:
            - 'symbol': TradingView symbol string (e.g. 'ZC1!')

    Returns:
        Dict with a 'status' key — always a dict, never None:
            {'status': 'no_rollover_needed'}
            {'status': 'warning', ...}   — near switch, no open position
            {'status': 'rolled', ...}    — position rolled to new contract
            {'status': 'closed', ...}    — position closed, not reopened
            {'status': 'error', ...}     — rollover attempt failed

    Raises:
        ValueError: If 'symbol' is missing from data, or if the symbol has no
            upcoming switch date in the YAML (propagated from check_and_rollover_position)
    """
```

**`check_and_rollover_position`** — core logic:

```python
def check_and_rollover_position(symbol):
    """
    Close (and optionally reopen) an open position before the contract switch date.

    Always returns a dict — never None.
    """
```

**Internal logic:**

```
1. next_switch_date = get_next_switch_date(symbol)
   (raises ValueError if symbol not in YAML or all dates exhausted)
2. days_until_switch = (next_switch_date - today).days
3. If days_until_switch > CLOSE_OUT_WARNING_DAYS:
   → return {'status': 'no_rollover_needed'}

# Within the danger window
4. old_contract, new_contract = get_contracts_for_rollover(symbol)
   (file-first lookup; sentinel logic encapsulated in contracts.py)
5. current_position = _get_contract_position(old_contract['conid'])

6. If current_position == 0:
   → Log warning, return {'status': 'warning', ...}

7. If current_position != 0:
   → close_side = 'S' if current_position > 0 else 'B'
   → close_result = place_order(old_contract['conid'], close_side)
   → If close failed: log error, return {'status': 'error', ...}

   → If REOPEN_ON_ROLLOVER is False:
      → return {'status': 'closed', 'old_conid': ..., 'message': '...'}

   → reopen_side = 'B' if current_position > 0 else 'S'
   → place_order(new_contract['conid'], reopen_side)
   → return {'status': 'rolled', 'old_conid': ..., 'new_conid': ..., 'side': reopen_side}
```

---

## Part 4 — Updated Folder Structure

```
app/ibkr/
├── connection.py       (unchanged)
├── contracts.py        (rewritten — front-month selection via switch dates)
├── trading.py          (renamed from ibkr_service.py — process_trading_data)
├── orders.py           (unchanged)
└── rollover.py         (new — process_rollover_data, check_and_rollover_position)

app/routes/
└── webhook.py          (route renamed /webhook → /trading, new /rollover route)

strategies/indicators/
└── contract_switch_warning.pine   (new)

data/historical_data/
└── contract_switch_dates.yaml     (extended with 2026+ dates)
```

---

## Part 5 — File Changes Summary

**Flow 1 — Trading Signal**

| File                                              | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|---------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `app/ibkr/contracts.py`                           | Remove `MIN_DAYS_UNTIL_EXPIRY`; add `MIN_BUFFER_DAYS`, `SWITCH_DATES_FILE_PATH`; add private helpers `_load_switch_dates`, `_get_next_switch_date`, `_resolve_next_switch_date`, `_get_contracts`, `_refresh_contracts`; rename `_get_closest_contract` → `_get_front_month_contract`; add public `get_next_switch_date`, `get_contracts_for_rollover`; update `get_front_month_conid`                                                                                                                                                                             |
| `data/historical_data/contract_switch_dates.yaml` | Extend with 2026+ dates (Part 0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `tests/ibkr/test_ibkr_service.py`                 | Rename → `test_trading.py`; update import path; rename `mock_get_contract_id` → `mock_get_front_month_conid` throughout                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `tests/ibkr/test_contracts.py`                    | Full rewrite — see Part 6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `tests/ibkr/conftest.py`                          | **Trading section:** rename `mock_logger_ibkr_service` → `mock_logger_trading` (path `app.ibkr.trading.logger`); rename `mock_get_contract_id` → `mock_get_front_month_conid` (path `app.ibkr.trading.get_front_month_conid`); update `mock_place_order` path to `app.ibkr.trading.place_order`. **Contracts section:** rename `mock_get_closest_contract` → `mock_get_front_month_contract` (path `app.ibkr.contracts._get_front_month_contract`); add `mock_load_switch_dates`, `mock_resolve_next_switch_date`, `mock_get_contracts`, `mock_refresh_contracts`. |

**Flow 2 — Contract Rollover**

| File                                                 | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `strategies/indicators/contract_switch_warning.pine` | New file — daily switch warning indicator                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `app/routes/webhook.py`                              | Rename existing `/webhook` route to `/trading`; add new `/rollover` route                                                                                                                                                                                                                                                                                                                                                                                  |
| `app/ibkr/trading.py`                                | No changes — trading signal flow only                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `app/ibkr/rollover.py`                               | New file — `process_rollover_data`, `check_and_rollover_position`, constants                                                                                                                                                                                                                                                                                                                                                                               |
| `tests/ibkr/test_rollover.py`                        | New file — see Part 6                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `tests/routes/test_webhook.py`                       | Rename `TestWebhookRoute` → `TestTradingRoute`; update all route paths `/webhook` → `/trading`; add `TestRolloverRoute`                                                                                                                                                                                                                                                                                                                                    |
| `tests/ibkr/conftest.py`                             | **New Rollover section:** add `mock_get_next_switch_date` (path `app.ibkr.rollover.get_next_switch_date`), `mock_get_contracts_for_rollover` (path `app.ibkr.rollover.get_contracts_for_rollover`), `mock_place_order_rollover` (path `app.ibkr.rollover.place_order`), `mock_get_contract_position_rollover` (path `app.ibkr.rollover._get_contract_position`), `mock_check_and_rollover_position` (path `app.ibkr.rollover.check_and_rollover_position`) |
| `tests/routes/conftest.py`                           | Add `mock_process_rollover_data` (path `app.routes.webhook.process_rollover_data`)                                                                                                                                                                                                                                                                                                                                                                         |

---

## Part 6 — Test Cases Required

### `test_contracts.py` (full rewrite)

#### `TestFetchContract` (unchanged behaviour — keep existing tests)

- Success returns contract list for symbol
- API error is caught, empty list returned
- API error is logged
- Empty API response returns empty list

#### `TestGetNextSwitchDate` (new — tests `_get_next_switch_date` directly with a dict)

- Symbol present, future date → returns correct date
- Mini/micro symbol resolved via `_symbol_mappings` → returns parent symbol's date
- Symbol not in YAML → raises `ValueError`
- All dates in the past → raises `ValueError`

#### `TestGetContracts` (new)

- Symbol present in file → returns stored list without calling API
- Symbol not in file → fetches from API, saves to file, returns result
- Symbol not in file, API returns empty → raises `ValueError`, logs error

#### `TestRefreshContracts` (new)

- API returns contracts → saves to file, returns list
- API returns empty → raises `ValueError`, logs error

#### `TestGetFrontMonthContract` (rename from `TestGetClosestContract`)

- Before switch date → returns earliest contract after `MIN_BUFFER_DAYS` cutoff
- After switch date → returns first contract expiring after switch date
- Exactly on switch date → follows the "after" branch
- All contracts within buffer → raises `ValueError`
- Contracts provided in reverse order → still returns correct front month

#### `TestGetFrontMonthConid` (rename + update from `TestGetContractId`)

- Returns conid from stored contracts via `_get_contracts` + `_get_front_month_contract`
- Stored contracts expired (ValueError) → calls `_refresh_contracts`, retries, returns conid
- `_refresh_contracts` raises → propagates
- File lookup uses IBKR symbol (not TV symbol)
- Fetch uses IBKR symbol

#### `TestGetContractsForRollover` (new)

- Returns `(current, next)` tuple; next contract has later expiry than current
- Stored contracts expired → calls `_refresh_contracts`, retries, returns both contracts
- `_resolve_next_switch_date` raises → propagates

#### `TestGetNextSwitchDatePublic` (new — tests public `get_next_switch_date`)

- Parses TV symbol, maps to IBKR, delegates to `_resolve_next_switch_date`
- `ValueError` from `_resolve_next_switch_date` propagates

---

### `test_trading.py` (rename from `test_ibkr_service.py` — update fixtures only)

- All existing test cases kept; only fixture name changes:
  `mock_get_contract_id` → `mock_get_front_month_conid` throughout
- `mock_logger_ibkr_service` → `mock_logger_trading`

---

### `test_rollover.py` (new)

#### `TestProcessRolloverData`

- Missing `symbol` key → raises `ValueError`
- Delegates to `check_and_rollover_position` with the symbol from payload
- Returns result from `check_and_rollover_position` unchanged
- `ValueError` from `check_and_rollover_position` propagates

#### `TestCheckAndRolloverPosition`

- Days until switch > `CLOSE_OUT_WARNING_DAYS` → returns `{'status': 'no_rollover_needed'}`
- Within warning window, position is 0 → logs warning, returns `{'status': 'warning'}`
- Long position, `REOPEN_ON_ROLLOVER=True` → closes with SELL, reopens with BUY, returns `{'status': 'rolled'}`
- Short position, `REOPEN_ON_ROLLOVER=True` → closes with BUY, reopens with SELL, returns `{'status': 'rolled'}`
- Long position, `REOPEN_ON_ROLLOVER=False` → closes only, returns `{'status': 'closed'}`
- Short position, `REOPEN_ON_ROLLOVER=False` → closes only, returns `{'status': 'closed'}`
- Close order returns `success: False` → logs error, returns `{'status': 'error'}`
- `get_next_switch_date` raises `ValueError` (symbol not in YAML) → propagates

---

### `test_webhook.py` (updates)

#### `TestTradingRoute` (rename from `TestWebhookRoute` — update route path only)

- All existing tests kept; route path updated `/webhook` → `/trading`

#### `TestRolloverRoute` (new)

- Valid POST → calls `process_rollover_data`, not `process_trading_data`
- Alert saved alongside trading alerts (calls `save_alert_data_to_file`)
- Unallowed IP → 403
- Non-JSON content type → 400
- Processing error still returns 200

---

## Open Questions

1. **Should rollover be opt-in per symbol?**
   Not initially — apply to all symbols. Can be restricted later if some symbols should never
   auto-rollover (e.g., cash-settled indices where expiry risk is different).

## Maintenance Notes

- **`CLOSE_OUT_WARNING_DAYS` and Pine's `warningDays` must stay in sync.** Both are set to 5.
  When changing one, change the other. There is no auto-sync between the Python constant and
  the Pine script input.
- **`contract_switch_dates.yaml` and `contract_switch_warning.pine` must stay in sync.**
  When adding new dates to the YAML (Part 0), add the same dates to the Pine script's
  `switch` statement. These are the only two places switch dates are stored.

---

## Implementation Order

**Flow 1 — Trading Signal**

1. [ ] **Part 0** — Extend YAML with 2026 dates (prerequisite for everything)
2. [ ] **Part 1** — Rewrite `contracts.py` + `test_contracts.py`; rename `test_ibkr_service.py` → `test_trading.py` +
   update fixtures; update `conftest.py` trading + contracts sections

**Flow 2 — Contract Rollover** (independent, can be done in parallel with Flow 1)

3. [ ] **Part 3.3** — Create `rollover.py`: implement `process_rollover_data` + `check_and_rollover_position` +
   constants
4. [ ] **Part 3.3** — Create `test_rollover.py`
5. [ ] **Part 3.2** — Update `webhook.py` routing + `test_webhook.py`
6. [ ] **Part 3.1** — Create `contract_switch_warning.pine`; add to each symbol chart in TradingView; configure daily 1D
   alert with webhook URL

**Final**

7. [ ] Run full test suite: `python -m pytest tests/ibkr/ tests/routes/`