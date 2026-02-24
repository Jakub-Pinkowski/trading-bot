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

### 1.2 Add switch-date loading

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
```

### 1.3 Rewrite `_get_closest_contract` → `_get_front_month_contract`

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
`_get_next_switch_date` always provides a date (raises if not found), so there is no
None branch.

### 1.4 Update `get_contract_id`

```python
def get_contract_id(symbol):
    parsed_symbol = parse_symbol(symbol)
    ibkr_symbol = map_tv_to_ibkr(parsed_symbol)
    switch_dates = _load_switch_dates()
    next_switch_date = _get_next_switch_date(ibkr_symbol, switch_dates)

    contracts_cache = load_file(CONTRACTS_FILE_PATH)
    contract_list = contracts_cache.get(ibkr_symbol)

    if isinstance(contract_list, list):
        try:
            contract = _get_front_month_contract(contract_list, next_switch_date)
            return contract['conid']
        except ValueError as err:
            logger.warning(f"Cache invalid for '{ibkr_symbol}': {err}")

    fresh_contracts = _fetch_contract(ibkr_symbol)
    if not fresh_contracts:
        logger.error(f'No contracts found for symbol: {ibkr_symbol}')
        raise ValueError(f'No contracts found for symbol: {ibkr_symbol}')

    contracts_cache[ibkr_symbol] = fresh_contracts
    save_file(contracts_cache, CONTRACTS_FILE_PATH)

    contract = _get_front_month_contract(fresh_contracts, next_switch_date)
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
  → get_contract_id(symbol)              [contracts.py]
  → place_order(conid, side)             [orders.py]

Flow 2 — Contract Rollover
──────────────────────────
contract_switch_warning.pine  (fires once per day during warning window)
  → POST /rollover  {"symbol": "ZC1!"}
  → process_rollover_data(data)           [rollover.py]
  → check_and_rollover_position(symbol)  [rollover.py]
  → _fetch_contract(symbol)              [contracts.py — get full list for old+new]
  → _get_front_month_contract(...)       [contracts.py — called twice: old, then new]
  → _get_contract_position(conid)        [orders.py ← shared]
  → place_order(conid, side)             [orders.py ← shared]
```

---

## Part 2 — Flow 1: Trading Signal

`process_trading_data` in `app/ibkr/trading.py` handles all trading signals.
No rollover logic lives here.

Part 1 (contracts.py rewrite) is the only change to this flow — `get_contract_id` now
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

**`SWITCH_DATES_FILE_PATH`** is imported from `contracts.py` — defined once, shared.

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
        ValueError: If 'symbol' is missing from data
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
1. Parse symbol → ibkr_symbol
2. Load switch dates, find next_switch_date
   (raises ValueError if symbol not in YAML or all dates exhausted)
3. days_until_switch = (next_switch_date - today).days
4. If days_until_switch > CLOSE_OUT_WARNING_DAYS:
   → return {'status': 'no_rollover_needed'}

# Within the danger window
5. Get fresh contracts via _fetch_contract(ibkr_symbol)
   (fetch directly — need full list to identify both old and new contract)
6. old_contract = _get_front_month_contract(contracts, next_switch_date)
   today < next_switch_date is guaranteed here (we are in the warning window,
   not yet past the switch) → this naturally returns the current front month
7. current_position = _get_contract_position(old_contract['conid'])

8. If current_position == 0:
   → Log warning, return {'status': 'warning', ...}

9. If current_position != 0:
   → close_side = 'S' if current_position > 0 else 'B'
   → close_result = place_order(old_contract['conid'], close_side)
   → If close failed: log error, return {'status': 'error', ...}

   → If REOPEN_ON_ROLLOVER is False:
      → return {'status': 'closed', 'old_conid': ..., 'message': '...'}

   → Advance next_switch_date to the one after, so today >= switch makes
     _get_front_month_contract return the NEW front month
   → new_contract = _get_front_month_contract(contracts, next_switch_date)
   → reopen_side = 'B' if current_position > 0 else 'S'
   → place_order(new_contract['conid'], reopen_side)
   → return {'status': 'rolled', 'old_conid': ..., 'new_conid': ..., 'side': reopen_side}
```

**Note on step 9 new contract selection:** passing `next_switch_date` with today still <
switch_date gives us the old front month again. To get the new one we need to simulate
being past the switch — the cleanest way is to pass `next_switch_date - 1 day` as a
sentinel cutoff. This is an implementation detail to resolve during coding; the plan marks
it as a known decision point.

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

| File                                              | Change                                                                                                                                                                                               |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `app/ibkr/contracts.py`                           | Remove `MIN_DAYS_UNTIL_EXPIRY`, add `MIN_BUFFER_DAYS`, add `_load_switch_dates`, add `_get_next_switch_date`, rename `_get_closest_contract` → `_get_front_month_contract`, update `get_contract_id` |
| `data/historical_data/contract_switch_dates.yaml` | Extend with 2026+ dates (Part 0)                                                                                                                                                                     |
| `tests/ibkr/test_contracts.py`                    | Rewrite tests for new function signatures; add switch-date fixture                                                                                                                                   |
| `tests/ibkr/conftest.py`                          | Add `mock_load_switch_dates`, `mock_get_next_switch_date` fixtures                                                                                                                                   |

**Flow 2 — Contract Rollover**

| File                                                 | Change                                                                              |
|------------------------------------------------------|-------------------------------------------------------------------------------------|
| `strategies/indicators/contract_switch_warning.pine` | New file — daily switch warning indicator                                           |
| `app/routes/webhook.py`                              | Rename existing `/webhook` route to `/trading`; add new `/rollover` route           |
| `app/ibkr/trading.py`                                | No changes — trading signal flow only                                               |
| `app/ibkr/rollover.py`                               | New file — `process_rollover_data`, `check_and_rollover_position`, constants        |
| `tests/ibkr/test_rollover.py`                        | New file — tests for both `process_rollover_data` and `check_and_rollover_position` |
| `tests/routes/test_webhook.py`                       | Update route path to `/trading`; add tests for `/rollover` route                    |
| `tests/ibkr/conftest.py`                             | Add `mock_check_and_rollover_position`, `mock_process_rollover_data` fixtures       |
| `tests/routes/conftest.py`                           | Add `mock_process_rollover_data` fixture                                            |

---

## Part 6 — Test Cases Required

### Flow 1 — Trading Signal

#### `test_contracts.py` updates

- `TestGetFrontMonthContract`:
    - Before switch date → returns earliest contract
    - After switch date → returns contract expiring after switch date
    - All contracts too close to expiry → raises `ValueError`
    - Contracts provided in reverse order → still returns correct front month

- `TestGetNextSwitchDate`:
    - Symbol present in YAML with a future date → returns correct date
    - Mini/micro symbol resolved via `_symbol_mappings` → returns parent symbol's date
    - Symbol not in YAML → raises `ValueError`
    - Symbol in YAML but all dates exhausted (past) → raises `ValueError`

- `TestGetContractId`:
    - Update all existing tests to remove `min_days_until_expiry` parameter
    - New: switch date is loaded and passed through to `_get_front_month_contract`
    - New: cache hit uses switch date for validation

### Flow 2 — Contract Rollover

#### `test_rollover.py` (new)

- `TestProcessRolloverData`:
    - Missing symbol → raises `ValueError`
    - Far from switch → returns `{'status': 'no_rollover_needed'}`
    - Delegates to `check_and_rollover_position` and passes result through
    - Exception from `check_and_rollover_position` propagates

- `TestCheckAndRolloverPosition`:
    - Far from switch date → returns `{'status': 'no_rollover_needed'}`
    - Near switch date, no position → returns `{'status': 'warning'}`
    - Near switch date, long position, `REOPEN_ON_ROLLOVER=True` → closes and reopens on new contract
    - Near switch date, short position, `REOPEN_ON_ROLLOVER=True` → closes and reopens on new contract
    - Near switch date, long position, `REOPEN_ON_ROLLOVER=False` → closes only, returns `{'status': 'closed'}`
    - Near switch date, short position, `REOPEN_ON_ROLLOVER=False` → closes only, returns `{'status': 'closed'}`
    - Rollover close fails → returns `{'status': 'error'}`
    - Symbol not in YAML → raises `ValueError` (propagated from `_get_next_switch_date`)

#### `test_webhook.py` additions

- POST to `/trading` → calls `process_trading_data`, not `process_rollover_data`
- POST to `/rollover` → calls `process_rollover_data`, not `process_trading_data`
- Both routes save alert to file
- Both routes return 200 even when the handler raises an exception

---

## Open Questions

1. **How to select the new contract in `check_and_rollover_position` step 9?**
   Calling `_get_front_month_contract(contracts, next_switch_date)` when today < switch_date
   returns the old front month again. The cleanest resolution identified so far: pass
   `next_switch_date - 1 day` as a cutoff sentinel so the function sees today >= switch_date
   and returns the next contract. Confirm this during implementation.

2. **Should rollover be opt-in per symbol?**
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
2. [ ] **Part 1** — Rewrite `contracts.py` + `test_contracts.py`

**Flow 2 — Contract Rollover** (independent, can be done in parallel with Flow 1)

3. [ ] **Part 3.4** — Create `rollover.py` + `test_rollover.py`
4. [ ] **Part 3.3** — Implement `process_rollover_data` + `check_and_rollover_position` in `rollover.py` +
   `test_rollover.py`
5. [ ] **Part 3.2** — Update `webhook.py` routing + `test_webhook.py`
6. [ ] **Part 3.1** — Create `contract_switch_warning.pine`; add to each symbol chart in TradingView; configure daily 1D
   alert with webhook URL

**Final**

7. [ ] Run full test suite: `python -m pytest tests/ibkr/ tests/routes/`