# Plan: Contracts Front-Month Rewrite + Near-Delivery Rollover

**Status:** In Progress
**Priority:** HIGH
**Date:** 2026-02-24 (updated 2026-02-26)

---

## Problem

`MIN_DAYS_UNTIL_EXPIRY = 60` in `_get_closest_contract` filtered out the front-month contract,
causing live trades to land on the second (or third) month instead. Additionally, there was no
handling for open positions approaching the IBKR close-out date.

---

## Two Independent Flows

```
Flow 1 — Trading Signal
───────────────────────
TradingView strategy alert
  → POST /trading  {"symbol": "ZC1!", "side": "B", ...}
  → process_trading_data(data)                         [trading.py]
  → ContractResolver(symbol).get_front_month_conid()   [contracts.py]
  → place_order(conid, side)                           [orders.py]

Flow 2 — Contract Rollover
──────────────────────────
contract_switch_warning.pine  (fires once per day during warning window)
  → POST /rollover  {"symbol": "ZC1!"}
  → process_rollover_data(data)                        [rollover.py]
  → check_and_rollover_position(symbol)                [rollover.py]
  → ContractResolver(symbol).next_switch_date          [contracts.py]
  → ContractResolver(symbol).get_rollover_pair()       [contracts.py]
  → _get_contract_position(conid)                      [orders.py]
  → place_order(conid, side)                           [orders.py]
```

---

## Part 0 — Extend `contract_switch_dates.yaml` [PARTIAL]

- [x] Grains: ZC, ZW, ZS, ZL — extended with 2026 dates
- [ ] Energy: CL, NG — all switch dates in the past, YAML needs updating
- [ ] Metals: GC, SI, HG, PL — all switch dates in the past, YAML needs updating

**Note:** `contract_switch_warning.pine` (Part 3.1) must be kept in sync with this YAML
whenever new dates are added.

---

## Part 1 — `app/ibkr/contracts.py` [DONE]

Rewritten as a `ContractResolver` class with lazy-loading cached properties.

**Module-level helpers:**

- `_parse_expiry(contract)` — parses `expirationDate` as string or integer into `datetime`
- `_select_front_month(contracts, last_switch_date, next_switch_date)` — returns earliest valid
  contract if no switch has occurred (`last_switch_date is None`); skips contracts expiring
  ≤ `next_switch_date` if a switch has occurred

**`ContractResolver` class:**

- `__init__(symbol)` — accepts TV symbol (e.g. `ZC1!`), resolves to `ibkr_symbol`
- `last_switch_date` property — most recently passed TV switch date, or None
- `next_switch_date` property — next upcoming TV switch date
- `contracts` property — full contract list for the symbol
- `front_month` property — current front-month contract dict
- `get_front_month_conid()` — returns `front_month['conid']`
- `get_rollover_pair()` — returns `(current_contract, new_contract)` tuple
- `_load_switch_context()` — loads both dates from YAML; handles quoted ISO strings and None entries
- `_load_contracts()` / `_fetch_contracts()` / `_store_contracts()` — file-first contract loading

**Key design decisions:**

- `_next_switch_date is None` is the "not yet loaded" sentinel
- Both `last_switch_date` and `next_switch_date` needed: `last` determines whether a TV roll has
  occurred; `next` is used to skip the old contract post-roll
- IBKR API returns `expirationDate` as integer — `_parse_expiry` always calls `str()` first

---

## Part 2 — Flow 1: Trading Signal [DONE]

`trading.py` updated to use `ContractResolver(symbol).get_front_month_conid()`.
No other changes to this flow.

---

## Part 3 — Flow 2: Contract Rollover [PENDING]

### 3.1 Pine Script — `strategies/indicators/contract_switch_warning.pine` [ ]

One universal script covering all traded symbols. Fires a daily alert (1D bar close) during
the warning window before each switch date.

```pine
//@version=6
indicator("Contract Switch Warning", overlay=true)

warningDays = input.int(1, title="Warning Days Before Switch")

// Switch dates per symbol — keep in sync with contract_switch_dates.yaml
// syminfo.root strips the contract suffix: "ZC1!" → "ZC"
dates = switch syminfo.root
    "ZC" => array.from(
        timestamp("2026-02-27 01:00:00"),
        timestamp("2026-04-30 01:00:00"),
        // ...
    )
    "ZW" => array.from(
        timestamp("2026-04-17 01:00:00"),
        // ...
    )
    // ... all symbols
    => array.new_int(0)  // unknown symbol — warning never fires

approachingSwitch = false
for i = 0 to array.size(dates) - 1
    daysUntil = (array.get(dates, i) - time) / 86400000
    if daysUntil >= 0 and daysUntil <= warningDays
        approachingSwitch := true

bgcolor(approachingSwitch ? color.new(color.orange, 80) : na,
    title="Switch Warning Background")

alertcondition(approachingSwitch,
    title="Contract Switch Warning",
    message='{"symbol": "{{ticker}}"}')
```

**TradingView alert setup (per chart):**

1. Add indicator to the chart for each active symbol (`ZC1!`, `ZW1!`, etc.)
2. Create alert on **"Contract Switch Warning"** condition
3. Frequency: **Once Per Bar Close** on the **1D** chart
4. Webhook URL: `/rollover`

**Maintenance:** When `contract_switch_dates.yaml` is extended, update this script in parallel.

### 3.2 Routes — `app/routes/webhook.py` [ ]

Rename existing `/webhook` route to `/trading`. Add `/rollover` route:

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

Both routes call `save_alert_data_to_file` — rollover alerts saved alongside trading alerts.

### 3.3 Rollover logic — `app/ibkr/rollover.py` [ ]

```python
from app.ibkr.contracts import ContractResolver
from app.ibkr.orders import place_order, _get_contract_position

# Keep CLOSE_OUT_WARNING_DAYS in sync with warningDays in contract_switch_warning.pine
CLOSE_OUT_WARNING_DAYS = 1

REOPEN_ON_ROLLOVER = True  # True  → close old position and reopen on new contract
# False → close old position only, do not reopen
```

**`process_rollover_data(data)`** — entry point called by `rollover_route`:

```
Validates 'symbol' key is present (raises ValueError if missing).
Delegates to check_and_rollover_position(symbol).
Returns result dict unchanged.
```

**`check_and_rollover_position(symbol)`** — core logic:

```
resolver = ContractResolver(symbol)
days_until_switch = (resolver.next_switch_date - today).days

If days_until_switch > CLOSE_OUT_WARNING_DAYS:
    → return {'status': 'no_rollover_needed'}

# Within the warning window
current_contract, new_contract = resolver.get_rollover_pair()
current_position = _get_contract_position(current_contract['conid'])

If current_position == 0:
    → log warning
    → return {'status': 'warning', 'days_until_switch': days_until_switch}

close_side = 'S' if current_position > 0 else 'B'
close_result = place_order(current_contract['conid'], close_side)

If close failed (success: False):
    → log error
    → return {'status': 'error', 'order': close_result}

If REOPEN_ON_ROLLOVER is False:
    → return {'status': 'closed', 'old_conid': ..., 'new_conid': ...}

reopen_side = 'B' if current_position > 0 else 'S'
place_order(new_contract['conid'], reopen_side)
return {'status': 'rolled', 'old_conid': ..., 'new_conid': ..., 'side': reopen_side}
```

---

## Part 4 — Folder Structure

```
app/ibkr/
├── connection.py       (unchanged)
├── contracts.py        ✅ rewritten — ContractResolver class
├── trading.py          ✅ renamed from ibkr_service.py
├── orders.py           (unchanged)
└── rollover.py         [ ] new

app/routes/
└── webhook.py          [ ] /webhook → /trading, add /rollover

strategies/indicators/
└── contract_switch_warning.pine   [ ] new

data/historical_data/
└── contract_switch_dates.yaml     ✅ grains done; energy/metals pending
```

---

## Part 5 — Tests

### `test_contracts.py` [DONE]

Full rewrite complete. 100% coverage. All 80 IBKR tests passing.

### `test_trading.py` [DONE]

Updated to use `mock_contract_resolver` fixture.

### `test_rollover.py` [ ] (new)

#### `TestProcessRolloverData`

- Missing `symbol` raises `ValueError`
- Delegates to `check_and_rollover_position` with symbol from payload
- Returns result unchanged
- `ValueError` from `check_and_rollover_position` propagates

#### `TestCheckAndRolloverPosition`

- `days_until_switch > CLOSE_OUT_WARNING_DAYS` → `{'status': 'no_rollover_needed'}`
- Within window, position is 0 → `{'status': 'warning', 'days_until_switch': N}`
- Long position, `REOPEN_ON_ROLLOVER=True` → closes SELL, reopens BUY, `{'status': 'rolled'}`
- Short position, `REOPEN_ON_ROLLOVER=True` → closes BUY, reopens SELL, `{'status': 'rolled'}`
- Long position, `REOPEN_ON_ROLLOVER=False` → closes only, `{'status': 'closed'}`
- Short position, `REOPEN_ON_ROLLOVER=False` → closes only, `{'status': 'closed'}`
- Close order returns `success: False` → `{'status': 'error'}`
- `ContractResolver.next_switch_date` raises `ValueError` → propagates

### `test_webhook.py` [ ] (updates)

#### `TestTradingRoute` (rename from `TestWebhookRoute`)
- All existing tests kept; route path updated `/webhook` → `/trading`

#### `TestRolloverRoute` (new)
- Valid POST → calls `process_rollover_data`, not `process_trading_data`
- Alert saved to daily file
- Unallowed IP → 403
- Non-JSON → 400
- Processing error still returns 200

### `conftest.py` [ ] (updates)

**New Rollover section in `tests/ibkr/conftest.py`:**

- `mock_logger_rollover` — `app.ibkr.rollover.logger`
- `mock_contract_resolver_rollover` — patches `ContractResolver` in rollover module
- `mock_place_order_rollover` — `app.ibkr.rollover.place_order`
- `mock_get_contract_position_rollover` — `app.ibkr.rollover._get_contract_position`
- `mock_check_and_rollover_position` — `app.ibkr.rollover.check_and_rollover_position`

**`tests/routes/conftest.py`:**

- Add `mock_process_rollover_data` — `app.routes.webhook.process_rollover_data`

---

## Maintenance Notes

- `CLOSE_OUT_WARNING_DAYS` (rollover.py) and `warningDays` (Pine script) **must stay in sync** — both are 1.
- `contract_switch_dates.yaml` and `contract_switch_warning.pine` **must stay in sync** — when adding new dates to the
  YAML, add the same dates to the Pine script.

---

## Implementation Order

**Flow 1 — Trading Signal**

- [x] **Part 0 (grains)** — Extend YAML with 2026 dates for ZC, ZW, ZS, ZL
- [x] **Pre-step** — Rename `ibkr_service.py` → `trading.py`; update routes, conftest, `__init__.py`
- [x] **Part 1** — Rewrite `contracts.py` as `ContractResolver`; full test suite, 100% coverage
- [x] **Part 2** — Update `trading.py` to use `ContractResolver(symbol).get_front_month_conid()`
- [ ] **Part 0 (energy/metals)** — Extend YAML for CL, NG, GC, SI, HG, PL

**Flow 2 — Contract Rollover**

- [ ] **Part 3.3** — Create `rollover.py` + `test_rollover.py`
- [ ] **Part 3.2** — Update `webhook.py` routing + `test_webhook.py` + `tests/routes/conftest.py`
- [ ] **Part 3.1** — Create `contract_switch_warning.pine`; add to each symbol chart; configure 1D daily alert

**Final**

- [ ] Run full test suite: `python -m pytest tests/ibkr/ tests/routes/`
