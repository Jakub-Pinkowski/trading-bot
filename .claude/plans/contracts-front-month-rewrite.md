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
  → process_trading_data(data)                          [trading.py]
  → ContractResolver(symbol).get_front_month_conid()    [contracts.py]
  → place_order(conid, side)                            [orders.py]

Flow 2 — Contract Rollover
──────────────────────────
contract_switch_warning.pine  (fires once per day during warning window)
  → POST /rollover  {"symbol": "ZC1!"}
  → process_rollover_data(data)                         [rollover.py]
  → _check_and_rollover_position(symbol)                [rollover.py]
  → ContractResolver(symbol).next_switch_date           [contracts.py]
  → ContractResolver(symbol).get_rollover_pair()        [contracts.py]
  → _get_contract_position(conid)                       [orders.py]
  → place_order(conid, side)                            [orders.py]
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
Logs resolved conid at INFO level: `Resolved front-month conid for {symbol}: {conid}`.

---

## Part 3 — Flow 2: Contract Rollover

### 3.1 Pine Script — `tv_scripts/indicators/contract_switch_warning.pine` [ ]

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

### 3.2 Routes — `app/routes/webhook.py` [DONE]

- Renamed `/webhook` → `/trading`
- Added `/rollover` route
- Extracted IP + Content-Type validation into `@webhook_blueprint.before_request`
- Both routes call `save_alert_data_to_file` before dispatching

### 3.3 Rollover logic — `app/ibkr/rollover.py` [DONE]

```python
CLOSE_OUT_WARNING_DAYS = 1   # Keep in sync with warningDays in contract_switch_warning.pine
REOPEN_ON_ROLLOVER = False   # True  → close old position and reopen on new contract
                             # False → close old position only, do not reopen
```

**`process_rollover_data(data)`** — public entry point called by `rollover_route`.

**`_check_and_rollover_position(symbol)`** — private core logic:

```
resolver = ContractResolver(symbol)
days_until_switch = (resolver.next_switch_date - today).days

If days_until_switch > CLOSE_OUT_WARNING_DAYS:
    → log info "No rollover needed for {symbol}: N day(s) until switch"
    → return {'status': 'no_rollover_needed'}

# Within the warning window
current_contract, new_contract = resolver.get_rollover_pair()
→ log info "Rollover pair for {symbol}: {conid} (expiry) → {conid} (expiry)"

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
reopen_result = place_order(new_contract['conid'], reopen_side)

If reopen failed (success: False):
    → log error
    → return {'status': 'reopen_failed', 'old_conid': ..., 'new_conid': ..., 'order': reopen_result}

→ return {'status': 'rolled', 'old_conid': ..., 'new_conid': ..., 'side': reopen_side}
```

---

## Part 4 — Folder Structure

```
app/ibkr/
├── connection.py       (unchanged)
├── contracts.py        ✅ rewritten — ContractResolver class
├── trading.py          ✅ renamed from ibkr_service.py
├── orders.py           (unchanged)
└── rollover.py         ✅ new

app/routes/
└── webhook.py          ✅ /webhook → /trading, /rollover added

tv_scripts/indicators/
└── contract_switch_warning.pine   [ ] new

data/historical_data/
└── contract_switch_dates.yaml     ✅ grains done; energy/metals pending
```

---

## Part 5 — Tests [DONE]

### `test_contracts.py` ✅

Full rewrite complete. 100% coverage. All 80 IBKR tests passing.

### `test_trading.py` ✅

Updated to use `mock_contract_resolver` fixture.

### `test_rollover.py` ✅

13 tests, 100% coverage.

#### `TestProcessRolloverData`
- Missing `symbol` raises `ValueError`
- Delegates to `_check_and_rollover_position` with symbol from payload
- Returns result unchanged
- `ValueError` from `_check_and_rollover_position` propagates

#### `TestCheckAndRolloverPosition`
- `days_until_switch > CLOSE_OUT_WARNING_DAYS` → `{'status': 'no_rollover_needed'}`
- Within window, position is 0 → `{'status': 'warning', 'days_until_switch': N}`
- Long position, `REOPEN_ON_ROLLOVER=True` → closes SELL, reopens BUY, `{'status': 'rolled'}`
- Short position, `REOPEN_ON_ROLLOVER=True` → closes BUY, reopens SELL, `{'status': 'rolled'}`
- Long position, `REOPEN_ON_ROLLOVER=False` → closes only, `{'status': 'closed'}`
- Short position, `REOPEN_ON_ROLLOVER=False` → closes only, `{'status': 'closed'}`
- Close order returns `success: False` → `{'status': 'error'}`
- Close succeeds but reopen fails → `{'status': 'reopen_failed'}`
- `ContractResolver.next_switch_date` raises `ValueError` → propagates

### `test_webhook.py` ✅

- `TestRequestValidation` — IP allowlist + Content-Type (blueprint-level, tested once)
- `TestTradingRoute` — renamed from `TestWebhookRoute`; route path updated `/webhook` → `/trading`
- `TestRolloverRoute` — dispatch, separation from trading, alert saved, error → 200

### `conftest.py` ✅

`tests/ibkr/conftest.py` — Rollover section added:

- `mock_logger_rollover`, `mock_contract_resolver_rollover`, `mock_place_order_rollover`
- `mock_get_contract_position_rollover`, `mock_check_and_rollover_position`

`tests/routes/conftest.py` — added `mock_process_rollover_data`

---

## Smoke Tests [DONE]

Live server tests against `http://127.0.0.1:5002` documented in
`scripts/smoke_trading_route_results.md`. 9 scenarios covering both routes verified.
INFO logs confirmed in `logs/info.log` — conid resolution and rollover pair visible per request.

---

## Maintenance Notes

- `CLOSE_OUT_WARNING_DAYS` (rollover.py) and `warningDays` (Pine script) **must stay in sync** — both are 1.
- `contract_switch_dates.yaml` and `contract_switch_warning.pine` **must stay in sync** — when adding new dates to the
  YAML, add the same dates to the Pine script.
- INFO logs go to `logs/info.log` (not terminal). Console shows WARNING and above only.

---

## Implementation Order

**Flow 1 — Trading Signal**

- [x] **Part 0 (grains)** — Extend YAML with 2026 dates for ZC, ZW, ZS, ZL
- [x] **Pre-step** — Rename `ibkr_service.py` → `trading.py`; update routes, conftest, `__init__.py`
- [x] **Part 1** — Rewrite `contracts.py` as `ContractResolver`; full test suite, 100% coverage
- [x] **Part 2** — Update `trading.py` to use `ContractResolver(symbol).get_front_month_conid()`
- [ ] **Part 0 (energy/metals)** — Extend YAML for CL, NG, GC, SI, HG, PL

**Flow 2 — Contract Rollover**

- [x] **Part 3.3** — Create `rollover.py` + `test_rollover.py`
- [x] **Part 3.2** — Update `webhook.py` routing + `test_webhook.py` + `tests/routes/conftest.py`
- [ ] **Part 3.1** — Create `contract_switch_warning.pine`; add to each symbol chart; configure 1D daily alert

**Final**

- [x] Run full test suite: `python -m pytest tests/ibkr/ tests/routes/`
- [x] Smoke tests with live server (IBKR not running) — all 9 scenarios passed
- [ ] Smoke tests with IBKR running — verify real order placement on correct conids
