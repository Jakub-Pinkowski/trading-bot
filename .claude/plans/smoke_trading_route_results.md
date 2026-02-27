# Smoke Test Results — `/trading` Route

Live server: `http://127.0.0.1:5002`
Date: 2026-02-26
IBKR connection: **not running** (simulates real deployment where IBKR gateway is down or unreachable)

> **Note (zsh):** always use single quotes around the JSON body — zsh expands `!` in double-quoted strings, which
> corrupts symbol names like `ZC1!`.

---

## Scenario 1 — Real buy signal, known symbol

```bash
curl -X POST http://127.0.0.1:5002/trading \
  -H 'Content-Type: application/json' \
  -d '{"symbol": "ZC1!", "side": "B", "dummy": "NO"}'
```

**Response**

```
HTTP 200
(empty body)
```

**What happened**

- `save_alert_data_to_file` wrote the alert to the daily JSON file in `data/alerts/ibkr_alerts/`
- `process_trading_data` was called
- `ContractResolver('ZC1!')` resolved the front-month conid from stored contracts + YAML switch dates
- `place_order` attempted to call the IBKR API — failed with a connection error (no IBKR running)
- Exception was caught by the route's `try/except`, logged, route returned 200

**Key assertion**: TradingView never receives a non-200, even when IBKR is down.

---

## Scenario 2 — Real sell signal, different symbol

```bash
curl -X POST http://127.0.0.1:5002/trading \
  -H 'Content-Type: application/json' \
  -d '{"symbol": "ZS1!", "side": "S", "dummy": "NO"}'
```

**Response**

```
HTTP 200
(empty body)
```

**What happened**

- Alert saved to file
- `ContractResolver('ZS1!')` resolved the front-month conid for Soybeans
- IBKR call failed (no connection), exception caught and logged
- Route returned 200

---

## Scenario 3 — Dummy signal (no order, no disk I/O)

```bash
curl -X POST http://127.0.0.1:5002/trading \
  -H 'Content-Type: application/json' \
  -d '{"symbol": "ZC1!", "side": "B", "dummy": "YES"}'
```

**Response**

```
HTTP 200
(empty body)
```

**What happened**

- `save_alert_data_to_file` returned early — dummy signals are not persisted
- `process_trading_data` was still called (route always dispatches)
- No IBKR API call was made (dummy mode is handled inside the trading logic)
- No file written

**Key assertion**: Dummy signals produce zero side-effects — no disk I/O, no orders.

---

## Scenario 4 — Missing symbol field

```bash
curl -X POST http://127.0.0.1:5002/trading \
  -H 'Content-Type: application/json' \
  -d '{"side": "B", "dummy": "NO"}'
```

**Response**

```
HTTP 200
(empty body)
```

**What happened**

- `save_alert_data_to_file` was called and saved the partial payload to disk (save happens before processing)
- `process_trading_data` raised `ValueError: Missing required field: symbol`
- Exception was caught by the route's `try/except`, logged with the full payload
- Route returned 200

**Key assertion**: Malformed payloads do not cause TradingView retries.

---

## Scenario 5 — Unknown symbol

```bash
curl -X POST http://127.0.0.1:5002/trading \
  -H 'Content-Type: application/json' \
  -d '{"symbol": "FAKE1!", "side": "B", "dummy": "NO"}'
```

**Response**

```
HTTP 200
(empty body)
```

**What happened**

- Alert saved to file
- `ContractResolver('FAKE1!')` raised because `FAKE1!` has no matching config in `futures_config/`
- Exception was caught by the route's `try/except`, logged
- Route returned 200

---

## Validation Checks (blueprint-level, always enforced)

### Non-JSON Content-Type → 400

```bash
curl -X POST http://127.0.0.1:5002/trading \
  -H 'Content-Type: text/plain' \
  -d 'hello'
```

**Response**

```
HTTP 400
Unsupported Content-Type
```

### Non-allowlisted IP → 403

Cannot be triggered from localhost via curl. Enforced in unit tests via `environ_base={'REMOTE_ADDR': '10.10.10.10'}`.

**Response**: `HTTP 403`

---

---

# Smoke Test Results — `/rollover` Route

Same server, same date. IBKR connection still not running.

---

## Scenario 6 — Symbol within warning window

ZC1! has a switch date of 2026-02-27 — one day away, within the `CLOSE_OUT_WARNING_DAYS = 1` threshold.

```bash
curl -X POST http://127.0.0.1:5002/rollover \
  -H 'Content-Type: application/json' \
  -d '{"symbol": "ZC1!"}'
```

**Response**

```
HTTP 200
(empty body)
```

**What happened**

- `save_alert_data_to_file` saved the alert to disk
- `process_rollover_data` was called
- `ContractResolver('ZC1!')` resolved `next_switch_date` → within warning window
- `get_rollover_pair()` returned the current and next contract conids
- `_get_contract_position` attempted IBKR API call — failed (no connection)
- Exception caught and logged, route returned 200

**With IBKR running**: would read the open position, place a close order on the current contract, and reopen on the next
contract (since `REOPEN_ON_ROLLOVER = True`).

---

## Scenario 7 — Symbol outside warning window

ZW1! has a switch date of 2026-06-16 — well beyond the 1-day threshold.

```bash
curl -X POST http://127.0.0.1:5002/rollover \
  -H 'Content-Type: application/json' \
  -d '{"symbol": "ZW1!"}'
```

**Response**

```
HTTP 200
(empty body)
```

**What happened**

- `save_alert_data_to_file` saved the alert to disk
- `ContractResolver('ZW1!')` resolved `next_switch_date` → outside warning window
- `_check_and_rollover_position` returned `{'status': 'no_rollover_needed'}` immediately
- No IBKR calls made

**Key assertion**: Rollover alerts sent outside the warning window are safe no-ops — no orders placed.

---

## Scenario 8 — Missing symbol field

```bash
curl -X POST http://127.0.0.1:5002/rollover \
  -H 'Content-Type: application/json' \
  -d '{"dummy": "NO"}'
```

**Response**

```
HTTP 200
(empty body)
```

**What happened**

- Alert saved to disk
- `process_rollover_data` raised `ValueError: Missing required field: symbol`
- Exception caught and logged, route returned 200

---

## Scenario 9 — Unknown symbol

```bash
curl -X POST http://127.0.0.1:5002/rollover \
  -H 'Content-Type: application/json' \
  -d '{"symbol": "FAKE1!"}'
```

**Response**

```
HTTP 200
(empty body)
```

**What happened**

- Alert saved to disk
- `ContractResolver('FAKE1!')` raised because `FAKE1!` has no matching config in `futures_config/`
- Exception caught and logged, route returned 200

---

## Notes

- All scenarios return `HTTP 200` regardless of downstream errors — this is intentional. TradingView retries any non-200
  response, which would cause duplicate orders.
- `save_alert_data_to_file` is called **before** processing. If processing fails, the raw alert is still persisted for
  manual inspection.
- IP and Content-Type validation are enforced at the blueprint level via `@webhook_blueprint.before_request` and apply
  to both `/trading` and `/rollover`.
