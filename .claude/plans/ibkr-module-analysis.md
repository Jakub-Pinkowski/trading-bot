# IBKR Module Analysis

> Analysis date: 2026-02-27
> Last updated: 2026-02-27 (minor fixes applied — see status column)

## Module Overview

The IBKR module integrates with the Interactive Brokers REST API for real-time futures trading. It handles:

- Webhook-triggered order execution
- Front-month contract resolution with rollover awareness
- IBKR session lifecycle (heartbeat scheduling)
- Position-aware order placement with message suppression

**Files analyzed**: `connection.py`, `contracts.py`, `orders.py`, `trading.py`, `rollover.py`, `app/utils/api_utils.py`,
`run.py`, and all associated tests.

---

## Critical Issues

### 1. IBKR Scheduler Is Commented Out (`run.py:10`)

```python
# start_ibkr_scheduler()  # <-- never called
```

The 60-second heartbeat that keeps the IBKR session alive is disabled. Without it, the IBKR session times out, making
the entire trading system non-functional in production until manually reconnected.

**Fix**: Uncomment the call or investigate why it was disabled (may be an intentional temporary measure).

---

### 2. SSL Verification Disabled (`app/utils/api_utils.py`)

```python
response = requests.get(url=url, verify=False, ...)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```

Both `api_get` and `api_post` disable SSL certificate validation and suppress the resulting warnings. This opens the
door to man-in-the-middle attacks against all IBKR API communication, including order placement.

**Fix**: Either use proper certificates (if IBKR uses a trusted CA) or pin IBKR's self-signed certificate explicitly
using `verify="/path/to/cert.pem"`.

---

## Logic Issues

### 3. ✅ `_get_contract_position()` Returns 0 on API Failure (`orders.py`)

**Fixed**: Now returns `None` on API error. `place_order()` guards against `None` and returns early with
`{'success': False, 'error': 'Position check failed: ...'}` rather than silently treating an API failure as a flat
position. `rollover.py` has the same guard. Tests updated.

---

### 4. For-Else Loop Edge Case in Suppression Retry (`orders.py:129–137`)

```python
for attempt in range(MAX_SUPPRESS_RETRIES):
    order_response = api_post(...)
    if ... 'messageIds' in order_response[0]:
        _suppress_messages(message_ids)
        continue  # retry
    break  # success path
else:
    logger.error('exceeded maximum suppression retries')
    return error
```

Python's `for...else` executes the `else` block only if the loop completes without a `break`. If all 3 attempts return
`messageIds`, it logs the error but `order_response` still holds the last retry's response (containing messageIds),
which may then be returned or parsed incorrectly. The error return value should be explicit and not rely on the state of
`order_response` after loop exhaustion.

---

### 5. ✅ Partial Rollover With No Recovery (`rollover.py`)

**Fixed**: Added `logger.critical(...)` when the reopen order fails after a successful close. The log message explicitly
names the affected contracts and states that manual intervention is required. Tests updated.

---

### 6. Silent Failure in `_tickle_ibkr_api()` (`connection.py`)

```python
except Exception:
    logger.error(...)
    # continues silently
```

The heartbeat catches all exceptions and logs them but does not raise, escalate, or trigger any recovery. If the IBKR
session is permanently lost, the scheduler will keep running and logging errors indefinitely with no alerting.

**Fix** (deferred): Track consecutive failure count; after N failures, emit a critical alert or stop the scheduler.

---

### 7. ✅ Silent Failure in `_suppress_messages()` (`orders.py`)

**Fixed**: `_suppress_messages()` now re-raises after logging. If suppression fails, the exception propagates to
`place_order()`'s outer handler, which logs it with a full traceback and returns
`{'success': False, 'error': 'An unexpected error occurred'}`. The root cause is now visible in logs rather than
silently swallowed. Tests updated.

---

## Missing Features

### 8. No Request Deduplication in `process_trading_data()` (`trading.py`)

TradingView retries webhook delivery on failure (up to 3 times). If the first delivery succeeds but TradingView doesn't
receive a timely response, it will send again. There is no request ID, timestamp, or deduplication key to detect and
skip duplicate signals.

**Suggestion**: Accept a unique signal ID from TradingView's alert body and cache processed IDs in memory (or a
short-lived file) for a rolling window (e.g., 5 minutes).

---

### 9. No Request Timeout in HTTP Calls (`api_utils.py`)

```python
response = requests.get(url=url, verify=False, headers=get_headers())
```

No `timeout` parameter. If the IBKR gateway becomes slow or unresponsive, every API call can hang indefinitely, blocking
the Flask request thread.

**Fix**: Add `timeout=(connect_timeout, read_timeout)`, e.g., `timeout=(3, 10)`.

---

### 10. No Retry Logic in HTTP Calls (`api_utils.py`)

A single transient network error causes the entire order flow to fail. There is no retry with backoff for connection
errors, 429 rate-limit responses, or 5xx server errors.

**Suggestion**: Use `urllib3.util.retry.Retry` or a simple retry decorator for idempotent requests (GET, and POST for
suppression). Order placement POST should not be retried blindly.

---

### 11. `REOPEN_ON_ROLLOVER` Is a Hardcoded Flag (`rollover.py`)

```python
REOPEN_ON_ROLLOVER = False  # must change code to enable
```

This is a behavioral flag that should be configurable without touching source code. Same applies to
`CLOSE_OUT_WARNING_DAYS`.

**Fix**: Move to environment variable or a config file (e.g., `config.py` reading from `.env`).

---

### 12. `MIN_BUFFER_DAYS` Is Hardcoded (`contracts.py`)

The 5-day buffer preventing trading of near-expiry contracts is hardcoded at the top of `contracts.py`. It should be
configurable per-symbol or globally in config.

---

### 13. No Graceful Scheduler Shutdown (`connection.py`)

There is no `atexit` handler or Flask teardown hook to cleanly shut down the APScheduler when the process exits. On
unclean shutdown, the scheduler thread may delay process exit.

**Fix**: Register `scheduler.shutdown()` via `atexit.register()` or Flask's `teardown_appcontext`.

---

### 14. No Response Validation in `api_get`/`api_post` (`api_utils.py`)

After checking HTTP status with `raise_for_status()`, the code calls `.json()` directly. If IBKR returns an HTML error
page (common with gateway restarts), this raises an unhandled `JSONDecodeError` with no useful context.

**Fix**: Wrap `.json()` call in a try-except and log the raw response body on failure.

---

## Code Quality Observations

### Positive

- **Excellent test coverage**: 97 test cases across all modules using consistent `monkeypatch` style.
- **Lazy-loading pattern in `ContractResolver`**: Clean sentinel-based caching avoids redundant I/O.
- **Clear separation of concerns**: Each file has one responsibility; logic is easy to follow.
- **Position-aware ordering**: The check-then-act pattern in `place_order()` prevents accidental double-entry.
- **Comprehensive docstrings**: All public functions document Args, Returns, and Raises.

### ✅ Minor Code Style (fixed)

- `_symbol_mappings` in `contracts.py`: Added a comment explaining that mini/micro symbols (e.g. `MES`, `MNQ`) share
  switch dates with their full-size equivalents via this mapping.
- `get_rollover_pair()`: The analysis initially flagged this as using `next()` with no default, but the code already
  provides a `None` default and raises a descriptive `ValueError`. No change needed.

---

## Summary Table

| #  | Severity     | File            | Issue                                                              | Status                       |
|----|--------------|-----------------|--------------------------------------------------------------------|------------------------------|
| 1  | **Critical** | `run.py`        | Scheduler commented out — no IBKR session heartbeat                | Open                         |
| 2  | **Critical** | `api_utils.py`  | SSL verification disabled — MITM vulnerability                     | Open                         |
| 3  | **High**     | `orders.py`     | `_get_contract_position()` returns 0 on API failure                | ✅ Fixed                      |
| 4  | **High**     | `orders.py`     | For-else loop logic in suppression retry                           | Open                         |
| 5  | **High**     | `rollover.py`   | No recovery if reopen fails after close succeeds                   | ✅ Fixed (critical log added) |
| 6  | **Medium**   | `connection.py` | Silent heartbeat failure with no recovery                          | Deferred                     |
| 7  | **Medium**   | `orders.py`     | Silent failure in `_suppress_messages()`                           | ✅ Fixed                      |
| 8  | **Medium**   | `trading.py`    | No deduplication for TradingView retries                           | Open                         |
| 9  | **Medium**   | `api_utils.py`  | No request timeout — can hang indefinitely                         | Open                         |
| 10 | **Medium**   | `api_utils.py`  | No retry logic for transient errors                                | Open                         |
| 11 | **Low**      | `rollover.py`   | `REOPEN_ON_ROLLOVER` hardcoded — not configurable                  | Open                         |
| 12 | **Low**      | `contracts.py`  | `MIN_BUFFER_DAYS` hardcoded — not configurable                     | Open                         |
| 13 | **Low**      | `connection.py` | No graceful scheduler shutdown                                     | Open                         |
| 14 | **Low**      | `api_utils.py`  | No response body validation — `JSONDecodeError` on HTML error page | Open                         |
| —  | **Low**      | `contracts.py`  | `_symbol_mappings` lacked explanatory comment                      | ✅ Fixed                      |
