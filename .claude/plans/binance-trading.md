# Binance Trading Integration Plan

## Overview

Add Binance Futures trading support alongside the existing IBKR integration. Signals arrive via the same TradingView
webhook format but hit a dedicated `/binance` route, keeping Binance completely separate from IBKR. Both brokers coexist
independently — Binance handles crypto perpetuals, IBKR handles traditional futures.

---

## 1. Current Architecture (IBKR)

```
TradingView → POST /trading → webhook.py → process_trading_data()
                                           ├─ ContractResolver (TV symbol → IBKR conid)
                                           └─ place_order(conid, side)
```

Key patterns to carry over:

- Always return HTTP 200 from webhooks
- Save all alerts to file before processing
- Check position before placing order (skip if already in direction)
- Return `{'success': bool, ...}` dicts, never raise on known failures

---

## 2. Signal Format

No changes to TradingView signal format. Binance signals use the same schema:

```json
{
  "symbol": "BINANCE:BTCUSDT.P",
  "side": "B",
  "dummy": "NO"
}
```

TradingView alerts for Binance are configured to point to `POST /binance` instead of `POST /trading`. The route itself
determines the broker — no symbol-based routing needed anywhere.

---

## 3. Target Architecture

```
                    webhook.py (single file, single blueprint)
                    ┌───────────────────────────────────────┐
TradingView (IBKR)  │  POST /trading                        │
        ──────────► │    ├─ Save to ibkr_alerts/            │
                    │    └─ process_trading_data(data)       │
                    │                                       │
TradingView (IBKR)  │  POST /rollover                       │
        ──────────► │    └─ process_rollover_data(data)     │
                    │                                       │
TradingView (Binance│  POST /binance                        │
        ──────────► │    ├─ Save to binance_alerts/         │
                    │    └─ process_binance_trading(data)   │
                    └───────────────────────────────────────┘
                    before_request validates IP + JSON for all routes
```

`/binance` is added to the existing `webhook.py` alongside `/trading` and `/rollover` — same blueprint, same
`before_request` validation, no new files in `app/routes/`.

---

## 4. Binance Module Structure

Unlike IBKR, no contract resolution module is needed — Binance symbol mapping is trivial string manipulation (strip
`BINANCE:` prefix and `.P` suffix), so it lives inline in `trading.py`.

```
app/binance/
├── __init__.py     # Module docstring
├── api.py          # Raw HTTP helpers with HMAC-SHA256 signing
├── config.py       # Per-symbol notional USDT and leverage
├── orders.py       # Position check, quantity calc, symbol setup, order placement
└── trading.py      # Orchestrator
```

No rollover module — perpetual futures never expire.

---

## 5. File-by-File Implementation Plan

### 5.1 `app/binance/__init__.py`

```python
"""Binance Futures trading integration."""
```

---

### 5.2 `app/binance/api.py`

Raw HTTP helpers for the Binance Futures REST API, mirroring `app/utils/api_utils.py`. Handles HMAC-SHA256 signing
transparently so `orders.py` never deals with auth directly.

```python
import hashlib
import hmac
import time
import requests
from config import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET

BASE_URL = 'https://demo-fapi.binance.com' if BINANCE_TESTNET else 'https://fapi.binance.com'


def _sign(params):
    # Copy to avoid mutating the caller's dict
    params = {**params}
    params['timestamp'] = int(time.time() * 1000)
    query = '&'.join(f'{k}={v}' for k, v in params.items())
    params['signature'] = hmac.new(
        BINANCE_API_SECRET.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    return params


def binance_get(endpoint, params=None, signed=True):
    params = params or {}
    if signed:
        params = _sign(params)
    headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
    response = requests.get(BASE_URL + endpoint, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


def binance_post(endpoint, params=None):
    params = _sign(params or {})
    headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
    response = requests.post(BASE_URL + endpoint, params=params, headers=headers)
    response.raise_for_status()
    return response.json()
```

Endpoint reference:

| Function                                             | Endpoint        | Signed |
|------------------------------------------------------|-----------------|--------|
| `binance_get('/fapi/v3/positionRisk', ...)`          | Position info   | Yes    |
| `binance_post('/fapi/v1/order', ...)`                | Place order     | Yes    |
| `binance_post('/fapi/v1/leverage', ...)`             | Set leverage    | Yes    |
| `binance_post('/fapi/v1/marginType', ...)`           | Set margin type | Yes    |
| `binance_get('/fapi/v1/exchangeInfo', signed=False)` | Exchange info   | No     |
| `binance_get('/fapi/v1/premiumIndex', signed=False)` | Mark price      | No     |

---

### 5.3 `app/binance/config.py`

Per-symbol trading configuration. Kept separate from `config.py` to avoid cluttering env-var config with trading data.

```python
NOTIONAL_USDT = {
    'BTCUSDT': 100,
    'ETHUSDT': 100,
    'SOLUSDT': 100,
}
DEFAULT_NOTIONAL_USDT = 100

LEVERAGE = {
    'BTCUSDT': 10,
    'ETHUSDT': 10,
}
DEFAULT_LEVERAGE = 5
```

---

### 5.4 `app/binance/orders.py`

The core module. Uses one-way mode — one net position per symbol, positive = long, negative = short.

**Key design difference from IBKR:** `_get_position()` returns the actual float quantity (e.g. `0.003`), not a
normalised `1/-1/0`. This is necessary because the close order must use the exact position size (`abs(positionAmt)`),
while the open order uses `_get_quantity()`.

Module-level state:

```python
_configured_symbols = set()  # Symbols already set up (leverage + margin mode)
_exchange_info_cache = None  # Cached exchangeInfo — fetched once, changes rarely
```

Key functions:

```python
def _get_position(symbol):
    """
    Returns current position quantity as float.
    Positive = long, negative = short, 0.0 = flat, None = API error.
    positionAmt from API is a STRING — cast with float().
    Calls GET /fapi/v3/positionRisk
    """


def _get_quantity(symbol):
    """
    Returns coin quantity for a new opening order, derived from USDT notional.
    1. Fetch mark price: binance_get('/fapi/v1/premiumIndex', {'symbol': symbol}, signed=False)
    2. raw_qty = NOTIONAL_USDT.get(symbol, DEFAULT_NOTIONAL_USDT) / float(response['markPrice'])
    3. Round DOWN to stepSize precision (from cached exchangeInfo LOT_SIZE filter)
    4. Return None if quantity < minQty (caller rejects with clear error)
    Only used for OPEN orders. Close orders always use abs(current positionAmt).
    """


def place_order(symbol, side):
    """
    Check position, place MARKET order if needed.
    side: 'B' (buy/long) or 'S' (sell/short)
    Returns {'success': bool, 'message': str, ...}

    Flow:
    1. _setup_symbol(symbol) — set leverage + margin type on first encounter (see section 7)
    2. _get_position(symbol) — None → error; already in direction → skip
    3. If flipping: close with abs(positionAmt) using reduceOnly='true', then open with _get_quantity()
    4. If flat: open directly with _get_quantity()
    5. Use newOrderRespType='RESULT' for full fill details (avgPrice, etc.)

    Position logic:
    - B + long  → skip
    - S + short → skip
    - B + short → CLOSE short (qty=abs(positionAmt), reduceOnly='true'), OPEN long (_get_quantity())
    - S + long  → CLOSE long (qty=abs(positionAmt), reduceOnly='true'), OPEN short (_get_quantity())
    - B/S + flat → OPEN directly
    """
```

Catch `requests.HTTPError` in `place_order` — parse `e.response.json()` for Binance error code/message and return a
failure dict. Never let it propagate.

---

### 5.5 `app/binance/trading.py`

Orchestrator. Symbol resolution is inlined here — it's 2 lines, not worth a separate module.

```python
from app.binance.orders import place_order
from app.utils.logger import get_logger

logger = get_logger('binance/trading')


def _resolve_symbol(tv_symbol):
    """'BINANCE:BTCUSDT.P' → 'BTCUSDT'"""
    return tv_symbol.split(':')[-1].removesuffix('.P')


def process_binance_trading(trading_data):
    """
    Process a TradingView signal for Binance Futures.

    Args:
        trading_data: Dict with 'symbol', 'side', optional 'dummy'.

    Returns:
        Status dict with 'status' key.
    """
    try:
        symbol = trading_data.get('symbol')
        side = trading_data.get('side')

        if not symbol or not side:
            logger.error('Missing required fields', extra={'data': trading_data})
            return {'status': 'error', 'message': 'Missing symbol or side'}

        if trading_data.get('dummy', 'NO').upper() == 'YES':
            logger.info('Dummy signal, skipping', extra={'symbol': symbol})
            return {'status': 'dummy_skip'}

        result = place_order(_resolve_symbol(symbol), side)

        if result.get('success'):
            return {'status': 'order_placed', **result}
        return {'status': 'order_failed', **result}

    except Exception:
        logger.exception('Unexpected error processing Binance signal', extra={'data': trading_data})
        return {'status': 'error', 'message': 'Unexpected error'}
```

---

### 5.6 `config.py` — additions

```python
# --- Binance Configuration ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
BINANCE_ALERTS_DIR = DATA_DIR / 'alerts' / 'binance_alerts'
```

---

### 5.7 `app/routes/webhook.py` — add `/binance` route

Additions only — existing routes, blueprint, and `before_request` are untouched:

```python
from app.binance.trading import process_binance_trading
from config import BINANCE_ALERTS_DIR


@webhook_blueprint.route('/binance', methods=['POST'])
def binance_trading_route():
    data = request.get_json()
    save_alert_data_to_file(data, BINANCE_ALERTS_DIR)
    process_binance_trading(data)
    return '', 200
```

`run.py` is unchanged — `/binance` is part of the same `webhook_blueprint` already registered.
Alerts are saved to `data/alerts/binance_alerts/alerts_YYYY-MM-DD.json` by the existing `save_alert_data_to_file()` — no
changes needed to that function.

---

## 6. Quantity Strategy

Binance MARKET orders require coin quantity (e.g. BTC), not USDT. Coin quantity is calculated at order time from a fixed
USDT notional:

```
coin_qty = NOTIONAL_USDT / mark_price   (rounded down to stepSize)
```

- **Mark price**: `GET /fapi/v1/premiumIndex` — public endpoint, no auth. Fetch fresh on every order.
- **stepSize / minQty**: from `GET /fapi/v1/exchangeInfo` LOT_SIZE filter — cache at module level, it changes rarely.
- Use `math.floor` to round down; never round up (would exceed notional intent).
- If `coin_qty < minQty`, reject before placing any order.

**Important:** `_get_quantity()` is only for opening new positions. Closing an existing position always uses
`abs(positionAmt)` from `_get_position()` — the exact current size, not a freshly calculated notional.

---

## 7. Symbol Setup (Leverage + Margin Mode)

Leverage and margin type are set lazily on first encounter per symbol, inside `_setup_symbol()` in `orders.py`:

```python
_configured_symbols = set()


def _setup_symbol(symbol):
    if symbol in _configured_symbols:
        return
    leverage = LEVERAGE.get(symbol, DEFAULT_LEVERAGE)
    binance_post('/fapi/v1/leverage', {'symbol': symbol, 'leverage': leverage})
    try:
        binance_post('/fapi/v1/marginType', {'symbol': symbol, 'marginType': 'CROSSED'})
    except requests.HTTPError as err:
        if err.response.json().get('code') == -4046:
            pass  # Already CROSSED — no-op
        else:
            raise
    _configured_symbols.add(symbol)
```

- `_configured_symbols` is module-level state — resets on server restart, which is fine since both API calls are
  idempotent
- Default Binance leverage varies by symbol — always set explicitly, never rely on account defaults
- Margin mode: **CROSSED** (shared margin across positions, simpler than ISOLATED)

---

## 8. Error Handling

| Scenario                | Return                                                                              |
|-------------------------|-------------------------------------------------------------------------------------|
| Position check fails    | `{'success': False, 'message': 'Failed to fetch position'}`                         |
| Already in position     | `{'success': True, 'message': 'Already long/short, skipping'}`                      |
| Quantity below minQty   | `{'success': False, 'message': 'Order quantity below minimum for {symbol}'}`        |
| Order placed            | `{'success': True, 'message': 'Order placed', 'order_id': ...}`                     |
| Binance API error (4xx) | Catch `HTTPError`, parse `e.response.json()` → `{'success': False, 'message': msg}` |
| Unexpected exception    | `logger.exception()`, return `{'success': False, 'message': 'Unexpected error'}`    |

`raise_for_status()` in `api.py` raises `requests.HTTPError` for 4xx/5xx. Catch it in `orders.py`, read
`e.response.json()` for the Binance error `code` and `msg`. The webhook always returns 200.

---

## 9. Testnet / Paper Trading

- Production base URL: `https://fapi.binance.com`
- Testnet base URL: `https://demo-fapi.binance.com` (official as of 2025; older `testnet.binancefuture.com` still
  responds but is no longer canonical)
- Controlled via `BINANCE_TESTNET=true` in `.env` — switches `BASE_URL` in `api.py`
- Testnet requires separate API keys registered at the Binance demo portal, not production keys

---

## 10. Dependencies

No new dependencies. `requests` is already in `requirements.txt`; signing uses stdlib `hmac` and `hashlib`.

---

## 11. API Gotchas (confirmed from docs)

1. **`positionAmt` is a string.** `/fapi/v3/positionRisk` returns it as `"0.001"` not `0.001`. Always:
   `float(pos['positionAmt'])`.

2. **`reduceOnly` must be a string.** Pass `'reduceOnly': 'true'`, not `True`. Query string serialisation turns Python
   `True` into `"True"` which Binance rejects.

3. **Margin type change errors if already set.** API returns `-4046` if the symbol is already in the requested mode.
   Catch and treat as a no-op.

4. **Use `stepSize` not `quantityPrecision`.** Quantity rounding must use `stepSize` from the LOT_SIZE filter in
   `exchangeInfo`. The docs explicitly warn against using the `quantityPrecision` field on the symbol object.

5. **Rate limiting is per IP.** HTTP 429 = rate limited; 418 = temporarily banned (2 min to 3 days for repeat
   violations). Header `X-MBX-USED-WEIGHT-1M` tracks current usage.

6. **Confirm one-way mode before going live.** Check `GET /fapi/v1/positionSide/dual` — if `"dualSidePosition": true`,
   the account is in hedge mode and `reduceOnly` will be rejected. This is a one-time account setting; verify manually
   before first deployment, not at runtime.

7. **Always pass `newOrderRespType='RESULT'`** on MARKET orders. Default `ACK` returns only the order ID; `RESULT`
   returns the full fill (avgPrice, cumQty) immediately, which is what you want for logging.

---

## 12. Testing Plan

```
tests/binance/
├── __init__.py
├── conftest.py         # Shared fixtures (mock HTTP responses, sample data)
├── test_orders.py      # place_order() with mocked binance_get/binance_post
└── test_trading.py     # process_binance_trading() end-to-end
```

All tests use `monkeypatch` to mock `binance_get` and `binance_post`. Mark integration tests that hit testnet with
`@pytest.mark.integration`.

Key test cases:

- `'BINANCE:BTCUSDT.P'` → resolves to `'BTCUSDT'` (via `_resolve_symbol` in trading.py)
- `positionAmt` as string `"0.001"` → parsed correctly to float
- Buy signal, flat → order placed (LONG), quantity from `_get_quantity()`
- Buy signal, already long → skipped
- Sell signal, already long → close uses `abs(positionAmt)`, open uses `_get_quantity()`
- Quantity below `minQty` → rejected before any order API call
- `dummy=YES` → skipped, no API call
- Missing `symbol` or `side` → error dict returned, no API call
- Binance API returns 4xx → `HTTPError` caught, error dict returned, no propagation
- Unexpected exception → caught in `process_binance_trading`, returns error dict
- `POST /binance` returns 200 regardless of outcome
- `_setup_symbol` called once per symbol, skipped on repeat orders

---

## 13. Implementation Order

1. **`.env`** — Add `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET`
2. **`config.py`** — Add Binance env vars and `BINANCE_ALERTS_DIR`
3. **`app/binance/config.py`** — Per-symbol `NOTIONAL_USDT` and `LEVERAGE`
4. **`app/binance/__init__.py`** — Module docstring
5. **`app/binance/api.py`** — Raw HTTP helpers with HMAC-SHA256 signing
6. **`app/binance/orders.py`** — Position check, quantity calc, symbol setup, order placement
7. **`app/binance/trading.py`** — Orchestrator with inlined symbol resolution
8. **`app/routes/webhook.py`** — Add `/binance` route
9. **Tests** — `tests/binance/`

---

## 14. Out of Scope (for now)

- Binance spot trading
- Coin-margined futures (COIN-M) — only USDT-margined
- Hedge mode — only one-way mode
- Trailing stop orders
- Strategy analysis for Binance positions
- Backtesting with Binance data

---

## 15. Open Questions

1. **Leverage per symbol**: What leverage should be used per instrument?
2. **Margin mode**: CROSSED or ISOLATED?
3. **Error notifications**: Should Binance order failures trigger any alerting beyond logging?
