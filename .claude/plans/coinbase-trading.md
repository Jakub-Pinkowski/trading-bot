# Coinbase Trading Integration Plan

## Overview

Add Coinbase Advanced Trade perpetuals support alongside the existing IBKR and Binance integrations. Signals arrive via
the same TradingView webhook format but hit a dedicated `/coinbase` route. All three brokers coexist independently —
Coinbase handles crypto perpetuals on the INTX exchange, Binance handles crypto perpetuals on Binance Futures, IBKR
handles traditional futures.

---

## 1. Current Architecture (IBKR + Binance)

```
TradingView → POST /trading → webhook.py → process_trading_data()        (IBKR)
TradingView → POST /binance → webhook.py → process_binance_trading(data)  (Binance)
```

Key patterns to carry over:

- Always return HTTP 200 from webhooks
- Save all alerts to file before processing
- Check position before placing order (skip if already in direction)
- Return `{'success': bool, ...}` dicts, never raise on known failures

---

## 2. Signal Format

No changes to TradingView signal format. Coinbase signals use the same schema:

```json
{
  "symbol": "COINBASE:BTCPERP",
  "side": "B",
  "dummy": "NO"
}
```

TradingView alerts for Coinbase are configured to point to `POST /coinbase`. The route determines the broker — no
symbol-based routing needed anywhere.

---

## 3. Target Architecture

```
                    webhook.py (single file, single blueprint)
                    ┌─────────────────────────────────────────┐
TradingView (IBKR)  │  POST /trading                          │
        ──────────► │    ├─ Save to ibkr_alerts/              │
                    │    └─ process_trading_data(data)         │
                    │                                         │
TradingView (IBKR)  │  POST /rollover                         │
        ──────────► │    └─ process_rollover_data(data)       │
                    │                                         │
TradingView (Binance│  POST /binance                          │
        ──────────► │    ├─ Save to binance_alerts/           │
                    │    └─ process_binance_trading(data)     │
                    │                                         │
TradingView (Coinb.)│  POST /coinbase                         │
        ──────────► │    ├─ Save to coinbase_alerts/          │
                    │    └─ process_coinbase_trading(data)    │
                    └─────────────────────────────────────────┘
                    before_request validates IP + JSON for all routes
```

`/coinbase` is added to the existing `webhook.py` alongside the other routes — same blueprint, same `before_request`
validation, no new files in `app/routes/`.

---

## 4. Coinbase Module Structure

Symbol mapping requires a lookup dict (Coinbase product IDs like `BTC-PERP-INTX` cannot be derived from TradingView
symbols by simple string manipulation), so it lives in `config.py`. Everything else mirrors the Binance layout.

```
app/coinbase/
├── __init__.py     # Module docstring
├── api.py          # JWT generation + raw HTTP helpers
├── config.py       # Per-symbol notional USD, leverage, symbol map
├── orders.py       # Position check, quantity calc, order placement
└── trading.py      # Orchestrator
```

No rollover module — INTX perpetuals never expire.

---

## 5. File-by-File Implementation Plan

### 5.1 `app/coinbase/__init__.py`

```python
"""Coinbase Advanced Trade perpetuals integration."""
```

---

### 5.2 `app/coinbase/api.py`

Raw HTTP helpers for the Coinbase Advanced Trade REST API. Handles JWT generation transparently so `orders.py` never
deals with auth directly.

**Authentication overview:** Every request requires a fresh JWT signed with an ECDSA private key. The JWT encodes the
exact HTTP method and path being called, so a single token is not reusable across different endpoints. Token lifespan
is 120 seconds.

```python
import secrets
import time
import uuid

import jwt
import requests
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from config import COINBASE_KEY_ID, COINBASE_ORG_ID, COINBASE_PRIVATE_KEY

BASE_URL = 'https://api.coinbase.com'
_API_PREFIX = '/api/v3/brokerage'


def _build_jwt(method, path):
    """
    Build a short-lived JWT for one Coinbase API request.

    Args:
        method: HTTP verb in uppercase, e.g. 'GET' or 'POST'.
        path:   Full path including /api/v3/brokerage prefix, e.g. '/api/v3/brokerage/orders'.

    Returns:
        Signed JWT string.
    """
    private_key = load_pem_private_key(COINBASE_PRIVATE_KEY.encode(), password=None)
    now = int(time.time())
    payload = {
        'sub': f'organizations/{COINBASE_ORG_ID}/apiKeys/{COINBASE_KEY_ID}',
        'iss': 'cdp',
        'nbf': now,
        'exp': now + 120,
        'uri': f'{method} api.coinbase.com{path}',
    }
    headers = {
        'kid': COINBASE_KEY_ID,
        'nonce': secrets.token_hex(16),
    }
    return jwt.encode(payload, private_key, algorithm='ES256', headers=headers)


def coinbase_get(path, params=None):
    token = _build_jwt('GET', path)
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(BASE_URL + path, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


def coinbase_post(path, body=None):
    token = _build_jwt('POST', path)
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    response = requests.post(BASE_URL + path, json=body or {}, headers=headers)
    response.raise_for_status()
    return response.json()


def generate_client_order_id():
    return str(uuid.uuid4())
```

Endpoint reference:

| Function                                                        | Endpoint                                   |
|-----------------------------------------------------------------|--------------------------------------------|
| `coinbase_get('/api/v3/brokerage/intx/positions/{product_id}')` | Current perpetual position for one product |
| `coinbase_post('/api/v3/brokerage/orders', body)`               | Place order                                |
| `coinbase_get('/api/v3/brokerage/products/{product_id}')`       | Product info (mid price, increments)       |
| `coinbase_get('/api/v3/brokerage/best_bid_ask', params)`        | Best bid/ask for price reference           |

---

### 5.3 `app/coinbase/config.py`

Per-symbol trading configuration and TradingView-to-Coinbase symbol mapping. Kept separate from `config.py` to avoid
cluttering env-var config with trading data.

```python
# Map TradingView symbol suffix → Coinbase product_id
# TradingView sends e.g. 'COINBASE:BTCPERP' → split on ':' → 'BTCPERP'
SYMBOL_MAP = {
    'BTCPERP': 'BTC-PERP-INTX',
    'ETHPERP': 'ETH-PERP-INTX',
    'SOLPERP': 'SOL-PERP-INTX',
}

NOTIONAL_USD = {
    'BTC-PERP-INTX': 100,
    'ETH-PERP-INTX': 100,
    'SOL-PERP-INTX': 100,
}
DEFAULT_NOTIONAL_USD = 100

LEVERAGE = {
    'BTC-PERP-INTX': 10,
    'ETH-PERP-INTX': 10,
}
DEFAULT_LEVERAGE = 5
```

---

### 5.4 `app/coinbase/orders.py`

The core module. Uses net position model — one net position per product, positive = long, negative = short.

**Key design:** Position size (`net_size`) is returned as a string by the API — always cast with `float()`. The close
order uses `abs(float(net_size))` (exact current size); the open order derives quantity from USD notional at current
price.

```python
def _get_position(product_id):
    """
    Returns current position net_size as float.
    Positive = long, negative = short, 0.0 = flat, None = API error.
    net_size from API is a STRING — cast with float().
    Calls GET /api/v3/brokerage/intx/positions/{product_id}
    """


def _get_base_size(product_id):
    """
    Returns base quantity for a new opening order, derived from USD notional.
    1. Fetch mid_market_price: coinbase_get('/api/v3/brokerage/products/{product_id}')
    2. raw_qty = NOTIONAL_USD.get(product_id, DEFAULT_NOTIONAL_USD) / float(mid_market_price)
    3. Round down to base_increment precision from product info
    4. Return None if quantity < base_min_size (caller rejects with clear error)
    Only used for OPEN orders. Close orders always use abs(float(net_size)).
    """


def place_order(product_id, side):
    """
    Check position, place MARKET (IOC) order if needed.
    side: 'B' (buy/long) or 'S' (sell/short)
    Returns {'success': bool, 'message': str, ...}

    Flow:
    1. _get_position(product_id) — None → error; already in direction → skip
    2. If flipping: close current side first (opposite side, abs(net_size) as base_size),
       then open new position (_get_base_size())
    3. If flat: open directly with _get_base_size()

    Position logic:
    - B + long  → skip
    - S + short → skip
    - B + short → SELL abs(net_size) to close, BUY _get_base_size() to open long
    - S + long  → BUY abs(net_size) to close, SELL _get_base_size() to open short
    - B/S + flat → OPEN directly
    """
```

**Order body shape (market IOC):**

```python
{
    'client_order_id': generate_client_order_id(),
    'product_id': product_id,
    'side': 'BUY',  # or 'SELL'
    'order_configuration': {
        'market_market_ioc': {
            'base_size': '0.001',  # string, not float
        }
    },
    'leverage': str(LEVERAGE.get(product_id, DEFAULT_LEVERAGE)),
    'margin_type': 'CROSS',
}
```

**Error handling:** Coinbase order errors do NOT use HTTP 4xx — the response is HTTP 200 with `"success": false`. Check
`response.json()['success']` rather than relying solely on `raise_for_status()`. Parse
`response.json()['error_response']` for the failure reason.

Catch `requests.HTTPError` only for genuine network/auth failures (5xx). Never let either propagate.

---

### 5.5 `app/coinbase/trading.py`

Orchestrator. Symbol resolution is inlined here — it's a single dict lookup.

```python
from app.coinbase.config import SYMBOL_MAP
from app.coinbase.orders import place_order
from app.utils.logger import get_logger

logger = get_logger('coinbase/trading')


def _resolve_symbol(tv_symbol):
    """'COINBASE:BTCPERP' → 'BTC-PERP-INTX'"""
    suffix = tv_symbol.split(':')[-1]
    return SYMBOL_MAP.get(suffix)


def process_coinbase_trading(trading_data):
    """
    Process a TradingView signal for Coinbase perpetuals.

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

        product_id = _resolve_symbol(symbol)
        if not product_id:
            logger.error('Unknown symbol', extra={'symbol': symbol})
            return {'status': 'error', 'message': f'Unknown symbol: {symbol}'}

        result = place_order(product_id, side)

        if result.get('success'):
            return {'status': 'order_placed', **result}
        return {'status': 'order_failed', **result}

    except Exception:
        logger.exception('Unexpected error processing Coinbase signal', extra={'data': trading_data})
        return {'status': 'error', 'message': 'Unexpected error'}
```

---

### 5.6 `config.py` — additions

```python
# --- Coinbase Configuration ---
COINBASE_KEY_ID = os.getenv('COINBASE_KEY_ID', '')
COINBASE_ORG_ID = os.getenv('COINBASE_ORG_ID', '')
# PEM private key stored with literal \n in .env; replace before use
COINBASE_PRIVATE_KEY = os.getenv('COINBASE_PRIVATE_KEY', '').replace('\\n', '\n')
COINBASE_ALERTS_DIR = DATA_DIR / 'alerts' / 'coinbase_alerts'
```

---

### 5.7 `app/routes/webhook.py` — add `/coinbase` route

Additions only — existing routes, blueprint, and `before_request` are untouched:

```python
from app.coinbase.trading import process_coinbase_trading
from config import COINBASE_ALERTS_DIR


@webhook_blueprint.route('/coinbase', methods=['POST'])
def coinbase_trading_route():
    data = request.get_json()
    save_alert_data_to_file(data, COINBASE_ALERTS_DIR)
    process_coinbase_trading(data)
    return '', 200
```

`run.py` is unchanged — `/coinbase` is part of the same `webhook_blueprint` already registered.
Alerts are saved to `data/alerts/coinbase_alerts/alerts_YYYY-MM-DD.json` by the existing
`save_alert_data_to_file()` — no changes needed to that function.

---

## 6. Quantity Strategy

Coinbase MARKET orders accept `base_size` (coin quantity) or `quote_size` (USD). For perpetuals, `base_size` is the
reliable choice — `quote_size` support on INTX perpetuals is unconfirmed. Calculate coin quantity from USD notional at
order time:

```
base_size = NOTIONAL_USD / mid_market_price   (rounded DOWN to base_increment)
```

- **Mid price**: `GET /api/v3/brokerage/products/{product_id}` — `mid_market_price` field. Fall back to
  `(best_bid + best_ask) / 2` from `GET /api/v3/brokerage/best_bid_ask` if `mid_market_price` is null.
- **base_increment / base_min_size**: also from the product endpoint — cache at module level, same as Binance's
  `_exchange_info_cache`.
- Use `math.floor` to round down to the nearest valid `base_increment` — this avoids exceeding the configured notional.
- If `base_size < base_min_size`, reject before placing any order.
- All numeric fields must be passed as **strings** in the order body — Coinbase rejects Python floats.

**Important:** `_get_base_size()` is only for opening new positions. Closing always uses `abs(float(net_size))` from
`_get_position()`.

---

## 7. Authentication Details

Coinbase uses ECDSA JWT auth (not HMAC). Key facts:

- **Key format**: ECDSA only — EdDSA/Ed25519 keys are explicitly not supported.
- **Key creation**: CDP portal → API Keys tab → Secret API Keys → algorithm: ECDSA. Required scopes: `view` + `trade`.
- **JWT lifespan**: 2 minutes (`exp = nbf + 120`). Generate a fresh JWT per API call.
- **`uri` claim**: Must match the exact request, e.g. `"GET api.coinbase.com/api/v3/brokerage/products/BTC-PERP-INTX"`.
  A mismatch causes a 401.
- **Private key in `.env`**: ECDSA private keys are multi-line PEM files. Store as a single line with literal `\n`
  characters. In `config.py`, call `.replace('\\n', '\n')` before use.
  ```
  COINBASE_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----\nMHQCAQEEI...\n-----END EC PRIVATE KEY-----\n"
  ```
- **`org_id`** vs **`key_id`**: Both are shown in the CDP portal after key creation. `key_id` goes in the JWT `kid`
  header; `org_id` is part of the `sub` claim.

---

## 8. Error Handling

| Scenario                 | Return                                                                           |
|--------------------------|----------------------------------------------------------------------------------|
| Unknown symbol           | `{'success': False, 'message': 'Unknown symbol: COINBASE:XYZPERP'}`              |
| Position check fails     | `{'success': False, 'message': 'Failed to fetch position'}`                      |
| Already in position      | `{'success': True, 'message': 'Already long/short, skipping'}`                   |
| Size below base_min_size | `{'success': False, 'message': 'Order size below minimum for {product_id}'}`     |
| Order placed             | `{'success': True, 'message': 'Order placed', 'order_id': ...}`                  |
| Coinbase order failure   | HTTP 200 with `success: false` — parse `error_response.new_order_failure_reason` |
| Network/auth failure     | Catch `HTTPError` (5xx/401) → `{'success': False, 'message': msg}`               |
| Unexpected exception     | `logger.exception()`, return `{'success': False, 'message': 'Unexpected error'}` |

**Critical difference from Binance:** Coinbase order endpoint returns HTTP 200 even when the order fails. Always check
`response.json()['success']` — do not rely on `raise_for_status()` alone for order placement logic.

---

## 9. Sandbox / Paper Trading

- **Production base URL**: `https://api.coinbase.com`
- **Sandbox**: Coinbase Advanced Trade does not offer a public sandbox URL equivalent to Binance's testnet. Testing
  options:
    1. Use the live API with very small sizes against real markets.
    2. Use API key permissions scoped to `view` only for dry-run validation.
- Controlled via `COINBASE_SANDBOX=true` in `.env` if/when a sandbox URL becomes available — add to `api.py`'s
  `BASE_URL` selection the same way `BINANCE_TESTNET` works.

---

## 10. Dependencies

Two new packages required — neither is currently in `requirements.txt`:

| Package        | Purpose                                    | Install                    |
|----------------|--------------------------------------------|----------------------------|
| `PyJWT`        | JWT encoding with ECDSA (ES256)            | `pip install PyJWT`        |
| `cryptography` | ECDSA key loading (`load_pem_private_key`) | `pip install cryptography` |

`requests` is already present. Both `PyJWT` and `cryptography` are stable, widely used packages with no transitive
conflicts expected.

---

## 11. API Gotchas (confirmed from docs)

1. **`net_size` is a string.** `GET /api/v3/brokerage/intx/positions/{product_id}` returns `net_size` as `"0.001"`,
   not `0.001`. Always: `float(pos['net_size'])`.

2. **Order errors are HTTP 200.** Unlike Binance (which uses 4xx for rejected orders), Coinbase returns HTTP 200 with
   `"success": false` and an `"error_response"` object. Never trust `raise_for_status()` alone to detect order failures.

3. **All numeric fields must be strings in the order body.** The Coinbase API rejects Python `float` or `int` types for
   `base_size`, `leverage`, `limit_price`, etc. Always stringify: `str(round(qty, 8))`.

4. **`uri` claim must match exactly.** The JWT `uri` claim encodes the specific endpoint being called. A `GET` JWT
   cannot be reused for a `POST` or a different path — generate a fresh JWT per call.

5. **`client_order_id` must be unique per order.** Submitting a duplicate `client_order_id` returns the original order
   rather than placing a new one. Always generate via `uuid.uuid4()`.

6. **`mid_market_price` can be null.** For illiquid or recently listed products, the product endpoint may return `null`
   for `mid_market_price`. Fall back to `(best_bid[0].price + best_ask[0].price) / 2` from the best bid/ask endpoint.

7. **Product IDs are exchange-specific.** Coinbase International Exchange perpetuals are `BTC-PERP-INTX` (INTX suffix).
   Standard spot pairs are `BTC-USD`. Never assume a mapping — maintain an explicit `SYMBOL_MAP` dict.

8. **Leverage is per-order, not per-symbol.** Unlike Binance (which requires a separate leverage-set API call before
   the first order), Coinbase accepts `leverage` directly in the order body. No symbol setup step needed.

9. **Private key newlines in `.env`.** PEM files contain literal newlines. When stored as a single `.env` line, the
   newlines must be escaped as `\n`. In `config.py`, call `.replace('\\n', '\n')` before passing to
   `load_pem_private_key()` — otherwise key parsing fails with a cryptographic error.

10. **JWT clock skew.** If the server clock is more than ~30 seconds behind, `nbf` validation will reject the token.
    Ensure NTP sync on the server. The `GET /api/v3/brokerage/time` endpoint returns server time for verification.

---

## 12. Testing Plan

```
tests/coinbase/
├── __init__.py
├── conftest.py         # Shared fixtures (mock HTTP responses, sample positions, sample products)
├── test_api.py         # JWT generation shape, coinbase_get/coinbase_post with mocked requests
├── test_orders.py      # place_order() with mocked coinbase_get/coinbase_post
└── test_trading.py     # process_coinbase_trading() end-to-end
```

All tests use `monkeypatch` to mock `coinbase_get`, `coinbase_post`, and `generate_client_order_id`. Mark integration
tests that hit production API with `@pytest.mark.integration`.

Key test cases:

- `'COINBASE:BTCPERP'` → resolves to `'BTC-PERP-INTX'` (via `_resolve_symbol`)
- `'COINBASE:UNKNOWN'` → returns error dict, no API call
- `net_size` as string `"0.001"` → parsed correctly to float
- Buy signal, flat → order placed (BUY), size from `_get_base_size()`
- Buy signal, already long → skipped
- Sell signal, already long → close (SELL abs(net_size)) then open (SELL _get_base_size())
- Size below `base_min_size` → rejected before any order API call
- `dummy=YES` → skipped, no API call
- Missing `symbol` or `side` → error dict returned, no API call
- Coinbase returns `{"success": false, "error_response": {...}}` → failure dict returned, no propagation
- Network error (HTTPError 5xx) → caught in `place_order`, returns error dict
- Unexpected exception → caught in `process_coinbase_trading`, returns error dict
- `POST /coinbase` returns 200 regardless of outcome
- `mid_market_price` is null → falls back to best bid/ask average

---

## 13. Implementation Order

1. **`requirements.txt`** — Add `PyJWT` and `cryptography`
2. **`.env`** — Add `COINBASE_KEY_ID`, `COINBASE_ORG_ID`, `COINBASE_PRIVATE_KEY`
3. **`config.py`** — Add Coinbase env vars and `COINBASE_ALERTS_DIR`
4. **`app/coinbase/config.py`** — `SYMBOL_MAP`, `NOTIONAL_USD`, `LEVERAGE`
5. **`app/coinbase/__init__.py`** — Module docstring
6. **`app/coinbase/api.py`** — JWT generation + raw HTTP helpers
7. **`app/coinbase/orders.py`** — Position check, size calc, order placement
8. **`app/coinbase/trading.py`** — Orchestrator with inlined symbol resolution
9. **`app/routes/webhook.py`** — Add `/coinbase` route
10. **Tests** — `tests/coinbase/`

---

## 14. Out of Scope (for now)

- Coinbase spot trading
- Limit orders / stop-loss orders
- Margin type ISOLATED — only CROSS
- Strategy analysis for Coinbase positions
- Backtesting with Coinbase data
- Webhook-based order fills (Coinbase WebSocket notifications)

---

## 15. Open Questions

1. **Leverage per symbol**: What leverage should be used per instrument?
2. **Notional size**: What USD notional per trade per symbol?
3. **Symbol map completeness**: Which perpetuals beyond BTC/ETH/SOL should be in `SYMBOL_MAP`?
4. **Sandbox testing**: How to validate orders without hitting live markets? Use very small sizes, or scope the key to
   `view`-only for dry runs?
5. **Error notifications**: Should Coinbase order failures trigger any alerting beyond logging?
6. **`quote_size` vs `base_size`**: Verify whether `quote_size` (USD notional direct) works for INTX perpetuals. If it
   does, the `_get_base_size()` price-fetch step can be eliminated in favour of passing USD notional directly.