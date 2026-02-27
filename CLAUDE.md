# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Algorithmic futures trading system with three operational modes: real-time webhook-based trading via Interactive
Brokers (IBKR), a parallel backtesting engine, and trade analysis. Built in Python with Flask, pandas, and multi-process
execution.

## Commands

```bash
# Run all tests
python -m pytest

# Run tests for a specific module
python -m pytest tests/backtesting/
python -m pytest tests/backtesting/strategies/

# Run a single test file
python -m pytest tests/backtesting/strategies/test_rsi.py

# Run a single test by name
python -m pytest -k "test_function_name"

# Run with coverage
pytest --cov=app --cov-report=html

# Skip slow or integration tests
python -m pytest -m "not slow"
python -m pytest -m "not integration"

# Start Flask webhook server
python run.py

# Run mass backtesting
python mass_backtest.py

# Fetch historical data from TradingView
python fetch_data.py

# Analyze strategy results
python analyze_strategies.py
```

## Architecture

### Three Operational Modes

1. **Real-time Trading** (`run.py`): Flask server receives TradingView webhook alerts at `/trading` and `/rollover`,
   validates source IPs, and executes orders through IBKR API. Connection maintained with 60-second heartbeat via
   APScheduler.

2. **Backtesting** (`mass_backtest.py`): `MassTester` orchestrates parallel strategy evaluation using
   `ProcessPoolExecutor`. Signals are detected at bar close and executed at next bar open (queued execution) to prevent
   look-ahead bias. Supports contract rollover, trailing stops, and slippage modeling.

3. **Analysis** (`analyze_strategies.py`): Strategy result post-processing. Note: the `app/analysis/` module is
   deprecated and pending rewrite.

### Backtesting Module (`app/backtesting/`)

This is the most complex module. Key components:

- **strategies/base/**: `BaseStrategy` abstract class with `PositionManager`, `TrailingStopManager`, and
  `ContractSwitchHandler`
- **strategies/**: Five strategy implementations (RSI, EMA Crossover, MACD, Bollinger Bands, Ichimoku Cloud) plus
  `StrategyFactory`
- **testing/**: `MassTester` → `orchestrator.py` (parallel dispatch) → `runner.py` (single test execution) →
  `reporting.py` (results aggregation to Parquet). `reporting.py` owns `save_to_parquet`.
- **cache/**: Two-tier LRU cache system (`DataFrameCache` for loaded data, `IndicatorsCache` for computed indicators)
  with file-locking for multi-process safety
- **indicators/**: Technical indicator calculations with hash-based caching
- **metrics/**: Per-trade metrics (points, dollar return, % of margin) and summary metrics (win rate, profit factor,
  Sharpe, Sortino, Calmar, VaR, max drawdown)
- **validators/**: Per-strategy parameter validation with a shared `CommonValidator`

### Analysis Module (`app/analysis/`)

Deprecated, pending rewrite. Contains `analysis_utils/` subfolder with helpers for data cleaning, fetching, and general
calculations. The main files (`analysis_runner.py`, `data_fetching.py`) own their own I/O helpers (`save_to_csv`,
`load_data_from_json_files`, `json_to_dataframe`) rather than delegating to utils.

### Futures Configuration (`futures_config/`)

Single source of truth for 50+ futures contracts across 7 categories (Grains, Softs, Energy, Metals, Crypto, Index,
Forex). Contains symbol specs (multiplier, tick size, margin), TradingView-to-IBKR symbol mapping, and correlation
groups.

### Routes (`app/routes/`)

- `webhook.py`: Flask Blueprint (`webhook_blueprint`) with two routes: `POST /trading` (order execution) and
  `POST /rollover` (contract rollover). A shared `before_request` hook validates source IP and JSON content type.
  Alerts are persisted to `data/alerts/ibkr_alerts/` as daily JSON files (dummy signals skipped). Always returns 200
  to prevent TradingView retry duplicates.

### Services (`app/ibkr/`)

IBKR API integration split across five files:

- `connection.py`: APScheduler setup; `_tickle_ibkr_api()` heartbeat keeps the session alive
- `contracts.py`: `ContractResolver` — front-month contract ID lookup with `MIN_BUFFER_DAYS` expiry guard and
  mini/micro symbol mapping
- `orders.py`: `place_order()` — position-aware order placement with message suppression retry loop
- `trading.py`: `process_trading_data()` — parses webhook payload and dispatches to `place_order()`
- `rollover.py`: `process_rollover_data()` — closes the front-month position and optionally reopens on the next
  contract; logs critical if reopen fails after a successful close

### Utilities (`app/utils/`)

- `api_utils.py`: IBKR REST API request helpers (`api_get`, `api_post`)
- `file_utils.py`: JSON-only I/O (`load_file`, `save_file`). CSV/Parquet helpers live in the modules that use them.
- `generic_utils.py`: Symbol parsing
- `logger.py`: Shared logger setup (skips file handlers during pytest)
- `math_utils.py`: Zero-safe math helpers (`safe_divide`, `safe_average`, `calculate_percentage`)

### Configuration (`config.py`)

Top-level config loaded from `.env` via `python-dotenv`. Provides `DEBUG`, `PORT`, `BASE_URL`, `ACCOUNT_ID`,
`ALLOWED_IPS` (TradingView + local IPs), and `DATA_DIR` / `BASE_DIR` paths.

### TradingView Scripts (`tv_scripts/`)

Pine Script files for TradingView. Not part of the Python runtime — used for signal generation and testing.

- `indicators/`: Five indicator scripts (RSI, EMA, MACD, Bollinger Bands, Ichimoku)
- `tv_strategies/`: Two strategy scripts (RSI, ATR+EMA+RSI)
- `contract_switch_warning.pine`: Alerts when a contract is near its switch date
- `webhook_test.pine`: Test script for validating webhook delivery

### Data Storage (`data/`)

- `historical_data/1!/`, `historical_data/2!/`: Front and next month contract data
- `backtesting/cache/`: Serialized indicator and DataFrame caches
- `backtesting/mass_test_results_all.parquet`: Aggregated backtest results
- `contracts/`: IBKR contract IDs
- `alerts/`, `analysis/`: Live trading data

## Code Conventions

- **Section headers**: `# ==================== Section Name ====================` (20 equals signs each side, Title
  Case)
- **Subsection headers**: `# --- Subsection Name ---` (3 dashes each side, Title Case)
- **Inline comments**: `# Explanation here` (Sentence case, no period, placed above code)
- **Docstrings**: Use structured format with `Args:`, `Returns:`, `Raises:` sections
- **No backward compatibility**: When deleting or updating fields, do not provide backward compatibility shims
- **Tests**: Update unit tests when modifying code. Test markers: `@pytest.mark.slow` (>0.5s),
  `@pytest.mark.integration`

## Test Structure

Tests mirror the app structure under `tests/`. Shared fixtures live in `tests/backtesting/fixtures/` (data fixtures for
mock OHLCV DataFrames, strategy fixtures for pre-configured instances). Global fixtures in `tests/conftest.py` include
Flask test client, module-level logger mocks, and IBKR service mocks.

**Mocking style**: Use `monkeypatch` (not `@patch` decorators or `with patch(...)` context managers). Test classes use
`monkeypatch.setattr(object, "attr", mock)` for object attributes and `monkeypatch.setattr("module.path.name", mock)`
for module-level names.

**Fixtures**: Module-level fixtures go in the nearest `conftest.py`, not defined locally inside test files.

## Key Dependencies

- `tvdatafeed`: Custom GitHub fork for TradingView data fetching
- `filelock`: Multi-process cache synchronization
- `pyarrow`: Parquet serialization for backtest results
- `APScheduler`: IBKR connection heartbeat
- `python-dotenv`: Environment variable loading from `.env`
