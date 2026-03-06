# CLAUDE.md

Algorithmic futures trading system: real-time IBKR trading, parallel backtesting, and trade analysis. Built in Python
with Flask, pandas, and multi-process execution.

## Commands

```bash
python -m pytest                          # Run all tests
python -m pytest tests/backtesting/       # Run module tests
python -m pytest -k "test_function_name"  # Run single test
python -m pytest -m "not slow"            # Skip slow tests
python run.py                             # Start Flask webhook server
python mass_backtest.py                   # Run mass backtesting
python fetch_data.py                      # Fetch historical data
python analyze_strategies.py              # Analyze strategy results
```

## Architecture

- `run.py`: Flask server receives TradingView webhooks, executes orders via IBKR API
- `mass_backtest.py`: `MassTester` runs parallel strategy evaluation with `ProcessPoolExecutor`
- `app/backtesting/`: Strategies, indicators, metrics, cache, validators, testing orchestration
- `app/ibkr/`: IBKR API integration (connection, contracts, orders, trading, rollover)
- `app/routes/webhook.py`: Two routes — `POST /trading`, `POST /rollover`
- `app/utils/`: Shared helpers (API, file I/O, math, logging)
- `app/analysis/`: Deprecated, pending rewrite
- `futures_config/`: Single source of truth for 50+ futures contracts (specs, symbol mapping)
- `config.py`: Env config via `.env` (`DEBUG`, `PORT`, `BASE_URL`, `ACCOUNT_ID`, `ALLOWED_IPS`, paths)

## Code Conventions

- **Section headers**: `# ==================== Section Name ====================`
- **Subsection headers**: `# --- Subsection Name ---`
- **Inline comments**: `# Explanation here` (sentence case, no period, placed above code)
- **Docstrings**: Structured with `Args:`, `Returns:`, `Raises:` sections
- **No backward compatibility shims** when deleting or updating fields

## Tests

- Mirror `app/` structure under `tests/`
- Shared fixtures in `tests/backtesting/fixtures/` and `tests/conftest.py`
- **Mocking**: Use `monkeypatch` only — no `@patch` decorators or `with patch(...)` context managers
- **Fixtures**: Define in nearest `conftest.py`, not inside test files
- **Markers**: `@pytest.mark.slow` (>0.5s), `@pytest.mark.integration`
- Update tests when modifying code
