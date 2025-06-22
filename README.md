# Trading Bot

## Overview

This project is an automated trading and analytics application designed for working with financial data and the Interactive Brokers (IBKR)
API. It focuses on streamlining the processes of trading, data fetching, analysis, backtesting, and performance evaluation.

## Key Features

- **Trading Automation**: Integrates with the IBKR API to execute, monitor, and manage trades automatically.
- **Webhook Support**: Receives trading alerts and commands (e.g., from TradingView) via webhooks.
- **Data Analysis**: Cleans, matches, and analyzes both alerts and trade data, providing per-trade and aggregate metrics to evaluate
  performance.
- **Backtesting Framework**: Tests multiple trading strategies (RSI, MACD, Bollinger Bands, EMA Crossover, Ichimoku
  Cloud) against historical data to
  evaluate performance.
- **Historical Data Management**: Organizes and processes historical market data for different contract months and time intervals.
- **Comprehensive Logging**: Maintains detailed logs for debugging, error tracking, and info-level monitoring.
- **Modular Architecture**: The codebase is organized into modules for routing, analysis, backtesting, IBKR services, and utilities,
  ensuring clarity and
  maintainability.
- **Extensive Utilities**: Includes a variety of utilities for data cleaning, fetching, file handling, logging, backtesting, and more.
- **Extensible Data Structure**: Separates raw and processed data, supports analysis on alerts, trades, and specialized alerts (TW Alerts).
- **Configuration and Security**: Uses environment variables and config files for easy setup and to keep sensitive data secure.

## Configuration

The application uses a combination of environment variables and configuration settings in `config.py`:

### Environment Variables (in `.env` file)

- `DEBUG`: Set to `True` for development, `False` for production
- `PORT`: The port on which the Flask application will run
- `BASE_URL`: The base URL of the application

### Configuration Settings (in `config.py`)

- IBKR setup (account ID, allowed IPs)
- Data directories
- File paths
- Strategy parameters (minimum days until expiry, quantity to trade, aggressive trading)
- Analysis parameters (timeframe to analyze)

## Usage

### Starting the Application

To start the Flask application that handles webhooks and interacts with IBKR:

```bash
python run.py
```

This will start a Flask server on the specified port, which will:

- Listen for webhook requests at `/webhook`
- Maintain a connection to the IBKR API
- Process trading alerts and execute trades

### Running Analysis

To run analysis on the collected data:

```bash
python analyze.py
```

This will:

- Fetch raw data (alerts, TW alerts, trades)
- Clean and process the data
- Match trades
- Calculate metrics
- Save the results to CSV files

### Backtesting

To run backtests on historical data using various trading strategies with multiple parameter combinations:

```bash
python mass_backtest.py
```

This will:

- Load historical data for specified symbols and time intervals
- Run multiple trading strategies (RSI, MACD, Bollinger Bands, EMA Crossover) with various parameter combinations
- Calculate per-trade metrics for each strategy
- Generate summary performance metrics
- Save results for strategy comparison

The mass_backtest.py script provides a powerful way to test many combinations of:

- Symbols (e.g., ZW, ZC, ZS)
- Timeframes (e.g., 1h, 4h, 1d)
- Strategies (RSI, EMA Crossover, MACD, Bollinger Bands)
- Strategy-specific parameters (e.g., RSI periods, EMA lengths)
- Common parameters (rollover, trailing stops)

#### Configuration

Instead of using command-line options, all settings are configured directly in the `mass_backtest.py` file. To customize your backtesting:

1. Open the `mass_backtest.py` file in a text editor
2. Modify the parameters in the MassTester initialization and strategy configuration sections
3. Run the script without any command-line arguments

Example configuration in `mass_backtest.py`:

class MassTester:
pass

class MassTester:
pass

```python
# Initialize the mass tester with multiple symbols and timeframes
tester = MassTester(
  tested_months=["1!"],  # Front month contracts
  symbols=["ZW", "ZC", "ZS"],  # Wheat, Corn, Soybeans
  intervals=["1h", "4h"]  # 1-hour and 4-hour timeframes
)

# Add RSI strategy tests with various parameter combinations
tester.add_rsi_tests(
  rsi_periods=[7, 14, 21],  # Test different RSI periods
  lower_thresholds=[20, 30],  # Test different oversold thresholds
  upper_thresholds=[70, 80],  # Test different overbought thresholds
  rollovers=[False, True],  # Test with and without a rollover
  trailing_stops=[None, 1.0, 2.0]  # Test with different trailing stops
)
```

#### Available Configuration Options

The following configuration options are available in the `mass_backtest.py` file:

- Data selection: `tested_months`, `symbols`, `intervals`
- Strategy methods:
  - `add_rsi_tests()`: Configure RSI strategy tests
  - `add_ema_crossover_tests()`: Configure EMA Crossover strategy tests
  - `add_macd_tests()`: Configure MACD strategy tests
  - `add_bollinger_bands_tests()`: Configure Bollinger Bands strategy tests
- Output options: `verbose`, `save_results` (in the `run_tests()` method)

#### Programmatic Usage

You can also use the MassTester class directly in your Python code:

```python
from app.backtesting.mass_testing import MassTester

# Initialize the tester
tester = MassTester(
  tested_months=["1!"],
  symbols=["ZW", "ZC"],
  intervals=["4h", "1d"]
)

# Add strategy tests
tester.add_rsi_tests(
  rsi_periods=[7, 14, 21],
  lower_thresholds=[20, 30],
  upper_thresholds=[70, 80],
  rollovers=[False, True],
  trailing_stops=[None, 1.0]
)

# Run tests
tester.run_tests()

# Analyze results
top_strategies = tester.get_top_strategies(metric="profit_factor")
print(top_strategies)
```

A complete example script is available in the `examples/mass_test_example.py` file, which demonstrates:

- Testing multiple strategies (RSI and EMA Crossover)
- Using various parameter combinations
- Analyzing results in different ways (by profit factor, win rate, symbol, timeframe)

### Webhook Integration

The application accepts POST requests at the `/webhook` endpoint. The request should contain trading alert data in JSON format. The
application will validate the IP address, parse the request data, save the alert data to a file, and process the trading data.

Example webhook payload:

```json
{
  "symbol": "ES",
  "action": "BUY",
  "quantity": 1,
  "price": 4500.50,
  "timestamp": "2023-05-01T12:34:56Z"
}
```

For a detailed breakdown of the project structure, see `structure.yaml`.

## Typical Use Cases

- Automated trading based on external alerts.
- Continuous analysis of trade and alert performance.
- Backtesting trading strategies against historical data to evaluate performance.
- Comparing different trading strategies to identify the most profitable approaches.
- Fetching and preparing financial datasets for further analysis.
- Logging trading activities and analysis operations for audit or debugging.
- Running analyses on historical data to derive actionable insights.

## Development

### Project Structure

The project follows a modular structure as detailed in `structure.yaml`. Key directories include:

- `app/`: Main application code
  - `analysis/`: Data analysis modules
  - `backtesting/`: Backtesting framework and strategies
  - `routes/`: Flask routes for webhooks
  - `services/`: Services for interacting with IBKR
  - `utils/`: Utility functions
- `tests/`: Test suite organized by module

### Running Tests

To run the test suite:

```bash
python -m pytest
```

To run tests for a specific module:

```bash
python -m pytest tests/analysis/
```

### Coding Standards

- Use PEP 8 style guidelines
- Add docstrings to all functions, classes, and modules
- Write unit tests for new functionality
- Update `structure.yaml` when adding new files or directories

## Known Issues

The following known issues are currently being tracked:

- **Backtesting**: Drawdown calculations need to be fixed for more accurate risk assessment
- **IBKR Authentication**: Periodic authentication issues with the IBKR API (see notes in `structure.yaml`)
- **Symbol Mapping**: Some symbols have different tickers on TradingView vs IBKR (e.g., YC vs XC)

## Future Enhancements

As noted in `structure.yaml`, there are several planned enhancements:

### Trading

- Handling near delivery cases where positions are still held close to the Close-Out date
- Automatic tool for IBKR login and session maintenance

### Backtesting

- Fixing drawdown calculations (high priority)
- Adding contract switch dates for missing symbols
- Updating margin requirements on a monthly basis
- Rewriting the switch contract logic to sell on the last possible day before the switch
- Improving visualization of backtesting results
- Advanced parameter optimization techniques

### Strategies

- Aligning TradingView strategies to match backtesting logic
- Updating TradingView strategies to BUY/SELL on n+1 candle after backtesting
- New trading strategies and improvements to existing ones

### Data Analysis

- Google Sheets integration for data analysis

### Development

- Database integration
- User interface
- Code refactoring:
  - Rewriting IBKR logic using classes
  - Rewriting analysis logic using classes
  - Adding comprehensive docstrings
  - Removing unnecessary comments
- Adding end-to-end tests
