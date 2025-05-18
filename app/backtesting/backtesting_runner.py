from app.backtesting.data_fetching import fetch_data

def run_backtesting():
    print("run backtesting")
    symbol = "ZW"

    market_data = fetch_data(symbol)
