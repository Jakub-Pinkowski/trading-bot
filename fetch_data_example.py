"""
Fetch historical futures data from TradingView.

Simple script to download and update historical market data
for configured symbols and timeframes.
"""
from app.backtesting import DataFetcher

# ==================== Configuration ====================

BASE_SYMBOLS = ['ZS']
CONTRACT_SUFFIX = '1!'
EXCHANGE = 'CBOT'
INTERVALS = ['5m', '15m', '30m', '1h', '2h', '4h', '1d']


# ==================== Main ====================

def main():
    """Fetch historical data for all configured symbols and intervals."""
    fetcher = DataFetcher(
        symbols=BASE_SYMBOLS,
        contract_suffix=CONTRACT_SUFFIX,
        exchange=EXCHANGE
    )

    fetcher.fetch_all_data(intervals=INTERVALS)


if __name__ == "__main__":
    main()
