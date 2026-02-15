"""
Fetch historical futures data from TradingView.

Bulk download and update historical market data for all configured
futures symbols organized by category.
"""
from app.backtesting import DataFetcher
from futures_config import CATEGORIES, get_exchange_for_symbol

# ==================== Configuration ====================

CONTRACT_SUFFIX = '1!'
INTERVALS = ['5m']

# Select which categories to fetch (see CATEGORIES.keys() for all available)
# Available categories: Grains, Softs, Energy, Metals, Crypto, Index, Forex
CATEGORIES_TO_FETCH = ['Grains', 'Softs']

# Build symbol list from selected categories
SYMBOLS_TO_FETCH = []
for category in CATEGORIES_TO_FETCH:
    SYMBOLS_TO_FETCH.extend(CATEGORIES[category])


# ==================== Main ====================

def main():
    """Fetch historical data for all configured symbols."""
    print(f"Fetching data for {len(SYMBOLS_TO_FETCH)} symbols...")
    print(f"Categories: {', '.join(CATEGORIES_TO_FETCH)}")
    print(f"Intervals: {', '.join(INTERVALS)}")
    print("-" * 60)

    # Group symbols by exchange for efficient fetching
    symbols_by_exchange = {}
    for symbol in SYMBOLS_TO_FETCH:
        exchange = get_exchange_for_symbol(symbol)
        if exchange not in symbols_by_exchange:
            symbols_by_exchange[exchange] = []
        symbols_by_exchange[exchange].append(symbol)

    # Fetch data for each exchange
    for exchange, symbols in symbols_by_exchange.items():
        print(f"\nFetching {len(symbols)} symbols from {exchange}: {symbols}")

        fetcher = DataFetcher(
            symbols=symbols,
            contract_suffix=CONTRACT_SUFFIX,
            exchange=exchange
        )

        fetcher.fetch_all_data(intervals=INTERVALS)

    print("\n" + "=" * 60)
    print("✅ Bulk data fetch completed!")


if __name__ == "__main__":
    main()
