"""
Automated TradingView Data Fetcher for CI/CD.

Fetches historical data for ALL TradingView-compatible symbols across
all configured intervals. Designed for scheduled runs via GitHub Actions.
"""
from app.backtesting import DataFetcher
from futures_config import CATEGORIES, get_exchange_for_symbol

# ==================== Configuration ====================

CONTRACT_SUFFIX = '1!'

# Fetch ALL intervals for comprehensive data coverage
INTERVALS = ['5m']

# Fetch ALL available categories
ALL_CATEGORIES = list(CATEGORIES.keys())

# Build symbol list from all categories
ALL_SYMBOLS = []
for category in ALL_CATEGORIES:
    ALL_SYMBOLS.extend(CATEGORIES[category])


# ==================== Main ====================

def main():
    """Fetch historical data for all TradingView-compatible symbols."""
    print("=" * 70)
    print("🚀 AUTOMATED TRADINGVIEW DATA FETCH")
    print("=" * 70)
    print(f"Total symbols: {len(ALL_SYMBOLS)}")
    print(f"Categories: {', '.join(ALL_CATEGORIES)}")
    print(f"Intervals: {', '.join(INTERVALS)}")
    print(f"Contract: {CONTRACT_SUFFIX}")
    print("-" * 70)

    # Group symbols by exchange for efficient fetching
    symbols_by_exchange = {}
    for symbol in ALL_SYMBOLS:
        exchange = get_exchange_for_symbol(symbol)
        if exchange not in symbols_by_exchange:
            symbols_by_exchange[exchange] = []
        symbols_by_exchange[exchange].append(symbol)

    print(f"\nExchanges: {len(symbols_by_exchange)}")
    for exchange, symbols in symbols_by_exchange.items():
        print(f"  - {exchange}: {len(symbols)} symbols")

    print("\n" + "=" * 70)
    print("STARTING DATA FETCH")
    print("=" * 70)

    # Fetch data for each exchange
    for idx, (exchange, symbols) in enumerate(symbols_by_exchange.items(), 1):
        print(f"\n[{idx}/{len(symbols_by_exchange)}] Exchange: {exchange}")
        print(f"Symbols: {', '.join(symbols)}")
        print("-" * 70)

        try:
            fetcher = DataFetcher(
                symbols=symbols,
                contract_suffix=CONTRACT_SUFFIX,
                exchange=exchange
            )

            fetcher.fetch_all_data(intervals=INTERVALS)
            print(f"✅ {exchange} completed successfully")

        except Exception as e:
            print(f"❌ {exchange} failed: {e}")
            # Continue with other exchanges even if one fails
            continue

    print("\n" + "=" * 70)
    print("✅ AUTOMATED DATA FETCH COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
