"""
Example script demonstrating DataFetcher usage.
This script shows how to use the new class-based DataFetcher
to download and manage historical futures data from TradingView.
"""
from app.backtesting import DataFetcher, create_data_fetcher
# ==================== Configuration ====================
# Symbols to fetch (base symbols without contract suffix)
BASE_SYMBOLS = ['MZL', 'MET', 'RTY']
# Contract suffix (1! = front month, 2! = second month, etc.)
CONTRACT_SUFFIX = '1!'
# Exchange
EXCHANGE = 'CBOT'
# Intervals to fetch
INTERVALS = ['5m', '15m', '30m', '1h', '2h', '4h', '1d']
# ==================== Usage Examples ====================
def example_fetch_all_data():
    """Example: Fetch all data for all configured symbols and intervals."""
    print("Example 1: Fetching all data for all symbols and intervals")
    print("-" * 60)
    # Create fetcher instance
    fetcher = DataFetcher(
        symbols=BASE_SYMBOLS,
        contract_suffix=CONTRACT_SUFFIX,
        exchange=EXCHANGE
    )
    # Fetch all data
    fetcher.fetch_all_data(intervals=INTERVALS)
    print("\nData fetch completed!")
def example_fetch_specific_symbol():
    """Example: Fetch data for a specific symbol only."""
    print("\nExample 2: Fetching data for a specific symbol")
    print("-" * 60)
    # Create fetcher using factory function
    fetcher = create_data_fetcher(
        symbols=['MZL'],
        contract_suffix=CONTRACT_SUFFIX,
        exchange=EXCHANGE
    )
    # Fetch only 1h and 1d intervals for this symbol
    fetcher.fetch_symbol_data('MZL', intervals=['1h', '1d'])
    print("\nSpecific symbol fetch completed!")
def example_get_data_info():
    """Example: Get information about stored data."""
    print("\nExample 3: Getting data information")
    print("-" * 60)
    fetcher = create_data_fetcher(
        symbols=BASE_SYMBOLS,
        contract_suffix=CONTRACT_SUFFIX
    )
    # Get info for each symbol
    for symbol in BASE_SYMBOLS:
        print(f"\n{symbol} data info:")
        for interval in ['1h', '4h', '1d']:
            info = fetcher.get_data_info(symbol, interval)
            if info:
                print(f"  {interval}: {info['row_count']} rows, "
                      f"{info['first_date']} to {info['last_date']}")
            else:
                print(f"  {interval}: No data available")
# ==================== Main Execution ====================
def main():
    """Main function to run examples."""
    print("=" * 60)
    print("DataFetcher Usage Examples")
    print("=" * 60)
    # Run the first example by default
    # Uncomment others to run different examples
    example_fetch_all_data()
    # Other examples (uncomment to run):
    # example_fetch_specific_symbol()
    # example_get_data_info()
if __name__ == "__main__":
    main()
