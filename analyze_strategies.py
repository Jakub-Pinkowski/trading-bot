"""
Example script demonstrating how to use StrategyAnalyzer to analyze existing strategy test results.
This script shows the workflow of analyzing strategy results from an existing parquet file
and saving them to a human-readable CSV format.
"""

import os

from app.backtesting.strategy_analysis import StrategyAnalyzer
from config import BACKTESTING_DATA_DIR


# TODO [HIGH]: Shave off this file
def main():
    # Analyze the results using StrategyAnalyzer
    print("Analyzing strategy results from existing parquet file...")

    # Initialize StrategyAnalyzer
    analyzer = StrategyAnalyzer()

    # Get top strategies based on a profit factor with minimum 5 trades
    top_strategies = analyzer.get_top_strategies(metric='profit_factor', min_trades=5)

    # Display top 5 strategies
    print("\nTop 5 strategies by profit factor:")
    if not top_strategies.empty:
        print(top_strategies.head(5)[['strategy', 'symbol', 'interval', 'total_trades', 'win_rate', 'profit_factor']])
    else:
        print("No strategies found with minimum 5 trades.")

    # Get top strategies based on win rate with minimum 10 trades
    top_win_rate = analyzer.get_top_strategies(metric='win_rate', min_trades=10)

    # Display the top 5 strategies by win rate
    print("\nTop 5 strategies by win rate:")
    if not top_win_rate.empty:
        print(top_win_rate.head(5)[['strategy', 'symbol', 'interval', 'total_trades', 'win_rate', 'profit_factor']])
    else:
        print("No strategies found with minimum 10 trades.")

    # Save top strategies to CSV files
    print("\nSaving top strategies to CSV files...")

    # Create a directory for CSV files if it doesn't exist
    csv_dir = os.path.join(BACKTESTING_DATA_DIR, 'csv_results')
    os.makedirs(csv_dir, exist_ok=True)

    # Save top strategies by profit factor to CSV (limited to 30 rows)
    if not top_strategies.empty:
        csv_top_profit = os.path.join(csv_dir, 'top_strategies_by_profit_factor.csv')
        # Use the df parameter to save the filtered DataFrame without modifying analyzer.results_df
        analyzer.save_results_to_csv(csv_top_profit, df=top_strategies, limit=30)
        print(f"Top strategies by profit factor saved to {csv_top_profit}")

    # Save top strategies by win rate to CSV (limited to 30 rows)
    if not top_win_rate.empty:
        csv_top_win_rate = os.path.join(csv_dir, 'top_strategies_by_win_rate.csv')
        # Use the df parameter to save the filtered DataFrame without modifying analyzer.results_df
        analyzer.save_results_to_csv(csv_top_win_rate, df=top_win_rate, limit=30)
        print(f"Top strategies by win rate saved to {csv_top_win_rate}")


if __name__ == "__main__":
    main()
