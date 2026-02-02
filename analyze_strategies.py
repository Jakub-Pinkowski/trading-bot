"""
Strategy Analysis Script

Analyze backtesting results and export top strategies to CSV files.
Uses StrategyAnalyzer to filter, aggregate, and rank strategies by various metrics.
"""

from app.backtesting.analysis import StrategyAnalyzer


def main():
    """Run strategy analysis and export results to CSV."""
    print("Analyzing strategy results from existing parquet file...\n")

    # Initialize StrategyAnalyzer (loads mass_test_results_all.parquet)
    analyzer = StrategyAnalyzer()

    # ==================== Example 1: Best Profit Factor (4h interval, weighted) ====================
    print("1. Getting top strategies by profit_factor (4h, weighted aggregation)...")
    top_profit_factor_4h = analyzer.get_top_strategies(
        metric='profit_factor',
        min_avg_trades_per_combination=20,
        limit=30,
        aggregate=True,
        interval='4h',
        weighted=True,
        min_slippage=0.05,
        min_symbol_count=3
    )
    print(f"   Found {len(top_profit_factor_4h)} strategies\n")

    # ==================== Example 2: Best Average Return (4h interval, weighted) ====================
    print("2. Getting top strategies by average_trade_return_percentage_of_contract (4h, weighted)...")
    top_avg_return_4h = analyzer.get_top_strategies(
        metric='average_trade_return_percentage_of_contract',
        min_avg_trades_per_combination=10,
        limit=30,
        aggregate=True,
        interval='4h',
        weighted=True,
        min_slippage=0.05,
        min_symbol_count=3
    )
    print(f"   Found {len(top_avg_return_4h)} strategies\n")

    # ==================== Example 3: Best Sharpe Ratio (all intervals, simple aggregation) ====================
    print("3. Getting top strategies by sharpe_ratio (all intervals, simple aggregation)...")
    top_sharpe = analyzer.get_top_strategies(
        metric='sharpe_ratio',
        min_avg_trades_per_combination=15,
        limit=30,
        aggregate=True,
        weighted=False,  # Simple average
        min_slippage=0.05,
        min_symbol_count=2
    )
    print(f"   Found {len(top_sharpe)} strategies\n")

    print("✅ Analysis complete! Check data/backtesting/csv_results/ for exported CSV files.")


# ==================== Available Metrics ====================
"""
Common metrics you can analyze:

Returns:
- profit_factor
- total_return_percentage_of_contract
- average_trade_return_percentage_of_contract
- average_win_percentage_of_contract
- average_loss_percentage_of_contract

Risk Metrics:
- maximum_drawdown_percentage
- sharpe_ratio
- sortino_ratio
- calmar_ratio
- value_at_risk
- expected_shortfall
- ulcer_index

Trade Statistics:
- win_rate
- total_trades
- average_trade_duration_hours
- max_consecutive_wins
- max_consecutive_losses
"""

if __name__ == "__main__":
    main()
