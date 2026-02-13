"""
Strategy Analysis Script

Analyze backtesting results and export top strategies to CSV files.
Uses StrategyAnalyzer to filter, aggregate, and rank strategies by various metrics.
"""

from app.backtesting.analysis import StrategyAnalyzer

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

# ==================== Configuration ====================

# List of analyses to run
# Add, remove, or modify analyses as needed
ANALYSES = [
    {
        'name': 'Best Profit Factor (4h, weighted)',
        'metric': 'profit_factor',
        'min_avg_trades_per_combination': 20,
        'limit': 30,
        'aggregate': True,
        'interval': '4h',
        'weighted': True,
        'min_slippage_ticks': 2,
        'min_symbol_count': 3
    },
    {
        'name': 'Best Average Return (4h, weighted)',
        'metric': 'average_trade_return_percentage_of_contract',
        'min_avg_trades_per_combination': 10,
        'limit': 30,
        'aggregate': True,
        'interval': '4h',
        'weighted': True,
        'min_slippage_ticks': 2,
        'min_symbol_count': 3
    },
    {
        'name': 'Best Sharpe Ratio (all intervals, simple)',
        'metric': 'sharpe_ratio',
        'min_avg_trades_per_combination': 15,
        'limit': 30,
        'aggregate': True,
        'weighted': False,
        'min_slippage_ticks': 2,
        'min_symbol_count': 2
    },
]


# ==================== Main ====================

def main():
    """Run strategy analysis and export results to CSV."""
    print("Analyzing strategy results from existing parquet file...\n")
    print(f"Running {len(ANALYSES)} analyses...\n")

    # Initialize StrategyAnalyzer (loads mass_test_results_all.parquet)
    analyzer = StrategyAnalyzer()

    # Run all configured analyses
    for i, analysis_config in enumerate(ANALYSES, start=1):
        # Create a copy to avoid mutating the original
        analysis = analysis_config.copy()
        name = analysis.pop('name')
        print(f"{i}. {name}...")
        analyzer.get_top_strategies(**analysis)
        print()

    print("✅ Analysis complete! Check data/backtesting/csv_results/ for exported CSV files.")


if __name__ == "__main__":
    main()
