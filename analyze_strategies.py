"""
Strategy Analysis Script

Analyze backtesting results and export top strategies to CSV files.
Uses StrategyAnalyzer to filter, aggregate, and rank strategies by various metrics.
"""

from app.backtesting.analysis import StrategyAnalyzer

# ==================== Configuration Guide ====================
"""
Configuration Parameters:

'name' (str): 
    Description label for the analysis (for logging purposes only)
    
'metric' (str): 
    Metric to rank strategies by
    Options: 
        Returns:
            - 'profit_factor'
            - 'total_return_percentage_of_contract'
            - 'average_trade_return_percentage_of_contract'
            - 'average_win_percentage_of_contract'
            - 'average_loss_percentage_of_contract'
        Risk Metrics:
            - 'maximum_drawdown_percentage'
            - 'sharpe_ratio'
            - 'sortino_ratio'
            - 'calmar_ratio'
            - 'value_at_risk'
            - 'expected_shortfall'
            - 'ulcer_index'
        Trade Statistics:
            - 'win_rate'
            - 'total_trades'
            - 'average_trade_duration_hours'
            - 'max_consecutive_wins'
            - 'max_consecutive_losses'
    
'min_avg_trades_per_combination' (int): 
    Minimum average trades per symbol/interval combination required
    Examples: 10, 15, 20, 30
    Higher values = more statistically significant results
    
'limit' (int): 
    Maximum number of top strategies to export to CSV
    Examples: 10, 20, 30, 50
    Use None for unlimited
    
'aggregate' (bool): 
    Whether to aggregate results across symbols and intervals
    - True: Combine all test runs into single metrics per strategy
    - False: Show individual results per symbol/interval combination
    Recommended: True for overall performance analysis
    
'interval' (str or None): 
    Filter by specific timeframe
    Options: '5m', '15m', '30m', '1h', '2h', '4h', '1d'
    Use None to include all intervals
    
'weighted' (bool): 
    Use weighted aggregation (only applies when aggregate=True)
    - True: Weight by number of trades (strategies with more trades have more influence)
    - False: Simple average across all symbol/interval combinations
    Recommended: True for more realistic results
    
'min_slippage_ticks' (int or None): 
    Minimum slippage level in ticks to filter by
    Examples: 1, 2, 3, 4
    Use None for no slippage filter
    Purpose: Filter to strategies tested with realistic transaction costs
    
'min_symbol_count' (int or None): 
    Minimum number of unique symbols strategy must work on
    Examples: 2, 3, 4
    Use None for no symbol count requirement
    Purpose: Filter out strategies that only work on one or two symbols (avoid overfitting)

'one_per_group' (bool):
    Filter to only one symbol per correlated group (e.g., keep ZC, exclude XC/MZC)
    - True: Avoid pseudo-replication from mini/micro contracts tracking same market
    - False: Include all symbols (may inflate symbol counts and skew aggregation)
    Recommended: True for accurate analysis
    Explanation: ZC (standard corn), XC (mini corn), and MZC (micro corn) all track
                 the same corn market with nearly identical candle patterns. Including
                 all three would give 3x weight to corn strategies vs other markets.
    
Optional Parameters (not shown in examples below):
    'symbol' (str or None): 
        Filter by specific symbol
        Examples: 'ES', 'NQ', 'ZS', 'CL', 'GC'
        Use None to include all symbols
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
        'interval': None,
        'weighted': True,
        'min_slippage_ticks': 2,
        'min_symbol_count': 3,
        'one_per_group': True
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
        'min_symbol_count': 3,
        'one_per_group': True
    },
    {
        'name': 'Best Sharpe Ratio (all intervals, simple)',
        'metric': 'sharpe_ratio',
        'min_avg_trades_per_combination': 15,
        'limit': 30,
        'aggregate': True,
        'weighted': False,
        'min_slippage_ticks': 2,
        'min_symbol_count': 2,
        'one_per_group': True
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
