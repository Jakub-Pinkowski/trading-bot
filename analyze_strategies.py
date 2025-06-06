from app.backtesting.strategy_analysis import StrategyAnalyzer


def main():
    # Analyze the results using StrategyAnalyzer
    print("Analyzing strategy results from existing parquet file...")

    # Initialize StrategyAnalyzer
    analyzer = StrategyAnalyzer()

    # Get top individual strategies based on a profit factor with a minimum 10 trades
    top_profit_factor = analyzer.get_top_strategies(metric='profit_factor', min_trades=10, limit=30, aggregate=False)

    # Get top individual strategies based on win rate with minimum 10 trades
    top_win_rate = analyzer.get_top_strategies(metric='win_rate', min_trades=10, limit=30, aggregate=False)

    # Get top aggregated strategies based on a profit factor with minimum 10 trades
    top_profit_factor_aggregated = analyzer.get_top_strategies(metric='profit_factor',
                                                               min_trades=10,
                                                               limit=30,
                                                               aggregate=True)

    # Get top aggregated strategies based on win rate with minimum 10 trades
    top_win_rate_aggregated = analyzer.get_top_strategies(metric='win_rate', min_trades=10, limit=30, aggregate=True)


if __name__ == "__main__":
    main()
