from app.backtesting.strategy_analysis import StrategyAnalyzer


def main():
    # Analyze the results using StrategyAnalyzer
    print("Analyzing strategy results from existing parquet file...")

    # Initialize StrategyAnalyzer
    analyzer = StrategyAnalyzer()

    # Get top strategies based on a profit factor with minimum 5 trades
    top_strategies = analyzer.get_top_strategies(metric='profit_factor', min_trades=10, limit=30)

    # Get top strategies based on win rate with minimum 10 trades
    top_win_rate = analyzer.get_top_strategies(metric='win_rate', min_trades=10, limit=30)


if __name__ == "__main__":
    main()
