from app.backtesting.strategy_analysis import StrategyAnalyzer


def main():
    # Analyze the results using StrategyAnalyzer
    print("Analyzing strategy results from existing parquet file...")

    # Initialize StrategyAnalyzer
    analyzer = StrategyAnalyzer()

    # Get top strategies for a profit factor for a specific timeframe
    top_profit_factor_1h = analyzer.get_top_strategies(metric='profit_factor',
                                                       min_trades=30,
                                                       limit=30,
                                                       aggregate=True,
                                                       interval='1h')

    # Get top strategies for a total return for a specific timeframe
    top_profit_factor_1h = analyzer.get_top_strategies(metric='total_return_percentage_of_margin',
                                                       min_trades=30,
                                                       limit=30,
                                                       aggregate=True,
                                                       interval='1h')


if __name__ == "__main__":
    main()
