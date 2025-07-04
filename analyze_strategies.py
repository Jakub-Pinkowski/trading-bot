from app.backtesting.strategy_analysis import StrategyAnalyzer


def main():
    # Analyze the results using StrategyAnalyzer
    print("Analyzing strategy results from existing parquet file...")

    # Initialize StrategyAnalyzer
    analyzer = StrategyAnalyzer()

    # Get top strategies for a profit factor for a specific timeframe (weighted)
    top_profit_factor_1h_weighted = analyzer.get_top_strategies(metric='profit_factor',
                                                                min_avg_trades_per_combination=20,
                                                                limit=30,
                                                                aggregate=True,
                                                                interval='4h',
                                                                weighted=True,
                                                                min_slippage=0.05,
                                                                min_symbol_count=3)

    # Get top strategies for an average win percentage of a margin for a specific timeframe (weighted)
    top_profit_factor_1h = analyzer.get_top_strategies(metric='average_trade_return_percentage_of_margin',
                                                       min_avg_trades_per_combination=10,
                                                       limit=30,
                                                       aggregate=True,
                                                       interval='4h',
                                                       weighted=True,
                                                       min_slippage=0.05,
                                                       min_symbol_count=3)

    # Get top strategies for a total return for a specific timeframe
    # top_profit_factor_1h = analyzer.get_top_strategies(metric='total_return_percentage_of_margin',
    #                                                    min_trades=30,
    #                                                    limit=30,
    #                                                    aggregate=True,
    #                                                    interval='1h',
    #                                                    min_slippage=0.1)


if __name__ == "__main__":
    main()
