import itertools
import json
import os
from datetime import datetime

import pandas as pd
import yaml

from app.backtesting.per_trade_metrics import calculate_trade_metrics
from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy
from app.backtesting.strategies.macd import MACDStrategy
from app.backtesting.strategies.rsi import RSIStrategy
from app.backtesting.summary_metrics import calculate_summary_metrics, print_summary_metrics
from app.utils.logger import get_logger
from config import HISTORICAL_DATA_DIR, SWITCH_DATES_FILE_PATH

logger = get_logger()


class MassTester:
    """A framework for mass-testing trading strategies with different parameter combinations."""

    def __init__(self, tested_months=None, symbols=None, intervals=None, output_dir="mass_test_results"):
        """Initialize the mass tester with lists of months, symbols, and intervals to test."""
        self.strategies = None
        self.tested_months = tested_months or ["1!"]
        self.symbols = symbols or ["ZW"]
        self.intervals = intervals or ["4h"]
        self.output_dir = output_dir

        # Load switch dates
        with open(SWITCH_DATES_FILE_PATH) as switch_dates_file:
            self.switch_dates_dict = yaml.safe_load(switch_dates_file)

        # Create an output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize results storage
        self.results = []

    def add_strategy_tests(self, strategy_class, param_grid: dict, name_template: str):
        """ Generic method for adding strategy tests with all combinations of given parameters. """

        # Ensure every parameter has at least one value (use [None] if empty)
        for param_name, values_list in param_grid.items():
            if not values_list:
                param_grid[param_name] = [None]

        # Extract the names of the parameters to preserve order.
        param_names = list(param_grid.keys())

        # Generate all possible combinations of parameter values using Cartesian product.
        param_value_combinations = itertools.product(*(param_grid[param_name] for param_name in param_names))

        # Loop over parameter values to create strategy instances and store them in self.strategies.
        for param_values in param_value_combinations:
            strategy_params = dict(zip(param_names, param_values))
            strategy_name = name_template.format(**strategy_params)
            strategy_instance = strategy_class(**strategy_params)

            self.strategies.append((strategy_name, strategy_instance))

    def add_rsi_tests(self, rsi_periods=None, lower_thresholds=None, upper_thresholds=None, rollovers=None, trailing_stops=None):
        self.add_strategy_tests(
            strategy_class=RSIStrategy,
            param_grid={
                'rsi_period': rsi_periods or [14],
                'lower': lower_thresholds or [30],
                'upper': upper_thresholds or [70],
                'rollover': rollovers or [False],
                'trailing': trailing_stops or [None]
            },
            name_template="RSI(period={rsi_period},lower={lower},upper={upper},rollover={rollover},trailing={trailing})"
        )

    def add_ema_crossover_tests(self, ema_shorts=None, ema_longs=None, rollovers=None, trailing_stops=None):
        self.add_strategy_tests(
            strategy_class=EMACrossoverStrategy,
            param_grid={
                'ema_short': ema_shorts or [9],
                'ema_long': ema_longs or [21],
                'rollover': rollovers or [False],
                'trailing': trailing_stops or [None]
            },
            name_template="EMA(short={ema_short},long={ema_long},rollover={rollover},trailing={trailing})"
        )

    def add_macd_tests(self, fast_periods=None, slow_periods=None, signal_periods=None, rollovers=None, trailing_stops=None):
        self.add_strategy_tests(
            strategy_class=MACDStrategy,
            param_grid={
                'fast_period': fast_periods or [12],
                'slow_period': slow_periods or [26],
                'signal_period': signal_periods or [9],
                'rollover': rollovers or [False],
                'trailing': trailing_stops or [None]
            },
            name_template="MACD(fast={fast_period},slow={slow_period},signal={signal_period},rollover={rollover},trailing={trailing})"
        )

    def add_bollinger_bands_tests(self, periods=None, num_stds=None, rollovers=None, trailing_stops=None):
        self.add_strategy_tests(
            strategy_class=BollingerBandsStrategy,
            param_grid={
                'period': periods or [20],
                'num_std': num_stds or [2.0],
                'rollover': rollovers or [False],
                'trailing': trailing_stops or [None]
            },
            name_template="BB(period={period},std={num_std},rollover={rollover},trailing={trailing})"
        )

    def run_tests(self, verbose=True, save_results=True):
        """ Run all tests with the configured parameters. """
        if not hasattr(self, 'strategies') or not self.strategies:
            raise ValueError("No strategies added for testing. Use add_*_tests methods first.")

        total_combinations = len(self.tested_months) * len(self.symbols) * len(self.intervals) * len(self.strategies)
        if verbose:
            print(f"Running {total_combinations} test combinations...")

        # Clear previous results
        self.results = []

        # Run tests for all combinations
        for tested_month in self.tested_months:
            for symbol in self.symbols:
                switch_dates = self.switch_dates_dict.get(symbol, [])
                switch_dates = [pd.to_datetime(switch_date) for switch_date in switch_dates]

                for interval in self.intervals:
                    if verbose:
                        print(f"\nProcessing: Month={tested_month}, Symbol={symbol}, Interval={interval}")

                    filepath = f"{HISTORICAL_DATA_DIR}/{tested_month}/{symbol}/{symbol}_{interval}.parquet"
                    try:
                        df = pd.read_parquet(filepath)
                    except Exception as error:
                        logger.error(f"Failed to read file: {filepath}\nReason: {error}")
                        continue

                    for strategy_name, strategy_instance in self.strategies:
                        if verbose:
                            print(f"\nRunning strategy: {strategy_name}")

                        trades_list = strategy_instance.run(df, switch_dates)

                        trades_with_metrics_list = []
                        for trade in trades_list:
                            trade_with_metrics = calculate_trade_metrics(trade, symbol)
                            trades_with_metrics_list.append(trade_with_metrics)

                        if trades_with_metrics_list:
                            summary_metrics = calculate_summary_metrics(trades_with_metrics_list)
                            if verbose:
                                print_summary_metrics(summary_metrics)

                            result = {
                                "month": tested_month,
                                "symbol": symbol,
                                "interval": interval,
                                "strategy": strategy_name,
                                "metrics": summary_metrics,
                                "timestamp": datetime.now().isoformat()
                            }
                            self.results.append(result)
                        else:
                            if verbose:
                                print("No trades generated by this strategy.")
                            logger.info(f"No trades generated by strategy {strategy_name} for {symbol} {interval} {tested_month}")

        if save_results and self.results:
            self._save_results()

        return self.results

    def _results_to_dataframe(self):
        """Convert results to a pandas DataFrame."""
        if not self.results:
            logger.warning("No results available to convert to DataFrame.")
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "month": result["month"],
                "symbol": result["symbol"],
                "interval": result["interval"],
                "strategy": result["strategy"],
                "total_trades": result["metrics"]["total_trades"],
                "win_rate": result["metrics"]["win_rate"],
                "profit_factor": result["metrics"]["profit_factor"],
                "total_gross_pnl": result["metrics"]["total_gross_pnl"],
                "avg_trade_return_pct": result["metrics"]["avg_trade_return_pct"]
            }
            for result in self.results
        ])

    def _save_results(self):
        """Save results to JSON and CSV files."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save individual test results to JSON
            json_filename = f"{self.output_dir}/mass_test_results_{timestamp}.json"
            with open(json_filename, 'w') as result_file:
                json.dump(self.results, result_file, indent=2)

            # Save summary to CSV
            results_df = self._results_to_dataframe()
            if not results_df.empty:
                # Save detailed results
                csv_filename = f"{self.output_dir}/mass_test_results_{timestamp}.csv"
                results_df.to_csv(csv_filename, index=False)

                # Save summary grouped by strategy
                summary_df = results_df.groupby('strategy').agg({
                    'total_trades': 'sum',
                    'win_rate': 'mean',
                    'profit_factor': 'mean',
                    'total_gross_pnl': 'sum',
                    'avg_trade_return_pct': 'mean'
                }).reset_index()
                summary_filename = f"{self.output_dir}/mass_test_summary_{timestamp}.csv"
                summary_df.to_csv(summary_filename, index=False)

                print(f"Results saved to {json_filename}, {csv_filename}, and summary to {summary_filename}")
            else:
                print(f"Results saved to {json_filename}")
        except Exception as error:
            logger.error(f"Failed to save results: {error}")

    def get_top_strategies(self, metric="profit_factor", ascending=False, min_trades=5):
        """ Get top-performing strategies based on a specific metric."""
        if not self.results:
            logger.error("No results available. Run tests first.")
            raise ValueError("No results available. Run tests first.")

        results_dataframe = self._results_to_dataframe()

        # Filter by minimum trades
        results_dataframe = results_dataframe[results_dataframe["total_trades"] >= min_trades]

        # Sort by metric
        results_dataframe = results_dataframe.sort_values(by=metric, ascending=ascending)

        return results_dataframe

    def compare_strategies(self, group_by=None, metrics=None):
        """ Compare strategies by grouping and averaging metrics. """
        if not self.results:
            logger.error("No results available. Run tests first.")
            raise ValueError("No results available. Run tests first.")

        group_by_columns = group_by or ["strategy"]
        metrics_list = metrics or ["total_trades", "win_rate", "profit_factor", "total_gross_pnl", "avg_trade_return_pct"]

        results_dataframe = self._results_to_dataframe()

        grouped = results_dataframe.groupby(group_by_columns)[metrics_list].mean().reset_index()
        grouped = grouped.sort_values(by="profit_factor", ascending=False)

        return grouped
