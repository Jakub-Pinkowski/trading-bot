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

    def __init__(self,
                 tested_months=None,
                 symbols=None,
                 intervals=None,
                 output_dir="mass_test_results"):
        """Initialize the mass tester with lists of months, symbols, and intervals to test."""
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

    def add_rsi_tests(self,
                      rsi_periods=None,
                      lower_thresholds=None,
                      upper_thresholds=None,
                      rollovers=None,
                      trailing_stops=None):
        """ Add RSI strategy tests with different parameter combinations. """
        rsi_periods = rsi_periods or [14]
        lower_thresholds = lower_thresholds or [30]
        upper_thresholds = upper_thresholds or [70]
        rollovers = rollovers or [False]
        trailing_stops = trailing_stops or [None]

        # Generate all combinations of parameters
        parameter_combinations = list(itertools.product(
            rsi_periods, lower_thresholds, upper_thresholds, rollovers, trailing_stops
        ))

        # Create strategy instances for each combination
        for rsi_period, lower_threshold, upper_threshold, rollover, trailing_stop in parameter_combinations:
            strategy_name = f"RSI(period={rsi_period},lower={lower_threshold},upper={upper_threshold},rollover={rollover},trailing={trailing_stop})"
            strategy_instance = RSIStrategy(
                rsi_period=rsi_period,
                lower=lower_threshold,
                upper=upper_threshold,
                rollover=rollover,
                trailing=trailing_stop
            )
            self._add_strategy(strategy_name, strategy_instance)

    def add_ema_crossover_tests(self,
                                ema_shorts=None,
                                ema_longs=None,
                                rollovers=None,
                                trailing_stops=None):
        """ Add EMA Crossover strategy tests with different parameter combinations."""
        ema_shorts = ema_shorts or [9]
        ema_longs = ema_longs or [21]
        rollovers = rollovers or [False]
        trailing_stops = trailing_stops or [None]

        # Generate all combinations of parameters
        parameter_combinations = list(itertools.product(
            ema_shorts, ema_longs, rollovers, trailing_stops
        ))

        # Create strategy instances for each combination
        for ema_short, ema_long, rollover, trailing_stop in parameter_combinations:
            strategy_name = f"EMA(short={ema_short},long={ema_long},rollover={rollover},trailing={trailing_stop})"
            strategy_instance = EMACrossoverStrategy(
                ema_short=ema_short,
                ema_long=ema_long,
                rollover=rollover,
                trailing=trailing_stop
            )
            self._add_strategy(strategy_name, strategy_instance)

    def add_macd_tests(self,
                       fast_periods=None,
                       slow_periods=None,
                       signal_periods=None,
                       rollovers=None,
                       trailing_stops=None):
        """ Add MACD strategy tests with different parameter combinations. """
        fast_periods = fast_periods or [12]
        slow_periods = slow_periods or [26]
        signal_periods = signal_periods or [9]
        rollovers = rollovers or [False]
        trailing_stops = trailing_stops or [None]

        # Generate all combinations of parameters
        parameter_combinations = list(itertools.product(
            fast_periods, slow_periods, signal_periods, rollovers, trailing_stops
        ))

        # Create strategy instances for each combination
        for fast_period, slow_period, signal_period, rollover, trailing_stop in parameter_combinations:
            strategy_name = f"MACD(fast={fast_period},slow={slow_period},signal={signal_period},rollover={rollover},trailing={trailing_stop})"
            strategy_instance = MACDStrategy(
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period,
                rollover=rollover,
                trailing=trailing_stop
            )
            self._add_strategy(strategy_name, strategy_instance)

    def add_bollinger_bands_tests(self,
                                  periods=None,
                                  num_stds=None,
                                  rollovers=None,
                                  trailing_stops=None):
        """ Add Bollinger Bands strategy tests with different parameter combinations. """
        periods = periods or [20]
        num_stds = num_stds or [2.0]
        rollovers = rollovers or [False]
        trailing_stops = trailing_stops or [None]

        # Generate all combinations of parameters
        parameter_combinations = list(itertools.product(
            periods, num_stds, rollovers, trailing_stops
        ))

        # Create strategy instances for each combination
        for period, num_std, rollover, trailing_stop in parameter_combinations:
            strategy_name = f"BB(period={period},std={num_std},rollover={rollover},trailing={trailing_stop})"
            strategy_instance = BollingerBandsStrategy(
                period=period,
                num_std=num_std,
                rollover=rollover,
                trailing=trailing_stop
            )
            self._add_strategy(strategy_name, strategy_instance)

    def _add_strategy(self, strategy_name, strategy_instance):
        """ Add a strategy to the list of strategies to test. """
        if not hasattr(self, 'strategies'):
            self.strategies = []

        self.strategies.append((strategy_name, strategy_instance))

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

        if save_results and self.results:
            self._save_results()

        return self.results

    def _save_results(self):
        """Save results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/mass_test_results_{timestamp}.json"

        with open(filename, 'w') as result_file:
            json.dump(self.results, result_file, indent=2)

        print(f"Results saved to {filename}")

    def get_top_strategies(self,
                           metric="profit_factor",
                           ascending=False,
                           min_trades=5):
        """ Get top-performing strategies based on a specific metric."""
        if not self.results:
            raise ValueError("No results available. Run tests first.")

        results_dataframe = pd.DataFrame([
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

        # Filter by minimum trades
        results_dataframe = results_dataframe[results_dataframe["total_trades"] >= min_trades]

        # Sort by metric
        results_dataframe = results_dataframe.sort_values(by=metric, ascending=ascending)

        return results_dataframe

    def compare_strategies(self,
                           group_by=None,
                           metrics=None):
        """ Compare strategies by grouping and averaging metrics. """
        if not self.results:
            raise ValueError("No results available. Run tests first.")

        group_by_columns = group_by or ["strategy"]
        metrics_list = metrics or ["total_trades", "win_rate", "profit_factor", "total_gross_pnl", "avg_trade_return_pct"]

        results_dataframe = pd.DataFrame([
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

        grouped = results_dataframe.groupby(group_by_columns)[metrics_list].mean().reset_index()
        grouped = grouped.sort_values(by="profit_factor", ascending=False)

        return grouped
