import concurrent.futures
import io
import itertools
import sys
from datetime import datetime

import pandas as pd
import yaml

from app.backtesting.per_trade_metrics import calculate_trade_metrics
from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy
from app.backtesting.strategies.macd import MACDStrategy
from app.backtesting.strategies.rsi import RSIStrategy
from app.backtesting.summary_metrics import calculate_summary_metrics, print_summary_metrics
from app.utils.file_utils import save_to_parquet
from app.utils.logger import get_logger
from config import HISTORICAL_DATA_DIR, SWITCH_DATES_FILE_PATH, BACKTESTING_DATA_DIR

logger = get_logger()


class MassTester:
    """A framework for mass-testing trading strategies with different parameter combinations."""

    def __init__(self, tested_months, symbols, intervals):
        """Initialize the mass tester with lists of months, symbols, and intervals to test."""
        self.strategies = []
        self.tested_months = tested_months
        self.symbols = symbols
        self.intervals = intervals

        # Load switch dates
        with open(SWITCH_DATES_FILE_PATH) as switch_dates_file:
            self.switch_dates_dict = yaml.safe_load(switch_dates_file)

        # Initialize results storage
        self.results = []

    def add_strategy_tests(self, strategy_class, param_grid, name_template):
        """ Generic method for adding strategy tests with all combinations of given parameters. """

        # Ensure every parameter has at least one value. Use [None] if empty
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

    # NOTE: Tested and approved
    def add_rsi_tests(self, rsi_periods, lower_thresholds, upper_thresholds, rollovers, trailing_stops):
        self.add_strategy_tests(
            strategy_class=RSIStrategy,
            param_grid={
                'rsi_period': rsi_periods,
                'lower': lower_thresholds,
                'upper': upper_thresholds,
                'rollover': rollovers,
                'trailing': trailing_stops
            },
            name_template='RSI(period={rsi_period},lower={lower},upper={upper},rollover={rollover},trailing={trailing})'
        )

    # NOTE: Tested and approved
    def add_ema_crossover_tests(self, ema_shorts, ema_longs, rollovers, trailing_stops):
        self.add_strategy_tests(
            strategy_class=EMACrossoverStrategy,
            param_grid={
                'ema_short': ema_shorts,
                'ema_long': ema_longs,
                'rollover': rollovers,
                'trailing': trailing_stops,
            },
            name_template='EMA(short={ema_short},long={ema_long},rollover={rollover},trailing={trailing})'
        )

    # TODO [MEDIUM]: Still to be tested
    def add_macd_tests(self, fast_periods, slow_periods, signal_periods, rollovers, trailing_stops):
        self.add_strategy_tests(
            strategy_class=MACDStrategy,
            param_grid={
                'fast_period': fast_periods,
                'slow_period': slow_periods,
                'signal_period': signal_periods,
                'rollover': rollovers,
                'trailing': trailing_stops,
            },
            name_template='MACD(fast={fast_period},slow={slow_period},signal={signal_period},rollover={rollover},trailing={trailing})'
        )

    # TODO [MEDIUM]: Still to be tested
    def add_bollinger_bands_tests(self, periods, num_stds, rollovers, trailing_stops):
        self.add_strategy_tests(
            strategy_class=BollingerBandsStrategy,
            param_grid={
                'period': periods,
                'num_std': num_stds,
                'rollover': rollovers,
                'trailing': trailing_stops,
            },
            name_template='BB(period={period},std={num_std},rollover={rollover},trailing={trailing})'
        )

    def run_tests(self, verbose=True, max_workers=None):
        """  Run all tests with the configured parameters in parallel."""
        if not hasattr(self, 'strategies') or not self.strategies:
            logger.error('No strategies added for testing. Use add_*_tests methods first.')
            raise ValueError('No strategies added for testing. Use add_*_tests methods first.')

        total_combinations = len(self.tested_months) * len(self.symbols) * len(self.intervals) * len(self.strategies)
        print(f'Running {total_combinations} test combinations ...')

        # Clear previous results
        self.results = []

        # Prepare all test combinations
        test_combinations = []
        for tested_month in self.tested_months:
            for symbol in self.symbols:
                for interval in self.intervals:
                    if verbose:
                        print(f'Preparing: Month={tested_month}, Symbol={symbol}, Interval={interval}')

                    for strategy_name, strategy_instance in self.strategies:
                        test_combinations.append((
                            tested_month,
                            symbol,
                            interval,
                            strategy_name,
                            strategy_instance,
                            verbose
                        ))

        # Run tests in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_test = {executor.submit(self._run_single_test, test_params): test_params for test_params in
                              test_combinations}

            for future in concurrent.futures.as_completed(future_to_test):
                result = future.result()
                if result:
                    # Print verbose output if available
                    if verbose and 'verbose_output' in result and result['verbose_output']:
                        print(result['verbose_output'])

                    # Remove verbose_output from a result before adding to a result list
                    if 'verbose_output' in result:
                        result_copy = result.copy()
                        del result_copy['verbose_output']
                        self.results.append(result_copy)
                    else:
                        self.results.append(result)

        if self.results:
            self._save_results()

        return self.results

    # --- Private methods ---
    def _run_single_test(self, test_params):
        """Run a single test with the given parameters."""
        tested_month, symbol, interval, strategy_name, strategy_instance, verbose = test_params
        output_buffer = []  # Buffer to collect verbose output

        switch_dates = self.switch_dates_dict.get(symbol, [])
        switch_dates = [pd.to_datetime(switch_date) for switch_date in switch_dates]

        filepath = f'{HISTORICAL_DATA_DIR}/{tested_month}/{symbol}/{symbol}_{interval}.parquet'
        try:
            df = pd.read_parquet(filepath)
        except Exception as error:
            logger.error(f'Failed to read file: {filepath}\nReason: {error}')
            return None

        if verbose:
            output_buffer.append(f'\nRunning strategy: {strategy_name} for {symbol} {interval} {tested_month}')

        trades_list = strategy_instance.run(df, switch_dates)

        trades_with_metrics_list = []
        for trade in trades_list:
            trade_with_metrics = calculate_trade_metrics(trade, symbol)
            trades_with_metrics_list.append(trade_with_metrics)

        if trades_with_metrics_list:
            summary_metrics = calculate_summary_metrics(trades_with_metrics_list)
            if verbose:
                original_stdout = sys.stdout
                string_io = io.StringIO()
                sys.stdout = string_io
                print_summary_metrics(summary_metrics)
                sys.stdout = original_stdout
                output_buffer.append(string_io.getvalue())

            result = {
                'month': tested_month,
                'symbol': symbol,
                'interval': interval,
                'strategy': strategy_name,
                'metrics': summary_metrics,
                'timestamp': datetime.now().isoformat(),
                'verbose_output': '\n'.join(output_buffer) if verbose else None
            }
            return result
        else:
            if verbose:
                output_buffer.append(f'No trades generated by strategy {strategy_name} for {symbol} {interval} {tested_month}')

            logger.info(f'No trades generated by strategy {strategy_name} for {symbol} {interval} {tested_month}')
            # Return a complete result dictionary even when no trades are generated
            return {
                'month': tested_month,
                'symbol': symbol,
                'interval': interval,
                'strategy': strategy_name,
                'metrics': {},  # Empty metrics
                'timestamp': datetime.now().isoformat(),
                'verbose_output': '\n'.join(output_buffer) if verbose else None
            }

    def _results_to_dataframe(self):
        """Convert results to a pandas DataFrame."""

        if not self.results:
            logger.warning('No results available to convert to DataFrame.')
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    'month': result['month'],
                    'symbol': result['symbol'],
                    'interval': result['interval'],
                    'strategy': result['strategy'],
                    # Trade counts
                    'total_trades': result['metrics']['total_trades'],
                    'win_rate': result['metrics']['win_rate'],
                    # Percentage-based metrics (for normalized comparison)
                    'profit_factor': result['metrics']['profit_factor'],
                    'total_return_percentage_of_margin': result['metrics'].get('total_return_percentage_of_margin', 0),
                    'average_trade_return_percentage_of_margin': result['metrics'][
                        'average_trade_return_percentage_of_margin'],
                    'average_win_percentage_of_margin': result['metrics'].get('average_win_percentage_of_margin', 0),
                    'average_loss_percentage_of_margin': result['metrics'].get('average_loss_percentage_of_margin', 0),
                    'maximum_drawdown_percentage': result['metrics']['maximum_drawdown_percentage'],
                    # Dollar-based metrics (for reference)
                    'total_net_pnl': result['metrics']['total_net_pnl'],
                    'avg_trade_net_pnl': result['metrics']['avg_trade_net_pnl']
                }
                for result in self.results
            ]
        )

    def _save_results(self):
        """Save results to one big parquet file."""
        try:
            # Convert results to DataFrame
            results_df = self._results_to_dataframe()
            if not results_df.empty:
                # Save all results to one big parquet file with unique entries
                parquet_filename = f'{BACKTESTING_DATA_DIR}/mass_test_results_all.parquet'
                save_to_parquet(results_df, parquet_filename)
                print(f'Results saved to {parquet_filename}')
            else:
                print('No results to save.')
        except Exception as error:
            logger.error(f'Failed to save results: {error}')
