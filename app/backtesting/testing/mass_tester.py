import itertools

import yaml

from app.backtesting.strategy_factory import create_strategy, get_strategy_name
from app.backtesting.testing.orchestrator import run_tests as orchestrator_run_tests
from app.utils.logger import get_logger
from config import SWITCH_DATES_FILE_PATH

logger = get_logger('backtesting/testing/mass_tester')


# ==================== MassTester Class ====================

class MassTester:
    """A framework for mass-testing trading strategies with different parameter combinations."""

    # ==================== Initialization ====================

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

    # ==================== Public API - Strategy Configuration ====================

    def add_bollinger_bands_tests(self, periods, num_stds, rollovers, trailing_stops, slippages=None):
        """Add Bollinger Bands strategy tests with all parameter combinations."""
        self._add_strategy_tests(
            strategy_type='bollinger',
            param_grid={
                'period': periods,
                'num_std': num_stds,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage': slippages
            }
        )

    def add_ema_crossover_tests(self, ema_shorts, ema_longs, rollovers, trailing_stops, slippages=None):
        """Add EMA Crossover strategy tests with all parameter combinations."""
        self._add_strategy_tests(
            strategy_type='ema',
            param_grid={
                'ema_short': ema_shorts,
                'ema_long': ema_longs,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage': slippages
            }
        )

    def add_ichimoku_cloud_tests(
        self,
        tenkan_periods,
        kijun_periods,
        senkou_span_b_periods,
        displacements,
        rollovers,
        trailing_stops,
        slippages=None
    ):
        """Add Ichimoku Cloud strategy tests with all parameter combinations."""
        self._add_strategy_tests(
            strategy_type='ichimoku',
            param_grid={
                'tenkan_period': tenkan_periods,
                'kijun_period': kijun_periods,
                'senkou_span_b_period': senkou_span_b_periods,
                'displacement': displacements,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage': slippages
            }
        )

    def add_macd_tests(self, fast_periods, slow_periods, signal_periods, rollovers, trailing_stops, slippages=None):
        """Add MACD strategy tests with all parameter combinations."""
        self._add_strategy_tests(
            strategy_type='macd',
            param_grid={
                'fast_period': fast_periods,
                'slow_period': slow_periods,
                'signal_period': signal_periods,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage': slippages
            }
        )

    def add_rsi_tests(self, rsi_periods, lower_thresholds, upper_thresholds, rollovers, trailing_stops, slippages=None):
        """Add RSI strategy tests with all parameter combinations."""
        self._add_strategy_tests(
            strategy_type='rsi',
            param_grid={
                'rsi_period': rsi_periods,
                'lower': lower_thresholds,
                'upper': upper_thresholds,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage': slippages
            }
        )

    # ==================== Public API - Execution ====================

    def run_tests(self, verbose=True, max_workers=None, skip_existing=True):
        """Run all tests with the configured parameters in parallel."""
        return orchestrator_run_tests(
            self,
            verbose=verbose,
            max_workers=max_workers,
            skip_existing=skip_existing
        )

    # ==================== Private Methods ====================

    def _add_strategy_tests(self, strategy_type, param_grid):
        """Generic method for adding strategy tests with all combinations of given parameters."""

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
            strategy_name = get_strategy_name(strategy_type, **strategy_params)
            strategy_instance = create_strategy(strategy_type, **strategy_params)

            self.strategies.append((strategy_name, strategy_instance))
