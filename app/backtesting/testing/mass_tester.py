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

    def add_strategy_tests(self, strategy_type, param_grid):
        """
        Generic method to add tests for ANY strategy.
        
        Usage:
            tester.add_strategy_tests('rsi', {
                'rsi_period': [13, 14, 21],
                'lower': [20, 30],
                'upper': [70, 80],
                'rollover': [False],
                'trailing': [None, 1, 2],
                'slippage': [0.05]
            })
        
        Arguments:
            strategy_type: Strategy identifier (e.g., 'rsi', 'ema')
            param_grid: Dictionary mapping parameter names to lists of values
        """
        self._add_strategy_tests(strategy_type, param_grid)

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
