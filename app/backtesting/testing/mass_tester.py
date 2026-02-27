import itertools

import yaml

from app.backtesting.strategies.strategy_factory import create_strategy, get_strategy_name
from app.backtesting.testing.orchestrator import run_tests as orchestrator_run_tests
from app.utils.logger import get_logger
from config import DATA_DIR

logger = get_logger('backtesting/testing/mass_tester')

# ==================== Module Paths ====================

HISTORICAL_DATA_DIR = DATA_DIR / "historical_data"
SWITCH_DATES_FILE_PATH = DATA_DIR / "contracts" / "contract_switch_dates.yaml"


# ==================== MassTester Class ====================

class MassTester:
    """
    Framework for mass-testing trading strategies across multiple parameters and timeframes.

    Orchestrates large-scale backtesting by generating all combinations of strategies,
    symbols, intervals, and months, then executing them in parallel. Manages strategy
    configuration, test execution coordination, and result aggregation. Supports skipping
    already-run tests for efficient incremental testing.
    """

    # ==================== Initialization ====================

    def __init__(self, tested_months, symbols, intervals):
        """
        Initialize the mass tester with test parameters.

        Loads contract switch dates from YAML configuration and prepares
        the testing framework for strategy addition and execution.

        Args:
            tested_months: List of month identifiers to test (e.g., ['1!', '2!'])
            symbols: List of futures symbols to test (e.g., ['ZS', 'CL', 'GC'])
            intervals: List of timeframes to test (e.g., ['15m', '1h', '4h', '1d'])
        """
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

    def add_bollinger_bands_tests(
        self,
        periods,
        number_of_standard_deviations_list,
        rollovers,
        trailing_stops,
        slippage_ticks_list
    ):
        """
        Add Bollinger Bands strategy tests with all parameter combinations.

        Creates test instances for every possible combination of the provided parameters.
        Bollinger Bands uses price volatility to identify potential entry/exit points.

        Args:
            periods: List of moving average periods (e.g., [10, 20, 30])
            number_of_standard_deviations_list: List of standard deviation multipliers (e.g., [1.5, 2.0, 2.5])
            rollovers: List of rollover flags (e.g., [True, False]). True = close positions at contract switches
            trailing_stops: List of trailing stop percentages (e.g., [None, 1, 2, 3]). None = disabled
            slippage_ticks_list: List of tick slippage values (e.g., [1, 2, 3])

        Returns:
            None. Strategies are added to the self.strategies list for later execution
        """
        self._add_strategy_tests(
            strategy_type='bollinger',
            param_grid={
                'period': periods,
                'number_of_standard_deviations': number_of_standard_deviations_list,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage_ticks': slippage_ticks_list,
                'symbol': [None]
            }
        )

    def add_ema_crossover_tests(
        self,
        short_ema_periods,
        long_ema_periods,
        rollovers,
        trailing_stops,
        slippage_ticks_list
    ):
        """
        Add EMA Crossover strategy tests with all parameter combinations.

        Creates test instances for every possible combination of the provided parameters.
        EMA Crossover generates signals when a fast EMA crosses above/below a slow EMA.

        Args:
            short_ema_periods: List of short (fast) EMA periods (e.g., [5, 9, 12])
            long_ema_periods: List of long (slow) EMA periods (e.g., [21, 26, 50])
            rollovers: List of rollover flags (e.g., [True, False]). True = close positions at contract switches
            trailing_stops: List of trailing stop percentages (e.g., [None, 1, 2, 3]). None = disabled
            slippage_ticks_list: List of tick slippage values (e.g., [1, 2, 3])

        Returns:
            None. Strategies are added to the self.strategies list for later execution
        """
        self._add_strategy_tests(
            strategy_type='ema',
            param_grid={
                'short_ema_period': short_ema_periods,
                'long_ema_period': long_ema_periods,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage_ticks': slippage_ticks_list,
                'symbol': [None]
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
        slippage_ticks_list
    ):
        """
        Add Ichimoku Cloud strategy tests with all parameter combinations.

        Creates test instances for every possible combination of the provided parameters.
        Ichimoku Cloud is a comprehensive indicator showing support/resistance, momentum, and trend direction.

        Args:
            tenkan_periods: List of Tenkan-sen (conversion line) periods (e.g., [7, 9, 11])
            kijun_periods: List of Kijun-sen (baseline) periods (e.g., [22, 26, 30])
            senkou_span_b_periods: List of Senkou Span B (leading span B) periods (e.g., [44, 52, 60])
            displacements: List of cloud displacement periods (e.g., [22, 26, 30])
            rollovers: List of rollover flags (e.g., [True, False]). True = close positions at contract switches
            trailing_stops: List of trailing stop percentages (e.g., [None, 1, 2, 3]). None = disabled
            slippage_ticks_list: List of tick slippage values (e.g., [1, 2, 3])

        Returns:
            None. Strategies are added to the self.strategies list for later execution
        """
        self._add_strategy_tests(
            strategy_type='ichimoku',
            param_grid={
                'tenkan_period': tenkan_periods,
                'kijun_period': kijun_periods,
                'senkou_span_b_period': senkou_span_b_periods,
                'displacement': displacements,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage_ticks': slippage_ticks_list,
                'symbol': [None]
            }
        )

    def add_macd_tests(
        self,
        fast_periods,
        slow_periods,
        signal_periods,
        rollovers,
        trailing_stops,
        slippage_ticks_list
    ):
        """
        Add MACD strategy tests with all parameter combinations.

        Creates test instances for every possible combination of the provided parameters.
        MACD (Moving Average Convergence Divergence) identifies changes in momentum, direction, and strength.

        Args:
            fast_periods: List of fast EMA periods (e.g., [8, 12, 16])
            slow_periods: List of slow EMA periods (e.g., [21, 26, 30])
            signal_periods: List of signal line periods (e.g., [6, 9, 12])
            rollovers: List of rollover flags (e.g., [True, False]). True = close positions at contract switches
            trailing_stops: List of trailing stop percentages (e.g., [None, 1, 2, 3]). None = disabled
            slippage_ticks_list: List of tick slippage values (e.g., [1, 2, 3])

        Returns:
            None. Strategies are added to the self.strategies list for later execution
        """
        self._add_strategy_tests(
            strategy_type='macd',
            param_grid={
                'fast_period': fast_periods,
                'slow_period': slow_periods,
                'signal_period': signal_periods,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage_ticks': slippage_ticks_list,
                'symbol': [None]
            }
        )

    def add_rsi_tests(
        self,
        rsi_periods,
        lower_thresholds,
        upper_thresholds,
        rollovers,
        trailing_stops,
        slippage_ticks_list
    ):
        """
        Add RSI strategy tests with all parameter combinations.

        Creates test instances for every possible combination of the provided parameters.
        RSI (Relative Strength Index) is a momentum oscillator measuring overbought/oversold conditions.

        Args:
            rsi_periods: List of RSI calculation periods (e.g., [7, 14, 21])
            lower_thresholds: List of oversold thresholds for buy signals (e.g., [20, 30, 40])
            upper_thresholds: List of overbought thresholds for sell signals (e.g., [60, 70, 80])
            rollovers: List of rollover flags (e.g., [True, False]). True = close positions at contract switches
            trailing_stops: List of trailing stop percentages (e.g., [None, 1, 2, 3]). None = disabled
            slippage_ticks_list: List of tick slippage values (e.g., [1, 2, 3])

        Returns:
            None. Strategies are added to the self.strategies list for later execution
        """
        self._add_strategy_tests(
            strategy_type='rsi',
            param_grid={
                'rsi_period': rsi_periods,
                'lower_threshold': lower_thresholds,
                'upper_threshold': upper_thresholds,
                'rollover': rollovers,
                'trailing': trailing_stops,
                'slippage_ticks': slippage_ticks_list,
                'symbol': [None]
            }
        )

    # ==================== Public API - Execution ====================

    def run_tests(self, verbose, max_workers, skip_existing):
        """
        Run all configured strategy tests in parallel across symbols, intervals, and months.

        Executes all strategies added via add_*_tests() methods using multiprocessing
        for parallel execution. Results are saved incrementally and can be analyzed later.

        Args:
            verbose: If True, print detailed progress information during execution.
                    If False, only print summary statistics
            max_workers: Maximum number of parallel worker processes. None = use all CPU cores.
                        Lower values reduce memory usage but increase execution time
            skip_existing: If True, skip tests that already have results in the database.
                          If False, re-run all tests (useful for parameter changes)

        Returns:
            List of result dictionaries. Each dict contains:
            - month: Tested month (e.g., '1!')
            - symbol: Tested symbol (e.g., 'ZS', 'CL', 'GC')
            - interval: Tested timeframe (e.g., '15m', '1h', '4h')
            - strategy: Strategy name with parameters
            - metrics: Dict of performance metrics (profit_factor, win_rate, etc.)
            - timestamp: ISO format timestamp of test execution
            - cache_stats: Cache hit/miss statistics
        """
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
