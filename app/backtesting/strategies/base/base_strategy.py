import pandas as pd

from app.backtesting.strategies.base.contract_switch_handler import ContractSwitchHandler
from app.backtesting.strategies.base.position_manager import PositionManager
from app.backtesting.strategies.base.trailing_stop_manager import TrailingStopManager
from app.utils.backtesting_utils.indicators_utils import hash_series

# ==================== Strategy Execution Constants ====================

# INDICATOR_WARMUP_PERIOD: Number of initial candles to skip before generating signals
# Rationale:
#   - Technical indicators (MA, EMA, RSI, etc.) need historical data to stabilize
#   - Example: A 100-period moving average needs 100 bars before it's valid
#   - Prevents generating signals based on incomplete/unstable indicator values
#   - 100 is chosen as a safe default that covers most common indicator periods:
#     * RSI (typically 14 periods)
#     * MACD (26 slow period)
#     * Bollinger Bands (typically 20 periods)
#     * Longer EMAs (up to 100 periods)
# Usage: Strategy execution starts from index INDICATOR_WARMUP_PERIOD in the DataFrame
INDICATOR_WARMUP_PERIOD = 100


# ==================== Helper Functions for Subclasses ====================

def precompute_hashes(df):
    """
    Pre-compute hashes for commonly used price series.

    This function calculates hashes once for price series that are typically
    passed to multiple indicator functions. This eliminates redundant hash
    computations and significantly improves performance when using multiple
    indicators.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary with pre-computed hashes for common price series:
        {
            'close': hash of df['close'],
            'high': hash of df['high'],
            'low': hash of df['low'],
            'open': hash of df['open'],
            'volume': hash of df['volume']
        }

    Example:
        def add_indicators(self, df):
            from app.backtesting.strategies.base.base_strategy import precompute_hashes
            # Pre-compute all hashes once
            hashes = precompute_hashes(df)

            # Pass pre-computed hashes to indicators (no redundant hashing!)
            df['rsi'] = calculate_rsi(df['close'], period=14,
                                     prices_hash=hashes['close'])
            df['rsi_long'] = calculate_rsi(df['close'], period=21,
                                          prices_hash=hashes['close'])
            df['ema'] = calculate_ema(df['close'], period=9,
                                     prices_hash=hashes['close'])
            return df
    """
    hashes = {}

    # Pre-compute hashes for OHLCV columns that exist in the DataFrame
    for col in ['close', 'high', 'low', 'open', 'volume']:
        if col in df.columns:
            hashes[col] = hash_series(df[col])

    return hashes


def detect_crossover(series1, series2, direction):
    """
    Detect when series1 crosses series2.

    Args:
        series1 (pd.Series): First series (e.g., fast EMA, MACD line)
        series2 (pd.Series): Second series (e.g., slow EMA, signal line)
        direction (str): 'above' for bullish crossover, 'below' for bearish crossover

    Returns:
        pd.Series: Boolean series indicating crossover points
    """
    prev_series1 = series1.shift(1)
    prev_series2 = series2.shift(1)

    if direction == 'above':
        # Series1 crosses above series2 (bullish)
        return (prev_series1 <= prev_series2) & (series1 > series2)
    else:  # direction == 'below'
        # Series1 crosses below series2 (bearish)
        return (prev_series1 >= prev_series2) & (series1 < series2)


def detect_threshold_cross(series, threshold, direction):
    """
    Detect when a series crosses a threshold value.

    Args:
        series (pd.Series): The series to check (e.g., RSI, price)
        threshold (float): The threshold value to cross
        direction (str): 'below' for crossing downward, 'above' for crossing upward

    Returns:
        pd.Series: Boolean series indicating threshold cross-points
    """
    prev_series = series.shift(1)

    if direction == 'below':
        # Series crosses below a threshold (bearish)
        return (prev_series > threshold) & (series <= threshold)
    else:  # direction == 'above'
        # Series crosses above a threshold (bullish)
        return (prev_series < threshold) & (series >= threshold)


class BaseStrategy:
    # ==================== Initialization ====================

    def __init__(self, rollover, trailing, slippage_ticks, symbol):
        """
        Initialize the base strategy.

        Args:
            rollover: Whether to handle contract rollovers
            trailing: Trailing stop percentage (None = disabled)
            slippage_ticks: Slippage in ticks (e.g., 2 = 2 ticks)
            symbol: The futures symbol (e.g., 'ZC', 'GC')
        """
        self.rollover = rollover
        self.trailing = trailing

        # Delegate to managers
        self.position_manager = PositionManager(slippage_ticks, symbol, trailing)
        self.trailing_stop_manager = TrailingStopManager(trailing) if trailing else None
        self.switch_handler = ContractSwitchHandler(None, rollover)

        # State variables for strategy execution
        self.prev_row = None
        self.prev_time = None
        self.queued_signal = None

    # ==================== Public API ====================

    def run(self, df, switch_dates):
        """
        Run the strategy.
        """
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        trades = self._extract_trades(df, switch_dates)

        return trades

    def add_indicators(self, df):
        """Add indicators to the dataframe. To be implemented by subclasses."""
        raise NotImplementedError('Subclasses must implement add_indicators method')

    def generate_signals(self, df):
        """
        Generate signals based on indicators. To be implemented by subclasses.
        Signals:
            1: Long entry
           -1: Short entry
            0: No action
        """
        raise NotImplementedError('Subclasses must implement generate_signals method')

    # ==================== Private Methods ====================

    # --- Trade Extraction & Execution ---

    def _extract_trades(self, df, switch_dates):
        """Extract trades based on signals.

        SIGNAL EXECUTION TIMING:
        ----------------------
        Signals are generated based on indicator values calculated from each bar's close.
        The signal is queued and executed at the OPEN of the NEXT bar, creating a 1-bar delay.

        Example Timeline (15-minute bars):
            10:15:00 - Bar N closes at 1074.25
                     → RSI = 29.8 (crosses below 30)
                     → Signal = 1 (BUY) - Generated and queued

            10:30:00 - Bar N+1 opens at 1074.25
                     → Queued signal executed
                     → Position opened at 1074.25 + slippage
        This is standard backtesting practice because:
        1. You can only see the signal AFTER the bar closes
        2. Execution happens at the next available opportunity (next bar's open)
        3. Slippage is applied to account for:
           - Execution delay between signal and fill
           - Bid-ask spread
           - Market movement during order placement

        This approach is:
        - Industry standard (used by TradingView, Backtrader, etc.)
        - Realistic and conservative
        - Not actual look-ahead bias (proper time sequencing)

        Args:
            df: DataFrame with OHLCV data and signals
            switch_dates: List of contract rollover dates

        Returns:
            List of trade dictionaries with entry/exit details
        """
        # Initialize managers
        self.switch_handler.set_switch_dates(switch_dates)
        self.switch_handler.reset()
        self.position_manager.reset()

        # Reset state variables
        self.prev_row = None
        self.prev_time = None
        self.queued_signal = None

        # Counter to skip the first candles for indicator warm-up
        candle_count = 0

        for idx, row in df.iterrows():
            current_time = pd.to_datetime(idx)
            signal = row['signal']
            price_open = row['open']
            price_high = row['high']
            price_low = row['low']

            # Increment candle counter
            candle_count += 1

            # Skip signal processing for the first candles to allow indicators to warm up
            if candle_count <= INDICATOR_WARMUP_PERIOD:
                self.prev_time = current_time
                self.prev_row = row
                continue

            # Handle trailing stop logic if enabled
            if self.trailing_stop_manager:
                self.trailing_stop_manager.handle_trailing_stop(self.position_manager, idx, price_high, price_low)

            # Handle contract switches - all switch logic is now in ContractSwitchHandler
            skip_signal = self.switch_handler.handle_contract_switch(
                current_time,
                self.position_manager,
                idx,
                price_open,
                self.prev_time,
                self.prev_row
            )

            # Skip signal for this bar if we are in a rollover position, and we are about to switch
            if skip_signal:
                self.prev_time = current_time
                self.prev_row = row
                continue

            # Execute queued signal from the previous bar
            self._execute_queued_signal(idx, price_open)

            # Set/overwrite queued_signal for next bar execution
            if signal != 0:
                self.queued_signal = signal

            self.prev_time = current_time
            self.prev_row = row

        return self.position_manager.get_trades()

    def _execute_queued_signal(self, idx, price_open):
        """Execute queued signal from the previous bar"""

        if self.queued_signal is not None:
            flip = None
            if self.queued_signal == 1 and self.position_manager.position != 1:
                flip = 1
            elif self.queued_signal == -1 and self.position_manager.position != -1:
                flip = -1

            if flip is not None:
                # Close if currently in position
                if self.position_manager.has_open_position():
                    self.position_manager.close_position(idx, price_open, switch=False)
                # Open a new position at this (current) bar
                self.position_manager.open_position(flip, idx, price_open)

            # Reset after using
            self.queued_signal = None
