from app.backtesting.indicators import calculate_macd
from app.backtesting.strategies.base.base_strategy import (BaseStrategy,
                                                           precompute_hashes,
                                                           detect_crossover)


class MACDStrategy(BaseStrategy):

    # ==================== Initialization ====================

    def __init__(
        self,
        fast_period=12,
        slow_period=26,
        signal_period=9,
        rollover=False,
        trailing=None,
        slippage=0
    ):
        super().__init__(rollover=rollover, trailing=trailing, slippage=slippage)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    # ==================== Public Methods ====================

    def add_indicators(self, df):
        # Pre-compute hash once
        hashes = precompute_hashes(df)

        # Calculate MACD using pre-computed hash
        macd_data = calculate_macd(
            df['close'],
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period,
            prices_hash=hashes['close']
        )

        df['macd_line'] = macd_data['macd_line']
        df['signal_line'] = macd_data['signal_line']
        df['histogram'] = macd_data['histogram']

        return df

    def generate_signals(self, df):
        """
        Signals:
            1: Long entry (MACD line crosses above signal line)
           -1: Short entry (MACD line crosses below signal line)
            0: No action
        """
        df['signal'] = 0

        # Buy signal: MACD line crosses above signal line
        df.loc[detect_crossover(df['macd_line'], df['signal_line'], 'above'), 'signal'] = 1

        # Sell signal: MACD line crosses below signal line
        df.loc[detect_crossover(df['macd_line'], df['signal_line'], 'below'), 'signal'] = -1

        return df
