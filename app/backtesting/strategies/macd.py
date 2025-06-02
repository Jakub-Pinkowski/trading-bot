from app.backtesting.indicators import calculate_macd
from app.backtesting.strategies.base_strategy import BaseStrategy

# Define parameters
FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9


class MACDStrategy(BaseStrategy):
    def __init__(self, fast_period=FAST_PERIOD, slow_period=SLOW_PERIOD, signal_period=SIGNAL_PERIOD, rollover=False, trailing=None):
        super().__init__(rollover=rollover, trailing=trailing)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def add_indicators(self, df):
        macd_data = calculate_macd(
            df['close'],
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period
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

        # Previous values for crossover detection
        prev_macd_line = df['macd_line'].shift(1)
        prev_signal_line = df['signal_line'].shift(1)

        # Buy signal: MACD line crosses the above signal line
        df.loc[(prev_macd_line <= prev_signal_line) & (df['macd_line'] > df['signal_line']), 'signal'] = 1

        # Sell signal: MACD line crosses the below signal line
        df.loc[(prev_macd_line >= prev_signal_line) & (df['macd_line'] < df['signal_line']), 'signal'] = -1

        return df
