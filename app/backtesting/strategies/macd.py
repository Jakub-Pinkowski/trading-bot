from app.backtesting.indicators import calculate_macd
from app.backtesting.strategies.base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
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
        buy_signals = (prev_macd_line <= prev_signal_line) & (df['macd_line'] > df['signal_line'])
        df.loc[buy_signals, 'signal'] = 1

        # Sell signal: MACD line crosses the below signal line
        sell_signals = (prev_macd_line >= prev_signal_line) & (df['macd_line'] < df['signal_line'])
        df.loc[sell_signals, 'signal'] = -1

        return df
