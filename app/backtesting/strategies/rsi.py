from app.backtesting.indicators import calculate_rsi
from app.backtesting.strategies.base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    def __init__(self, rsi_period=14, lower=30, upper=70, rollover=False, trailing=None, slippage=0):
        super().__init__(rollover=rollover, trailing=trailing, slippage=slippage)
        self.rsi_period = rsi_period
        self.lower = lower
        self.upper = upper

    def add_indicators(self, df):
        df['rsi'] = calculate_rsi(df['close'], period=self.rsi_period)
        return df

    def generate_signals(self, df):
        """
        Signals:
            1: Long entry (RSI crosses below a lower threshold)
           -1: Short entry (RSI crosses above an upper threshold)
            0: No action
        """
        df['signal'] = 0

        # Buy signal: RSI crosses below a lower threshold
        df.loc[self._detect_threshold_cross(df['rsi'], self.lower, 'below'), 'signal'] = 1

        # Sell signal: RSI crosses above an upper threshold
        df.loc[self._detect_threshold_cross(df['rsi'], self.upper, 'above'), 'signal'] = -1

        return df
