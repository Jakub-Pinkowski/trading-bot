from app.backtesting.indicators import calculate_rsi
from app.backtesting.strategies.base_strategy import BaseStrategy

# Define parameters
RSI_PERIOD = 14
LOWER = 30
UPPER = 70


class RSIStrategy(BaseStrategy):
    def __init__(self, rsi_period=RSI_PERIOD, lower=LOWER, upper=UPPER, rollover=False):
        super().__init__(rollover=rollover)
        self.rsi_period = rsi_period
        self.lower = lower
        self.upper = upper

    def add_indicators(self, df):
        df['rsi'] = calculate_rsi(df["close"], period=self.rsi_period)
        return df

    def generate_signals(self, df):
        """
        Signals:
            1: Long entry
           -1: Short entry
            0: No action
        """
        df['signal'] = 0
        prev_rsi = df['rsi'].shift(1)

        # Buy signal: RSI crosses below a lower threshold
        df.loc[(prev_rsi > self.lower) & (df['rsi'] <= self.lower), 'signal'] = 1

        # Sell signal: RSI crosses above an upper threshold
        df.loc[(prev_rsi < self.upper) & (df['rsi'] >= self.upper), 'signal'] = -1

        return df
