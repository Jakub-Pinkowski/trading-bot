from app.backtesting.indicators import calculate_ema
from app.backtesting.strategies.base_strategy import BaseStrategy

# Define parameters
EMA_SHORT = 9
EMA_LONG = 21


class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, ema_short=EMA_SHORT, ema_long=EMA_LONG, rollover=False):
        super().__init__(rollover=rollover)
        self.ema_short = ema_short
        self.ema_long = ema_long

    def add_indicators(self, df):
        df['ema_short'] = calculate_ema(df["close"], period=self.ema_short)
        df['ema_long'] = calculate_ema(df["close"], period=self.ema_long)
        return df

    def generate_signals(self, df):
        """
        Signals:
            1: Long entry (short EMA crosses above long EMA)
           -1: Short entry (short EMA crosses below long EMA)
            0: No action
        """
        df['signal'] = 0

        # Previous values for crossover detection
        prev_ema_short = df['ema_short'].shift(1)
        prev_ema_long = df['ema_long'].shift(1)

        # Buy signal: Short EMA crosses above Long EMA
        df.loc[(prev_ema_short <= prev_ema_long) & (df['ema_short'] > df['ema_long']), 'signal'] = 1

        # Sell signal: Short EMA crosses below Long EMA
        df.loc[(prev_ema_short >= prev_ema_long) & (df['ema_short'] < df['ema_long']), 'signal'] = -1

        return df
