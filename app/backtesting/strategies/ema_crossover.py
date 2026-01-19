from app.backtesting.indicators import calculate_ema
from app.backtesting.strategies.base_strategy import BaseStrategy


class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, ema_short=9, ema_long=21, rollover=False, trailing=None, slippage=0):
        super().__init__(rollover=rollover, trailing=trailing, slippage=slippage)
        self.ema_short = ema_short
        self.ema_long = ema_long

    def add_indicators(self, df):
        df['ema_short'] = calculate_ema(df['close'], period=self.ema_short)
        df['ema_long'] = calculate_ema(df['close'], period=self.ema_long)
        return df

    def generate_signals(self, df):
        """
        Signals:
            1: Long entry (short EMA crosses above long EMA)
           -1: Short entry (short EMA crosses below long EMA)
            0: No action
        """
        df['signal'] = 0

        # Buy signal: Short EMA crosses above Long EMA
        df.loc[self._detect_crossover(df['ema_short'], df['ema_long'], 'above'), 'signal'] = 1

        # Sell signal: Short EMA crosses below Long EMA
        df.loc[self._detect_crossover(df['ema_short'], df['ema_long'], 'below'), 'signal'] = -1

        return df
