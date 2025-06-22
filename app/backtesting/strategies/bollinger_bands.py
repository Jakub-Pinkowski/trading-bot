from app.backtesting.indicators import calculate_bollinger_bands
from app.backtesting.strategies.base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, period=20, num_std=2, rollover=False, trailing=None, slippage=0):
        super().__init__(rollover=rollover, trailing=trailing, slippage=slippage)
        self.period = period
        self.num_std = num_std

    def add_indicators(self, df):
        bb_data = calculate_bollinger_bands(df['close'], period=self.period, num_std=self.num_std)

        df['middle_band'] = bb_data['middle_band']
        df['upper_band'] = bb_data['upper_band']
        df['lower_band'] = bb_data['lower_band']

        return df

    def generate_signals(self, df):
        """
        Signals:
            1: Long entry (price bounces back from the lower band - crosses back above after being below)
           -1: Short entry (price falls back from the upper band - crosses back below after being above)
            0: No action
        """
        df['signal'] = 0

        # Previous values for crossover detection
        prev_close = df['close'].shift(1)
        prev_lower_band = df['lower_band'].shift(1)
        prev_upper_band = df['upper_band'].shift(1)

        # Buy signal: Price bounces back from the lower band (crosses back above after being below)
        df.loc[(prev_close < prev_lower_band) & (df['close'] >= df['lower_band']), 'signal'] = 1

        # Sell signal: Price falls back from the upper band (crosses back below after being above)
        df.loc[(prev_close > prev_upper_band) & (df['close'] <= df['upper_band']), 'signal'] = -1

        return df
