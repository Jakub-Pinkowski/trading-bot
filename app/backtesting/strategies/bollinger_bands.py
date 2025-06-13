from app.backtesting.indicators import calculate_bollinger_bands
from app.backtesting.strategies.base_strategy import BaseStrategy


# TODO [MEDIUM]: To be tested
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
            1: Long entry (price crosses below a lower band)
           -1: Short entry (price crosses above an upper band)
            0: No action
        """
        df['signal'] = 0

        # Previous values for crossover detection
        prev_close = df['close'].shift(1)
        prev_lower_band = df['lower_band'].shift(1)
        prev_upper_band = df['upper_band'].shift(1)

        # Buy signal: Price crosses below a lower band
        df.loc[(prev_close >= prev_lower_band) & (df['close'] < df['lower_band']), 'signal'] = 1

        # Sell signal: Price crosses above an upper band
        df.loc[(prev_close <= prev_upper_band) & (df['close'] > df['upper_band']), 'signal'] = -1

        return df
