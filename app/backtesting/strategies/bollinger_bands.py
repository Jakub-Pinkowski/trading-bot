from app.backtesting.indicators import calculate_bollinger_bands
from app.backtesting.strategies.base.base_strategy import (BaseStrategy,
                                                           precompute_hashes,
                                                           detect_crossover)


class BollingerBandsStrategy(BaseStrategy):

    # ==================== Initialization ====================

    def __init__(
        self,
        period=20,
        number_of_standard_deviations=2,
        rollover=False,
        trailing=None,
        slippage=0,
        symbol=None
    ):
        super().__init__(rollover=rollover,
                         trailing=trailing,
                         slippage=slippage,
                         symbol=symbol)
        self.period = period
        self.number_of_standard_deviations = number_of_standard_deviations

    # ==================== Public Methods ====================

    def add_indicators(self, df):
        # Pre-compute hash once
        hashes = precompute_hashes(df)

        # Calculate Bollinger Bands using pre-computed hash
        bb_data = calculate_bollinger_bands(df['close'], period=self.period,
                                            number_of_standard_deviations=self.number_of_standard_deviations,
                                            prices_hash=hashes['close'])

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

        # Buy signal: Price crosses back above lower band
        df.loc[detect_crossover(df['close'], df['lower_band'], 'above'), 'signal'] = 1

        # Sell signal: Price crosses back below upper band
        df.loc[detect_crossover(df['close'], df['upper_band'], 'below'), 'signal'] = -1

        return df
