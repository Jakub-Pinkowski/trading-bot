from app.backtesting.indicators import calculate_ema
from app.backtesting.strategies.base.base_strategy import (BaseStrategy,
                                                           precompute_hashes,
                                                           detect_crossover)
from app.backtesting.strategies.base.registry import register_strategy
from app.backtesting.validators import EMAValidator


@register_strategy('ema', EMAValidator)
class EMACrossoverStrategy(BaseStrategy):

    # ==================== Initialization ====================

    def __init__(
        self,
        ema_short=9,
        ema_long=21,
        rollover=False,
        trailing=None,
        slippage=0,
        symbol=None
    ):
        super().__init__(rollover=rollover,
                         trailing=trailing,
                         slippage=slippage,
                         symbol=symbol)
        self.ema_short = ema_short
        self.ema_long = ema_long

    # ==================== Public Methods ====================

    def add_indicators(self, df):
        # Pre-compute hash once
        hashes = precompute_hashes(df)

        # Calculate both EMAs using pre-computed hash
        df['ema_short'] = calculate_ema(df['close'], period=self.ema_short,
                                        prices_hash=hashes['close'])
        df['ema_long'] = calculate_ema(df['close'], period=self.ema_long,
                                       prices_hash=hashes['close'])
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
        df.loc[detect_crossover(df['ema_short'], df['ema_long'], 'above'), 'signal'] = 1

        # Sell signal: Short EMA crosses below Long EMA
        df.loc[detect_crossover(df['ema_short'], df['ema_long'], 'below'), 'signal'] = -1

        return df
