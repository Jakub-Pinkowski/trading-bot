from app.backtesting.indicators import calculate_rsi
from app.backtesting.strategies.base.base_strategy import (BaseStrategy,
                                                           precompute_hashes,
                                                           detect_threshold_cross)
from app.backtesting.strategies.base.registry import register_strategy
from app.backtesting.validators import RSIValidator


@register_strategy('rsi', RSIValidator)
class RSIStrategy(BaseStrategy):

    # ==================== Initialization ====================

    def __init__(
        self,
        rsi_period=14,
        lower=30,
        upper=70,
        rollover=False,
        trailing=None,
        slippage=0,
        symbol=None
    ):
        super().__init__(rollover=rollover,
                         trailing=trailing,
                         slippage=slippage,
                         symbol=symbol)
        self.rsi_period = rsi_period
        self.lower = lower
        self.upper = upper

    # ==================== Public Methods ====================

    def add_indicators(self, df):
        # Pre-compute hash once
        hashes = precompute_hashes(df)

        # Calculate RSI using pre-computed hash
        df['rsi'] = calculate_rsi(df['close'], period=self.rsi_period,
                                  prices_hash=hashes['close'])
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
        df.loc[detect_threshold_cross(df['rsi'], self.lower, 'below'), 'signal'] = 1

        # Sell signal: RSI crosses above an upper threshold
        df.loc[detect_threshold_cross(df['rsi'], self.upper, 'above'), 'signal'] = -1

        return df
