from app.backtesting.indicators import calculate_rsi
from app.backtesting.strategies.base.base_strategy import BaseStrategy, precompute_hashes, detect_threshold_cross


class RSIStrategy(BaseStrategy):

    # ==================== Initialization ====================

    def __init__(self, rsi_period, lower_threshold, upper_threshold, rollover, trailing, slippage_ticks, symbol):
        super().__init__(rollover=rollover, trailing=trailing, slippage_ticks=slippage_ticks, symbol=symbol)
        self.rsi_period = rsi_period
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    # ==================== Name Formatting ====================

    @staticmethod
    def format_name(rsi_period, lower_threshold, upper_threshold, rollover, trailing, slippage_ticks, **kwargs):
        """Generate standardized strategy name."""
        return f'RSI(period={rsi_period},lower={lower_threshold},upper={upper_threshold},rollover={rollover},trailing={trailing},slippage={slippage_ticks})'

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
        df.loc[detect_threshold_cross(df['rsi'], self.lower_threshold, 'below'), 'signal'] = 1

        # Sell signal: RSI crosses above an upper threshold
        df.loc[detect_threshold_cross(df['rsi'], self.upper_threshold, 'above'), 'signal'] = -1

        return df
