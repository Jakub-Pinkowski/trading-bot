from app.backtesting.indicators import calculate_ichimoku
from app.backtesting.strategies.base.base_strategy import (BaseStrategy,
                                                           precompute_hashes,
                                                           detect_crossover)


class IchimokuCloudStrategy(BaseStrategy):

    # ==================== Initialization ====================

    def __init__(
        self,
        tenkan_period=9,
        kijun_period=26,
        senkou_span_b_period=52,
        displacement=26,
        rollover=False,
        trailing=None,
        slippage=0
    ):
        super().__init__(rollover=rollover, trailing=trailing, slippage=slippage)
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period
        self.displacement = displacement

    # ==================== Public Methods ====================

    def add_indicators(self, df):
        # Pre-compute all hashes once
        hashes = precompute_hashes(df)

        # Calculate Ichimoku using pre-computed hashes
        ichimoku_data = calculate_ichimoku(
            df['high'],
            df['low'],
            df['close'],
            tenkan_period=self.tenkan_period,
            kijun_period=self.kijun_period,
            senkou_span_b_period=self.senkou_span_b_period,
            displacement=self.displacement,
            high_hash=hashes['high'],
            low_hash=hashes['low'],
            close_hash=hashes['close']
        )

        df['tenkan_sen'] = ichimoku_data['tenkan_sen']
        df['kijun_sen'] = ichimoku_data['kijun_sen']
        df['senkou_span_a'] = ichimoku_data['senkou_span_a']
        df['senkou_span_b'] = ichimoku_data['senkou_span_b']
        df['chikou_span'] = ichimoku_data['chikou_span']

        return df

    def generate_signals(self, df):
        """
        Signals:
            1: Long entry (Tenkan-sen crosses above Kijun-sen AND price is above the cloud)
           -1: Short entry (Tenkan-sen crosses below Kijun-sen AND price is below the cloud)
            0: No action
        """
        df['signal'] = 0

        # Detect Tenkan-Kijun crossovers
        tenkan_crosses_above = detect_crossover(df['tenkan_sen'], df['kijun_sen'], 'above')
        tenkan_crosses_below = detect_crossover(df['tenkan_sen'], df['kijun_sen'], 'below')

        # Determine if the price is above or below the cloud
        above_cloud = (df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b'])
        below_cloud = (df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b'])

        # Buy signal: Tenkan-sen crosses above Kijun-sen AND price is above the cloud
        df.loc[tenkan_crosses_above & above_cloud, 'signal'] = 1

        # Sell signal: Tenkan-sen crosses below Kijun-sen AND price is below the cloud
        df.loc[tenkan_crosses_below & below_cloud, 'signal'] = -1

        return df
