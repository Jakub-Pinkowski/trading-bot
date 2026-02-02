from app.backtesting.indicators import calculate_ichimoku_cloud
from app.backtesting.strategies.base.base_strategy import BaseStrategy, precompute_hashes, detect_crossover


class IchimokuCloudStrategy(BaseStrategy):

    # ==================== Initialization ====================

    def __init__(
        self,
        tenkan_period,
        kijun_period,
        senkou_span_b_period,
        displacement,
        rollover,
        trailing,
        slippage,
        symbol
    ):
        super().__init__(rollover=rollover, trailing=trailing, slippage=slippage, symbol=symbol)
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period
        self.displacement = displacement

    # ==================== Name Formatting ====================

    @staticmethod
    def format_name(
        tenkan_period, kijun_period, senkou_span_b_period, displacement,
        rollover, trailing, slippage, **kwargs
    ):
        """Generate standardized strategy name."""
        return f'Ichimoku(tenkan={tenkan_period},kijun={kijun_period},senkou_b={senkou_span_b_period},displacement={displacement},rollover={rollover},trailing={trailing},slippage={slippage})'

    # ==================== Public Methods ====================

    def add_indicators(self, df):
        # Pre-compute all hashes once
        hashes = precompute_hashes(df)

        # Calculate Ichimoku using pre-computed hashes
        ichimoku_data = calculate_ichimoku_cloud(
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
