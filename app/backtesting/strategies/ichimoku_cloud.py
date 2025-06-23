from app.backtesting.indicators import calculate_ichimoku
from app.backtesting.strategies.base_strategy import BaseStrategy


class IchimokuCloudStrategy(BaseStrategy):
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

    def add_indicators(self, df):
        ichimoku_data = calculate_ichimoku(
            df['high'],
            df['low'],
            df['close'],
            tenkan_period=self.tenkan_period,
            kijun_period=self.kijun_period,
            senkou_span_b_period=self.senkou_span_b_period,
            displacement=self.displacement
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

        # Previous values for crossover detection
        prev_tenkan = df['tenkan_sen'].shift(1)
        prev_kijun = df['kijun_sen'].shift(1)

        # Determine if the price is above or below the cloud
        above_cloud = (df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b'])
        below_cloud = (df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b'])

        # Buy signal: Tenkan-sen crosses above Kijun-sen AND price is above the cloud
        buy_condition = (prev_tenkan <= prev_kijun) & (df['tenkan_sen'] > df['kijun_sen']) & above_cloud
        df.loc[buy_condition, 'signal'] = 1

        # Sell signal: Tenkan-sen crosses below Kijun-sen AND price is below the cloud
        sell_condition = (prev_tenkan >= prev_kijun) & (df['tenkan_sen'] < df['kijun_sen']) & below_cloud
        df.loc[sell_condition, 'signal'] = -1

        return df
