import numpy as np
import pandas as pd

from app.backtesting.strategies import IchimokuCloudStrategy
from tests.backtesting.strategies.conftest import create_test_df


class TestIchimokuCloudStrategy:
    def test_initialization(self):
        """Test that the Ichimoku Cloud strategy initializes with correct default parameters."""
        strategy = IchimokuCloudStrategy(tenkan_period=9,
                                         kijun_period=26,
                                         senkou_span_b_period=52,
                                         displacement=26,
                                         rollover=False,
                                         trailing=None,
                                         slippage=0,
                                         symbol=None)
        assert strategy.tenkan_period == 9
        assert strategy.kijun_period == 26
        assert strategy.senkou_span_b_period == 52
        assert strategy.displacement == 26

        # Test with custom parameters
        strategy = IchimokuCloudStrategy(tenkan_period=5,
                                         kijun_period=15,
                                         senkou_span_b_period=30,
                                         displacement=15,
                                         rollover=False,
                                         trailing=None,
                                         slippage=0,
                                         symbol=None)
        assert strategy.tenkan_period == 5
        assert strategy.kijun_period == 15
        assert strategy.senkou_span_b_period == 30
        assert strategy.displacement == 15

    def test_add_indicators(self):
        """Test that the add_indicators method correctly adds Ichimoku components to the dataframe."""
        strategy = IchimokuCloudStrategy(tenkan_period=9,
                                         kijun_period=26,
                                         senkou_span_b_period=52,
                                         displacement=26,
                                         rollover=False,
                                         trailing=None,
                                         slippage=0,
                                         symbol=None)
        # Create a dataframe with enough data for Ichimoku calculations
        df = create_test_df(length=200)

        # Apply the strategy's add_indicators method
        df_with_indicators = strategy.add_indicators(df)

        # Verify Ichimoku columns were added
        assert 'tenkan_sen' in df_with_indicators.columns
        assert 'kijun_sen' in df_with_indicators.columns
        assert 'senkou_span_a' in df_with_indicators.columns
        assert 'senkou_span_b' in df_with_indicators.columns
        assert 'chikou_span' in df_with_indicators.columns

        # Verify that we have some valid values
        assert not df_with_indicators['tenkan_sen'].isna().all()
        assert not df_with_indicators['kijun_sen'].isna().all()
        assert not df_with_indicators['senkou_span_a'].isna().all()
        assert not df_with_indicators['senkou_span_b'].isna().all()
        assert not df_with_indicators['chikou_span'].isna().all()

        # Verify initial values are NaN due to calculation requirements
        assert df_with_indicators['tenkan_sen'].iloc[:8].isna().all()
        assert df_with_indicators['kijun_sen'].iloc[:25].isna().all()
        assert df_with_indicators['senkou_span_b'].iloc[:51].isna().all()

    def test_generate_signals_default_params(self):
        """Test that the generate_signals method correctly identifies buy/sell signals with default parameters."""
        strategy = IchimokuCloudStrategy(tenkan_period=9,
                                         kijun_period=26,
                                         senkou_span_b_period=52,
                                         displacement=26,
                                         rollover=False,
                                         trailing=None,
                                         slippage=0,
                                         symbol=None)
        df = create_test_df(length=200)
        df = strategy.add_indicators(df)

        # Apply the strategy's generate_signals method
        df_with_signals = strategy.generate_signals(df)

        # Verify signal column was added
        assert 'signal' in df_with_signals.columns

        # Verify signals are either -1, 0, or 1
        assert df_with_signals['signal'].isin([-1, 0, 1]).all()

        # Find buy signals (Tenkan-sen crosses above Kijun-sen AND price is above the cloud)
        buy_signals = df_with_signals[df_with_signals['signal'] == 1]

        # Find sell signals (Tenkan-sen crosses below Kijun-sen AND price is below the cloud)
        sell_signals = df_with_signals[df_with_signals['signal'] == -1]

        # If we have buy signals, verify the conditions
        if len(buy_signals) > 0:
            for idx in buy_signals.index:
                # Get previous index
                prev_idx = df_with_signals.index[df_with_signals.index.get_loc(idx) - 1]

                # Verify Tenkan-sen crossed above Kijun-sen
                assert df_with_signals.loc[prev_idx, 'tenkan_sen'] <= df_with_signals.loc[prev_idx, 'kijun_sen']
                assert df_with_signals.loc[idx, 'tenkan_sen'] > df_with_signals.loc[idx, 'kijun_sen']

                # Verify price is above the cloud
                assert df_with_signals.loc[idx, 'close'] > df_with_signals.loc[idx, 'senkou_span_a']
                assert df_with_signals.loc[idx, 'close'] > df_with_signals.loc[idx, 'senkou_span_b']

        # If we have sell signals, verify the conditions
        if len(sell_signals) > 0:
            for idx in sell_signals.index:
                # Get previous index
                prev_idx = df_with_signals.index[df_with_signals.index.get_loc(idx) - 1]

                # Verify Tenkan-sen crossed below Kijun-sen
                assert df_with_signals.loc[prev_idx, 'tenkan_sen'] >= df_with_signals.loc[prev_idx, 'kijun_sen']
                assert df_with_signals.loc[idx, 'tenkan_sen'] < df_with_signals.loc[idx, 'kijun_sen']

                # Verify price is below the cloud
                assert df_with_signals.loc[idx, 'close'] < df_with_signals.loc[idx, 'senkou_span_a']
                assert df_with_signals.loc[idx, 'close'] < df_with_signals.loc[idx, 'senkou_span_b']

    def test_generate_signals_custom_params(self):
        """Test that the generate_signals method correctly identifies buy/sell signals with custom parameters."""
        strategy = IchimokuCloudStrategy(tenkan_period=5,
                                         kijun_period=15,
                                         senkou_span_b_period=30,
                                         displacement=15,
                                         rollover=False,
                                         trailing=None,
                                         slippage=0,
                                         symbol=None)
        df = create_test_df(length=200)
        df = strategy.add_indicators(df)

        # Apply the strategy's generate_signals method
        df_with_signals = strategy.generate_signals(df)

        # Verify signal column was added
        assert 'signal' in df_with_signals.columns

        # Verify signals are either -1, 0, or 1
        assert df_with_signals['signal'].isin([-1, 0, 1]).all()

        # Find buy signals (Tenkan-sen crosses above Kijun-sen AND price is above the cloud)
        buy_signals = df_with_signals[df_with_signals['signal'] == 1]

        # Find sell signals (Tenkan-sen crosses below Kijun-sen AND price is below the cloud)
        sell_signals = df_with_signals[df_with_signals['signal'] == -1]

        # If we have buy signals, verify the conditions
        if len(buy_signals) > 0:
            for idx in buy_signals.index:
                # Get previous index
                prev_idx = df_with_signals.index[df_with_signals.index.get_loc(idx) - 1]

                # Verify Tenkan-sen crossed above Kijun-sen
                assert df_with_signals.loc[prev_idx, 'tenkan_sen'] <= df_with_signals.loc[prev_idx, 'kijun_sen']
                assert df_with_signals.loc[idx, 'tenkan_sen'] > df_with_signals.loc[idx, 'kijun_sen']

                # Verify price is above the cloud
                assert df_with_signals.loc[idx, 'close'] > df_with_signals.loc[idx, 'senkou_span_a']
                assert df_with_signals.loc[idx, 'close'] > df_with_signals.loc[idx, 'senkou_span_b']

        # If we have sell signals, verify the conditions
        if len(sell_signals) > 0:
            for idx in sell_signals.index:
                # Get previous index
                prev_idx = df_with_signals.index[df_with_signals.index.get_loc(idx) - 1]

                # Verify Tenkan-sen crossed below Kijun-sen
                assert df_with_signals.loc[prev_idx, 'tenkan_sen'] >= df_with_signals.loc[prev_idx, 'kijun_sen']
                assert df_with_signals.loc[idx, 'tenkan_sen'] < df_with_signals.loc[idx, 'kijun_sen']

                # Verify price is below the cloud
                assert df_with_signals.loc[idx, 'close'] < df_with_signals.loc[idx, 'senkou_span_a']
                assert df_with_signals.loc[idx, 'close'] < df_with_signals.loc[idx, 'senkou_span_b']

    def test_run_end_to_end(self):
        """Test the full strategy workflow from data to trades."""
        strategy = IchimokuCloudStrategy(tenkan_period=9,
                                         kijun_period=26,
                                         senkou_span_b_period=52,
                                         displacement=26,
                                         rollover=False,
                                         trailing=None,
                                         slippage=0,
                                         symbol=None)
        df = create_test_df(length=200)

        # Run the strategy
        trades = strategy.run(df, [])

        # Verify the strategy ran without errors
        assert isinstance(trades, list)

    def test_no_signals_with_flat_prices(self):
        """Test that no signals are generated when prices are flat."""
        strategy = IchimokuCloudStrategy(tenkan_period=9,
                                         kijun_period=26,
                                         senkou_span_b_period=52,
                                         displacement=26,
                                         rollover=False,
                                         trailing=None,
                                         slippage=0,
                                         symbol=None)

        # Create a dataframe with flat prices
        dates = pd.date_range(start='2020-01-01', periods=200)
        data = {
            'open': [100] * 200,
            'high': [101] * 200,
            'low': [99] * 200,
            'close': [100] * 200,
        }
        df = pd.DataFrame(data, index=dates)

        # Add indicators and generate signals
        df = strategy.add_indicators(df)
        df = strategy.generate_signals(df)

        # With flat prices, there should be no signals
        assert (df['signal'] == 0).all()

    def test_with_trailing_stop(self):
        """Test the strategy with trailing stop."""
        strategy = IchimokuCloudStrategy(tenkan_period=9,
                                         kijun_period=26,
                                         senkou_span_b_period=52,
                                         displacement=26,
                                         rollover=False,
                                         trailing=0.05,
                                         slippage=0,
                                         symbol=None)  # 5% trailing stop
        df = create_test_df(length=200)

        # Run the strategy
        trades = strategy.run(df, [])

        # Verify the strategy ran without errors
        assert isinstance(trades, list)

        # If there are trades, verify trailing stop logic
        # Note: The actual trade dictionaries don't have 'exit_reason' or 'direction' keys
        # Instead, they have 'entry_time', 'entry_price', 'exit_time', 'exit_price', and 'side' keys
        for trade in trades:
            if trade['side'] == 'long':
                # For long trades, check if the exit price could be due to a trailing stop
                highest_price = df.loc[trade['entry_time']:trade['exit_time'], 'high'].max()
                # If the exit price is significantly lower than the highest price during the trade,
                # it might be due to a trailing stop
                if trade['exit_price'] < highest_price * 0.97:  # Allow some buffer
                    # Verify the exit price is not lower than what the trailing stop would allow
                    assert trade['exit_price'] >= highest_price * (1 - 0.05 - 0.01)  # 5% trailing + 1% buffer
            else:  # short
                # For short trades, check if the exit price could be due to a trailing stop
                lowest_price = df.loc[trade['entry_time']:trade['exit_time'], 'low'].min()
                # If the exit price is significantly higher than the lowest price during the trade,
                # it might be due to a trailing stop
                if trade['exit_price'] > lowest_price * 1.03:  # Allow some buffer
                    # Verify the exit price is not higher than what the trailing stop would allow
                    assert trade['exit_price'] <= lowest_price * (1 + 0.05 + 0.01)  # 5% trailing + 1% buffer

    def test_with_slippage(self):
        """Test the strategy with slippage."""
        slippage = 0.01  # 1% slippage
        strategy = IchimokuCloudStrategy(tenkan_period=9,
                                         kijun_period=26,
                                         senkou_span_b_period=52,
                                         displacement=26,
                                         rollover=False,
                                         trailing=None,
                                         slippage=slippage,
                                         symbol=None)
        df = create_test_df(length=200)

        # Run the strategy
        trades = strategy.run(df, [])

        # Verify the strategy ran without errors
        assert isinstance(trades, list)

        # If there are trades, verify slippage is applied
        for trade in trades:
            if trade['side'] == 'long':
                # For long trades, entry price should be higher than the close price
                entry_time = trade['entry_time']
                entry_price = trade['entry_price']
                close_price = df.loc[entry_time, 'close']
                assert entry_price >= close_price
                assert entry_price <= close_price * (1 + slippage)
            else:  # short
                # For short trades, entry price should be lower than the close price
                entry_time = trade['entry_time']
                entry_price = trade['entry_price']
                close_price = df.loc[entry_time, 'close']
                assert entry_price <= close_price
                assert entry_price >= close_price * (1 - slippage)

    def test_extreme_market_conditions(self):
        """Test the strategy under extreme market conditions."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=5,  # Use shorter periods to generate more signals
            kijun_period=10,
            senkou_span_b_period=20,
            displacement=10,
            rollover=False,
            trailing=None,
            slippage=0,
            symbol=None
        )

        # Create a dataframe with extreme market conditions
        # We'll create a more complex price pattern to ensure signal generation
        dates = pd.date_range(start='2020-01-01', periods=300)

        # Create a price series with multiple trend changes to trigger signals
        # Start with a flat period
        flat_period = np.ones(50) * 100

        # Then a sharp rise
        rise_period = np.linspace(100, 200, 50)

        # Then a consolidation
        consolidation = np.ones(50) * 200

        # Then a sharp fall
        fall_period = np.linspace(200, 100, 50)

        # Another consolidation
        consolidation2 = np.ones(50) * 100

        # Final rise
        final_rise = np.linspace(100, 150, 50)

        # Combine all periods
        base_prices = np.concatenate([flat_period, rise_period, consolidation, fall_period, consolidation2, final_rise])

        # Add some noise to create crossovers
        noise = np.random.normal(0, 5, 300)
        close_prices = base_prices + noise

        # Create the dataframe
        data = {
            'open': close_prices - 1,
            'high': close_prices + 2,
            'low': close_prices - 2,
            'close': close_prices,
        }

        df = pd.DataFrame(data, index=dates)

        # Add indicators and generate signals
        df = strategy.add_indicators(df)
        df = strategy.generate_signals(df)

        # Verify that signals are generated under extreme market conditions
        # The test output shows that signals are being generated, which is what we want to test

        # Count the number of buy and sell signals
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        total_signals = buy_signals + sell_signals

        # Verify that at least some signals are generated
        assert total_signals > 0, "No signals were generated under extreme market conditions"

        # Verify that both buy and sell signals are generated
        assert buy_signals > 0, "No buy signals were generated"
        assert sell_signals > 0, "No sell signals were generated"

        # Verify that the Ichimoku components are calculated correctly
        # Check that Tenkan-sen and Kijun-sen are not all NaN
        assert not df['tenkan_sen'].isna().all()
        assert not df['kijun_sen'].isna().all()

        # Check that the cloud components are calculated correctly
        assert not df['senkou_span_a'].isna().all()
        assert not df['senkou_span_b'].isna().all()

        # 3. Run the strategy to generate trades
        trades = strategy.run(df, [])

        # Verify the strategy ran without errors
        assert isinstance(trades, list)

    def test_ichimoku_cloud_color(self):
        """Test the Ichimoku cloud color (bullish/bearish) under different market conditions."""
        strategy = IchimokuCloudStrategy(tenkan_period=9,
                                         kijun_period=26,
                                         senkou_span_b_period=52,
                                         displacement=26,
                                         rollover=False,
                                         trailing=None,
                                         slippage=0,
                                         symbol=None)

        # Create a dataframe with different market conditions
        dates = pd.date_range(start='2020-01-01', periods=200)

        # First 100 days: uptrend
        up_close = np.linspace(100, 200, 100)

        # Next 100 days: downtrend
        down_close = np.linspace(200, 100, 100)

        close_prices = np.concatenate([up_close, down_close])

        data = {
            'open': close_prices - 1,
            'high': close_prices + 2,
            'low': close_prices - 2,
            'close': close_prices,
        }

        df = pd.DataFrame(data, index=dates)

        # Add indicators
        df = strategy.add_indicators(df)

        # In an uptrend, the cloud should be bullish (Senkou Span A > Senkou Span B)
        uptrend_idx = 80  # Index in the uptrend
        assert df['senkou_span_a'].iloc[uptrend_idx] > df['senkou_span_b'].iloc[uptrend_idx]

        # In a downtrend, the cloud should be bearish (Senkou Span A < Senkou Span B)
        downtrend_idx = 180  # Index in the downtrend
        assert df['senkou_span_a'].iloc[downtrend_idx] < df['senkou_span_b'].iloc[downtrend_idx]
