from datetime import datetime, timedelta

import pandas as pd

from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy


# Helper function to create a test dataframe with price patterns suitable for Bollinger Bands testing
def create_test_df(length=50):
    dates = [datetime.now() + timedelta(days=i) for i in range(length)]

    # Create a price series that will generate clear Bollinger Bands signals
    close_prices = []

    # Start with a steady uptrend
    for i in range(20):
        close_prices.append(100 + i)

    # Then a sharp move above the upper band
    for i in range(5):
        close_prices.append(120 + i * 3)

    # Then a reversion to the mean
    for i in range(10):
        close_prices.append(135 - i * 2)

    # Then a sharp move below the lower band
    for i in range(5):
        close_prices.append(115 - i * 3)

    # Then a reversion to the mean
    for i in range(10):
        close_prices.append(100 + i)

    # Ensure the length matches the requested length
    while len(close_prices) < length:
        close_prices.append(close_prices[-1])

    # Create OHLC data
    data = {
        'open': close_prices,
        'high': [p + 1 for p in close_prices],
        'low': [p - 1 for p in close_prices],
        'close': close_prices,
    }

    df = pd.DataFrame(data, index=dates)
    return df


class TestBollingerBandsStrategy:
    def test_initialization(self):
        """Test that the Bollinger Bands strategy initializes with correct default parameters."""
        strategy = BollingerBandsStrategy()
        assert strategy.period == 20
        assert strategy.num_std == 2

        # Test with custom parameters
        strategy = BollingerBandsStrategy(period=10, num_std=3, rollover=True, trailing=2.0, slippage=1.0)
        assert strategy.period == 10
        assert strategy.num_std == 3
        assert strategy.rollover == True
        assert strategy.trailing == 2.0
        assert strategy.slippage == 1.0

    def test_add_indicators(self):
        """Test that the add_indicators method correctly adds Bollinger Bands to the dataframe."""
        strategy = BollingerBandsStrategy()
        # Create a larger dataframe to ensure we have valid Bollinger Bands values
        df = create_test_df(length=100)

        # Apply the strategy's add_indicators method
        df_with_indicators = strategy.add_indicators(df)

        # Verify Bollinger Bands columns were added
        assert 'middle_band' in df_with_indicators.columns
        assert 'upper_band' in df_with_indicators.columns
        assert 'lower_band' in df_with_indicators.columns

        # Clear the cache to ensure we're not using cached values
        from app.backtesting.cache.indicators_cache import indicator_cache
        indicator_cache.clear()

        # Apply the strategy's add_indicators method again
        df_with_indicators = strategy.add_indicators(df)

        # Skip the initial NaN values and verify that we have some valid Bollinger Bands values
        valid_middle_band = df_with_indicators['middle_band'].iloc[strategy.period:].dropna()
        assert len(valid_middle_band) > 0, "No valid middle band values calculated"

        # Verify that upper band is always higher than middle band
        assert (df_with_indicators['upper_band'].iloc[strategy.period:] >=
                df_with_indicators['middle_band'].iloc[strategy.period:]).all()

        # Verify that lower band is always lower than middle band
        assert (df_with_indicators['lower_band'].iloc[strategy.period:] <=
                df_with_indicators['middle_band'].iloc[strategy.period:]).all()

        # Verify bands are NaN for the first few periods
        assert df_with_indicators['middle_band'].iloc[:strategy.period - 1].isna().all()
        assert df_with_indicators['upper_band'].iloc[:strategy.period - 1].isna().all()
        assert df_with_indicators['lower_band'].iloc[:strategy.period - 1].isna().all()

    def test_generate_signals_default_params(self):
        """Test that the generate_signals method correctly identifies buy/sell signals with default parameters."""
        strategy = BollingerBandsStrategy()
        df = create_test_df()
        df = strategy.add_indicators(df)

        # Apply the strategy's generate_signals method
        df_with_signals = strategy.generate_signals(df)

        # Verify signal column was added
        assert 'signal' in df_with_signals.columns

        # Find where price bounces back from the lower band (buy signals)
        buy_signals = df_with_signals[
            (df_with_signals['close'].shift(1) < df_with_signals['lower_band'].shift(1)) &
            (df_with_signals['close'] >= df_with_signals['lower_band'])
            ]

        # Find where price falls back from the upper band (sell signals)
        sell_signals = df_with_signals[
            (df_with_signals['close'].shift(1) > df_with_signals['upper_band'].shift(1)) &
            (df_with_signals['close'] <= df_with_signals['upper_band'])
            ]

        # Verify all buy signals have signal value of 1
        assert (buy_signals['signal'] == 1).all()

        # Verify all sell signals have signal value of -1
        assert (sell_signals['signal'] == -1).all()

        # Verify no other signals exist
        other_signals = df_with_signals[
            ~df_with_signals.index.isin(buy_signals.index) &
            ~df_with_signals.index.isin(sell_signals.index)
            ]
        assert (other_signals['signal'] == 0).all()

    def test_generate_signals_custom_params(self):
        """Test that the generate_signals method correctly identifies buy/sell signals with custom parameters."""
        # Use more extreme thresholds
        strategy = BollingerBandsStrategy(period=10, num_std=3)
        df = create_test_df()
        df = strategy.add_indicators(df)

        # Apply the strategy's generate_signals method
        df_with_signals = strategy.generate_signals(df)

        # Find where price bounces back from the lower band (buy signals)
        buy_signals = df_with_signals[
            (df_with_signals['close'].shift(1) < df_with_signals['lower_band'].shift(1)) &
            (df_with_signals['close'] >= df_with_signals['lower_band'])
            ]

        # Find where price falls back from the upper band (sell signals)
        sell_signals = df_with_signals[
            (df_with_signals['close'].shift(1) > df_with_signals['upper_band'].shift(1)) &
            (df_with_signals['close'] <= df_with_signals['upper_band'])
            ]

        # Verify all buy signals have signal value of 1
        assert (buy_signals['signal'] == 1).all()

        # Verify all sell signals have signal value of -1
        assert (sell_signals['signal'] == -1).all()

    def test_run_end_to_end(self):
        """Test the full strategy workflow from data to trades."""
        strategy = BollingerBandsStrategy()
        df = create_test_df()

        # Run the strategy
        trades = strategy.run(df, [])

        # Verify the strategy ran without errors
        assert isinstance(trades, list)

        # If trades were generated, verify their structure
        if trades:
            for trade in trades:
                assert 'entry_time' in trade
                assert 'entry_price' in trade
                assert 'exit_time' in trade
                assert 'exit_price' in trade
                assert 'side' in trade
                assert trade['side'] in ['long', 'short']

    def test_no_signals_with_flat_prices(self):
        """Test that no signals are generated with flat prices."""
        strategy = BollingerBandsStrategy()

        # Create a dataframe with constant prices
        dates = [datetime.now() + timedelta(days=i) for i in range(30)]
        data = {
            'open': [100] * 30,
            'high': [101] * 30,
            'low': [99] * 30,
            'close': [100] * 30,
        }
        df = pd.DataFrame(data, index=dates)

        # Run the strategy
        trades = strategy.run(df, [])

        # Verify no trades were generated
        assert len(trades) == 0

    def test_with_trailing_stop(self):
        """Test Bollinger Bands strategy with trailing stop."""
        strategy = BollingerBandsStrategy(trailing=2.0)
        df = create_test_df()

        # Run the strategy
        trades = strategy.run(df, [])

        # Verify the strategy ran without errors
        assert isinstance(trades, list)

        # If trades were generated, verify their structure
        if trades:
            for trade in trades:
                assert 'entry_time' in trade
                assert 'entry_price' in trade
                assert 'exit_time' in trade
                assert 'exit_price' in trade
                assert 'side' in trade

    def test_with_contract_switch(self):
        """Test Bollinger Bands strategy with a contract switch."""
        strategy = BollingerBandsStrategy(rollover=True)
        df = create_test_df()

        # Create a switch date in the middle of the dataframe
        switch_date = df.index[25]

        # Run the strategy
        trades = strategy.run(df, [switch_date])

        # Verify the strategy ran without errors
        assert isinstance(trades, list)

        # If trades were generated, verify their structure
        if trades:
            for trade in trades:
                assert 'entry_time' in trade
                assert 'entry_price' in trade
                assert 'exit_time' in trade
                assert 'exit_price' in trade
                assert 'side' in trade

            # If there are trades with the switch flag, verify them
            switch_trades = [trade for trade in trades if trade.get('switch')]
            if switch_trades:
                for trade in switch_trades:
                    assert trade['switch'] is True

    def test_slippage(self):
        """Test that slippage is correctly applied to entry and exit prices in the Bollinger Bands strategy."""
        # Create a strategy with 2% slippage
        strategy = BollingerBandsStrategy(slippage=2.0)
        df = create_test_df()

        # Add indicators and generate signals
        df = strategy.add_indicators(df)
        df = strategy.generate_signals(df)

        # Extract trades
        trades = strategy._extract_trades(df, [])

        # Should have at least one trade
        if len(trades) > 0:
            # Find long and short trades
            long_trades = [t for t in trades if t['side'] == 'long']
            short_trades = [t for t in trades if t['side'] == 'short']

            # Verify slippage is applied correctly for long trades
            for trade in long_trades:
                # Get the original entry and exit prices from the dataframe
                entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
                exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]

                original_entry_price = df.iloc[entry_idx]['open']
                original_exit_price = df.iloc[exit_idx]['open']

                # For long positions:
                # - Entry price should be higher than the original price (pay more on entry)
                # - Exit price should be lower than the original price (receive less on exit)
                expected_entry_price = round(original_entry_price * (1 + strategy.slippage / 100), 2)
                expected_exit_price = round(original_exit_price * (1 - strategy.slippage / 100), 2)

                assert trade['entry_price'] == expected_entry_price
                assert trade['exit_price'] == expected_exit_price

            # Verify slippage is applied correctly for short trades
            for trade in short_trades:
                # Get the original entry and exit prices from the dataframe
                entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
                exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]

                original_entry_price = df.iloc[entry_idx]['open']
                original_exit_price = df.iloc[exit_idx]['open']

                # For short positions:
                # - Entry price should be lower than the original price (receive less on entry)
                # - Exit price should be higher than the original price (pay more on exit)
                expected_entry_price = round(original_entry_price * (1 - strategy.slippage / 100), 2)
                expected_exit_price = round(original_exit_price * (1 + strategy.slippage / 100), 2)

                assert trade['entry_price'] == expected_entry_price
                assert trade['exit_price'] == expected_exit_price
