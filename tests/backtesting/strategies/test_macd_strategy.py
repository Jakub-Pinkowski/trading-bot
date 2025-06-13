from datetime import datetime, timedelta

import pandas as pd

from app.backtesting.strategies.macd import MACDStrategy


# Helper function to create a test dataframe with price patterns suitable for MACD testing
def create_test_df(length=60):
    dates = [datetime.now() + timedelta(days=i) for i in range(length)]

    # Create a price series that will generate clear MACD signals
    close_prices = []

    # Start with a downtrend
    for i in range(15):
        close_prices.append(100 - i)

    # Then an uptrend to create a bullish crossover
    for i in range(15):
        close_prices.append(85 + i * 1.5)

    # Then a downtrend to create a bearish crossover
    for i in range(15):
        close_prices.append(107.5 - i * 1.5)

    # Then another uptrend
    for i in range(15):
        close_prices.append(85 + i)

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


class TestMACDStrategy:
    def test_initialization(self):
        """Test that the MACD strategy initializes with correct default parameters."""
        strategy = MACDStrategy()
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.signal_period == 9

        # Test with custom parameters
        strategy = MACDStrategy(
            fast_period=8,
            slow_period=21,
            signal_period=5,
            rollover=True,
            trailing=2.0,
            slippage=1.0
        )
        assert strategy.fast_period == 8
        assert strategy.slow_period == 21
        assert strategy.signal_period == 5
        assert strategy.rollover == True
        assert strategy.trailing == 2.0
        assert strategy.slippage == 1.0

    def test_add_indicators(self):
        """Test that the add_indicators method correctly adds MACD indicators to the dataframe."""
        strategy = MACDStrategy()
        # Create a larger dataframe to ensure we have valid MACD values
        df = create_test_df(length=100)

        # Apply the strategy's add_indicators method
        df_with_indicators = strategy.add_indicators(df)

        # Verify MACD columns were added
        assert 'macd_line' in df_with_indicators.columns
        assert 'signal_line' in df_with_indicators.columns
        assert 'histogram' in df_with_indicators.columns

        # Skip the initial NaN values and verify that we have some valid MACD values
        valid_macd = df_with_indicators['macd_line'].iloc[strategy.slow_period:].dropna()
        assert len(valid_macd) > 0, "No valid MACD values calculated"

        # Verify that histogram is the difference between MACD line and signal line
        # Allow for small floating point differences
        for i in range(strategy.slow_period + strategy.signal_period, len(df_with_indicators)):
            expected_histogram = df_with_indicators.iloc[i]['macd_line'] - df_with_indicators.iloc[i]['signal_line']
            actual_histogram = df_with_indicators.iloc[i]['histogram']
            assert abs(expected_histogram - actual_histogram) < 1e-10, f"Histogram calculation incorrect at index {i}"

        # Verify indicators are NaN for the first few periods
        assert df_with_indicators['macd_line'].iloc[:strategy.slow_period - 1].isna().all()
        assert df_with_indicators['signal_line'].iloc[:strategy.slow_period - 1].isna().all()
        assert df_with_indicators['histogram'].iloc[:strategy.slow_period - 1].isna().all()

    def test_generate_signals_default_params(self):
        """Test that the generate_signals method correctly identifies buy/sell signals with default parameters."""
        strategy = MACDStrategy()
        df = create_test_df()
        df = strategy.add_indicators(df)

        # Apply the strategy's generate_signals method
        df_with_signals = strategy.generate_signals(df)

        # Verify signal column was added
        assert 'signal' in df_with_signals.columns

        # Find where MACD line crosses above signal line (buy signals)
        buy_signals = df_with_signals[
            (df_with_signals['macd_line'].shift(1) <= df_with_signals['signal_line'].shift(1)) &
            (df_with_signals['macd_line'] > df_with_signals['signal_line'])
            ]

        # Find where MACD line crosses below signal line (sell signals)
        sell_signals = df_with_signals[
            (df_with_signals['macd_line'].shift(1) >= df_with_signals['signal_line'].shift(1)) &
            (df_with_signals['macd_line'] < df_with_signals['signal_line'])
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
        # Use different periods
        strategy = MACDStrategy(fast_period=8, slow_period=21, signal_period=5)
        df = create_test_df()
        df = strategy.add_indicators(df)

        # Apply the strategy's generate_signals method
        df_with_signals = strategy.generate_signals(df)

        # Find where MACD line crosses above signal line (buy signals)
        buy_signals = df_with_signals[
            (df_with_signals['macd_line'].shift(1) <= df_with_signals['signal_line'].shift(1)) &
            (df_with_signals['macd_line'] > df_with_signals['signal_line'])
            ]

        # Find where MACD line crosses below signal line (sell signals)
        sell_signals = df_with_signals[
            (df_with_signals['macd_line'].shift(1) >= df_with_signals['signal_line'].shift(1)) &
            (df_with_signals['macd_line'] < df_with_signals['signal_line'])
            ]

        # Verify all buy signals have signal value of 1
        assert (buy_signals['signal'] == 1).all()

        # Verify all sell signals have signal value of -1
        assert (sell_signals['signal'] == -1).all()

    def test_run_end_to_end(self):
        """Test the full strategy workflow from data to trades."""
        strategy = MACDStrategy()
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
        strategy = MACDStrategy()

        # Create a dataframe with constant prices
        dates = [datetime.now() + timedelta(days=i) for i in range(50)]
        data = {
            'open': [100] * 50,
            'high': [101] * 50,
            'low': [99] * 50,
            'close': [100] * 50,
        }
        df = pd.DataFrame(data, index=dates)

        # Run the strategy
        trades = strategy.run(df, [])

        # Verify no trades were generated
        assert len(trades) == 0

    def test_with_trailing_stop(self):
        """Test MACD strategy with trailing stop."""
        strategy = MACDStrategy(trailing=2.0)
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
        """Test MACD strategy with a contract switch."""
        strategy = MACDStrategy(rollover=True)
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
        """Test that slippage is correctly applied to entry and exit prices in the MACD strategy."""
        # Create a strategy with 2% slippage
        strategy = MACDStrategy(slippage=2.0)
        df = create_test_df()

        # Add indicators and generate signals
        df = strategy.add_indicators(df)
        df = strategy.generate_signals(df)

        # Extract trades
        trades = strategy.extract_trades(df, [])

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
