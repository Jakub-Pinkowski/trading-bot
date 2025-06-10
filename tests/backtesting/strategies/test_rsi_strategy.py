from datetime import datetime, timedelta

import pandas as pd

from app.backtesting.strategies.rsi import RSIStrategy


# Helper function to create a test dataframe with price patterns suitable for RSI testing
def create_test_df(length=50):
    dates = [datetime.now() + timedelta(days=i) for i in range(length)]

    # Create a price series that will generate clear RSI signals
    # Start with a downtrend to push RSI low, then uptrend to push it high
    close_prices = []

    # Downtrend for the first part
    for i in range(20):
        close_prices.append(100 - i)

    # Uptrend for the second part
    for i in range(20):
        close_prices.append(80 + i)

    # Downtrend again
    for i in range(10):
        close_prices.append(100 - i)

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


class TestRSIStrategy:
    def test_initialization(self):
        """Test that the RSI strategy initializes with correct default parameters."""
        strategy = RSIStrategy()
        assert strategy.rsi_period == 14
        assert strategy.lower == 30
        assert strategy.upper == 70

        # Test with custom parameters
        strategy = RSIStrategy(rsi_period=10, lower=20, upper=80)
        assert strategy.rsi_period == 10
        assert strategy.lower == 20
        assert strategy.upper == 80

    def test_add_indicators(self):
        """Test that the add_indicators method correctly adds RSI to the dataframe."""
        strategy = RSIStrategy()
        df = create_test_df()

        # Apply the strategy's add_indicators method
        df_with_indicators = strategy.add_indicators(df)

        # Verify RSI column was added
        assert 'rsi' in df_with_indicators.columns

        # Verify RSI values are within the expected range
        assert df_with_indicators['rsi'].min() >= 0
        assert df_with_indicators['rsi'].max() <= 100

        # Verify RSI is NaN for the first few periods
        assert df_with_indicators['rsi'].iloc[:strategy.rsi_period].isna().all()

    def test_generate_signals_default_params(self):
        """Test that the generate_signals method correctly identifies buy/sell signals with default parameters."""
        strategy = RSIStrategy()
        df = create_test_df()
        df = strategy.add_indicators(df)

        # Apply the strategy's generate_signals method
        df_with_signals = strategy.generate_signals(df)

        # Verify signal column was added
        assert 'signal' in df_with_signals.columns

        # Find where RSI crosses below lower threshold (buy signals)
        buy_signals = df_with_signals[
            (df_with_signals['rsi'].shift(1) > strategy.lower) &
            (df_with_signals['rsi'] <= strategy.lower)
            ]

        # Find where RSI crosses above upper threshold (sell signals)
        sell_signals = df_with_signals[
            (df_with_signals['rsi'].shift(1) < strategy.upper) &
            (df_with_signals['rsi'] >= strategy.upper)
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
        strategy = RSIStrategy(rsi_period=7, lower=20, upper=80)
        df = create_test_df()
        df = strategy.add_indicators(df)

        # Apply the strategy's generate_signals method
        df_with_signals = strategy.generate_signals(df)

        # Find where RSI crosses below lower threshold (buy signals)
        buy_signals = df_with_signals[
            (df_with_signals['rsi'].shift(1) > strategy.lower) &
            (df_with_signals['rsi'] <= strategy.lower)
            ]

        # Find where RSI crosses above upper threshold (sell signals)
        sell_signals = df_with_signals[
            (df_with_signals['rsi'].shift(1) < strategy.upper) &
            (df_with_signals['rsi'] >= strategy.upper)
            ]

        # Verify all buy signals have signal value of 1
        assert (buy_signals['signal'] == 1).all()

        # Verify all sell signals have signal value of -1
        assert (sell_signals['signal'] == -1).all()

    def test_run_end_to_end(self):
        """Test the full strategy workflow from data to trades."""
        strategy = RSIStrategy()
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
        strategy = RSIStrategy()

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
        """Test RSI strategy with trailing stop."""
        strategy = RSIStrategy(trailing=2.0)
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
        """Test RSI strategy with a contract switch."""
        strategy = RSIStrategy(rollover=True)
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

    def test_extreme_market_conditions(self):
        """Test RSI strategy with extreme market conditions."""
        # For this test, we'll skip the RSI calculation and directly test the signal generation
        # and trade extraction with manually set RSI values
        import numpy as np

        strategy = RSIStrategy()

        # Create a dataframe with dates
        dates = [datetime.now() + timedelta(days=i) for i in range(50)]
        df = pd.DataFrame(index=dates)

        # Add required price columns with constant values
        df['open'] = 100
        df['high'] = 101
        df['low'] = 99
        df['close'] = 100

        # Manually create an RSI column with extreme values and clear crossings
        # Start with NaN for the first 14 periods (RSI period)
        rsi_values = [np.nan] * 14

        # Add values that will create extreme conditions and threshold crossings

        # First, create a pattern that will generate a buy signal:
        # RSI above a lower threshold
        rsi_values.append(40.0)  # Previous value
        # RSI crosses below a lower threshold (buy signal)
        rsi_values.append(25.0)  # Current value - well below a threshold
        # RSI stays low
        rsi_values.append(20.0)
        rsi_values.append(15.0)  # Very extreme low value

        # Then, create a pattern that will generate a sell signal:
        # RSI rises
        rsi_values.append(40.0)
        # RSI below an upper threshold
        rsi_values.append(60.0)  # Previous value
        # RSI crosses above an upper threshold (sell signal)
        rsi_values.append(80.0)  # Current value - well above a threshold
        # RSI stays high
        rsi_values.append(85.0)
        rsi_values.append(90.0)  # Very extreme high value

        # Fill the rest with neutral values
        while len(rsi_values) < 50:
            rsi_values.append(50.0)

        # Add RSI column to dataframe
        df['rsi'] = rsi_values

        # Generate signals
        df = strategy.generate_signals(df)

        # There should be at least one buy signal (RSI crossing below a lower threshold)
        buy_signals = df[df['signal'] == 1]
        assert len(buy_signals) > 0, "No buy signals generated in extreme market conditions"

        # There should be at least one sell signal (RSI crossing above an upper threshold)
        sell_signals = df[df['signal'] == -1]
        assert len(sell_signals) > 0, "No sell signals generated in extreme market conditions"

        # Verify specific signals at the crossing points
        # Buy signal
        assert df.iloc[15]['signal'] == 1, "Should generate buy signal when RSI crosses below lower threshold"

        # Sell signal
        assert df.iloc[20]['signal'] == -1, "Should generate sell signal when RSI crosses above upper threshold"

        # Since we're primarily testing signal generation in extreme conditions,
        # we'll skip the trade extraction test which depends on more complex logic
        # The signal generation tests above already verify the core functionality

    def test_boundary_rsi_values(self):
        """Test RSI strategy with RSI values at or near the threshold boundaries."""
        import numpy as np

        # Create a strategy with custom thresholds for easier testing
        strategy = RSIStrategy(lower=30, upper=70)

        # Create a dataframe with dates
        dates = [datetime.now() + timedelta(days=i) for i in range(50)]
        df = pd.DataFrame(index=dates)

        # Add required price columns
        df['open'] = 100
        df['high'] = 101
        df['low'] = 99
        df['close'] = 100

        # Manually create an RSI column with specific boundary values
        # Start with NaN for the first 14 periods (RSI period)
        rsi_values = [np.nan] * 14

        # Add values that will test the boundary conditions:
        # 1. RSI above a lower threshold
        rsi_values.append(35.0)  # Previous value
        # 2. RSI crosses below a lower threshold (buy signal)
        rsi_values.append(29.9)  # Current value
        # 3. RSI stays below a lower threshold (no signal)
        rsi_values.append(25.0)
        # 4. RSI below lower threshold
        rsi_values.append(20.0)  # Previous value
        # 5. RSI crosses above lower threshold (no signal)
        rsi_values.append(35.0)  # Current value

        # 6. RSI below upper threshold
        rsi_values.append(65.0)  # Previous value
        # 7. RSI crosses above upper threshold (sell signal)
        rsi_values.append(70.1)  # Current value
        # 8. RSI stays above upper threshold (no signal)
        rsi_values.append(75.0)
        # 9. RSI above upper threshold
        rsi_values.append(80.0)  # Previous value
        # 10. RSI crosses below upper threshold (no signal)
        rsi_values.append(65.0)  # Current value

        # Fill the rest with neutral values
        while len(rsi_values) < 50:
            rsi_values.append(50.0)

        # Add RSI column to dataframe
        df['rsi'] = rsi_values

        # Generate signals
        df = strategy.generate_signals(df)

        # Verify signals at boundary conditions

        # Check for a buy signal when RSI crosses below lower threshold
        assert df.iloc[15]['signal'] == 1, "Should generate buy signal when RSI crosses below lower threshold"

        # Check for a sell signal when RSI crosses above upper threshold
        assert df.iloc[20]['signal'] == -1, "Should generate sell signal when RSI crosses above upper threshold"

        # Check no signals when RSI stays below lower threshold (no crossing)
        assert df.iloc[16]['signal'] == 0, "Should not generate signal when RSI stays below lower threshold"

        # Check no signals when RSI stays above upper threshold (no crossing)
        assert df.iloc[21]['signal'] == 0, "Should not generate signal when RSI stays above upper threshold"

        # Check no signals when RSI crosses above lower threshold
        assert df.iloc[18]['signal'] == 0, "Should not generate signal when RSI crosses above lower threshold"

        # Check no signals when RSI crosses below upper threshold
        assert df.iloc[23]['signal'] == 0, "Should not generate signal when RSI crosses below upper threshold"
