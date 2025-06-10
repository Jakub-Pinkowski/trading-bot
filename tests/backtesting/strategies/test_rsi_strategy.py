from datetime import datetime, timedelta

import pandas as pd

from app.backtesting.strategies.rsi import RSIStrategy


# Helper function to create test dataframe with price patterns suitable for RSI testing
def create_test_df(length=50):
    dates = [datetime.now() + timedelta(days=i) for i in range(length)]

    # Create a price series that will generate clear RSI signals
    # Start with a downtrend to push RSI low, then uptrend to push it high
    close_prices = []

    # Downtrend for first part
    for i in range(20):
        close_prices.append(100 - i)

    # Uptrend for second part
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

        # Verify RSI values are within expected range
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
