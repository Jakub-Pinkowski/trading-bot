from datetime import datetime, timedelta

import pandas as pd

from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy


# Helper function to create test dataframe with price patterns suitable for EMA crossover testing
def create_test_df(length=50):
    dates = [datetime.now() + timedelta(days=i) for i in range(length)]

    # Create a price series that will generate clear EMA crossover signals
    close_prices = []

    # Start with an uptrend
    for i in range(15):
        close_prices.append(100 + i)

    # Then a downtrend
    for i in range(15):
        close_prices.append(115 - i)

    # Then another uptrend
    for i in range(15):
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


class TestEMACrossoverStrategy:
    def test_initialization(self):
        """Test that the EMA Crossover strategy initializes with correct default parameters."""
        strategy = EMACrossoverStrategy()
        assert strategy.ema_short == 9
        assert strategy.ema_long == 21

        # Test with custom parameters
        strategy = EMACrossoverStrategy(ema_short=5, ema_long=15)
        assert strategy.ema_short == 5
        assert strategy.ema_long == 15

    def test_add_indicators(self):
        """Test that the add_indicators method correctly adds EMAs to the dataframe."""
        strategy = EMACrossoverStrategy()
        df = create_test_df()

        # Apply the strategy's add_indicators method
        df_with_indicators = strategy.add_indicators(df)

        # Verify EMA columns were added
        assert 'ema_short' in df_with_indicators.columns
        assert 'ema_long' in df_with_indicators.columns

        # Verify EMA values are within expected range
        assert not df_with_indicators['ema_short'].isnull().all()
        assert not df_with_indicators['ema_long'].isnull().all()

        # Verify EMA is NaN for the first few periods
        assert df_with_indicators['ema_short'].iloc[:strategy.ema_short - 1].isna().all()
        assert df_with_indicators['ema_long'].iloc[:strategy.ema_long - 1].isna().all()

        # Verify that both EMAs have valid values after the initialization period
        assert not df_with_indicators['ema_short'].iloc[strategy.ema_long:].isnull().any()
        assert not df_with_indicators['ema_long'].iloc[strategy.ema_long:].isnull().any()

    def test_generate_signals_default_params(self):
        """Test that the generate_signals method correctly identifies buy/sell signals with default parameters."""
        strategy = EMACrossoverStrategy()
        df = create_test_df()
        df = strategy.add_indicators(df)

        # Apply the strategy's generate_signals method
        df_with_signals = strategy.generate_signals(df)

        # Verify signal column was added
        assert 'signal' in df_with_signals.columns

        # Find where short EMA crosses above long EMA (buy signals)
        buy_signals = df_with_signals[
            (df_with_signals['ema_short'].shift(1) <= df_with_signals['ema_long'].shift(1)) &
            (df_with_signals['ema_short'] > df_with_signals['ema_long'])
            ]

        # Find where short EMA crosses below long EMA (sell signals)
        sell_signals = df_with_signals[
            (df_with_signals['ema_short'].shift(1) >= df_with_signals['ema_long'].shift(1)) &
            (df_with_signals['ema_short'] < df_with_signals['ema_long'])
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
        # Use different EMA periods
        strategy = EMACrossoverStrategy(ema_short=5, ema_long=15)
        df = create_test_df()
        df = strategy.add_indicators(df)

        # Apply the strategy's generate_signals method
        df_with_signals = strategy.generate_signals(df)

        # Find where short EMA crosses above long EMA (buy signals)
        buy_signals = df_with_signals[
            (df_with_signals['ema_short'].shift(1) <= df_with_signals['ema_long'].shift(1)) &
            (df_with_signals['ema_short'] > df_with_signals['ema_long'])
            ]

        # Find where short EMA crosses below long EMA (sell signals)
        sell_signals = df_with_signals[
            (df_with_signals['ema_short'].shift(1) >= df_with_signals['ema_long'].shift(1)) &
            (df_with_signals['ema_short'] < df_with_signals['ema_long'])
            ]

        # Verify all buy signals have signal value of 1
        assert (buy_signals['signal'] == 1).all()

        # Verify all sell signals have signal value of -1
        assert (sell_signals['signal'] == -1).all()

    def test_run_end_to_end(self):
        """Test the full strategy workflow from data to trades."""
        strategy = EMACrossoverStrategy()
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
        strategy = EMACrossoverStrategy()

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
        """Test EMA Crossover strategy with trailing stop."""
        strategy = EMACrossoverStrategy(trailing=2.0)
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
        """Test EMA Crossover strategy with contract switch."""
        strategy = EMACrossoverStrategy(rollover=True)
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
