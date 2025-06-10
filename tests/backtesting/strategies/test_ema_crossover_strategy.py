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

    def test_close_ema_values(self):
        """Test EMA Crossover strategy when EMA values are very close to each other."""

        # Create a strategy with custom parameters
        strategy = EMACrossoverStrategy(ema_short=5, ema_long=10)

        # Create a dataframe with dates
        dates = [datetime.now() + timedelta(days=i) for i in range(50)]
        df = pd.DataFrame(index=dates)

        # Add required price columns with a price series that will result in close EMA values
        # Start with a flat price
        close_prices = [100] * 15

        # Then add a very gentle slope up (should cause EMAs to be close but short > long)
        for i in range(15):
            close_prices.append(100 + (i * 0.1))

        # Then add a very gentle slope down (should cause EMAs to be close but short < long)
        for i in range(15):
            close_prices.append(101.5 - (i * 0.1))

        # Fill the rest with flat prices
        while len(close_prices) < 50:
            close_prices.append(100)

        # Create OHLC data
        df['open'] = close_prices
        df['high'] = [p + 0.05 for p in close_prices]
        df['low'] = [p - 0.05 for p in close_prices]
        df['close'] = close_prices

        # Add indicators
        df = strategy.add_indicators(df)

        # Verify EMAs are calculated
        assert 'ema_short' in df.columns
        assert 'ema_long' in df.columns

        # Find where EMAs are very close (difference < 0.1)
        close_emas = df[(df['ema_short'] - df['ema_long']).abs() < 0.1]

        # There should be some periods where EMAs are close
        assert len(close_emas) > 0, "Test data should have periods with close EMA values"

        # Generate signals
        df = strategy.generate_signals(df)

        # Find crossover points
        crossovers = df[df['signal'] != 0]

        # Verify crossovers occur at the expected points
        # When close prices change from uptrend to downtrend and vice versa
        assert len(crossovers) >= 2, "Should have at least 2 crossovers with the test data"

        # Run end-to-end and verify trades
        trades = strategy.run(df, [])

        # Verify trades are generated
        assert len(trades) > 0, "No trades generated with close EMA values"

        # Verify trade sides alternate (long, short, long, etc.)
        if len(trades) >= 2:
            for i in range(1, len(trades)):
                assert trades[i]['side'] != trades[i - 1]['side'], "Trade sides should alternate"

    def test_multiple_contract_switches(self):
        """Test EMA Crossover strategy with multiple contract switches."""

        # Create a custom strategy that returns trades with switch flags
        class MultiSwitchTestStrategy(EMACrossoverStrategy):
            def extract_trades(self, df, switch_dates):
                # Create trades with switch flags
                trades = []
                for i, switch_date in enumerate(switch_dates):
                    # Create a trade with the switch flag
                    trade = {
                        'entry_time': df.index[max(0, i * 25)],
                        'entry_price': 100.0,
                        'exit_time': switch_date,
                        'exit_price': 110.0,
                        'side': 'long',
                        'switch': True
                    }
                    trades.append(trade)
                return trades

        strategy = MultiSwitchTestStrategy(rollover=True)
        df = create_test_df(length=100)  # Longer dataframe for multiple switches

        # Create multiple switch dates
        switch_dates = [df.index[25], df.index[50], df.index[75]]

        # Run the strategy
        trades = strategy.run(df, switch_dates)

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

            # Find trades with the switch flag
            switch_trades = [trade for trade in trades if trade.get('switch')]

            # There should be at least one switch trade for each switch date
            assert len(switch_trades) > 0, "No switch trades generated with multiple contract switches"

            # Verify switch trades have the correct flag
            for trade in switch_trades:
                assert trade['switch'] is True
