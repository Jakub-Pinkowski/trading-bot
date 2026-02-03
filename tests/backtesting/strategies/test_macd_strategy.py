from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.backtesting.strategies import MACDStrategy
from tests.backtesting.strategies.conftest import create_test_df


class TestMACDStrategy:
    def test_initialization(self):
        """Test that the MACD strategy initializes with correct default parameters."""
        strategy = MACDStrategy(fast_period=12,
                                slow_period=26,
                                signal_period=9,
                                rollover=False,
                                trailing=None,
                                slippage_ticks=0,
                                symbol=None)
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
            slippage_ticks=1.0,
            symbol=None
        )
        assert strategy.fast_period == 8
        assert strategy.slow_period == 21
        assert strategy.signal_period == 5
        assert strategy.rollover == True
        assert strategy.trailing == 2.0
        assert strategy.position_manager.slippage_ticks == 1.0

    def test_add_indicators(self):
        """Test that the add_indicators method correctly adds MACD indicators to the dataframe."""
        strategy = MACDStrategy(fast_period=12,
                                slow_period=26,
                                signal_period=9,
                                rollover=False,
                                trailing=None,
                                slippage_ticks=0,
                                symbol=None)

        # Create a simple dataframe with a clear trend
        dates = [datetime.now() + timedelta(days=i) for i in range(100)]

        # Create a simple price series with a clear trend
        prices = [100 + i for i in range(100)]

        # Create OHLC data
        data = {
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
        }

        df = pd.DataFrame(data, index=dates)

        # Apply the strategy's add_indicators method
        df_with_indicators = strategy.add_indicators(df)

        # Verify MACD columns were added
        assert 'macd_line' in df_with_indicators.columns
        assert 'signal_line' in df_with_indicators.columns
        assert 'histogram' in df_with_indicators.columns

        # Verify that we have non-NaN values after the initialization period
        # The MACD line should be available after the slow period
        assert not df_with_indicators['macd_line'].iloc[strategy.slow_period:].isna().all()

        # The signal line should be available after the slow period + signal period
        assert not df_with_indicators['signal_line'].iloc[strategy.slow_period + strategy.signal_period:].isna().all()

        # The histogram should be available when both MACD line and signal line are available
        assert not df_with_indicators['histogram'].iloc[strategy.slow_period + strategy.signal_period:].isna().all()

        # Verify that histogram is the difference between MACD line and signal line
        # Allow for small floating point differences
        for i in range(strategy.slow_period + strategy.signal_period, len(df_with_indicators)):
            if not pd.isna(df_with_indicators.iloc[i]['macd_line']) and not pd.isna(
                    df_with_indicators.iloc[i]['signal_line']):
                expected_histogram = df_with_indicators.iloc[i]['macd_line'] - df_with_indicators.iloc[i]['signal_line']
                actual_histogram = df_with_indicators.iloc[i]['histogram']
                assert abs(expected_histogram - actual_histogram) < 1e-10, f"Histogram calculation incorrect at index {i}"

        # Verify indicators are NaN for the first few periods
        # The MACD line should be NaN for the first slow_period - 1 periods
        assert df_with_indicators['macd_line'].iloc[:strategy.slow_period - 1].isna().all()

        # The signal line should be NaN for at least the first slow_period - 1 periods
        # (it might be NaN for more periods depending on the implementation)
        assert df_with_indicators['signal_line'].iloc[:strategy.slow_period - 1].isna().all()

        # The histogram should be NaN for at least the first slow_period - 1 periods
        # (it might be NaN for more periods depending on the implementation)
        assert df_with_indicators['histogram'].iloc[:strategy.slow_period - 1].isna().all()

    def test_generate_signals_default_params(self):
        """Test that the generate_signals method correctly identifies buy/sell signals with default parameters."""
        strategy = MACDStrategy(fast_period=12,
                                slow_period=26,
                                signal_period=9,
                                rollover=False,
                                trailing=None,
                                slippage_ticks=0,
                                symbol=None)
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
        strategy = MACDStrategy(fast_period=8,
                                slow_period=21,
                                signal_period=5,
                                rollover=False,
                                trailing=None,
                                slippage_ticks=0,
                                symbol=None)
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
        strategy = MACDStrategy(fast_period=12,
                                slow_period=26,
                                signal_period=9,
                                rollover=False,
                                trailing=None,
                                slippage_ticks=0,
                                symbol=None)
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
        strategy = MACDStrategy(fast_period=12,
                                slow_period=26,
                                signal_period=9,
                                rollover=False,
                                trailing=None,
                                slippage_ticks=0,
                                symbol=None)

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
        strategy = MACDStrategy(fast_period=12,
                                slow_period=26,
                                signal_period=9,
                                rollover=False,
                                trailing=2.0,
                                slippage_ticks=0,
                                symbol=None)
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
        strategy = MACDStrategy(fast_period=12,
                                slow_period=26,
                                signal_period=9,
                                rollover=True,
                                trailing=None,
                                slippage_ticks=0,
                                symbol=None)
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
        from config import TICK_SIZES, DEFAULT_TICK_SIZE
        
        # Create a strategy with 2 ticks slippage
        slippage_ticks = 2
        strategy = MACDStrategy(fast_period=12,
                                slow_period=26,
                                signal_period=9,
                                rollover=False,
                                trailing=None,
                                slippage_ticks=slippage_ticks,
                                symbol=None)
        df = create_test_df()

        # Add indicators and generate signals
        df = strategy.add_indicators(df)
        df = strategy.generate_signals(df)

        # Extract trades
        trades = strategy._extract_trades(df, [])

        # Get tick size
        tick_size = TICK_SIZES.get(None, DEFAULT_TICK_SIZE)
        slippage_amount = slippage_ticks * tick_size

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
                expected_entry_price = round(original_entry_price + slippage_amount, 2)
                expected_exit_price = round(original_exit_price - slippage_amount, 2)

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
                expected_entry_price = round(original_entry_price - slippage_amount, 2)
                expected_exit_price = round(original_exit_price + slippage_amount, 2)

                assert trade['entry_price'] == expected_entry_price
                assert trade['exit_price'] == expected_exit_price

    def test_extreme_market_conditions(self):
        """Test MACD strategy with extreme market conditions."""
        # Create a dataframe with extreme price movements
        dates = [datetime.now() + timedelta(days=i) for i in range(100)]
        df = pd.DataFrame(index=dates)

        # Create a price series with extreme movements
        close_prices = []

        # Start with a stable price
        for i in range(30):
            close_prices.append(100)

        # Add a sharp drop (crash)
        for i in range(10):
            close_prices.append(100 - i * 5)  # Drop by 5 each day

        # Add a sharp recovery
        for i in range(10):
            close_prices.append(50 + i * 5)  # Recover by 5 each day

        # Add a period of high volatility
        for i in range(20):
            if i % 2 == 0:
                close_prices.append(close_prices[-1] * 1.05)  # Up 5%
            else:
                close_prices.append(close_prices[-1] * 0.95)  # Down 5%

        # Add a period of low volatility
        last_price = close_prices[-1]
        for i in range(30):
            close_prices.append(last_price + np.random.normal(0, 0.5))  # Small random movements

        # Create OHLC data
        data = {
            'open': close_prices,
            'high': [p * 1.01 for p in close_prices],  # 1% higher than close
            'low': [p * 0.99 for p in close_prices],  # 1% lower than close
            'close': close_prices,
        }

        df = pd.DataFrame(data, index=dates)

        # Create a strategy with default parameters
        strategy = MACDStrategy(fast_period=12,
                                slow_period=26,
                                signal_period=9,
                                rollover=False,
                                trailing=None,
                                slippage_ticks=0,
                                symbol=None)

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

    def test_boundary_macd_values(self):
        """Test MACD strategy with MACD values at or near the crossover boundaries."""
        # Create a strategy with default parameters
        strategy = MACDStrategy(fast_period=12,
                                slow_period=26,
                                signal_period=9,
                                rollover=False,
                                trailing=None,
                                slippage_ticks=0,
                                symbol=None)

        # Create a dataframe with dates
        dates = [datetime.now() + timedelta(days=i) for i in range(50)]
        df = pd.DataFrame(index=dates)

        # Add required price columns
        df['open'] = 100
        df['high'] = 101
        df['low'] = 99
        df['close'] = 100

        # Manually create MACD columns with specific boundary values
        # Start with NaN for the first 26 periods (slow period)
        macd_line = [np.nan] * 26
        signal_line = [np.nan] * 26
        histogram = [np.nan] * 26

        # Add values that will test the boundary conditions:

        # 1. MACD line below signal line
        macd_line.append(-1.0)  # Previous value
        signal_line.append(1.0)  # Previous value
        histogram.append(-2.0)  # Previous value

        # 2. MACD line crosses above signal line (buy signal)
        macd_line.append(1.1)  # Current value
        signal_line.append(1.0)  # Current value
        histogram.append(0.1)  # Current value

        # 3. MACD line stays above signal line (no signal)
        macd_line.append(2.0)
        signal_line.append(1.0)
        histogram.append(1.0)

        # 4. MACD line above signal line
        macd_line.append(2.0)  # Previous value
        signal_line.append(1.0)  # Previous value
        histogram.append(1.0)  # Previous value

        # 5. MACD line crosses below signal line (sell signal)
        macd_line.append(0.9)  # Current value
        signal_line.append(1.0)  # Current value
        histogram.append(-0.1)  # Current value

        # 6. MACD line stays below signal line (no signal)
        macd_line.append(0.5)
        signal_line.append(1.0)
        histogram.append(-0.5)

        # 7. Test exact equality (no crossover)
        macd_line.append(1.0)  # Equal to signal line
        signal_line.append(1.0)
        histogram.append(0.0)

        # 8. Test tiny crossover (just above)
        macd_line.append(1.000001)  # Just above signal line
        signal_line.append(1.0)
        histogram.append(0.000001)

        # 9. Test tiny crossover (just below)
        macd_line.append(0.999999)  # Just below signal line
        signal_line.append(1.0)
        histogram.append(-0.000001)

        # Fill the rest with neutral values
        while len(macd_line) < 50:
            macd_line.append(0.0)
            signal_line.append(0.0)
            histogram.append(0.0)

        # Add MACD columns to dataframe
        df['macd_line'] = macd_line
        df['signal_line'] = signal_line
        df['histogram'] = histogram

        # Generate signals
        df = strategy.generate_signals(df)

        # Verify signals at boundary conditions

        # Check for a buy signal when MACD line crosses above signal line
        assert df.iloc[27]['signal'] == 1, "Should generate buy signal when MACD line crosses above signal line"

        # Check for a sell signal when MACD line crosses below signal line
        assert df.iloc[30]['signal'] == -1, "Should generate sell signal when MACD line crosses below signal line"

        # Check no signals when MACD line stays above signal line (no crossing)
        assert df.iloc[28]['signal'] == 0, "Should not generate signal when MACD line stays above signal line"

        # Check no signals when MACD line stays below signal line (no crossing)
        assert df.iloc[31]['signal'] == 0, "Should not generate signal when MACD line stays below signal line"

        # Check no signals when MACD line equals signal line (no crossing)
        assert df.iloc[32]['signal'] == 0, "Should not generate signal when MACD line equals signal line"

        # Check for signals with tiny crossovers
        assert df.iloc[33]['signal'] == 1, "Should generate buy signal with tiny crossover above"
        assert df.iloc[34]['signal'] == -1, "Should generate sell signal with tiny crossover below"

    def test_macd_divergence_patterns(self):
        """Test MACD strategy with price-MACD divergence patterns."""
        # Create a strategy with default parameters
        strategy = MACDStrategy(fast_period=12,
                                slow_period=26,
                                signal_period=9,
                                rollover=False,
                                trailing=None,
                                slippage_ticks=0,
                                symbol=None)

        # Create a dataframe with dates
        dates = [datetime.now() + timedelta(days=i) for i in range(100)]
        df = pd.DataFrame(index=dates)

        # Create a price series with a divergence pattern
        # Price making higher highs, but MACD making lower highs (bearish divergence)
        base_price = 100
        prices = []

        # Initial uptrend
        for i in range(30):
            prices.append(base_price + i)

        # First peak
        prices.append(prices[-1] + 5)  # Day 30

        # Pullback
        for i in range(10):
            prices.append(prices[-1] - 2)

        # Second peak (higher high in price)
        for i in range(20):
            prices.append(prices[-1] + 1)
        prices[-1] += 10  # Make sure it's a higher high

        # Decline after divergence
        for i in range(39):
            prices.append(prices[-1] * 0.99)

        # Create OHLC data
        df['open'] = prices
        df['high'] = [p * 1.01 for p in prices]
        df['low'] = [p * 0.99 for p in prices]
        df['close'] = prices

        # Add indicators
        df = strategy.add_indicators(df)

        # Manually modify MACD to create a divergence
        # We'll make the MACD at the second peak lower than at the first peak
        # This is a bearish divergence (price higher high, MACD lower high)

        # Find the MACD at the first peak (around day 30)
        first_peak_idx = 30
        first_peak_macd = df['macd_line'].iloc[first_peak_idx]

        # Make the MACD at the second peak (around day 60) lower
        second_peak_idx = 60

        # Ensure there's a crossover at the second peak
        # Set the previous day's MACD below the signal line
        df.loc[df.index[second_peak_idx - 1], 'macd_line'] = df['signal_line'].iloc[second_peak_idx - 1] - 0.5

        # Set the current day's MACD above the signal line but lower than the first peak
        df.loc[df.index[second_peak_idx], 'macd_line'] = min(first_peak_macd * 0.8,
                                                             df['signal_line'].iloc[second_peak_idx] + 0.5)

        # Update histogram
        df.loc[df.index[second_peak_idx - 1], 'histogram'] = df['macd_line'].iloc[second_peak_idx - 1] - \
                                                             df['signal_line'].iloc[second_peak_idx - 1]
        df.loc[df.index[second_peak_idx], 'histogram'] = df['macd_line'].iloc[second_peak_idx] - df['signal_line'].iloc[
            second_peak_idx]

        # Generate signals
        df = strategy.generate_signals(df)

        # Extract trades
        trades = strategy._extract_trades(df, [])

        # Verify the strategy ran without errors
        assert isinstance(trades, list)

        # Verify that there's a buy signal at the second peak due to MACD crossover
        # even though there's a bearish divergence
        assert df.iloc[second_peak_idx][
                   'signal'] == 1, "Should generate buy signal at MACD crossover despite divergence"

        # If there are trades that entered long near the second peak, they should be documented
        divergence_trades = []
        for trade in trades:
            # Find the index in the dataframe that matches the entry_time
            entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
            if second_peak_idx - 2 <= entry_idx <= second_peak_idx + 2 and trade['side'] == 'long':
                # Calculate profit/loss
                pnl = trade['exit_price'] / trade['entry_price'] - 1
                divergence_trades.append((trade, pnl))

        # Document the behavior during divergence
        # The standard MACD strategy doesn't account for divergences
        # This is a known limitation - it might generate false signals during divergences
        if divergence_trades:
            # Verify the structure of divergence trades
            for trade, pnl in divergence_trades:
                assert 'entry_time' in trade, "Trade should have entry_time"
                assert 'entry_price' in trade, "Trade should have entry_price"
                assert 'exit_time' in trade, "Trade should have exit_time"
                assert 'exit_price' in trade, "Trade should have exit_price"
                assert 'side' in trade, "Trade should have side"
                assert trade['side'] == 'long', "Divergence trade should be long"
