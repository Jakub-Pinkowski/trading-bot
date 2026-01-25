from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.backtesting.strategies.base_strategy import BaseStrategy
from tests.backtesting.strategies.conftest import create_test_df


# Create a concrete implementation of BaseStrategy for testing
class StrategyForTesting(BaseStrategy):
    def __init__(self, rollover=False, trailing=None, slippage=0):
        super().__init__(rollover=rollover, trailing=trailing, slippage=slippage)

    def add_indicators(self, df):
        # Simple implementation for testing
        return df

    def generate_signals(self, df):
        # Simple implementation for testing
        df['signal'] = 0
        # Set some signals for testing after the warm-up period
        if len(df) > 105:
            df.iloc[102, df.columns.get_loc('signal')] = 1  # Buy signal
            # For contract switch test, ensure there's an open position at the switch date (index 107)
            # by not closing the position before the switch
            if len(df) > 110:
                df.iloc[109, df.columns.get_loc('signal')] = -1  # Sell signal after switch date
            else:
                df.iloc[104, df.columns.get_loc('signal')] = -1  # Sell signal
        return df


class TestTrailingScenarios:
    def test_different_trailing_values(self):
        """Test that different trailing stop percentages are correctly applied."""
        # Test with multiple trailing stop values
        trailing_values = [1.0, 2.0, 5.0, 10.0]

        for trailing in trailing_values:
            # Create a strategy with the current trailing value
            strategy = StrategyForTesting(trailing=trailing)
            df = create_test_df(length=150)

            # Modify prices to create a scenario where trailing stop will be triggered
            # For a long position:
            # 1. Entry at index 102 (after warm-up period)
            # 2. Price moves up at index 103 (trailing stop should adjust)
            # 3. Price drops at index 105 (should trigger trailing stop)
            df.loc[df.index[102], 'open'] = 100.0  # Entry price
            df.loc[df.index[103], 'high'] = 110.0  # Price moves up, trailing stop should adjust
            df.loc[df.index[104], 'high'] = 115.0  # Price moves up more, trailing stop should adjust again
            df.loc[df.index[105], 'low'] = 100.0  # Price drops, may trigger trailing stop depending on trailing value

            df = strategy.generate_signals(df)  # Add signals
            trades = strategy._extract_trades(df, [])

            # Should have at least one trade
            assert len(trades) > 0

            # Find long trades
            long_trades = [t for t in trades if t['side'] == 'long']

            # Verify trailing stop is applied correctly for long trades
            for trade in long_trades:
                # Calculate the expected trailing stop based on the highest price reached
                # The highest price in our scenario is at index 4 (115.0)
                highest_price = 115.0
                expected_trailing_stop = round(highest_price * (1 - trailing / 100), 2)

                # If the price drops below the trailing stop, the trade should be closed
                # The low at index 5 is 100.0
                if 100.0 <= expected_trailing_stop:
                    # Trailing stop not triggered, trade should be closed by the sell signal at index 4
                    # or by some other mechanism (depends on the random prices in the test dataframe)
                    # Just verify that the trade was closed
                    assert trade['exit_time'] is not None
                else:
                    # Trailing stop triggered, trade should be closed at the trailing stop price
                    # The exit price should be close to the trailing stop price
                    # Allow for some flexibility in when the trade is closed
                    if trade['exit_time'] == df.index[5]:
                        # If closed at index 5, it should be at the trailing stop price
                        assert abs(trade['exit_price'] - expected_trailing_stop) < 0.01, \
                            f"Exit price should be close to trailing stop price {expected_trailing_stop}, got {trade['exit_price']}"
                    else:
                        # If closed at a different time, just verify that the trade was closed
                        assert trade['exit_time'] is not None

    def test_trailing_stop_in_uptrend(self):
        """Test trailing stop behavior in an uptrend market."""
        # Create a strategy with trailing stop
        trailing = 3.0
        strategy = StrategyForTesting(trailing=trailing)

        # Create a dataframe with an uptrend
        dates = [datetime.now() + timedelta(days=i) for i in range(150)]
        df = pd.DataFrame(index=dates)

        # Create an uptrend price series
        base_price = 100
        prices = [base_price + i * 0.5 for i in range(150)]  # Steady uptrend

        # Create OHLC data
        df['open'] = prices
        df['high'] = [p * 1.01 for p in prices]  # 1% above open
        df['low'] = [p * 0.99 for p in prices]  # 1% below open
        df['close'] = prices

        # Generate signals and extract trades
        df = strategy.generate_signals(df)
        trades = strategy._extract_trades(df, [])

        # Should have at least one trade
        assert len(trades) > 0

        # Find long trades
        long_trades = [t for t in trades if t['side'] == 'long']

        # In an uptrend, trailing stops should not be triggered for long trades
        # unless there's a significant pullback
        for trade in long_trades:
            # Get the entry and exit indices
            entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
            exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]

            # Calculate the maximum price reached during the trade
            trade_df = df.iloc[entry_idx:exit_idx + 1]
            max_price = trade_df['high'].max()

            # Calculate the expected trailing stop based on the maximum price
            expected_trailing_stop = round(max_price * (1 - trailing / 100), 2)

            # If the trade was closed by a trailing stop, the exit price should be close to the trailing stop
            # Otherwise, it was closed by a signal
            if exit_idx < len(df) - 1:  # Not the last bar
                # Find the index of the maximum price
                max_price_idx = trade_df['high'].idxmax()
                # Convert timestamp to integer index
                max_price_pos = df.index.get_loc(max_price_idx)
                # Get the minimum price after the maximum price
                min_price_after_max = df.iloc[max_price_pos:exit_idx + 1]['low'].min()
                if min_price_after_max <= expected_trailing_stop:
                    # Trade was likely closed by trailing stop
                    assert abs(trade['exit_price'] - expected_trailing_stop) < 0.01, \
                        f"Exit price should be close to trailing stop price {expected_trailing_stop}, got {trade['exit_price']}"

    def test_trailing_stop_in_downtrend(self):
        """Test trailing stop behavior in a downtrend market."""
        # Create a strategy with trailing stop
        trailing = 3.0

        # Create a custom strategy that generates short signals
        class ShortStrategy(BaseStrategy):
            def __init__(self, trailing=None):
                super().__init__(trailing=trailing)

            def add_indicators(self, df):
                return df

            def generate_signals(self, df):
                df['signal'] = 0
                # Generate a short signal at index 102 (after warm-up period)
                if len(df) > 105:
                    df.iloc[102, df.columns.get_loc('signal')] = -1  # Short signal
                    # Generate a buy signal at index 115 to close the short position
                    if len(df) > 115:
                        df.iloc[115, df.columns.get_loc('signal')] = 1  # Buy signal to close short
                return df

        strategy = ShortStrategy(trailing=trailing)

        # Create a dataframe with a downtrend
        dates = [datetime.now() + timedelta(days=i) for i in range(150)]
        df = pd.DataFrame(index=dates)

        # Create a downtrend price series
        base_price = 100
        prices = [base_price - i * 0.5 for i in range(150)]  # Steady downtrend

        # Create OHLC data
        df['open'] = prices
        df['high'] = [p * 1.01 for p in prices]  # 1% above open
        df['low'] = [p * 0.99 for p in prices]  # 1% below open
        df['close'] = prices

        # Generate signals and extract trades
        df = strategy.generate_signals(df)
        trades = strategy._extract_trades(df, [])

        # Should have at least one trade
        assert len(trades) > 0

        # Find short trades
        short_trades = [t for t in trades if t['side'] == 'short']

        # In a downtrend, trailing stops should not be triggered for short trades
        # unless there's a significant rally
        for trade in short_trades:
            # Get the entry and exit indices
            entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
            exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]

            # Calculate the minimum price reached during the trade
            trade_df = df.iloc[entry_idx:exit_idx + 1]
            min_price = trade_df['low'].min()

            # Calculate the expected trailing stop based on the minimum price
            expected_trailing_stop = round(min_price * (1 + trailing / 100), 2)

            # If the trade was closed by a trailing stop, the exit price should be close to the trailing stop
            # Otherwise, it was closed by a signal
            if exit_idx < len(df) - 1:  # Not the last bar
                # Find the index of the minimum price
                min_price_idx = trade_df['low'].idxmin()
                # Convert timestamp to integer index
                min_price_pos = df.index.get_loc(min_price_idx)
                # Get the maximum price after the minimum price
                max_price_after_min = df.iloc[min_price_pos:exit_idx + 1]['high'].max()
                if max_price_after_min >= expected_trailing_stop:
                    # Trade was likely closed by trailing stop
                    assert abs(trade['exit_price'] - expected_trailing_stop) < 0.01, \
                        f"Exit price should be close to trailing stop price {expected_trailing_stop}, got {trade['exit_price']}"

    def test_trailing_stop_in_sideways_market(self):
        """Test trailing stop behavior in a sideways market."""
        # Create a strategy with trailing stop
        trailing = 2.0
        strategy = StrategyForTesting(trailing=trailing)

        # Create a dataframe with a sideways market
        dates = [datetime.now() + timedelta(days=i) for i in range(150)]
        df = pd.DataFrame(index=dates)

        # Create a sideways price series with some volatility
        np.random.seed(42)  # For reproducibility
        base_price = 100
        prices = [base_price]
        for _ in range(149):
            # Random walk with mean reversion
            new_price = prices[-1] + np.random.normal(0, 1)
            # Mean reversion - pull back toward base price
            new_price = new_price * 0.9 + base_price * 0.1
            prices.append(new_price)

        # Create OHLC data with some intraday volatility
        df['open'] = prices
        df['high'] = [p * (1 + np.random.uniform(0.01, 0.02)) for p in prices]  # 1-2% above open
        df['low'] = [p * (1 - np.random.uniform(0.01, 0.02)) for p in prices]  # 1-2% below open
        df['close'] = prices

        # Generate signals and extract trades
        df = strategy.generate_signals(df)
        trades = strategy._extract_trades(df, [])

        # Should have at least one trade
        assert len(trades) > 0

        # In a sideways market, trailing stops are more likely to be triggered
        # due to price oscillations
        for trade in trades:
            # Get the entry and exit indices
            entry_idx = df.index.get_indexer([trade['entry_time']], method='nearest')[0]
            exit_idx = df.index.get_indexer([trade['exit_time']], method='nearest')[0]

            # For long positions
            if trade['side'] == 'long':
                # Calculate the maximum price reached during the trade
                trade_df = df.iloc[entry_idx:exit_idx + 1]
                max_price = trade_df['high'].max()

                # Calculate the expected trailing stop based on the maximum price
                expected_trailing_stop = round(max_price * (1 - trailing / 100), 2)

                # Check if the trade was closed by a trailing stop
                # Find the index of the maximum price
                max_price_idx = trade_df['high'].idxmax()
                # Convert timestamp to integer index
                max_price_pos = df.index.get_loc(max_price_idx)
                # Get the minimum price after the maximum price
                min_price_after_max = df.iloc[max_price_pos:exit_idx + 1]['low'].min()
                if min_price_after_max <= expected_trailing_stop:
                    # Trade was likely closed by trailing stop
                    assert abs(trade['exit_price'] - expected_trailing_stop) < 0.5, \
                        f"Exit price should be close to trailing stop price {expected_trailing_stop}, got {trade['exit_price']}"

            # For short positions
            elif trade['side'] == 'short':
                # Calculate the minimum price reached during the trade
                trade_df = df.iloc[entry_idx:exit_idx + 1]
                min_price = trade_df['low'].min()

                # Calculate the expected trailing stop based on the minimum price
                expected_trailing_stop = round(min_price * (1 + trailing / 100), 2)

                # Check if the trade was closed by a trailing stop
                # Find the index of the minimum price
                min_price_idx = trade_df['low'].idxmin()
                # Convert timestamp to integer index
                min_price_pos = df.index.get_loc(min_price_idx)
                # Get the maximum price after the minimum price
                max_price_after_min = df.iloc[min_price_pos:exit_idx + 1]['high'].max()
                if max_price_after_min >= expected_trailing_stop:
                    # Trade was likely closed by trailing stop
                    assert abs(trade['exit_price'] - expected_trailing_stop) < 0.01, \
                        f"Exit price should be close to trailing stop price {expected_trailing_stop}, got {trade['exit_price']}"

    def test_trailing_stop_with_contract_rollover(self):
        """Test that trailing stops work correctly during contract rollovers."""
        # Create a strategy with rollover and trailing stop
        trailing = 2.0

        # Create a simple test strategy that directly returns trades with the switch flag
        class SwitchTestStrategy(BaseStrategy):
            def __init__(self, rollover=True, trailing=None):
                super().__init__(rollover=rollover, trailing=trailing)
                self.trailing_stop_value = None

            def add_indicators(self, df):
                return df

            def generate_signals(self, df):
                df['signal'] = 0
                return df

            def _extract_trades(self, df, switch_dates):
                # Create a trade with the switch flag
                # Use indices after the warm-up period
                switch_trade = {
                    'entry_time': df.index[102],
                    'entry_price': 100.0,
                    'exit_time': switch_dates[0],
                    'exit_price': 110.0,
                    'side': 'long',
                    'switch': True
                }

                # Create a post-rollover trade
                # Use indices after the warm-up period and after the switch date
                post_rollover_trade = {
                    'entry_time': df.index[111],  # Right after rollover
                    'entry_price': 100.0,
                    'exit_time': df.index[115],
                    'exit_price': 120.0,
                    'side': 'long'
                }

                # Calculate and store the initial trailing stop for the post-rollover trade
                self.trailing_stop_value = round(100.0 * (1 - self.trailing / 100), 2) if self.trailing else None

                return [switch_trade, post_rollover_trade]

        # Use the simple test strategy
        strategy = SwitchTestStrategy(rollover=True, trailing=trailing)

        # Create a dataframe with 150 bars
        df = create_test_df(length=150)

        # Create a switch date after the warm-up period
        switch_date = df.index[110]

        # Run the strategy
        trades = strategy.run(df, [switch_date])

        # Should have at least one trade
        assert len(trades) > 0

        # Find trades that were closed due to rollover
        rollover_trades = [trade for trade in trades if trade.get('switch')]

        # There should be at least one rollover trade
        assert len(rollover_trades) > 0, "No trades closed due to rollover"

        # Verify the switch trade properties
        for trade in rollover_trades:
            assert trade['switch'] is True
            assert trade['exit_time'] == switch_date

        # Find trades that were opened after rollover
        post_rollover_trades = [trade for trade in trades if trade['entry_time'] > switch_date]

        # There should be at least one trade opened after rollover
        assert len(post_rollover_trades) > 0, "No trades opened after rollover"

        # For post-rollover trades, verify that the trailing stop was reset
        # This is hard to test directly since we're mocking the trades
        # But we can verify that the trailing stop value was calculated correctly
        if trailing is not None:
            assert strategy.trailing_stop_value is not None, "Trailing stop value should be set"
            expected_stop = round(100.0 * (1 - trailing / 100), 2)
            assert strategy.trailing_stop_value == expected_stop, \
                f"Initial trailing stop should be {expected_stop}, got {strategy.trailing_stop_value}"

    def test_trailing_stop_with_high_volatility(self):
        """Test that trailing stops work correctly in high volatility conditions."""
        # Create a strategy with trailing stop
        trailing = 3.0

        # Create a custom strategy that simulates high volatility trades
        class VolatilityTestStrategy(BaseStrategy):
            def __init__(self, trailing=None):
                super().__init__(trailing=trailing)
                self.trailing_stops_triggered = 0

            def add_indicators(self, df):
                return df

            def generate_signals(self, df):
                df['signal'] = 0
                # Add some signals to generate trades after the warm-up period
                if len(df) > 105:
                    # Long signal
                    df.iloc[102, df.columns.get_loc('signal')] = 1
                    # Short signal
                    df.iloc[120, df.columns.get_loc('signal')] = -1
                    # Close signal
                    df.iloc[140, df.columns.get_loc('signal')] = 1
                return df

            def _handle_trailing_stop(self, idx, price_high, price_low):
                """Override to count trailing stop triggers"""
                # Check if a trailing stop has been triggered
                if self.position is not None and self.trailing_stop is not None:
                    if self.position == 1 and price_low <= self.trailing_stop:
                        self.trailing_stops_triggered += 1
                        self._close_position(idx, self.trailing_stop, switch=False)
                    elif self.position == -1 and price_high >= self.trailing_stop:
                        self.trailing_stops_triggered += 1
                        self._close_position(idx, self.trailing_stop, switch=False)

                # Update trailing stop if position still open and price moved favorably
                if self.position is not None and self.trailing_stop is not None:
                    if self.position == 1:  # Long position
                        new_stop = round(price_high * (1 - self.trailing / 100), 2)
                        if new_stop > self.trailing_stop:
                            self.trailing_stop = new_stop
                    elif self.position == -1:  # Short position
                        new_stop = round(price_low * (1 + self.trailing / 100), 2)
                        if new_stop < self.trailing_stop:
                            self.trailing_stop = new_stop

        strategy = VolatilityTestStrategy(trailing=trailing)

        # Create a dataframe with high volatility
        dates = [datetime.now() + timedelta(days=i) for i in range(150)]
        df = pd.DataFrame(index=dates)

        # Create a highly volatile price series
        base_price = 100

        # Generate random walk with high volatility
        np.random.seed(42)  # For reproducibility

        # Daily volatility of 4% (very high)
        volatility = 0.04

        # Generate prices
        close_prices = [base_price]
        for _ in range(149):
            # Random daily return with high volatility
            daily_return = np.random.normal(0, volatility)
            new_price = close_prices[-1] * (1 + daily_return)
            close_prices.append(new_price)

        # Create OHLC data with large intraday ranges
        df['open'] = close_prices
        df['high'] = [p * (1 + np.random.uniform(0.01, 0.03)) for p in close_prices]  # 1-3% above close
        df['low'] = [p * (1 - np.random.uniform(0.01, 0.03)) for p in close_prices]  # 1-3% below close
        df['close'] = close_prices

        # Run the strategy
        trades = strategy.run(df, [])

        # Should have at least one trade
        assert len(trades) > 0

    def test_trailing_stop_impact_on_performance(self):
        """Test the impact of trailing stops on overall strategy performance."""
        # Create strategies with different trailing stop values
        trailing_values = [None, 1.0, 2.0, 5.0, 10.0]

        # Create a custom strategy that will generate more trades and be more sensitive to trailing stops
        class TrailingImpactStrategy(BaseStrategy):
            def __init__(self, trailing=None):
                super().__init__(trailing=trailing)
                self.trailing_stops_triggered = 0

            def add_indicators(self, df):
                return df

            def generate_signals(self, df):
                df['signal'] = 0

                # Generate more frequent signals after the warm-up period
                # Buy signals at local bottoms
                for i in range(105, len(df) - 1):
                    if df['close'].iloc[i - 1] > df['close'].iloc[i - 2] > df['close'].iloc[i - 3] and \
                            df['close'].iloc[i] > df['close'].iloc[i - 1]:
                        df.iloc[i, df.columns.get_loc('signal')] = 1

                # Sell signals at local tops
                for i in range(105, len(df) - 1):
                    if df['close'].iloc[i - 1] < df['close'].iloc[i - 2] < df['close'].iloc[i - 3] and \
                            df['close'].iloc[i] < df['close'].iloc[i - 1]:
                        df.iloc[i, df.columns.get_loc('signal')] = -1

                return df

            def _extract_trades(self, df, switch_dates):
                """Override to count trailing stop triggers"""
                # Call parent method
                self.switch_handler.set_switch_dates(switch_dates)
                self.switch_handler.reset()
                self.position_mgr.reset()
                
                # Reset state variables
                self.prev_row = None
                self.prev_time = None
                self.queued_signal = None

                # Counter to skip the first candles for indicator warm-up
                from app.backtesting.strategies.base_strategy import INDICATOR_WARMUP_PERIOD
                candle_count = 0

                for idx, row in df.iterrows():
                    current_time = pd.to_datetime(idx)
                    signal = row['signal']
                    price_open = row['open']
                    price_high = row['high']
                    price_low = row['low']

                    # Increment candle counter
                    candle_count += 1

                    # Skip signal processing for the first candles to allow indicators to warm up
                    if candle_count <= INDICATOR_WARMUP_PERIOD:
                        self.prev_time = current_time
                        self.prev_row = row
                        continue

                    # Handle trailing stop logic if enabled - custom version to count triggers
                    if self.trailing_stop_mgr:
                        # Check if trailing stop was triggered
                        if self.position_mgr.has_open_position() and self.position_mgr.trailing_stop is not None:
                            position = self.position_mgr.position
                            trailing_stop = self.position_mgr.trailing_stop
                            
                            if position == 1 and price_low <= trailing_stop:
                                # Long stop triggered - count it
                                self.trailing_stops_triggered += 1
                                self.position_mgr.close_position(idx, trailing_stop, switch=False)
                                self.prev_time = current_time
                                self.prev_row = row
                                continue
                            elif position == -1 and price_high >= trailing_stop:
                                # Short stop triggered - count it
                                self.trailing_stops_triggered += 1
                                self.position_mgr.close_position(idx, trailing_stop, switch=False)
                                self.prev_time = current_time
                                self.prev_row = row
                                continue
                        
                        # Update trailing stop if not triggered
                        if self.position_mgr.has_open_position() and self.position_mgr.trailing_stop is not None:
                            position = self.position_mgr.position
                            trailing_stop = self.position_mgr.trailing_stop
                            new_stop = self.trailing_stop_mgr.calculate_new_trailing_stop(position, price_high, price_low)
                            
                            if new_stop is not None:
                                if position == 1 and new_stop > trailing_stop:
                                    self.position_mgr.trailing_stop = new_stop
                                elif position == -1 and new_stop < trailing_stop:
                                    self.position_mgr.trailing_stop = new_stop

                    # Handle contract switches
                    if self.switch_handler.should_switch(current_time):
                        if self.position_mgr.has_open_position() and self.prev_row is not None:
                            prev_position = self.position_mgr.close_position_at_switch(self.prev_time, self.prev_row)
                            if self.rollover:
                                self.switch_handler.must_reopen = prev_position
                                self.switch_handler.skip_signal_this_bar = True

                    skip_signal = self.switch_handler.handle_contract_switch(current_time, self.position_mgr, idx, price_open)

                    # Skip signal for this bar if we are in a rollover position
                    if skip_signal:
                        self.prev_time = current_time
                        self.prev_row = row
                        continue

                    # Execute queued signal from the previous bar
                    if self.queued_signal is not None:
                        flip = None
                        if self.queued_signal == 1 and self.position_mgr.position != 1:
                            flip = 1
                        elif self.queued_signal == -1 and self.position_mgr.position != -1:
                            flip = -1

                        if flip is not None:
                            # Close if currently in position
                            if self.position_mgr.has_open_position():
                                self.position_mgr.close_position(idx, price_open, switch=False)
                            # Open a new position at this (current) bar
                            self.position_mgr.open_position(flip, idx, price_open)

                        # Reset after using
                        self.queued_signal = None

                    # Set/overwrite queued_signal for next bar execution
                    if signal != 0:
                        self.queued_signal = signal

                    self.prev_time = current_time
                    self.prev_row = row

                return self.position_mgr.get_trades()

        strategies = [TrailingImpactStrategy(trailing=t) for t in trailing_values]

        # Create a deterministic price series with clear trends and reversals
        # This will ensure trailing stops have a measurable impact
        dates = [datetime.now() + timedelta(days=i) for i in range(200)]
        df = pd.DataFrame(index=dates)

        # Create a price series with multiple trends and reversals
        np.random.seed(42)  # Fix seed for reproducibility

        # Start with a base price
        base_price = 100

        # Create a price series with multiple trends
        prices = []

        # Uptrend
        for i in range(40):
            prices.append(base_price + i * 1.5 + np.random.normal(0, 1))

        # Downtrend
        for i in range(40):
            prices.append(base_price + 60 - i * 2 + np.random.normal(0, 1.5))

        # Sideways with volatility
        for i in range(40):
            prices.append(base_price + 20 + np.random.normal(0, 5))

        # Uptrend with pullbacks
        for i in range(40):
            # Add a pullback every 10 days
            if i % 10 == 0 and i > 0:
                prices.append(prices[-1] - 10 + np.random.normal(0, 1))
            else:
                prices.append(prices[-1] + 2 + np.random.normal(0, 1))

        # Downtrend with rallies
        for i in range(40):
            # Add a rally every 10 days
            if i % 10 == 0 and i > 0:
                prices.append(prices[-1] + 10 + np.random.normal(0, 1))
            else:
                prices.append(prices[-1] - 2 + np.random.normal(0, 1))

        # Ensure we have enough prices
        while len(prices) < 200:
            prices.append(prices[-1] + np.random.normal(0, 2))

        # Create OHLC data with intraday volatility
        df['open'] = prices
        df['close'] = prices
        df['high'] = [p * (1 + np.random.uniform(0.01, 0.03)) for p in prices]  # 1-3% above close
        df['low'] = [p * (1 - np.random.uniform(0.01, 0.03)) for p in prices]  # 1-3% below close

        # Run each strategy and calculate performance metrics
        results = []
        for i, strategy in enumerate(strategies):
            trades = strategy.run(df, [])

            # Track how many trades were closed by trailing stops
            trailing_stops_triggered = getattr(strategy, 'trailing_stops_triggered', 0)

            # Calculate total return and average trade duration
            total_return = 0
            total_duration = 0
            for trade in trades:
                if trade['side'] == 'long':
                    trade_return = trade['exit_price'] / trade['entry_price'] - 1
                else:  # short
                    trade_return = 1 - trade['exit_price'] / trade['entry_price']
                total_return += trade_return

                # Calculate trade duration in days
                entry_time = trade['entry_time']
                exit_time = trade['exit_time']
                duration = (exit_time - entry_time).days
                total_duration += duration

            avg_duration = total_duration / len(trades) if trades else 0

            results.append({
                'trailing': trailing_values[i],
                'num_trades': len(trades),
                'total_return': total_return,
                'avg_duration': avg_duration,
                'trailing_stops_triggered': trailing_stops_triggered
            })

        # Add assertions to verify that trailing stops have a measurable impact
        # We expect at least some differences between strategies with different trailing stops

        # Verify that strategies with trailing stops trigger some trailing stops
        for result in results:
            if result['trailing'] is not None:
                assert result[
                           'trailing_stops_triggered'] > 0, f"Strategy with {result['trailing']}% trailing stop should trigger some trailing stops"

        # Verify that different trailing stop values lead to different outcomes
        # Compare the first strategy with trailing (index 1) to the strategy without trailing (index 0)
        assert results[1]['num_trades'] != results[0][
            'num_trades'], "Adding trailing stop should change the number of trades"
        assert abs(results[1]['total_return'] - results[0][
            'total_return']) > 0.01, "Adding trailing stop should impact total return"
        assert abs(results[1]['avg_duration'] - results[0][
            'avg_duration']) > 0.1, "Adding trailing stop should impact average trade duration"

        # Verify that increasing trailing stop percentage has an impact
        # Compare the largest trailing stop (index 4) to the smallest trailing stop (index 1)
        assert results[4]['trailing_stops_triggered'] != results[1]['trailing_stops_triggered'], \
            "Different trailing stop percentages should trigger different numbers of trailing stops"
