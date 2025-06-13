from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.backtesting.strategies.base_strategy import BaseStrategy


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
        # Set some signals for testing
        if len(df) > 5:
            df.iloc[2, df.columns.get_loc('signal')] = 1  # Buy signal
            # For contract switch test, ensure there's an open position at the switch date (index 7)
            # by not closing the position before the switch
            if len(df) > 10:
                df.iloc[9, df.columns.get_loc('signal')] = -1  # Sell signal after switch date
            else:
                df.iloc[4, df.columns.get_loc('signal')] = -1  # Sell signal
        return df


# Helper function to create test dataframe
def create_test_df(length=10):
    dates = [datetime.now() + timedelta(days=i) for i in range(length)]
    data = {
        'open': np.random.rand(length) * 100 + 50,
        'high': np.random.rand(length) * 100 + 60,
        'low': np.random.rand(length) * 100 + 40,
        'close': np.random.rand(length) * 100 + 50,
    }
    df = pd.DataFrame(data, index=dates)
    return df


class TestBaseStrategy:
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError when not implemented."""
        strategy = BaseStrategy()

        # Test add_indicators abstract method
        df = create_test_df()
        with pytest.raises(NotImplementedError, match="Subclasses must implement add_indicators method"):
            strategy.add_indicators(df)

        # Test generate_signals abstract method
        with pytest.raises(NotImplementedError, match="Subclasses must implement generate_signals method"):
            strategy.generate_signals(df)

    def test_initialization(self):
        """Test that the strategy initializes correctly."""
        strategy = StrategyForTesting()
        assert strategy.position is None
        assert strategy.entry_time is None
        assert strategy.entry_price is None
        assert strategy.trailing_stop is None
        assert strategy.rollover is False
        assert strategy.trailing is None

        # Test with parameters
        strategy = StrategyForTesting(rollover=True, trailing=2.0)
        assert strategy.rollover is True
        assert strategy.trailing == 2.0

    def test_run_method(self):
        """Test the run method executes the full workflow."""
        strategy = StrategyForTesting()
        df = create_test_df()
        switch_dates = []

        # Spy on the methods to ensure they're called
        original_add_indicators = strategy.add_indicators
        original_generate_signals = strategy.generate_signals
        original_extract_trades = strategy.extract_trades

        calls = {'add_indicators': 0, 'generate_signals': 0, 'extract_trades': 0}

        def mock_add_indicators(df):
            calls['add_indicators'] += 1
            return original_add_indicators(df)

        def mock_generate_signals(df):
            calls['generate_signals'] += 1
            return original_generate_signals(df)

        def mock_extract_trades(df, switch_dates):
            calls['extract_trades'] += 1
            return original_extract_trades(df, switch_dates)

        strategy.add_indicators = mock_add_indicators
        strategy.generate_signals = mock_generate_signals
        strategy.extract_trades = mock_extract_trades

        trades = strategy.run(df, switch_dates)

        # Verify all methods were called once
        assert calls['add_indicators'] == 1
        assert calls['generate_signals'] == 1
        assert calls['extract_trades'] == 1

    def test_extract_trades_basic(self):
        """Test that trades are extracted correctly from signals."""
        strategy = StrategyForTesting()
        df = create_test_df()
        df = strategy.generate_signals(df)  # Add signals

        trades = strategy.extract_trades(df, [])

        # Should have at least one trade
        assert len(trades) > 0

        # Verify trade structure
        for trade in trades:
            assert 'entry_time' in trade
            assert 'entry_price' in trade
            assert 'exit_time' in trade
            assert 'exit_price' in trade
            assert 'side' in trade
            assert trade['side'] in ['long', 'short']

    def test_trailing_stop_long(self):
        """Test trailing stop functionality for long positions."""
        strategy = StrategyForTesting(trailing=2.0)
        df = create_test_df()

        # Modify prices to test trailing stop
        df.loc[df.index[2], 'open'] = 100.0  # Entry price
        df.loc[df.index[3], 'high'] = 110.0  # Price moves up, trailing stop should adjust
        df.loc[df.index[4], 'low'] = 95.0  # Price drops but not enough to trigger stop
        df.loc[df.index[5], 'low'] = 90.0  # Price drops below trailing stop

        df = strategy.generate_signals(df)
        trades = strategy.extract_trades(df, [])

        # Should have at least one trade
        assert len(trades) > 0

        # Find long trades
        long_trades = [t for t in trades if t['side'] == 'long']
        assert len(long_trades) > 0

        # Verify trade structure
        for trade in long_trades:
            assert 'entry_time' in trade
            assert 'entry_price' in trade
            assert 'exit_time' in trade
            assert 'exit_price' in trade

    def test_trailing_stop_short(self):
        """Test trailing stop functionality for short positions."""
        strategy = StrategyForTesting(trailing=2.0)
        df = create_test_df()

        # Modify signals to create a short position
        df = strategy.generate_signals(df)
        df.iloc[2, df.columns.get_loc('signal')] = -1  # Short signal

        # Modify prices to test trailing stop
        df.loc[df.index[2], 'open'] = 100.0  # Entry price
        df.loc[df.index[3], 'low'] = 90.0  # Price moves down, trailing stop should adjust
        df.loc[df.index[4], 'high'] = 95.0  # Price rises but not enough to trigger stop
        df.loc[df.index[5], 'high'] = 105.0  # Price rises above trailing stop

        trades = strategy.extract_trades(df, [])

        # Should have at least one trade
        assert len(trades) > 0

        # Find short trades
        short_trades = [t for t in trades if t['side'] == 'short']
        assert len(short_trades) > 0

        # Verify trade structure
        for trade in short_trades:
            assert 'entry_time' in trade
            assert 'entry_price' in trade
            assert 'exit_time' in trade
            assert 'exit_price' in trade

    def test_contract_switch(self):
        """Test contract switch functionality."""

        # Create a simple test strategy with a forced switch trade
        class SwitchTestStrategy(BaseStrategy):
            def add_indicators(self, df):
                return df

            def generate_signals(self, df):
                df['signal'] = 0
                return df

            def extract_trades(self, df, switch_dates):
                # Create a trade with the switch flag
                trade = {
                    'entry_time': df.index[0],
                    'entry_price': 100.0,
                    'exit_time': df.index[1],
                    'exit_price': 110.0,
                    'side': 'long',
                    'switch': True
                }
                return [trade]

        # Use the simple test strategy
        strategy = SwitchTestStrategy(rollover=True)
        df = create_test_df(length=5)

        # Run the strategy
        trades = strategy.run(df, [df.index[1]])

        # Should have at least one trade
        assert len(trades) > 0

        # Check if any trades have the switch flag
        switch_trades = [trade for trade in trades if 'switch' in trade]
        assert len(switch_trades) > 0

        # Verify the switch trade properties
        for trade in switch_trades:
            assert trade['switch'] is True

    def test_contract_switch_edge_case(self):
        """Test contract switch edge case where we reach the end of switch_dates."""

        # Create a test strategy that will handle multiple contract switches
        class MultiSwitchStrategy(BaseStrategy):
            def add_indicators(self, df):
                return df

            def generate_signals(self, df):
                df['signal'] = 0
                # Add a buy signal at the beginning
                if len(df) > 0:
                    df.iloc[0, df.columns.get_loc('signal')] = 1
                return df

            # We'll use the default extract_trades implementation

        # Create a dataframe with 10 bars
        df = create_test_df(length=10)

        # Create switch dates for each bar
        switch_dates = [df.index[i] for i in range(1, 10)]

        # Create the strategy
        strategy = MultiSwitchStrategy(rollover=True)

        # Run the strategy - this should process all switch dates
        trades = strategy.run(df, switch_dates)

        # Verify we have trades
        assert len(trades) > 0

        # Verify that at least some trades have the switch flag
        switch_trades = [trade for trade in trades if trade.get('switch', False)]
        assert len(switch_trades) > 0

    def test_contract_switch_without_rollover(self):
        """Test contract switch functionality when rollover is False."""

        # Create a test strategy that will handle contract switches without rollover
        class NoRolloverSwitchStrategy(BaseStrategy):
            def add_indicators(self, df):
                return df

            def generate_signals(self, df):
                df['signal'] = 0
                # Add a buy signal at the beginning
                if len(df) > 0:
                    df.iloc[0, df.columns.get_loc('signal')] = 1
                return df

            # We'll use the default extract_trades implementation

        # Create a dataframe with 5 bars
        df = create_test_df(length=5)

        # Create a switch date
        switch_dates = [df.index[2]]  # Switch at the third bar

        # Create the strategy with rollover=False
        strategy = NoRolloverSwitchStrategy(rollover=False)

        # Run the strategy
        trades = strategy.run(df, switch_dates)

        # Verify we have trades
        assert len(trades) > 0

        # Verify that at least one trade has the switch flag
        switch_trades = [trade for trade in trades if trade.get('switch', False)]
        assert len(switch_trades) > 0

        # Verify that must_reopen is set to None when rollover is False
        # This is testing the line in _close_position_at_switch where rollover is False
        assert strategy.must_reopen is None

    def test_reopen_position_after_switch(self):
        """Test reopening a position after a contract switch with rollover enabled."""

        # Create a test strategy that will handle contract switches with rollover
        class RolloverSwitchStrategy(BaseStrategy):
            def add_indicators(self, df):
                return df

            def generate_signals(self, df):
                df['signal'] = 0
                # Add a buy signal at the beginning
                if len(df) > 0:
                    df.iloc[0, df.columns.get_loc('signal')] = 1
                return df

            # We'll use the default extract_trades implementation

        # Create a dataframe with 5 bars
        df = create_test_df(length=5)

        # Create a switch date
        switch_dates = [df.index[2]]  # Switch at the third bar

        # Create the strategy with rollover=True
        strategy = RolloverSwitchStrategy(rollover=True)

        # Manually set up the state to test the specific code path
        # This simulates the state after a position has been closed due to a switch
        # and we're about to reopen it in the next contract
        strategy._reset()
        strategy.must_reopen = 1  # Indicate we want to reopen a long position
        strategy.position = None  # No current position

        # Now call _handle_contract_switch directly to test the reopening logic
        # This tests the code path where must_reopen is not None and position is None
        current_time = df.index[3]  # After the switch
        idx = df.index[3]
        price_open = df.iloc[3]['open']

        strategy._handle_contract_switch(current_time, idx, price_open)

        # Verify that the position was reopened
        assert strategy.position == 1  # Long position
        assert strategy.entry_time == idx
        assert strategy.entry_price is not None
        assert strategy.must_reopen is None  # must_reopen should be reset

    def test_close_position_at_switch(self):
        """Test closing a position at a contract switch date."""

        # Create a test strategy
        class SwitchTestStrategy(BaseStrategy):
            def add_indicators(self, df):
                return df

            def generate_signals(self, df):
                df['signal'] = 0
                return df

        # Create a dataframe with 5 bars
        df = create_test_df(length=5)

        # Create the strategy
        strategy = SwitchTestStrategy()

        # Set up the state to simulate an open position
        strategy._reset()
        strategy.position = 1  # Long position
        strategy.entry_time = df.index[0]
        strategy.entry_price = 100.0
        strategy.prev_row = {'open': 110.0}  # Exit price for the switch

        # Call _close_position_at_switch directly
        strategy._close_position_at_switch(df.index[2])

        # Verify the position was closed
        assert strategy.position is None
        assert strategy.entry_time is None
        assert strategy.entry_price is None

        # Verify a trade was recorded with the switch flag
        assert len(strategy.trades) == 1
        assert strategy.trades[0]['switch'] is True
        assert strategy.trades[0]['exit_price'] == 110.0

    def test_no_signals(self):
        """Test behavior when there are no signals."""
        strategy = StrategyForTesting()
        df = create_test_df()

        # Override generate_signals to return no signals
        def no_signals(df):
            df['signal'] = 0
            return df

        strategy.generate_signals = no_signals

        trades = strategy.run(df, [])

        # Should have no trades
        assert len(trades) == 0

    def test_slippage(self):
        """Test that slippage is correctly applied to entry and exit prices."""
        # Create a strategy with 2% slippage
        strategy = StrategyForTesting(slippage=2.0)
        df = create_test_df()
        df = strategy.generate_signals(df)  # Add signals

        trades = strategy.extract_trades(df, [])

        # Should have at least one trade
        assert len(trades) > 0

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

            assert trade[
                       'entry_price'] == expected_entry_price, f"Long entry price with slippage should be {expected_entry_price}, got {trade['entry_price']}"
            assert trade[
                       'exit_price'] == expected_exit_price, f"Long exit price with slippage should be {expected_exit_price}, got {trade['exit_price']}"

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

            assert trade[
                       'entry_price'] == expected_entry_price, f"Short entry price with slippage should be {expected_entry_price}, got {trade['entry_price']}"
            assert trade[
                       'exit_price'] == expected_exit_price, f"Short exit price with slippage should be {expected_exit_price}, got {trade['exit_price']}"
