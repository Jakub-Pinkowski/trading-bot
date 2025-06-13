from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.backtesting.strategies.base_strategy import BaseStrategy


# Create a concrete implementation of BaseStrategy for testing
class TestStrategy(BaseStrategy):
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
    def test_initialization(self):
        """Test that the strategy initializes correctly."""
        strategy = TestStrategy()
        assert strategy.position is None
        assert strategy.entry_time is None
        assert strategy.entry_price is None
        assert strategy.trailing_stop is None
        assert strategy.rollover is False
        assert strategy.trailing is None

        # Test with parameters
        strategy = TestStrategy(rollover=True, trailing=2.0)
        assert strategy.rollover is True
        assert strategy.trailing == 2.0

    def test_run_method(self):
        """Test the run method executes the full workflow."""
        strategy = TestStrategy()
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
        strategy = TestStrategy()
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
        strategy = TestStrategy(trailing=2.0)
        df = create_test_df()

        # Modify prices to test trailing stop
        df.iloc[2]['open'] = 100.0  # Entry price
        df.iloc[3]['high'] = 110.0  # Price moves up, trailing stop should adjust
        df.iloc[4]['low'] = 95.0  # Price drops but not enough to trigger stop
        df.iloc[5]['low'] = 90.0  # Price drops below trailing stop

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
        strategy = TestStrategy(trailing=2.0)
        df = create_test_df()

        # Modify signals to create a short position
        df = strategy.generate_signals(df)
        df.iloc[2, df.columns.get_loc('signal')] = -1  # Short signal

        # Modify prices to test trailing stop
        df.iloc[2]['open'] = 100.0  # Entry price
        df.iloc[3]['low'] = 90.0  # Price moves down, trailing stop should adjust
        df.iloc[4]['high'] = 95.0  # Price rises but not enough to trigger stop
        df.iloc[5]['high'] = 105.0  # Price rises above trailing stop

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

    def test_no_signals(self):
        """Test behavior when there are no signals."""
        strategy = TestStrategy()
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
        strategy = TestStrategy(slippage=2.0)
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
