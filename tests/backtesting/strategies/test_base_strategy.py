from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.backtesting.strategies.base_strategy import BaseStrategy


# Create a concrete implementation of BaseStrategy for testing
class TestStrategy(BaseStrategy):
    def __init__(self, rollover=False, trailing=None):
        super().__init__(rollover=rollover, trailing=trailing)

    def add_indicators(self, df):
        # Simple implementation for testing
        return df

    def generate_signals(self, df):
        # Simple implementation for testing
        df['signal'] = 0
        # Set some signals for testing
        if len(df) > 5:
            df.iloc[2, df.columns.get_loc('signal')] = 1  # Buy signal
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
        strategy = TestStrategy(rollover=True)
        df = create_test_df(length=15)

        # Create a switch date in the middle of the dataframe
        switch_date = df.index[7]

        df = strategy.generate_signals(df)
        trades = strategy.extract_trades(df, [switch_date])

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
