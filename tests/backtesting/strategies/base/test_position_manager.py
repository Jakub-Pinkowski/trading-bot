"""
Tests for PositionManager.

Tests cover:
- Initialization and configuration
- Position state management
- Slippage calculations (entry and exit)
- Opening and closing positions
- Trade recording and retrieval
- Trailing stop integration
- Contract switch handling
- Edge cases and error conditions
"""
import pandas as pd
import pytest

from app.backtesting.strategies.base.position_manager import PositionManager
from config import TICK_SIZES


# ==================== Fixtures ====================

@pytest.fixture
def position_manager():
    """Standard position manager instance for ZS."""
    return PositionManager(
        slippage_ticks=1,
        symbol='ZS',
        trailing=None
    )


@pytest.fixture
def position_manager_with_trailing():
    """Position manager with trailing stop enabled."""
    return PositionManager(
        slippage_ticks=1,
        symbol='ZS',
        trailing=2.0
    )


@pytest.fixture
def position_manager_no_slippage():
    """Position manager with zero slippage."""
    return PositionManager(
        slippage_ticks=0,
        symbol='ZS',
        trailing=None
    )


@pytest.fixture
def sample_timestamp():
    """Sample timestamp for testing."""
    return pd.Timestamp('2025-01-01 10:00:00')


@pytest.fixture
def sample_row():
    """Sample OHLCV row for testing."""
    return pd.Series({
        'open': 100.0,
        'high': 102.0,
        'low': 98.0,
        'close': 101.0,
        'volume': 1000
    })


# ==================== Test Classes ====================

class TestPositionManagerInitialization:
    """Test PositionManager initialization and configuration."""

    def test_initialization_with_basic_parameters(self):
        """Test position manager initializes with basic parameters."""
        pm = PositionManager(
            slippage_ticks=2,
            symbol='CL',
            trailing=None
        )

        assert pm.slippage_ticks == 2
        assert pm.symbol == 'CL'
        assert pm.trailing is None
        assert pm.position is None
        assert pm.entry_time is None
        assert pm.entry_price is None
        assert pm.trailing_stop is None
        assert pm.trades == []

    def test_initialization_with_trailing_stop(self):
        """Test position manager initializes with trailing stop enabled."""
        pm = PositionManager(
            slippage_ticks=1,
            symbol='ES',
            trailing=3.0
        )

        assert pm.trailing == 3.0
        assert pm.trailing_stop is None  # Not set until position opened

    def test_initialization_with_zero_slippage(self):
        """Test position manager initializes with zero slippage."""
        pm = PositionManager(
            slippage_ticks=0,
            symbol='ZS',
            trailing=None
        )

        assert pm.slippage_ticks == 0

    def test_initialization_different_symbols(self):
        """Test position manager works with different symbols."""
        symbols = ['ZS', 'CL', 'ES', 'GC', 'NQ']

        for symbol in symbols:
            pm = PositionManager(
                slippage_ticks=1,
                symbol=symbol,
                trailing=None
            )
            assert pm.symbol == symbol


class TestStateManagement:
    """Test position state management operations."""

    def test_reset_clears_all_state(self, position_manager, sample_timestamp):
        """Test reset() clears all state variables."""
        # Open a position first
        position_manager.open_position(1, sample_timestamp, 100.0)
        position_manager.trades.append({'test': 'trade'})

        # Reset
        position_manager.reset()

        # All state should be cleared
        assert position_manager.position is None
        assert position_manager.entry_time is None
        assert position_manager.entry_price is None
        assert position_manager.trailing_stop is None
        assert position_manager.trades == []

    def test_reset_position_internal(self, position_manager, sample_timestamp):
        """Test _reset_position() clears position but not trades."""
        # Open a position
        position_manager.open_position(1, sample_timestamp, 100.0)
        position_manager.trades.append({'test': 'trade'})

        # Reset position only
        position_manager._reset_position()

        # Position cleared but trades remain
        assert position_manager.position is None
        assert position_manager.entry_time is None
        assert position_manager.entry_price is None
        assert position_manager.trailing_stop is None
        assert len(position_manager.trades) == 1

    def test_has_open_position_initially_false(self, position_manager):
        """Test has_open_position() returns False initially."""
        assert position_manager.has_open_position() is False

    def test_has_open_position_true_after_opening(self, position_manager, sample_timestamp):
        """Test has_open_position() returns True after opening position."""
        position_manager.open_position(1, sample_timestamp, 100.0)
        assert position_manager.has_open_position() is True

    def test_has_open_position_false_after_closing(self, position_manager, sample_timestamp):
        """Test has_open_position() returns False after closing position."""
        position_manager.open_position(1, sample_timestamp, 100.0)
        position_manager.close_position(sample_timestamp, 105.0, switch=False)
        assert position_manager.has_open_position() is False

    def test_get_trades_returns_empty_list_initially(self, position_manager):
        """Test get_trades() returns empty list initially."""
        trades = position_manager.get_trades()
        assert trades == []
        assert isinstance(trades, list)

    def test_get_trades_returns_completed_trades(self, position_manager, sample_timestamp):
        """Test get_trades() returns completed trades."""
        # Create a trade
        position_manager.open_position(1, sample_timestamp, 100.0)
        exit_time = sample_timestamp + pd.Timedelta(hours=1)
        position_manager.close_position(exit_time, 105.0, switch=False)

        trades = position_manager.get_trades()
        assert len(trades) == 1
        assert trades[0]['entry_time'] == sample_timestamp
        assert trades[0]['exit_time'] == exit_time


class TestSlippageCalculations:
    """Test slippage calculation methods."""

    # --- Entry Slippage ---

    def test_entry_slippage_long_position(self, position_manager):
        """Test entry slippage increases price for long positions."""
        base_price = 100.0
        adjusted = position_manager.apply_slippage_to_entry_price(1, base_price)

        # Long pays more on entry (slippage added)
        tick_size = TICK_SIZES.get('ZS', 0.01)
        expected = base_price + (1 * tick_size)
        assert adjusted == pytest.approx(expected, abs=0.01)
        assert adjusted > base_price

    def test_entry_slippage_short_position(self, position_manager):
        """Test entry slippage decreases price for short positions."""
        base_price = 100.0
        adjusted = position_manager.apply_slippage_to_entry_price(-1, base_price)

        # Short receives less on entry (slippage subtracted)
        tick_size = TICK_SIZES.get('ZS', 0.01)
        expected = base_price - (1 * tick_size)
        assert adjusted == pytest.approx(expected, abs=0.01)
        assert adjusted < base_price

    def test_entry_slippage_with_zero_ticks(self, position_manager_no_slippage):
        """Test entry slippage with zero slippage ticks."""
        base_price = 100.0

        # Long
        adjusted_long = position_manager_no_slippage.apply_slippage_to_entry_price(1, base_price)
        assert adjusted_long == base_price

        # Short
        adjusted_short = position_manager_no_slippage.apply_slippage_to_entry_price(-1, base_price)
        assert adjusted_short == base_price

    def test_entry_slippage_with_multiple_ticks(self):
        """Test entry slippage with multiple slippage ticks."""
        pm = PositionManager(slippage_ticks=3, symbol='ZS', trailing=None)
        base_price = 100.0

        adjusted_long = pm.apply_slippage_to_entry_price(1, base_price)
        tick_size = TICK_SIZES.get('ZS', 0.01)
        expected = base_price + (3 * tick_size)
        assert adjusted_long == pytest.approx(expected, abs=0.01)

    def test_entry_slippage_different_symbols(self):
        """Test entry slippage with different symbols (different tick sizes)."""
        test_symbols = [
            ('ZS', 0.25),  # 1/4 cent per bushel
            ('CL', 0.01),  # 1 cent per barrel
            ('ES', 0.25),  # 0.25 index points
        ]

        for symbol, expected_tick in test_symbols:
            pm = PositionManager(slippage_ticks=1, symbol=symbol, trailing=None)
            base_price = 100.0

            adjusted = pm.apply_slippage_to_entry_price(1, base_price)
            expected = base_price + expected_tick
            assert adjusted == pytest.approx(expected, abs=0.01)

    # --- Exit Slippage ---

    def test_exit_slippage_long_position(self, position_manager):
        """Test exit slippage decreases price for long positions."""
        base_price = 105.0
        adjusted = position_manager.apply_slippage_to_exit_price(1, base_price)

        # Long receives less on exit (slippage subtracted)
        tick_size = TICK_SIZES.get('ZS', 0.01)
        expected = base_price - (1 * tick_size)
        assert adjusted == pytest.approx(expected, abs=0.01)
        assert adjusted < base_price

    def test_exit_slippage_short_position(self, position_manager):
        """Test exit slippage increases price for short positions."""
        base_price = 95.0
        adjusted = position_manager.apply_slippage_to_exit_price(-1, base_price)

        # Short pays more on exit (slippage added)
        tick_size = TICK_SIZES.get('ZS', 0.01)
        expected = base_price + (1 * tick_size)
        assert adjusted == pytest.approx(expected, abs=0.01)
        assert adjusted > base_price

    def test_exit_slippage_with_zero_ticks(self, position_manager_no_slippage):
        """Test exit slippage with zero slippage ticks."""
        base_price = 105.0

        # Long
        adjusted_long = position_manager_no_slippage.apply_slippage_to_exit_price(1, base_price)
        assert adjusted_long == base_price

        # Short
        adjusted_short = position_manager_no_slippage.apply_slippage_to_exit_price(-1, base_price)
        assert adjusted_short == base_price

    def test_slippage_reduces_profit_for_winning_long_trade(self, position_manager):
        """Test slippage reduces profit for winning long trade."""
        entry_price = 100.0
        exit_price = 110.0

        entry_with_slip = position_manager.apply_slippage_to_entry_price(1, entry_price)
        exit_with_slip = position_manager.apply_slippage_to_exit_price(1, exit_price)

        # Profit without slippage
        profit_no_slip = exit_price - entry_price

        # Profit with slippage
        profit_with_slip = exit_with_slip - entry_with_slip

        # Slippage should reduce profit
        assert profit_with_slip < profit_no_slip

    def test_slippage_increases_loss_for_losing_long_trade(self, position_manager):
        """Test slippage increases loss for losing long trade."""
        entry_price = 100.0
        exit_price = 95.0

        entry_with_slip = position_manager.apply_slippage_to_entry_price(1, entry_price)
        exit_with_slip = position_manager.apply_slippage_to_exit_price(1, exit_price)

        # Loss without slippage
        loss_no_slip = entry_price - exit_price

        # Loss with slippage
        loss_with_slip = entry_with_slip - exit_with_slip

        # Slippage should increase loss
        assert loss_with_slip > loss_no_slip


class TestOpeningPositions:
    """Test opening position operations."""

    def test_open_long_position(self, position_manager, sample_timestamp):
        """Test opening a long position."""
        position_manager.open_position(1, sample_timestamp, 100.0)

        assert position_manager.position == 1
        assert position_manager.entry_time == sample_timestamp
        assert position_manager.entry_price > 100.0  # Slippage applied
        assert position_manager.has_open_position() is True

    def test_open_short_position(self, position_manager, sample_timestamp):
        """Test opening a short position."""
        position_manager.open_position(-1, sample_timestamp, 100.0)

        assert position_manager.position == -1
        assert position_manager.entry_time == sample_timestamp
        assert position_manager.entry_price < 100.0  # Slippage applied
        assert position_manager.has_open_position() is True

    def test_open_position_with_trailing_stop_long(self, position_manager_with_trailing, sample_timestamp):
        """Test opening long position sets trailing stop."""
        entry_price = 100.0
        position_manager_with_trailing.open_position(1, sample_timestamp, entry_price)

        # Trailing stop should be set below entry price for long
        assert position_manager_with_trailing.trailing_stop is not None
        assert position_manager_with_trailing.trailing_stop < position_manager_with_trailing.entry_price

        # Calculate expected trailing stop (2% below entry after slippage)
        expected_stop = position_manager_with_trailing.entry_price * (1 - 2.0 / 100)
        assert position_manager_with_trailing.trailing_stop == pytest.approx(expected_stop, abs=0.01)

    def test_open_position_with_trailing_stop_short(self, position_manager_with_trailing, sample_timestamp):
        """Test opening short position sets trailing stop."""
        entry_price = 100.0
        position_manager_with_trailing.open_position(-1, sample_timestamp, entry_price)

        # Trailing stop should be set above entry price for short
        assert position_manager_with_trailing.trailing_stop is not None
        assert position_manager_with_trailing.trailing_stop > position_manager_with_trailing.entry_price

        # Calculate expected trailing stop (2% above entry after slippage)
        expected_stop = position_manager_with_trailing.entry_price * (1 + 2.0 / 100)
        assert position_manager_with_trailing.trailing_stop == pytest.approx(expected_stop, abs=0.01)

    def test_open_position_without_trailing_stop(self, position_manager, sample_timestamp):
        """Test opening position without trailing stop."""
        position_manager.open_position(1, sample_timestamp, 100.0)

        # Trailing stop should remain None
        assert position_manager.trailing_stop is None

    def test_open_position_applies_slippage_correctly(self, position_manager, sample_timestamp):
        """Test slippage is applied correctly on position open."""
        base_price = 100.0
        position_manager.open_position(1, sample_timestamp, base_price)

        tick_size = TICK_SIZES.get('ZS', 0.01)
        expected_entry = base_price + (1 * tick_size)
        assert position_manager.entry_price == pytest.approx(expected_entry, abs=0.01)


class TestClosingPositions:
    """Test closing position operations."""

    def test_close_long_position(self, position_manager, sample_timestamp):
        """Test closing a long position."""
        # Open position
        position_manager.open_position(1, sample_timestamp, 100.0)
        entry_price = position_manager.entry_price

        # Close position
        exit_time = sample_timestamp + pd.Timedelta(hours=1)
        position_manager.close_position(exit_time, 105.0, switch=False)

        # Position should be closed
        assert position_manager.has_open_position() is False
        assert position_manager.position is None

        # Trade should be recorded
        assert len(position_manager.trades) == 1
        trade = position_manager.trades[0]
        assert trade['entry_time'] == sample_timestamp
        assert trade['entry_price'] == entry_price
        assert trade['exit_time'] == exit_time
        assert trade['exit_price'] < 105.0  # Exit slippage applied
        assert trade['side'] == 'long'

    def test_close_short_position(self, position_manager, sample_timestamp):
        """Test closing a short position."""
        # Open position
        position_manager.open_position(-1, sample_timestamp, 100.0)
        entry_price = position_manager.entry_price

        # Close position
        exit_time = sample_timestamp + pd.Timedelta(hours=1)
        position_manager.close_position(exit_time, 95.0, switch=False)

        # Position should be closed
        assert position_manager.has_open_position() is False

        # Trade should be recorded
        assert len(position_manager.trades) == 1
        trade = position_manager.trades[0]
        assert trade['entry_price'] == entry_price
        assert trade['exit_price'] > 95.0  # Exit slippage applied
        assert trade['side'] == 'short'

    def test_close_position_with_switch_flag(self, position_manager, sample_timestamp):
        """Test closing position with switch flag set."""
        position_manager.open_position(1, sample_timestamp, 100.0)
        exit_time = sample_timestamp + pd.Timedelta(hours=1)

        # Close with switch=True
        position_manager.close_position(exit_time, 105.0, switch=True)

        # Trade should have switch flag
        trade = position_manager.trades[0]
        assert 'switch' in trade
        assert trade['switch'] is True

    def test_close_position_without_switch_flag(self, position_manager, sample_timestamp):
        """Test closing position without switch flag."""
        position_manager.open_position(1, sample_timestamp, 100.0)
        exit_time = sample_timestamp + pd.Timedelta(hours=1)

        # Close with switch=False
        position_manager.close_position(exit_time, 105.0, switch=False)

        # Trade should not have switch flag
        trade = position_manager.trades[0]
        assert 'switch' not in trade

    def test_close_position_applies_slippage_correctly(self, position_manager, sample_timestamp):
        """Test slippage is applied correctly on position close."""
        position_manager.open_position(1, sample_timestamp, 100.0)

        base_exit_price = 105.0
        exit_time = sample_timestamp + pd.Timedelta(hours=1)
        position_manager.close_position(exit_time, base_exit_price, switch=False)

        # Exit price should have slippage applied
        tick_size = TICK_SIZES.get('ZS', 0.01)
        expected_exit = base_exit_price - (1 * tick_size)
        trade = position_manager.trades[0]
        assert trade['exit_price'] == pytest.approx(expected_exit, abs=0.01)

    def test_multiple_trades_recorded(self, position_manager, sample_timestamp):
        """Test multiple trades are recorded correctly."""
        # Trade 1
        position_manager.open_position(1, sample_timestamp, 100.0)
        position_manager.close_position(sample_timestamp, 105.0, switch=False)

        # Trade 2
        time2 = sample_timestamp + pd.Timedelta(hours=2)
        position_manager.open_position(-1, time2, 105.0)
        position_manager.close_position(time2, 100.0, switch=False)

        # Both trades should be recorded
        assert len(position_manager.trades) == 2
        assert position_manager.trades[0]['side'] == 'long'
        assert position_manager.trades[1]['side'] == 'short'


class TestContractSwitchHandling:
    """Test contract switch position closing."""

    def test_close_position_at_switch_long(self, position_manager, sample_timestamp, sample_row):
        """Test closing long position at contract switch."""
        # Open long position
        position_manager.open_position(1, sample_timestamp, 100.0)

        # Close at switch
        prev_position = position_manager.close_position_at_switch(sample_timestamp, sample_row)

        # Should return previous position direction
        assert prev_position == 1

        # Position should be closed
        assert position_manager.has_open_position() is False

        # Trade should be recorded with switch flag
        assert len(position_manager.trades) == 1
        trade = position_manager.trades[0]
        assert trade['switch'] is True
        assert trade['side'] == 'long'
        assert trade['exit_time'] == sample_timestamp

    def test_close_position_at_switch_short(self, position_manager, sample_timestamp, sample_row):
        """Test closing short position at contract switch."""
        # Open short position
        position_manager.open_position(-1, sample_timestamp, 100.0)

        # Close at switch
        prev_position = position_manager.close_position_at_switch(sample_timestamp, sample_row)

        # Should return previous position direction
        assert prev_position == -1

        # Trade should have switch flag
        trade = position_manager.trades[0]
        assert trade['switch'] is True
        assert trade['side'] == 'short'

    def test_close_at_switch_uses_open_price(self, position_manager, sample_timestamp, sample_row):
        """Test switch close uses open price of last bar."""
        position_manager.open_position(1, sample_timestamp, 100.0)

        # Close at switch (should use 'open' from sample_row)
        position_manager.close_position_at_switch(sample_timestamp, sample_row)

        # Exit price should be based on row's open price
        trade = position_manager.trades[0]
        tick_size = TICK_SIZES.get('ZS', 0.01)
        expected_exit = sample_row['open'] - (1 * tick_size)  # Long exit with slippage
        assert trade['exit_price'] == pytest.approx(expected_exit, abs=0.01)


class TestTradeRecording:
    """Test trade recording and data structure."""

    def test_trade_structure_long(self, position_manager, sample_timestamp):
        """Test trade structure for long position."""
        position_manager.open_position(1, sample_timestamp, 100.0)
        exit_time = sample_timestamp + pd.Timedelta(hours=1)
        position_manager.close_position(exit_time, 105.0, switch=False)

        trade = position_manager.trades[0]

        # Verify all required fields
        assert 'entry_time' in trade
        assert 'entry_price' in trade
        assert 'exit_time' in trade
        assert 'exit_price' in trade
        assert 'side' in trade

        # Verify values
        assert trade['entry_time'] == sample_timestamp
        assert trade['exit_time'] == exit_time
        assert trade['side'] == 'long'
        assert isinstance(trade['entry_price'], (int, float))
        assert isinstance(trade['exit_price'], (int, float))

    def test_trade_structure_short(self, position_manager, sample_timestamp):
        """Test trade structure for short position."""
        position_manager.open_position(-1, sample_timestamp, 100.0)
        exit_time = sample_timestamp + pd.Timedelta(hours=1)
        position_manager.close_position(exit_time, 95.0, switch=False)

        trade = position_manager.trades[0]
        assert trade['side'] == 'short'

    def test_trades_list_order(self, position_manager, sample_timestamp):
        """Test trades are recorded in chronological order."""
        timestamps = [
            sample_timestamp,
            sample_timestamp + pd.Timedelta(hours=1),
            sample_timestamp + pd.Timedelta(hours=2)
        ]

        for i, ts in enumerate(timestamps):
            position_manager.open_position(1, ts, 100.0)
            position_manager.close_position(ts, 105.0, switch=False)

        # Trades should be in order
        trades = position_manager.get_trades()
        assert len(trades) == 3
        for i in range(len(trades)):
            assert trades[i]['entry_time'] == timestamps[i]


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_position_manager_with_very_small_prices(self, position_manager, sample_timestamp):
        """Test position manager with very small prices."""
        small_price = 0.01
        position_manager.open_position(1, sample_timestamp, small_price)

        # Should handle small prices correctly
        assert position_manager.entry_price >= small_price
        assert position_manager.has_open_position() is True

    def test_position_manager_with_very_large_prices(self, position_manager, sample_timestamp):
        """Test position manager with very large prices."""
        large_price = 100000.0
        position_manager.open_position(1, sample_timestamp, large_price)

        # Should handle large prices correctly
        assert position_manager.entry_price > large_price
        assert position_manager.has_open_position() is True

    def test_high_slippage_ticks(self, sample_timestamp):
        """Test position manager with high slippage."""
        pm = PositionManager(slippage_ticks=10, symbol='ZS', trailing=None)
        base_price = 100.0

        pm.open_position(1, sample_timestamp, base_price)

        # Entry should reflect high slippage
        tick_size = TICK_SIZES.get('ZS', 0.01)
        expected = base_price + (10 * tick_size)
        assert pm.entry_price == pytest.approx(expected, abs=0.01)

    def test_reset_after_multiple_operations(self, position_manager, sample_timestamp):
        """Test reset works correctly after multiple operations."""
        # Create multiple trades
        for i in range(3):
            ts = sample_timestamp + pd.Timedelta(hours=i)
            position_manager.open_position(1, ts, 100.0)
            position_manager.close_position(ts, 105.0, switch=False)

        # Open a position
        position_manager.open_position(1, sample_timestamp, 100.0)

        # Reset should clear everything
        position_manager.reset()

        assert position_manager.position is None
        assert position_manager.entry_time is None
        assert position_manager.entry_price is None
        assert position_manager.trailing_stop is None
        assert len(position_manager.trades) == 0

    def test_trailing_stop_with_zero_percent(self, sample_timestamp):
        """Test trailing stop with 0% (essentially disabled)."""
        pm = PositionManager(slippage_ticks=1, symbol='ZS', trailing=0.0)

        pm.open_position(1, sample_timestamp, 100.0)

        # Trailing stop should be set but equal to entry price
        assert pm.trailing_stop is not None
        assert pm.trailing_stop == pytest.approx(pm.entry_price, abs=0.01)

    def test_symbol_not_in_tick_sizes(self, sample_timestamp):
        """Test position manager with unknown symbol (uses default tick size)."""
        pm = PositionManager(slippage_ticks=1, symbol='UNKNOWN', trailing=None)
        base_price = 100.0

        pm.open_position(1, sample_timestamp, base_price)

        # Should use default tick size
        from config import DEFAULT_TICK_SIZE
        expected = base_price + (1 * DEFAULT_TICK_SIZE)
        assert pm.entry_price == pytest.approx(expected, abs=0.01)


class TestPositionManagerIntegration:
    """Test position manager in realistic scenarios."""

    def test_complete_long_trade_lifecycle(self, position_manager, sample_timestamp):
        """Test complete lifecycle of a long trade."""
        # Open long at 100
        position_manager.open_position(1, sample_timestamp, 100.0)
        assert position_manager.has_open_position() is True
        assert position_manager.position == 1

        # Close long at 110 (profit)
        exit_time = sample_timestamp + pd.Timedelta(hours=2)
        position_manager.close_position(exit_time, 110.0, switch=False)
        assert position_manager.has_open_position() is False

        # Verify trade
        trades = position_manager.get_trades()
        assert len(trades) == 1
        trade = trades[0]
        assert trade['side'] == 'long'
        assert trade['exit_price'] > trade['entry_price']  # Profitable even with slippage

    def test_complete_short_trade_lifecycle(self, position_manager, sample_timestamp):
        """Test complete lifecycle of a short trade."""
        # Open short at 100
        position_manager.open_position(-1, sample_timestamp, 100.0)
        assert position_manager.has_open_position() is True
        assert position_manager.position == -1

        # Close short at 90 (profit)
        exit_time = sample_timestamp + pd.Timedelta(hours=2)
        position_manager.close_position(exit_time, 90.0, switch=False)
        assert position_manager.has_open_position() is False

        # Verify trade
        trades = position_manager.get_trades()
        assert len(trades) == 1
        trade = trades[0]
        assert trade['side'] == 'short'
        assert trade['entry_price'] > trade['exit_price']  # Profitable

    def test_alternating_long_short_trades(self, position_manager, sample_timestamp):
        """Test alternating between long and short positions."""
        timestamps = [
            sample_timestamp,
            sample_timestamp + pd.Timedelta(hours=1),
            sample_timestamp + pd.Timedelta(hours=2),
            sample_timestamp + pd.Timedelta(hours=3)
        ]

        # Long -> Short -> Long -> Short
        directions = [1, -1, 1, -1]

        for i, (ts, direction) in enumerate(zip(timestamps, directions)):
            position_manager.open_position(direction, ts, 100.0 + i)
            position_manager.close_position(ts, 105.0 + i, switch=False)

        # Should have 4 trades
        trades = position_manager.get_trades()
        assert len(trades) == 4

        # Verify alternating sides
        assert trades[0]['side'] == 'long'
        assert trades[1]['side'] == 'short'
        assert trades[2]['side'] == 'long'
        assert trades[3]['side'] == 'short'

    def test_position_reopen_after_close(self, position_manager, sample_timestamp):
        """Test reopening position after closing."""
        # First trade
        position_manager.open_position(1, sample_timestamp, 100.0)
        position_manager.close_position(sample_timestamp, 105.0, switch=False)

        # Second trade
        time2 = sample_timestamp + pd.Timedelta(hours=1)
        position_manager.open_position(1, time2, 105.0)

        # Should have open position again
        assert position_manager.has_open_position() is True
        assert position_manager.position == 1

        # First trade should still be recorded
        assert len(position_manager.trades) == 1
