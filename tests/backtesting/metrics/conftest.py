"""
Shared fixtures for backtesting metrics tests.

Provides factory fixtures for creating test trades and trade sequences,
along with common assertion helpers and test data.
"""
from datetime import datetime, timedelta

import pytest

from app.backtesting.metrics.per_trade_metrics import calculate_trade_metrics


# ==================== Trade Factory ====================

@pytest.fixture
def trade_factory():
    """
    Factory fixture for creating test trades with calculated metrics.

    Creates individual trades with automatic metric calculation.
    Simplifies test setup by providing sensible defaults.

    Usage:
        trade = trade_factory(symbol='ZS', entry=1200, exit=1210)
        trade = trade_factory('CL', 75.0, 74.8, side='short')
        trade = trade_factory('ES', 5000, 5010, duration_hours=2.5)

    Args:
        symbol: Futures symbol (e.g., 'ZS', 'CL', 'ES')
        entry_price: Entry price for the trade
        exit_price: Exit price for the trade
        side: 'long' or 'short' (default: 'long')
        entry_time: Entry datetime (default: 2024-01-15 10:00)
        exit_time: Exit datetime (default: entry_time + duration_hours)
        duration_hours: Trade duration in hours (default: 4.0)

    Returns:
        Dictionary with calculated trade metrics
    """

    def _create_trade(
        symbol,
        entry_price,
        exit_price,
        side='long',
        entry_time=None,
        exit_time=None,
        duration_hours=4.0
    ):
        if entry_time is None:
            entry_time = datetime(2024, 1, 15, 10, 0)
        if exit_time is None:
            exit_time = entry_time + timedelta(hours=duration_hours)

        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'side': side
        }
        return calculate_trade_metrics(trade, symbol)

    return _create_trade


@pytest.fixture
def trades_factory(trade_factory):
    """
    Factory for creating sequences of trades.

    Provides convenient methods for creating common trade patterns:
    - Sequences from price specifications
    - All winning trades
    - All losing trades
    - Mixed winning and losing trades

    Usage:
        # Create sequence from price specs
        trades = trades_factory.create_sequence([
            (1200, 1210),
            (1210, 1205),
        ], symbol='ZS')

        # Create winning/losing sequences
        winning = trades_factory.all_winning(count=5)
        losing = trades_factory.all_losing(count=3)
        mixed = trades_factory.mixed(win_count=3, loss_count=2)

    Returns:
        TradesFactory instance with helper methods
    """

    class TradesFactory:
        def __init__(self, trade_factory):
            self.trade_factory = trade_factory

        def create_sequence(self, specs, symbol='ZS', side='long', start_date=None):
            """
            Create multiple trades from (entry, exit) price specs.

            Args:
                specs: List of tuples (entry, exit) or (entry, exit, side)
                symbol: Futures symbol for all trades (default: 'ZS')
                side: Default side if not specified in spec (default: 'long')
                start_date: Starting date for sequence (default: 2024-01-10)

            Returns:
                List of calculated trade metrics
            """
            if start_date is None:
                start_date = datetime(2024, 1, 10, 10, 0)

            trades = []
            for i, spec in enumerate(specs):
                entry, exit = spec[:2]
                trade_side = spec[2] if len(spec) > 2 else side

                entry_time = start_date + timedelta(days=i)
                exit_time = entry_time + timedelta(hours=4)

                trades.append(self.trade_factory(
                    symbol=symbol,
                    entry_price=entry,
                    exit_price=exit,
                    side=trade_side,
                    entry_time=entry_time,
                    exit_time=exit_time
                ))
            return trades

        def all_winning(self, count=5, symbol='ZS', base_price=1200.0):
            """
            Create sequence of winning trades.

            Args:
                count: Number of winning trades to create (default: 5)
                symbol: Futures symbol (default: 'ZS')
                base_price: Base entry price (default: 1200.0)

            Returns:
                List of winning trades with increasing profits
            """
            return [
                self.trade_factory(
                    symbol=symbol,
                    entry_price=base_price,
                    exit_price=base_price + (i + 1) * 5,
                    entry_time=datetime(2024, 1, 10, 10, 0) + timedelta(days=i),
                    exit_time=datetime(2024, 1, 10, 14, 0) + timedelta(days=i)
                )
                for i in range(count)
            ]

        def all_losing(self, count=5, symbol='ZS', base_price=1200.0):
            """
            Create sequence of losing trades.

            Args:
                count: Number of losing trades to create (default: 5)
                symbol: Futures symbol (default: 'ZS')
                base_price: Base entry price (default: 1200.0)

            Returns:
                List of losing trades with increasing losses
            """
            return [
                self.trade_factory(
                    symbol=symbol,
                    entry_price=base_price,
                    exit_price=base_price - (i + 1) * 5,
                    entry_time=datetime(2024, 1, 10, 10, 0) + timedelta(days=i),
                    exit_time=datetime(2024, 1, 10, 14, 0) + timedelta(days=i)
                )
                for i in range(count)
            ]

        def mixed(self, win_count=3, loss_count=2, symbol='ZS', base_price=1200.0):
            """
            Create mixed winning and losing trades.

            Args:
                win_count: Number of winning trades (default: 3)
                loss_count: Number of losing trades (default: 2)
                symbol: Futures symbol (default: 'ZS')
                base_price: Base entry price (default: 1200.0)

            Returns:
                List of mixed trades (wins first, then losses)
            """
            trades = self.all_winning(win_count, symbol, base_price)
            trades.extend(self.all_losing(loss_count, symbol, base_price))
            return trades

    return TradesFactory(trade_factory)


# ==================== Symbol Data ====================

@pytest.fixture
def symbol_test_data():
    """
    Common symbol test data with expected categories and margins.

    Provides reference data for testing symbol-specific calculations
    including margin ratios and typical price levels.

    Usage:
        data = symbol_test_data
        assert data['ZS']['category'] == 'grains'
        assert data['CL']['margin_ratio'] == 0.25

    Returns:
        Dictionary mapping symbols to their test properties
    """
    return {
        'ZS': {'category': 'grains', 'margin_ratio': 0.08, 'test_price': 1200.0},
        'CL': {'category': 'energies', 'margin_ratio': 0.25, 'test_price': 75.0},
        'ES': {'category': 'indices', 'margin_ratio': 0.08, 'test_price': 5000.0},
        'GC': {'category': 'metals', 'margin_ratio': 0.12, 'test_price': 2050.0},
        'NG': {'category': 'energies', 'margin_ratio': 0.25, 'test_price': 3.0},
        '6E': {'category': 'forex', 'margin_ratio': 0.04, 'test_price': 1.10},
        'BTC': {'category': 'crypto', 'margin_ratio': 0.40, 'test_price': 45000.0},
    }
