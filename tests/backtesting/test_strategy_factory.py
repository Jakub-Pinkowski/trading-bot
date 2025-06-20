import unittest
from unittest.mock import patch

import pytest

from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy
from app.backtesting.strategies.macd import MACDStrategy
from app.backtesting.strategies.rsi import RSIStrategy
from app.backtesting.strategy_factory import create_strategy, get_strategy_name


class TestStrategyFactory(unittest.TestCase):
    """Tests for the StrategyFactory class."""

    def test_create_strategy_rsi(self):
        """Test creating an RSI strategy."""
        # Test with default parameters
        strategy = create_strategy('rsi')

        self.assertIsInstance(strategy, RSIStrategy)
        self.assertEqual(strategy.rsi_period, 14)
        self.assertEqual(strategy.lower, 30)
        self.assertEqual(strategy.upper, 70)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)

        # Test with custom parameters
        strategy = create_strategy(
            'rsi',
            rsi_period=21,
            lower=35,
            upper=75,
            rollover=True,
            trailing=2.0
        )

        self.assertIsInstance(strategy, RSIStrategy)
        self.assertEqual(strategy.rsi_period, 21)
        self.assertEqual(strategy.lower, 35)
        self.assertEqual(strategy.upper, 75)
        self.assertEqual(strategy.rollover, True)
        self.assertEqual(strategy.trailing, 2.0)

    def test_create_strategy_ema(self):
        """Test creating an EMA Crossover strategy."""
        # Test with default parameters
        strategy = create_strategy('ema')

        self.assertIsInstance(strategy, EMACrossoverStrategy)
        self.assertEqual(strategy.ema_short, 9)
        self.assertEqual(strategy.ema_long, 21)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)

        # Test with custom parameters
        strategy = create_strategy(
            'ema',
            ema_short=12,
            ema_long=26,
            rollover=True,
            trailing=2.0
        )

        self.assertIsInstance(strategy, EMACrossoverStrategy)
        self.assertEqual(strategy.ema_short, 12)
        self.assertEqual(strategy.ema_long, 26)
        self.assertEqual(strategy.rollover, True)
        self.assertEqual(strategy.trailing, 2.0)

    def test_create_strategy_macd(self):
        """Test creating a MACD strategy."""
        # Test with default parameters
        strategy = create_strategy('macd')

        self.assertIsInstance(strategy, MACDStrategy)
        self.assertEqual(strategy.fast_period, 12)
        self.assertEqual(strategy.slow_period, 26)
        self.assertEqual(strategy.signal_period, 9)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)

        # Test with custom parameters
        strategy = create_strategy(
            'macd',
            fast_period=10,
            slow_period=30,
            signal_period=7,
            rollover=True,
            trailing=2.0
        )

        self.assertIsInstance(strategy, MACDStrategy)
        self.assertEqual(strategy.fast_period, 10)
        self.assertEqual(strategy.slow_period, 30)
        self.assertEqual(strategy.signal_period, 7)
        self.assertEqual(strategy.rollover, True)
        self.assertEqual(strategy.trailing, 2.0)

    def test_create_strategy_bollinger(self):
        """Test creating a Bollinger Bands strategy."""
        # Test with default parameters
        strategy = create_strategy('bollinger')

        self.assertIsInstance(strategy, BollingerBandsStrategy)
        self.assertEqual(strategy.period, 20)
        self.assertEqual(strategy.num_std, 2)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)

        # Test with custom parameters
        strategy = create_strategy(
            'bollinger',
            period=15,
            num_std=2.5,
            rollover=True,
            trailing=2.0
        )

        self.assertIsInstance(strategy, BollingerBandsStrategy)
        self.assertEqual(strategy.period, 15)
        self.assertEqual(strategy.num_std, 2.5)
        self.assertEqual(strategy.rollover, True)
        self.assertEqual(strategy.trailing, 2.0)

    def test_create_strategy_unknown_type(self):
        """Test creating a strategy with an unknown type."""
        with pytest.raises(ValueError, match="Unknown strategy type"):
            create_strategy('unknown')

    @patch('app.backtesting.strategy_factory.STRATEGY_TYPES', ['rsi', 'ema', 'macd', 'bollinger', 'custom'])
    def test_create_strategy_fallback(self):
        """Test the fallback return statement when a strategy type is in STRATEGY_TYPES but has no specific handler."""
        # This test is specifically for line 79 in strategy_factory.py
        # The 'custom' strategy type is added to STRATEGY_TYPES but has no specific handler
        result = create_strategy('custom')
        self.assertIsNone(result)

    def test_create_rsi_strategy_invalid_parameters(self):
        """Test creating an RSI strategy with invalid parameters."""
        # Invalid RSI period
        with pytest.raises(ValueError, match="rsi period must be a positive integer"):
            create_strategy('rsi', rsi_period=-1)

        # Invalid lower threshold
        with pytest.raises(ValueError, match="lower threshold must be between 0 and 100"):
            create_strategy('rsi', lower=-10)

        # Invalid upper threshold
        with pytest.raises(ValueError, match="upper threshold must be between 0 and 100"):
            create_strategy('rsi', upper=110)

        # Lower >= upper
        with pytest.raises(ValueError, match="Lower threshold must be less than upper threshold"):
            create_strategy('rsi', lower=70, upper=30)

    def test_create_ema_strategy_invalid_parameters(self):
        """Test creating an EMA strategy with invalid parameters."""
        # Invalid short EMA period
        with pytest.raises(ValueError, match="short EMA period must be a positive integer"):
            create_strategy('ema', ema_short=-1)

        # Invalid long EMA period
        with pytest.raises(ValueError, match="long EMA period must be a positive integer"):
            create_strategy('ema', ema_long=-1)

        # Short >= long
        with pytest.raises(ValueError, match="Short EMA period must be less than long EMA period"):
            create_strategy('ema', ema_short=21, ema_long=9)

    def test_create_macd_strategy_invalid_parameters(self):
        """Test creating a MACD strategy with invalid parameters."""
        # Invalid fast period
        with pytest.raises(ValueError, match="fast period must be a positive integer"):
            create_strategy('macd', fast_period=-1)

        # Invalid slow period
        with pytest.raises(ValueError, match="slow period must be a positive integer"):
            create_strategy('macd', slow_period=-1)

        # Invalid signal period
        with pytest.raises(ValueError, match="signal period must be a positive integer"):
            create_strategy('macd', signal_period=-1)

        # Fast >= slow
        with pytest.raises(ValueError, match="Fast period must be less than slow period"):
            create_strategy('macd', fast_period=26, slow_period=12)

    def test_create_bollinger_strategy_invalid_parameters(self):
        """Test creating a Bollinger Bands strategy with invalid parameters."""
        # Invalid period
        with pytest.raises(ValueError, match="period must be a positive integer"):
            create_strategy('bollinger', period=-1)

        # Invalid number of standard deviations
        with pytest.raises(ValueError, match="number of standard deviations must be positive"):
            create_strategy('bollinger', num_std=-1)

    def test_create_strategy_invalid_common_parameters(self):
        """Test creating a strategy with invalid common parameters."""
        # Invalid rollover (not a boolean)
        with pytest.raises(ValueError, match="rollover must be a boolean"):
            create_strategy('rsi', rollover="True")  # String instead of boolean

        # Invalid trailing (not a positive number)
        with pytest.raises(ValueError, match="trailing must be None or a positive number"):
            create_strategy('rsi', trailing=0)  # Zero is not positive

        with pytest.raises(ValueError, match="trailing must be None or a positive number"):
            create_strategy('rsi', trailing=-1)  # Negative number

        with pytest.raises(ValueError, match="trailing must be None or a positive number"):
            create_strategy('rsi', trailing="2.0")  # String instead of number

        # Invalid slippage (not a non-negative number)
        with pytest.raises(ValueError, match="slippage must be None or a non-negative number"):
            create_strategy('rsi', slippage=-1)  # Negative number

        with pytest.raises(ValueError, match="slippage must be None or a non-negative number"):
            create_strategy('rsi', slippage="0.5")  # String instead of number

    def test_get_strategy_name(self):
        """Test getting a standardized name for a strategy."""
        # RSI strategy
        name = get_strategy_name('rsi',
                                 rsi_period=14,
                                 lower=30,
                                 upper=70,
                                 rollover=False,
                                 trailing=None,
                                 slippage=None)
        self.assertEqual(name, 'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage=None)')

        # EMA strategy
        name = get_strategy_name('ema',
                                 ema_short=9,
                                 ema_long=21,
                                 rollover=True,
                                 trailing=2.0,
                                 slippage=None)
        self.assertEqual(name, 'EMA(short=9,long=21,rollover=True,trailing=2.0,slippage=None)')

        # MACD strategy
        name = get_strategy_name('macd',
                                 fast_period=12,
                                 slow_period=26,
                                 signal_period=9,
                                 rollover=False,
                                 trailing=None,
                                 slippage=None)
        self.assertEqual(name, 'MACD(fast=12,slow=26,signal=9,rollover=False,trailing=None,slippage=None)')

        # Bollinger Bands strategy
        name = get_strategy_name('bollinger',
                                 period=20,
                                 num_std=2,
                                 rollover=True,
                                 trailing=None,
                                 slippage=None)
        self.assertEqual(name, 'BB(period=20,std=2,rollover=True,trailing=None,slippage=None)')

        # Unknown strategy type
        name = get_strategy_name('unknown')
        self.assertEqual(name, 'Unknown(unknown)')


if __name__ == '__main__':
    unittest.main()
