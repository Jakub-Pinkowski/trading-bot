import unittest

import pytest

from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy
from app.backtesting.strategies.macd import MACDStrategy
from app.backtesting.strategies.rsi import RSIStrategy
from app.backtesting.strategy_factory import StrategyFactory


class TestStrategyFactory(unittest.TestCase):
    """Tests for the StrategyFactory class."""

    def test_create_strategy_rsi(self):
        """Test creating an RSI strategy."""
        # Test with default parameters
        strategy = StrategyFactory.create_strategy('rsi')

        self.assertIsInstance(strategy, RSIStrategy)
        self.assertEqual(strategy.rsi_period, 14)
        self.assertEqual(strategy.lower, 30)
        self.assertEqual(strategy.upper, 70)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)
        
        # Test with custom parameters
        strategy = StrategyFactory.create_strategy(
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
        strategy = StrategyFactory.create_strategy('ema')

        self.assertIsInstance(strategy, EMACrossoverStrategy)
        self.assertEqual(strategy.ema_short, 9)
        self.assertEqual(strategy.ema_long, 21)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)

        # Test with custom parameters
        strategy = StrategyFactory.create_strategy(
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
        strategy = StrategyFactory.create_strategy('macd')

        self.assertIsInstance(strategy, MACDStrategy)
        self.assertEqual(strategy.fast_period, 12)
        self.assertEqual(strategy.slow_period, 26)
        self.assertEqual(strategy.signal_period, 9)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)

        # Test with custom parameters
        strategy = StrategyFactory.create_strategy(
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
        strategy = StrategyFactory.create_strategy('bollinger')

        self.assertIsInstance(strategy, BollingerBandsStrategy)
        self.assertEqual(strategy.period, 20)
        self.assertEqual(strategy.num_std, 2)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)

        # Test with custom parameters
        strategy = StrategyFactory.create_strategy(
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
            StrategyFactory.create_strategy('unknown')

    def test_create_rsi_strategy_invalid_parameters(self):
        """Test creating an RSI strategy with invalid parameters."""
        # Invalid RSI period
        with pytest.raises(ValueError, match="RSI period must be a positive integer"):
            StrategyFactory.create_strategy('rsi', rsi_period=-1)

        # Invalid lower threshold
        with pytest.raises(ValueError, match="Lower threshold must be between 0 and 100"):
            StrategyFactory.create_strategy('rsi', lower=-10)

        # Invalid upper threshold
        with pytest.raises(ValueError, match="Upper threshold must be between 0 and 100"):
            StrategyFactory.create_strategy('rsi', upper=110)

        # Lower >= upper
        with pytest.raises(ValueError, match="Lower threshold must be less than upper threshold"):
            StrategyFactory.create_strategy('rsi', lower=70, upper=30)

    def test_create_ema_strategy_invalid_parameters(self):
        """Test creating an EMA strategy with invalid parameters."""
        # Invalid short EMA period
        with pytest.raises(ValueError, match="Short EMA period must be a positive integer"):
            StrategyFactory.create_strategy('ema', ema_short=-1)

        # Invalid long EMA period
        with pytest.raises(ValueError, match="Long EMA period must be a positive integer"):
            StrategyFactory.create_strategy('ema', ema_long=-1)

        # Short >= long
        with pytest.raises(ValueError, match="Short EMA period must be less than long EMA period"):
            StrategyFactory.create_strategy('ema', ema_short=21, ema_long=9)

    def test_create_macd_strategy_invalid_parameters(self):
        """Test creating a MACD strategy with invalid parameters."""
        # Invalid fast period
        with pytest.raises(ValueError, match="Fast period must be a positive integer"):
            StrategyFactory.create_strategy('macd', fast_period=-1)

        # Invalid slow period
        with pytest.raises(ValueError, match="Slow period must be a positive integer"):
            StrategyFactory.create_strategy('macd', slow_period=-1)

        # Invalid signal period
        with pytest.raises(ValueError, match="Signal period must be a positive integer"):
            StrategyFactory.create_strategy('macd', signal_period=-1)

        # Fast >= slow
        with pytest.raises(ValueError, match="Fast period must be less than slow period"):
            StrategyFactory.create_strategy('macd', fast_period=26, slow_period=12)

    def test_create_bollinger_strategy_invalid_parameters(self):
        """Test creating a Bollinger Bands strategy with invalid parameters."""
        # Invalid period
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            StrategyFactory.create_strategy('bollinger', period=-1)

        # Invalid number of standard deviations
        with pytest.raises(ValueError, match="Number of standard deviations must be positive"):
            StrategyFactory.create_strategy('bollinger', num_std=-1)

    def test_get_strategy_name(self):
        """Test getting a standardized name for a strategy."""
        # RSI strategy
        name = StrategyFactory.get_strategy_name('rsi',
                                                 rsi_period=14,
                                                 lower=30,
                                                 upper=70,
                                                 rollover=False,
                                                 trailing=None)
        self.assertEqual(name, 'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None)')

        # EMA strategy
        name = StrategyFactory.get_strategy_name('ema', ema_short=9, ema_long=21, rollover=True, trailing=2.0)
        self.assertEqual(name, 'EMA(short=9,long=21,rollover=True,trailing=2.0)')

        # MACD strategy
        name = StrategyFactory.get_strategy_name('macd',
                                                 fast_period=12,
                                                 slow_period=26,
                                                 signal_period=9,
                                                 rollover=False,
                                                 trailing=None)
        self.assertEqual(name, 'MACD(fast=12,slow=26,signal=9,rollover=False,trailing=None)')

        # Bollinger Bands strategy
        name = StrategyFactory.get_strategy_name('bollinger', period=20, num_std=2, rollover=True, trailing=None)
        self.assertEqual(name, 'BB(period=20,std=2,rollover=True,trailing=None)')

        # Unknown strategy type
        name = StrategyFactory.get_strategy_name('unknown')
        self.assertEqual(name, 'Unknown(unknown)')


if __name__ == '__main__':
    unittest.main()
