import unittest
from unittest.mock import patch

import pytest

from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy
from app.backtesting.strategies.ichimoku_cloud import IchimokuCloudStrategy
from app.backtesting.strategies.macd import MACDStrategy
from app.backtesting.strategies.rsi import RSIStrategy
from app.backtesting.strategy_factory import (
    create_strategy, get_strategy_name, _extract_common_params,
    _validate_positive_integer, _validate_positive_number, _validate_range, _format_common_params,
    _validate_rsi_parameters, _validate_ema_parameters, _validate_macd_parameters,
    _validate_bollinger_parameters, _validate_ichimoku_parameters, _validate_common_parameters
)


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

    def test_create_strategy_ichimoku(self):
        """Test creating an Ichimoku Cloud strategy."""
        # Test with default parameters
        strategy = create_strategy('ichimoku')

        self.assertIsInstance(strategy, IchimokuCloudStrategy)
        self.assertEqual(strategy.tenkan_period, 9)
        self.assertEqual(strategy.kijun_period, 26)
        self.assertEqual(strategy.senkou_span_b_period, 52)
        self.assertEqual(strategy.displacement, 26)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)

        # Test with custom parameters
        strategy = create_strategy(
            'ichimoku',
            tenkan_period=7,
            kijun_period=22,
            senkou_span_b_period=44,
            displacement=22,
            rollover=True,
            trailing=2.0
        )

        self.assertIsInstance(strategy, IchimokuCloudStrategy)
        self.assertEqual(strategy.tenkan_period, 7)
        self.assertEqual(strategy.kijun_period, 22)
        self.assertEqual(strategy.senkou_span_b_period, 44)
        self.assertEqual(strategy.displacement, 22)
        self.assertEqual(strategy.rollover, True)
        self.assertEqual(strategy.trailing, 2.0)

    def test_create_ichimoku_strategy_invalid_parameters(self):
        """Test creating an Ichimoku strategy with invalid parameters."""
        # Invalid tenkan period
        with pytest.raises(ValueError, match="tenkan period must be a positive integer"):
            create_strategy('ichimoku', tenkan_period=-1)

        # Invalid kijun period
        with pytest.raises(ValueError, match="kijun period must be a positive integer"):
            create_strategy('ichimoku', kijun_period=-1)

        # Invalid senkou span B period
        with pytest.raises(ValueError, match="senkou span B period must be a positive integer"):
            create_strategy('ichimoku', senkou_span_b_period=-1)

        # Invalid displacement
        with pytest.raises(ValueError, match="displacement must be a positive integer"):
            create_strategy('ichimoku', displacement=-1)

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

        # Ichimoku strategy
        name = get_strategy_name('ichimoku',
                                 tenkan_period=9,
                                 kijun_period=26,
                                 senkou_span_b_period=52,
                                 displacement=26,
                                 rollover=False,
                                 trailing=None,
                                 slippage=None)
        self.assertEqual(name,
                         'Ichimoku(tenkan=9,kijun=26,senkou_b=52,displacement=26,rollover=False,trailing=None,slippage=None)')

        # Unknown strategy type
        name = get_strategy_name('unknown')
        self.assertEqual(name, 'Unknown(unknown)')


class TestPrivateHelperFunctions(unittest.TestCase):
    """Tests for private helper functions in strategy_factory."""

    def test_extract_common_params_valid(self):
        """Test _extract_common_params with valid parameters."""
        params = {
            'rollover': True,
            'trailing': 2.5,
            'slippage': 0.1,
            'other_param': 'ignored'
        }
        result = _extract_common_params(**params)

        self.assertEqual(result['rollover'], True)
        self.assertEqual(result['trailing'], 2.5)
        self.assertEqual(result['slippage'], 0.1)
        self.assertNotIn('other_param', result)

    def test_extract_common_params_defaults(self):
        """Test _extract_common_params with default values."""
        result = _extract_common_params()

        self.assertEqual(result['rollover'], False)
        self.assertIsNone(result['trailing'])
        self.assertIsNone(result['slippage'])

    def test_extract_common_params_invalid_rollover(self):
        """Test _extract_common_params with invalid rollover."""
        with pytest.raises(ValueError, match="rollover must be a boolean"):
            _extract_common_params(rollover="True")

    def test_extract_common_params_invalid_trailing(self):
        """Test _extract_common_params with invalid trailing."""
        with pytest.raises(ValueError, match="trailing must be None or a positive number"):
            _extract_common_params(trailing=0)

        with pytest.raises(ValueError, match="trailing must be None or a positive number"):
            _extract_common_params(trailing=-1)

        with pytest.raises(ValueError, match="trailing must be None or a positive number"):
            _extract_common_params(trailing="2.0")

    def test_extract_common_params_invalid_slippage(self):
        """Test _extract_common_params with invalid slippage."""
        with pytest.raises(ValueError, match="slippage must be None or a non-negative number"):
            _extract_common_params(slippage=-1)

        with pytest.raises(ValueError, match="slippage must be None or a non-negative number"):
            _extract_common_params(slippage="0.5")

    def test_validate_positive_integer_valid(self):
        """Test _validate_positive_integer with valid values."""
        # Should not raise any exception
        _validate_positive_integer(1, "test_param")
        _validate_positive_integer(100, "test_param")

    def test_validate_positive_integer_invalid(self):
        """Test _validate_positive_integer with invalid values."""
        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            _validate_positive_integer(0, "test_param")

        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            _validate_positive_integer(-1, "test_param")

        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            _validate_positive_integer(1.5, "test_param")

        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            _validate_positive_integer("1", "test_param")

    def test_validate_positive_number_valid(self):
        """Test _validate_positive_number with valid values."""
        # Should not raise any exception
        _validate_positive_number(1, "test_param")
        _validate_positive_number(1.5, "test_param")
        _validate_positive_number(100.0, "test_param")

    def test_validate_positive_number_invalid(self):
        """Test _validate_positive_number with invalid values."""
        with pytest.raises(ValueError, match="test_param must be positive"):
            _validate_positive_number(0, "test_param")

        with pytest.raises(ValueError, match="test_param must be positive"):
            _validate_positive_number(-1, "test_param")

        with pytest.raises(ValueError, match="test_param must be positive"):
            _validate_positive_number(-1.5, "test_param")

        with pytest.raises(ValueError, match="test_param must be positive"):
            _validate_positive_number("1", "test_param")

    def test_validate_range_valid(self):
        """Test _validate_range with valid values."""
        # Should not raise any exception
        _validate_range(5, "test_param", 0, 10)
        _validate_range(0, "test_param", 0, 10)
        _validate_range(10, "test_param", 0, 10)
        _validate_range(5.5, "test_param", 0, 10)

    def test_validate_range_invalid(self):
        """Test _validate_range with invalid values."""
        with pytest.raises(ValueError, match="test_param must be between 0 and 10"):
            _validate_range(-1, "test_param", 0, 10)

        with pytest.raises(ValueError, match="test_param must be between 0 and 10"):
            _validate_range(11, "test_param", 0, 10)

        with pytest.raises(ValueError, match="test_param must be between 0 and 10"):
            _validate_range("5", "test_param", 0, 10)

    def test_format_common_params(self):
        """Test _format_common_params function."""
        params = {
            'rollover': True,
            'trailing': 2.5,
            'slippage': 0.1
        }
        result = _format_common_params(**params)
        expected = "rollover=True,trailing=2.5,slippage=0.1"
        self.assertEqual(result, expected)

        # Test with defaults
        result = _format_common_params()
        expected = "rollover=False,trailing=None,slippage=None"
        self.assertEqual(result, expected)


class TestParameterValidation(unittest.TestCase):
    """Tests for the enhanced parameter validation functions."""

    def test_validate_rsi_parameters_optimal(self):
        """Test RSI parameter validation with optimal parameters."""
        # Standard parameters should generate no warnings
        warnings = _validate_rsi_parameters(14, 30, 70)
        self.assertEqual(len(warnings), 0)

    def test_validate_rsi_parameters_warnings(self):
        """Test RSI parameter validation with parameters that generate warnings."""
        # Very short period
        warnings = _validate_rsi_parameters(5, 30, 70)
        self.assertEqual(len(warnings), 1)
        self.assertIn("quite short", warnings[0])

        # Very long period
        warnings = _validate_rsi_parameters(35, 30, 70)
        self.assertEqual(len(warnings), 1)
        self.assertIn("quite long", warnings[0])

        # Very aggressive lower threshold
        warnings = _validate_rsi_parameters(14, 15, 70)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("very aggressive" in w for w in warnings))
        self.assertTrue(any("very wide" in w for w in warnings))

        # Very conservative lower threshold
        warnings = _validate_rsi_parameters(14, 45, 70)
        self.assertEqual(len(warnings), 1)
        self.assertIn("very conservative", warnings[0])

        # Very aggressive upper threshold
        warnings = _validate_rsi_parameters(14, 30, 55)
        self.assertEqual(len(warnings), 1)
        self.assertIn("very aggressive", warnings[0])

        # Very conservative upper threshold
        warnings = _validate_rsi_parameters(14, 30, 85)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("very conservative" in w for w in warnings))
        self.assertTrue(any("very wide" in w for w in warnings))

        # Very narrow gap
        warnings = _validate_rsi_parameters(14, 40, 50)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("very aggressive" in w for w in warnings))
        self.assertTrue(any("quite narrow" in w for w in warnings))

        # Very wide gap
        warnings = _validate_rsi_parameters(14, 20, 80)
        self.assertEqual(len(warnings), 1)
        self.assertIn("very wide", warnings[0])

    def test_validate_ema_parameters_optimal(self):
        """Test EMA parameter validation with optimal parameters."""
        # Standard parameters should generate no warnings
        warnings = _validate_ema_parameters(9, 21)
        self.assertEqual(len(warnings), 0)

        warnings = _validate_ema_parameters(12, 26)
        self.assertEqual(len(warnings), 0)

    def test_validate_ema_parameters_warnings(self):
        """Test EMA parameter validation with parameters that generate warnings."""
        # Very short periods
        warnings = _validate_ema_parameters(3, 15)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("very sensitive" in w for w in warnings))
        self.assertTrue(any("very wide" in w for w in warnings))

        # Very long periods
        warnings = _validate_ema_parameters(25, 60)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("too slow for crossover" in w for w in warnings))
        self.assertTrue(any("too slow and miss trend" in w for w in warnings))

        # Too close ratio
        warnings = _validate_ema_parameters(12, 15)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too close", warnings[0])

        # Very wide ratio
        warnings = _validate_ema_parameters(5, 50)
        self.assertEqual(len(warnings), 1)
        self.assertIn("very wide", warnings[0])

    def test_validate_macd_parameters_optimal(self):
        """Test MACD parameter validation with optimal parameters."""
        # Standard parameters should generate a positive note
        warnings = _validate_macd_parameters(12, 26, 9)
        self.assertEqual(len(warnings), 1)
        self.assertIn("standard MACD parameters", warnings[0])

    def test_validate_macd_parameters_warnings(self):
        """Test MACD parameter validation with parameters that generate warnings."""
        # Very short fast period
        warnings = _validate_macd_parameters(5, 26, 9)
        self.assertEqual(len(warnings), 1)
        self.assertIn("very short", warnings[0])

        # Very long fast period
        warnings = _validate_macd_parameters(20, 26, 9)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too slow for responsive", warnings[0])

        # Very short slow period
        warnings = _validate_macd_parameters(12, 15, 9)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too short for trend", warnings[0])

        # Very long slow period
        warnings = _validate_macd_parameters(12, 35, 9)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too slow and miss trend", warnings[0])

        # Very short signal period
        warnings = _validate_macd_parameters(12, 26, 5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("very short", warnings[0])

        # Very long signal period
        warnings = _validate_macd_parameters(12, 26, 15)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too slow for timely", warnings[0])

    def test_validate_bollinger_parameters_optimal(self):
        """Test Bollinger Bands parameter validation with optimal parameters."""
        # Standard parameters should generate a positive note
        warnings = _validate_bollinger_parameters(20, 2.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("standard Bollinger Bands", warnings[0])

    def test_validate_bollinger_parameters_warnings(self):
        """Test Bollinger Bands parameter validation with parameters that generate warnings."""
        # Very short period
        warnings = _validate_bollinger_parameters(10, 2.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("quite short", warnings[0])

        # Very long period
        warnings = _validate_bollinger_parameters(30, 2.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("quite long", warnings[0])

        # Very narrow bands
        warnings = _validate_bollinger_parameters(20, 1.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("quite narrow", warnings[0])

        # Very wide bands
        warnings = _validate_bollinger_parameters(20, 3.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("quite wide", warnings[0])

    def test_validate_ichimoku_parameters_optimal(self):
        """Test Ichimoku parameter validation with optimal parameters."""
        # Traditional parameters should generate a positive note
        warnings = _validate_ichimoku_parameters(9, 26, 52, 26)
        self.assertEqual(len(warnings), 1)
        self.assertIn("traditional Ichimoku parameters", warnings[0])

    def test_validate_ichimoku_parameters_warnings(self):
        """Test Ichimoku parameter validation with parameters that generate warnings."""
        # Very short tenkan period
        warnings = _validate_ichimoku_parameters(5, 26, 52, 26)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("quite short" in w for w in warnings))
        self.assertTrue(any("deviates from traditional" in w for w in warnings))

        # Very long tenkan period
        warnings = _validate_ichimoku_parameters(15, 26, 52, 26)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("too slow for conversion" in w for w in warnings))
        self.assertTrue(any("deviates from traditional" in w for w in warnings))

        # Very short kijun period
        warnings = _validate_ichimoku_parameters(9, 20, 52, 26)
        self.assertEqual(len(warnings), 4)
        self.assertTrue(any("too short for baseline" in w for w in warnings))
        self.assertTrue(any("deviates from traditional" in w for w in warnings))
        self.assertTrue(any("differs from Kijun period" in w for w in warnings))

        # Very long kijun period
        warnings = _validate_ichimoku_parameters(9, 35, 52, 26)
        self.assertEqual(len(warnings), 4)
        self.assertTrue(any("too slow for trend" in w for w in warnings))
        self.assertTrue(any("deviates from traditional" in w for w in warnings))
        self.assertTrue(any("differs from Kijun period" in w for w in warnings))

        # Different displacement
        warnings = _validate_ichimoku_parameters(9, 26, 52, 20)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("differs from Kijun period" in w for w in warnings))
        self.assertTrue(any("may be too short for proper cloud projection" in w for w in warnings))

        # Non-traditional ratios
        warnings = _validate_ichimoku_parameters(12, 24, 48, 24)
        self.assertEqual(len(warnings), 1)
        self.assertTrue(any("deviates from traditional" in w for w in warnings))

    def test_validate_common_parameters_optimal(self):
        """Test common parameter validation with optimal parameters."""
        # Reasonable parameters should generate no warnings
        warnings = _validate_common_parameters(False, 2.5, 0.15)
        self.assertEqual(len(warnings), 0)

    def test_validate_common_parameters_warnings(self):
        """Test common parameter validation with parameters that generate warnings."""
        # Very tight trailing stop
        warnings = _validate_common_parameters(False, 0.5, None)
        self.assertEqual(len(warnings), 1)
        self.assertIn("very tight", warnings[0])

        # Very wide trailing stop
        warnings = _validate_common_parameters(False, 8.0, None)
        self.assertEqual(len(warnings), 1)
        self.assertIn("very wide", warnings[0])

        # Zero slippage
        warnings = _validate_common_parameters(False, None, 0.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("unrealistic", warnings[0])

        # Very high slippage
        warnings = _validate_common_parameters(False, None, 0.8)
        self.assertEqual(len(warnings), 1)
        self.assertIn("very high", warnings[0])

    def test_validation_integration_with_strategy_creation(self):
        """Test that validation warnings are properly logged during strategy creation."""
        # This test verifies that the validation functions are called during strategy creation
        # We can't easily test the logging output, but we can verify strategies are still created

        # RSI with suboptimal parameters should still create strategy
        strategy = create_strategy('rsi', rsi_period=5, lower=15, upper=85)
        self.assertIsInstance(strategy, RSIStrategy)
        self.assertEqual(strategy.rsi_period, 5)
        self.assertEqual(strategy.lower, 15)
        self.assertEqual(strategy.upper, 85)

        # EMA with suboptimal parameters should still create strategy
        strategy = create_strategy('ema', ema_short=3, ema_long=60)
        self.assertIsInstance(strategy, EMACrossoverStrategy)
        self.assertEqual(strategy.ema_short, 3)
        self.assertEqual(strategy.ema_long, 60)

        # MACD with suboptimal parameters should still create strategy
        strategy = create_strategy('macd', fast_period=5, slow_period=35, signal_period=15)
        self.assertIsInstance(strategy, MACDStrategy)
        self.assertEqual(strategy.fast_period, 5)
        self.assertEqual(strategy.slow_period, 35)
        self.assertEqual(strategy.signal_period, 15)

        # Bollinger with suboptimal parameters should still create strategy
        strategy = create_strategy('bollinger', period=10, num_std=3.0)
        self.assertIsInstance(strategy, BollingerBandsStrategy)
        self.assertEqual(strategy.period, 10)
        self.assertEqual(strategy.num_std, 3.0)

        # Ichimoku with suboptimal parameters should still create strategy
        strategy = create_strategy('ichimoku', tenkan_period=5, kijun_period=20,
                                   senkou_span_b_period=40, displacement=20)
        self.assertIsInstance(strategy, IchimokuCloudStrategy)
        self.assertEqual(strategy.tenkan_period, 5)
        self.assertEqual(strategy.kijun_period, 20)
        self.assertEqual(strategy.senkou_span_b_period, 40)
        self.assertEqual(strategy.displacement, 20)


if __name__ == '__main__':
    unittest.main()
