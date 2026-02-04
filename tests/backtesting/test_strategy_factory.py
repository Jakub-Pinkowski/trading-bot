import unittest
from unittest.mock import patch

import pytest

import app.backtesting.strategy_factory as strategy_factory
from app.backtesting.strategies import (
    BollingerBandsStrategy,
    EMACrossoverStrategy,
    IchimokuCloudStrategy,
    MACDStrategy,
    RSIStrategy,
)
from app.backtesting.strategy_factory import (
    create_strategy, get_strategy_name, _log_warnings_once, _logged_warnings
)
from app.backtesting.validators import (
    BollingerValidator, CommonValidator, EMAValidator,
    IchimokuValidator, MACDValidator, RSIValidator
)


class TestStrategyFactory(unittest.TestCase):
    """Tests for the StrategyFactory class."""

    def test_create_strategy_rsi(self):
        """Test creating an RSI strategy."""
        # Test with default parameters
        strategy = create_strategy('rsi',
                                   rsi_period=14,
                                   lower_threshold=30,
                                   upper_threshold=70,
                                   rollover=False,
                                   trailing=None,
                                   slippage_ticks=0,
                                   symbol=None)

        self.assertIsInstance(strategy, RSIStrategy)
        self.assertEqual(strategy.rsi_period, 14)
        self.assertEqual(strategy.lower_threshold, 30)
        self.assertEqual(strategy.upper_threshold, 70)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)

        # Test with custom parameters
        strategy = create_strategy(
            'rsi',
            rsi_period=21,
            lower_threshold=35,
            upper_threshold=75,
            rollover=True,
            trailing=2.0,
            slippage_ticks=0,
            symbol=None
        )

        self.assertIsInstance(strategy, RSIStrategy)
        self.assertEqual(strategy.rsi_period, 21)
        self.assertEqual(strategy.lower_threshold, 35)
        self.assertEqual(strategy.upper_threshold, 75)
        self.assertEqual(strategy.rollover, True)
        self.assertEqual(strategy.trailing, 2.0)

    def test_create_strategy_ema(self):
        """Test creating an EMA Crossover strategy."""
        # Test with default parameters
        strategy = create_strategy('ema',
                                   short_ema_period=9,
                                   long_ema_period=21,
                                   rollover=False,
                                   trailing=None,
                                   slippage_ticks=0,
                                   symbol=None)

        self.assertIsInstance(strategy, EMACrossoverStrategy)
        self.assertEqual(strategy.short_ema_period, 9)
        self.assertEqual(strategy.long_ema_period, 21)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)

        # Test with custom parameters
        strategy = create_strategy(
            'ema',
            short_ema_period=12,
            long_ema_period=26,
            rollover=True,
            trailing=2.0,
            slippage_ticks=0,
            symbol=None
        )

        self.assertIsInstance(strategy, EMACrossoverStrategy)
        self.assertEqual(strategy.short_ema_period, 12)
        self.assertEqual(strategy.long_ema_period, 26)
        self.assertEqual(strategy.rollover, True)
        self.assertEqual(strategy.trailing, 2.0)

    def test_create_strategy_macd(self):
        """Test creating a MACD strategy."""
        # Test with default parameters
        strategy = create_strategy('macd',
                                   fast_period=12,
                                   slow_period=26,
                                   signal_period=9,
                                   rollover=False,
                                   trailing=None,
                                   slippage_ticks=0,
                                   symbol=None)

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
            trailing=2.0,
            slippage_ticks=0,
            symbol=None
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
        strategy = create_strategy('bollinger',
                                   period=20,
                                   number_of_standard_deviations=2,
                                   rollover=False,
                                   trailing=None,
                                   slippage_ticks=0,
                                   symbol=None)

        self.assertIsInstance(strategy, BollingerBandsStrategy)
        self.assertEqual(strategy.period, 20)
        self.assertEqual(strategy.number_of_standard_deviations, 2)
        self.assertEqual(strategy.rollover, False)
        self.assertIsNone(strategy.trailing)

        # Test with custom parameters
        strategy = create_strategy(
            'bollinger',
            period=15,
            number_of_standard_deviations=2.5,
            rollover=True,
            trailing=2.0,
            slippage_ticks=0,
            symbol=None
        )

        self.assertIsInstance(strategy, BollingerBandsStrategy)
        self.assertEqual(strategy.period, 15)
        self.assertEqual(strategy.number_of_standard_deviations, 2.5)
        self.assertEqual(strategy.rollover, True)
        self.assertEqual(strategy.trailing, 2.0)

    def test_create_strategy_unknown_type(self):
        """Test creating a strategy with an unknown type."""
        with pytest.raises(ValueError, match="Unknown strategy type"):
            create_strategy('unknown')

    def test_create_strategy_fallback(self):
        """Test that unknown strategy types raise ValueError."""
        # Unknown strategy type should raise ValueError
        with pytest.raises(ValueError, match="Unknown strategy type"):
            create_strategy('custom')

    def test_create_rsi_strategy_invalid_parameters(self):
        """Test creating an RSI strategy with invalid parameters."""
        # Invalid RSI period
        with pytest.raises(ValueError, match="rsi period must be a positive integer"):
            create_strategy('rsi',
                            rsi_period=-1,
                            lower_threshold=30,
                            upper_threshold=70,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Invalid lower threshold
        with pytest.raises(ValueError, match="lower threshold must be between 0 and 100"):
            create_strategy('rsi',
                            rsi_period=14,
                            lower_threshold=-10,
                            upper_threshold=70,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Invalid upper threshold
        with pytest.raises(ValueError, match="upper threshold must be between 0 and 100"):
            create_strategy('rsi',
                            rsi_period=14,
                            lower_threshold=30,
                            upper_threshold=110,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Lower >= upper
        with pytest.raises(ValueError, match="Lower threshold .* must be less than upper threshold"):
            create_strategy('rsi',
                            rsi_period=14,
                            lower_threshold=70,
                            upper_threshold=30,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

    def test_create_ema_strategy_invalid_parameters(self):
        """Test creating an EMA strategy with invalid parameters."""
        # Invalid short EMA period
        with pytest.raises(ValueError, match="short EMA period must be a positive integer"):
            create_strategy('ema',
                            short_ema_period=-1,
                            long_ema_period=21,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Invalid long EMA period
        with pytest.raises(ValueError, match="long EMA period must be a positive integer"):
            create_strategy('ema',
                            short_ema_period=9,
                            long_ema_period=-1,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Short >= long
        with pytest.raises(ValueError, match="Short EMA period .* must be less than long EMA period"):
            create_strategy('ema',
                            short_ema_period=21,
                            long_ema_period=9,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

    def test_create_macd_strategy_invalid_parameters(self):
        """Test creating a MACD strategy with invalid parameters."""
        # Invalid fast period
        with pytest.raises(ValueError, match="fast period must be a positive integer"):
            create_strategy('macd',
                            fast_period=-1,
                            slow_period=26,
                            signal_period=9,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Invalid slow period
        with pytest.raises(ValueError, match="slow period must be a positive integer"):
            create_strategy('macd',
                            fast_period=12,
                            slow_period=-1,
                            signal_period=9,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Invalid signal period
        with pytest.raises(ValueError, match="signal period must be a positive integer"):
            create_strategy('macd',
                            fast_period=12,
                            slow_period=26,
                            signal_period=-1,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Fast >= slow
        with pytest.raises(ValueError, match="Fast period .* must be less than slow period"):
            create_strategy('macd',
                            fast_period=26,
                            slow_period=12,
                            signal_period=9,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

    def test_create_bollinger_strategy_invalid_parameters(self):
        """Test creating a Bollinger Bands strategy with invalid parameters."""
        # Invalid period
        with pytest.raises(ValueError, match="period must be a positive integer"):
            create_strategy('bollinger',
                            period=-1,
                            number_of_standard_deviations=2,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Invalid number of standard deviations
        with pytest.raises(ValueError, match="number of standard deviations must be positive"):
            create_strategy('bollinger',
                            period=20,
                            number_of_standard_deviations=-1,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

    def test_create_strategy_ichimoku(self):
        """Test creating an Ichimoku Cloud strategy."""
        # Test with default parameters
        strategy = create_strategy('ichimoku',
                                   tenkan_period=9,
                                   kijun_period=26,
                                   senkou_span_b_period=52,
                                   displacement=26,
                                   rollover=False,
                                   trailing=None,
                                   slippage_ticks=0,
                                   symbol=None)

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
            trailing=2.0,
            slippage_ticks=0,
            symbol=None
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
            create_strategy('ichimoku',
                            tenkan_period=-1,
                            kijun_period=26,
                            senkou_span_b_period=52,
                            displacement=26,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Invalid kijun period
        with pytest.raises(ValueError, match="kijun period must be a positive integer"):
            create_strategy('ichimoku',
                            tenkan_period=9,
                            kijun_period=-1,
                            senkou_span_b_period=52,
                            displacement=26,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Invalid senkou span B period
        with pytest.raises(ValueError, match="senkou span B period must be a positive integer"):
            create_strategy('ichimoku',
                            tenkan_period=9,
                            kijun_period=26,
                            senkou_span_b_period=-1,
                            displacement=26,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Invalid displacement
        with pytest.raises(ValueError, match="displacement must be a positive integer"):
            create_strategy('ichimoku',
                            tenkan_period=9,
                            kijun_period=26,
                            senkou_span_b_period=52,
                            displacement=-1,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

    def test_create_strategy_invalid_common_parameters(self):
        """Test creating a strategy with invalid common parameters."""
        # Invalid rollover (not a boolean)
        with pytest.raises(ValueError, match="rollover must be a boolean"):
            create_strategy('rsi',
                            rsi_period=14,
                            lower_threshold=30,
                            upper_threshold=70,
                            rollover="True",
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)  # String instead of boolean

        # Invalid trailing (not a positive number)
        with pytest.raises(ValueError, match="trailing must be positive"):
            create_strategy('rsi',
                            rsi_period=14,
                            lower_threshold=30,
                            upper_threshold=70,
                            rollover=False,
                            trailing=0,
                            slippage_ticks=0,
                            symbol=None)  # Zero is not positive

        with pytest.raises(ValueError, match="trailing must be positive"):
            create_strategy('rsi',
                            rsi_period=14,
                            lower_threshold=30,
                            upper_threshold=70,
                            rollover=False,
                            trailing=-1,
                            slippage_ticks=0,
                            symbol=None)  # Negative number

        with pytest.raises(ValueError, match="trailing must be positive"):
            create_strategy('rsi',
                            rsi_period=14,
                            lower_threshold=30,
                            upper_threshold=70,
                            rollover=False,
                            trailing="2.0",
                            slippage_ticks=0,
                            symbol=None)  # String instead of number

        # Invalid slippage (not a non-negative number)
        with pytest.raises(ValueError, match="slippage_ticks must be a non-negative number"):
            create_strategy('rsi',
                            rsi_period=14,
                            lower_threshold=30,
                            upper_threshold=70,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=-1,
                            symbol=None)  # Negative number

        with pytest.raises(ValueError, match="slippage_ticks must be a non-negative number"):
            create_strategy('rsi',
                            rsi_period=14,
                            lower_threshold=30,
                            upper_threshold=70,
                            rollover=False,
                            trailing=None,
                            slippage_ticks="0.5",
                            symbol=None)  # String instead of number
    def test_get_strategy_name(self):
        """Test getting a standardized name for a strategy."""
        # Bollinger Bands strategy
        name = get_strategy_name('bollinger',
                                 period=20,
                                 number_of_standard_deviations=2,
                                 rollover=True,
                                 trailing=None,
                                 slippage_ticks=None,
                                 symbol=None)
        self.assertEqual(name, 'BB(period=20,std=2,rollover=True,trailing=None,slippage_ticks=None)')

        # EMA strategy
        name = get_strategy_name('ema',
                                 short_ema_period=9,
                                 long_ema_period=21,
                                 rollover=True,
                                 trailing=2.0,
                                 slippage_ticks=None,
                                 symbol=None)
        self.assertEqual(name, 'EMA(short=9,long=21,rollover=True,trailing=2.0,slippage_ticks=None)')

        # Ichimoku strategy
        name = get_strategy_name('ichimoku',
                                 tenkan_period=9,
                                 kijun_period=26,
                                 senkou_span_b_period=52,
                                 displacement=26,
                                 rollover=False,
                                 trailing=None,
                                 slippage_ticks=None,
                                 symbol=None)
        self.assertEqual(name,
                         'Ichimoku(tenkan=9,kijun=26,senkou_b=52,displacement=26,rollover=False,trailing=None,slippage_ticks=None)')

        # MACD strategy
        name = get_strategy_name('macd',
                                 fast_period=12,
                                 slow_period=26,
                                 signal_period=9,
                                 rollover=False,
                                 trailing=None,
                                 slippage_ticks=None,
                                 symbol=None)
        self.assertEqual(name, 'MACD(fast=12,slow=26,signal=9,rollover=False,trailing=None,slippage_ticks=None)')

        # RSI strategy
        name = get_strategy_name('rsi',
                                 rsi_period=14,
                                 lower_threshold=30,
                                 upper_threshold=70,
                                 rollover=False,
                                 trailing=None,
                                 slippage_ticks=None,
                                 symbol=None)
        self.assertEqual(name, 'RSI(period=14,lower=30,upper=70,rollover=False,trailing=None,slippage_ticks=None)')

        # Unknown strategy type should raise ValueError
        with pytest.raises(ValueError, match="Unknown strategy type"):
            get_strategy_name('unknown', rollover=False, trailing=None, slippage_ticks=0, symbol=None)


class TestPrivateHelperFunctions(unittest.TestCase):
    """Tests for private helper functions in strategy_factory."""

    # Removed: _extract_common_params no longer exists (refactored to _extract_params with different signature)
    # def test_extract_common_params_valid(self):
    # def test_extract_common_params_defaults(self):

    def test_validate_positive_integer_valid(self):
        """Test validate_positive_integer with valid values."""
        from app.backtesting.validators import Validator
        validator = Validator()
        # Should not raise any exception
        validate_positive_integer(1, "test_param")
        validate_positive_integer(100, "test_param")

    def test_validate_positive_integer_invalid(self):
        """Test validate_positive_integer with invalid values."""
        from app.backtesting.validators import Validator
        validator = Validator()

        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            validate_positive_integer(0, "test_param")

        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            validate_positive_integer(-1, "test_param")

        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            validate_positive_integer(1.5, "test_param")

        with pytest.raises(ValueError, match="test_param must be a positive integer"):
            validate_positive_integer("1", "test_param")

    def test_validate_positive_number_valid(self):
        """Test validate_positive_number with valid values."""
        from app.backtesting.validators import Validator
        validator = Validator()
        # Should not raise any exception
        validate_positive_number(1, "test_param")
        validate_positive_number(1.5, "test_param")
        validate_positive_number(100.0, "test_param")

    def test_validate_positive_number_invalid(self):
        """Test validate_positive_number with invalid values."""
        from app.backtesting.validators import Validator
        validator = Validator()

        with pytest.raises(ValueError, match="test_param must be positive"):
            validate_positive_number(0, "test_param")

        with pytest.raises(ValueError, match="test_param must be positive"):
            validate_positive_number(-1, "test_param")

        with pytest.raises(ValueError, match="test_param must be positive"):
            validate_positive_number(-1.5, "test_param")

        with pytest.raises(ValueError, match="test_param must be positive"):
            validate_positive_number("1", "test_param")

    def test_validate_range_valid(self):
        """Test validate_type_and_range with valid values."""
        from app.backtesting.validators import Validator
        validator = Validator()
        # Should not raise any exception
        validate_type_and_range(5, "test_param", 0, 10)
        validate_type_and_range(0, "test_param", 0, 10)
        validate_type_and_range(10, "test_param", 0, 10)
        validate_type_and_range(5.5, "test_param", 0, 10)

    def test_validate_range_invalid(self):
        """Test validate_type_and_range with invalid values."""
        from app.backtesting.validators import Validator
        validator = Validator()

        with pytest.raises(ValueError, match="test_param must be between 0 and 10"):
            validate_type_and_range(-1, "test_param", 0, 10)

        with pytest.raises(ValueError, match="test_param must be between 0 and 10"):
            validate_type_and_range(11, "test_param", 0, 10)

        with pytest.raises(ValueError, match="test_param must be between 0 and 10"):
            validate_type_and_range("5", "test_param", 0, 10)

    # Removed: _format_common_params no longer exists (name formatting now uses strategy's format_name() method)
    # def test_format_common_params(self):


class TestParameterValidation(unittest.TestCase):
    """Tests for the enhanced parameter validation functions."""

    def testvalidate_rsi_parameters_optimal(self):
        """Test RSI parameter validation with optimal parameters."""
        # Standard parameters should generate no warnings
        validator = RSIValidator()
        warnings = validator.validate(rsi_period=14, lower_threshold=30, upper_threshold=70)
        self.assertEqual(len(warnings), 0)

    def testvalidate_rsi_parameters_warnings(self):
        """Test RSI parameter validation with parameters that generate warnings."""
        # Very short period
        validator = RSIValidator()
        warnings = validator.validate(rsi_period=5, lower_threshold=30, upper_threshold=70)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too short", warnings[0])

        # Very long period
        validator = RSIValidator()
        warnings = validator.validate(rsi_period=35, lower_threshold=30, upper_threshold=70)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too long", warnings[0])

        # Very aggressive lower threshold
        validator = RSIValidator()
        warnings = validator.validate(rsi_period=14, lower_threshold=15, upper_threshold=70)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("too aggressive" in w for w in warnings))
        self.assertTrue(any("too wide" in w for w in warnings))

        # Very conservative lower threshold
        validator = RSIValidator()
        warnings = validator.validate(rsi_period=14, lower_threshold=45, upper_threshold=70)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too conservative", warnings[0])

        # Very aggressive upper threshold
        validator = RSIValidator()
        warnings = validator.validate(rsi_period=14, lower_threshold=30, upper_threshold=55)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too aggressive", warnings[0])

        # Very conservative upper threshold
        validator = RSIValidator()
        warnings = validator.validate(rsi_period=14, lower_threshold=30, upper_threshold=85)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("too conservative" in w for w in warnings))
        self.assertTrue(any("too wide" in w for w in warnings))

        # Very narrow gap
        validator = RSIValidator()
        warnings = validator.validate(rsi_period=14, lower_threshold=40, upper_threshold=50)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("too aggressive" in w for w in warnings))
        self.assertTrue(any("too narrow" in w for w in warnings))

        # Very wide gap
        validator = RSIValidator()
        warnings = validator.validate(rsi_period=14, lower_threshold=20, upper_threshold=80)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too wide", warnings[0])

    def testvalidate_ema_parameters_optimal(self):
        """Test EMA parameter validation with optimal parameters."""
        # Standard parameters should generate no warnings
        validator = EMAValidator()
        warnings = validator.validate(short_ema_period=9, long_ema_period=21)
        self.assertEqual(len(warnings), 0)

        validator = EMAValidator()
        warnings = validator.validate(short_ema_period=12, long_ema_period=26)
        self.assertEqual(len(warnings), 0)

    def testvalidate_ema_parameters_warnings(self):
        """Test EMA parameter validation with parameters that generate warnings."""
        # Very short periods
        validator = EMAValidator()
        warnings = validator.validate(short_ema_period=3, long_ema_period=15)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("too short" in w for w in warnings))
        self.assertTrue(any("too wide" in w for w in warnings))

        # Very long periods
        validator = EMAValidator()
        warnings = validator.validate(short_ema_period=25, long_ema_period=60)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("too long" in w for w in warnings))
        self.assertTrue(any("miss trend changes" in w for w in warnings))

        # Too close ratio
        validator = EMAValidator()
        warnings = validator.validate(short_ema_period=12, long_ema_period=15)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too close", warnings[0])

        # Very wide ratio
        validator = EMAValidator()
        warnings = validator.validate(short_ema_period=5, long_ema_period=50)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too wide", warnings[0])

    def testvalidate_macd_parameters_optimal(self):
        """Test MACD parameter validation with optimal parameters."""
        # Standard parameters should generate no warnings (optimal)
        validator = MACDValidator()
        warnings = validator.validate(fast_period=12, slow_period=26, signal_period=9)
        self.assertEqual(len(warnings), 0)

    def testvalidate_macd_parameters_warnings(self):
        """Test MACD parameter validation with parameters that generate warnings."""
        # Very short fast period
        validator = MACDValidator()
        warnings = validator.validate(fast_period=5, slow_period=26, signal_period=9)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too short", warnings[0])

        # Very long fast period
        validator = MACDValidator()
        warnings = validator.validate(fast_period=20, slow_period=26, signal_period=9)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too long", warnings[0])

        # Very short slow period
        validator = MACDValidator()
        warnings = validator.validate(fast_period=12, slow_period=15, signal_period=9)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too short", warnings[0])

        # Very long slow period
        validator = MACDValidator()
        warnings = validator.validate(fast_period=12, slow_period=35, signal_period=9)
        self.assertEqual(len(warnings), 1)
        self.assertIn("miss trend changes", warnings[0])

        # Very short signal period
        validator = MACDValidator()
        warnings = validator.validate(fast_period=12, slow_period=26, signal_period=5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too short", warnings[0])

        # Very long signal period
        validator = MACDValidator()
        warnings = validator.validate(fast_period=12, slow_period=26, signal_period=15)
        self.assertEqual(len(warnings), 1)
        self.assertIn("delay signals", warnings[0])

    def testvalidate_bollinger_parameters_optimal(self):
        """Test Bollinger Bands parameter validation with optimal parameters."""
        # Standard parameters should generate no warnings (optimal)
        validator = BollingerValidator()
        warnings = validator.validate(period=20, number_of_standard_deviations=2.0)
        self.assertEqual(len(warnings), 0)

    def testvalidate_bollinger_parameters_warnings(self):
        """Test Bollinger Bands parameter validation with parameters that generate warnings."""
        # Very short period
        validator = BollingerValidator()
        warnings = validator.validate(period=10, number_of_standard_deviations=2.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too short", warnings[0])

        # Very long period
        validator = BollingerValidator()
        warnings = validator.validate(period=30, number_of_standard_deviations=2.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too long", warnings[0])

        # Very narrow bands
        validator = BollingerValidator()
        warnings = validator.validate(period=20, number_of_standard_deviations=1.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too narrow", warnings[0])

        # Very wide bands
        validator = BollingerValidator()
        warnings = validator.validate(period=20, number_of_standard_deviations=3.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too wide", warnings[0])

    def testvalidate_ichimoku_parameters_optimal(self):
        """Test Ichimoku parameter validation with optimal parameters."""
        # Traditional parameters should generate no warnings (optimal)
        validator = IchimokuValidator()
        warnings = validator.validate(tenkan_period=9, kijun_period=26, senkou_span_b_period=52, displacement=26)
        self.assertEqual(len(warnings), 0)

    def testvalidate_ichimoku_parameters_warnings(self):
        """Test Ichimoku parameter validation with parameters that generate warnings."""
        # Very short tenkan period
        validator = IchimokuValidator()
        warnings = validator.validate(tenkan_period=5, kijun_period=26, senkou_span_b_period=52, displacement=26)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("too short" in w for w in warnings))
        self.assertTrue(any("deviates from traditional" in w for w in warnings))

        # Very long tenkan period
        validator = IchimokuValidator()
        warnings = validator.validate(tenkan_period=15, kijun_period=26, senkou_span_b_period=52, displacement=26)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("too long" in w for w in warnings))
        self.assertTrue(any("deviates from traditional" in w for w in warnings))

        # Very short kijun period
        validator = IchimokuValidator()
        warnings = validator.validate(tenkan_period=9, kijun_period=20, senkou_span_b_period=52, displacement=26)
        self.assertEqual(len(warnings), 4)
        self.assertTrue(any("too short" in w for w in warnings))
        self.assertTrue(any("deviates from traditional" in w for w in warnings))
        self.assertTrue(any("differs from Kijun period" in w for w in warnings))

        # Very long kijun period
        validator = IchimokuValidator()
        warnings = validator.validate(tenkan_period=9, kijun_period=35, senkou_span_b_period=52, displacement=26)
        self.assertEqual(len(warnings), 4)
        self.assertTrue(any("too long" in w for w in warnings))
        self.assertTrue(any("deviates from traditional" in w for w in warnings))
        self.assertTrue(any("differs from Kijun period" in w for w in warnings))

        # Different displacement
        validator = IchimokuValidator()
        warnings = validator.validate(tenkan_period=9, kijun_period=26, senkou_span_b_period=52, displacement=20)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("differs from Kijun period" in w for w in warnings))
        self.assertTrue(any("too short" in w for w in warnings))

        # Non-traditional ratios
        validator = IchimokuValidator()
        warnings = validator.validate(tenkan_period=12, kijun_period=24, senkou_span_b_period=48, displacement=24)
        self.assertEqual(len(warnings), 1)
        self.assertTrue(any("deviates from traditional" in w for w in warnings))

    def testvalidate_common_parameters_optimal(self):
        """Test common parameter validation with optimal parameters."""
        # Reasonable parameters should generate no warnings
        validator = CommonValidator()
        warnings = validator.validate(rollover=False, trailing=2.5, slippage_ticks=1)
        self.assertEqual(len(warnings), 0)

    def testvalidate_common_parameters_warnings(self):
        """Test common parameter validation with parameters that generate warnings."""
        # Very tight trailing stop
        validator = CommonValidator()
        warnings = validator.validate(rollover=False, trailing=0.5, slippage_ticks=1)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too tight", warnings[0])

        # Very wide trailing stop
        validator = CommonValidator()
        warnings = validator.validate(rollover=False, trailing=8.0, slippage_ticks=1)
        self.assertEqual(len(warnings), 1)
        self.assertIn("too wide", warnings[0])

        # Zero slippage (acceptable in testing - no warning expected)
        validator = CommonValidator()
        warnings = validator.validate(rollover=False, trailing=None, slippage_ticks=0.0)
        self.assertEqual(len(warnings), 0)

        # High slippage (6 ticks)
        validator = CommonValidator()
        warnings = validator.validate(rollover=False, trailing=None, slippage_ticks=6)
        self.assertEqual(len(warnings), 1)
        self.assertIn("high", warnings[0].lower())

        # Very high slippage (12 ticks)
        validator = CommonValidator()
        warnings = validator.validate(rollover=False, trailing=None, slippage_ticks=12)
        self.assertEqual(len(warnings), 1)
        self.assertIn("high", warnings[0].lower())

    def test_validate_common_parameters_invalid_rollover_type(self):
        """Test that CommonValidator raises ValueError for non-boolean rollover."""
        # String instead of boolean
        validator = CommonValidator()
        with pytest.raises(ValueError, match="rollover must be a boolean"):
            validator.validate(rollover="true", trailing=None, slippage_ticks=None)

        # Integer instead of boolean
        validator = CommonValidator()
        with pytest.raises(ValueError, match="rollover must be a boolean"):
            validator.validate(rollover=1, trailing=None, slippage_ticks=None)

        # None instead of boolean
        validator = CommonValidator()
        with pytest.raises(ValueError, match="rollover must be a boolean"):
            validator.validate(rollover=None, trailing=None, slippage_ticks=None)

    def test_validation_integration_with_strategy_creation(self):
        """Test that validation warnings are properly logged during strategy creation."""
        # This test verifies that the validation functions are called during strategy creation
        # We can't easily test the logging output, but we can verify strategies are still created

        # RSI with suboptimal parameters should still create strategy
        strategy = create_strategy('rsi',
                                   rsi_period=5,
                                   lower_threshold=15,
                                   upper_threshold=85,
                                   rollover=False,
                                   trailing=None,
                                   slippage_ticks=0,
                                   symbol=None)
        self.assertIsInstance(strategy, RSIStrategy)
        self.assertEqual(strategy.rsi_period, 5)
        self.assertEqual(strategy.lower_threshold, 15)
        self.assertEqual(strategy.upper_threshold, 85)

        # EMA with suboptimal parameters should still create strategy
        strategy = create_strategy('ema',
                                   short_ema_period=3,
                                   long_ema_period=60,
                                   rollover=False,
                                   trailing=None,
                                   slippage_ticks=0,
                                   symbol=None)
        self.assertIsInstance(strategy, EMACrossoverStrategy)
        self.assertEqual(strategy.short_ema_period, 3)
        self.assertEqual(strategy.long_ema_period, 60)

        # MACD with suboptimal parameters should still create strategy
        strategy = create_strategy('macd',
                                   fast_period=5,
                                   slow_period=35,
                                   signal_period=15,
                                   rollover=False,
                                   trailing=None,
                                   slippage_ticks=0,
                                   symbol=None)
        self.assertIsInstance(strategy, MACDStrategy)
        self.assertEqual(strategy.fast_period, 5)
        self.assertEqual(strategy.slow_period, 35)
        self.assertEqual(strategy.signal_period, 15)

        # Bollinger with suboptimal parameters should still create strategy
        strategy = create_strategy('bollinger',
                                   period=10,
                                   number_of_standard_deviations=3.0,
                                   rollover=False,
                                   trailing=None,
                                   slippage_ticks=0,
                                   symbol=None)
        self.assertIsInstance(strategy, BollingerBandsStrategy)
        self.assertEqual(strategy.period, 10)
        self.assertEqual(strategy.number_of_standard_deviations, 3.0)

        # Ichimoku with suboptimal parameters should still create strategy
        strategy = create_strategy('ichimoku', tenkan_period=5, kijun_period=20,
                                   senkou_span_b_period=40,
                                   displacement=20,
                                   rollover=False,
                                   trailing=None,
                                   slippage_ticks=0,
                                   symbol=None)
        self.assertIsInstance(strategy, IchimokuCloudStrategy)
        self.assertEqual(strategy.tenkan_period, 5)
        self.assertEqual(strategy.kijun_period, 20)
        self.assertEqual(strategy.senkou_span_b_period, 40)
        self.assertEqual(strategy.displacement, 20)


class TestWarningDeduplication(unittest.TestCase):
    """Tests for warning deduplication functionality."""

    def setUp(self):
        """Store the original warning state, enable warnings, and clear logged warnings before each test."""
        self.original_warnings_enabled = strategy_factory._log_warnings_enabled
        strategy_factory._log_warnings_enabled = True  # Enable warnings for these tests
        _logged_warnings.clear()

    def tearDown(self):
        """Restore the original warning state and clear logged warnings after each test."""
        strategy_factory._log_warnings_enabled = self.original_warnings_enabled
        _logged_warnings.clear()

    @patch('app.backtesting.strategy_factory.logger')
    def test_log_warnings_once_single_warning(self, mock_logger):
        """Test that a single warning is logged only once."""
        warnings = ["Test warning message"]
        strategy_type = "TEST"

        # Call the function twice with the same warning
        _log_warnings_once(warnings, strategy_type)
        _log_warnings_once(warnings, strategy_type)

        # Verify the warning was logged only once
        self.assertEqual(mock_logger.warning.call_count, 1)
        mock_logger.warning.assert_called_with("TEST Strategy Parameter Guidance: Test warning message")

    @patch('app.backtesting.strategy_factory.logger')
    def test_log_warnings_once_multiple_warnings(self, mock_logger):
        """Test that multiple different warnings are logged."""
        warnings1 = ["Warning 1", "Warning 2"]
        warnings2 = ["Warning 3"]
        strategy_type = "TEST"

        # Call with different warnings
        _log_warnings_once(warnings1, strategy_type)
        _log_warnings_once(warnings2, strategy_type)

        # Verify all warnings were logged
        self.assertEqual(mock_logger.warning.call_count, 3)
        expected_calls = [
            unittest.mock.call("TEST Strategy Parameter Guidance: Warning 1"),
            unittest.mock.call("TEST Strategy Parameter Guidance: Warning 2"),
            unittest.mock.call("TEST Strategy Parameter Guidance: Warning 3")
        ]
        mock_logger.warning.assert_has_calls(expected_calls)

    @patch('app.backtesting.strategy_factory.logger')
    def test_log_warnings_once_different_strategy_types(self, mock_logger):
        """Test that the same warning for different strategy types is logged for both strategies."""
        warnings = ["Same warning message"]

        # Call with different strategy types
        _log_warnings_once(warnings, "RSI")
        _log_warnings_once(warnings, "EMA")

        # Verify both warnings were logged (different strategy types)
        self.assertEqual(mock_logger.warning.call_count, 2)
        expected_calls = [
            unittest.mock.call("RSI Strategy Parameter Guidance: Same warning message"),
            unittest.mock.call("EMA Strategy Parameter Guidance: Same warning message")
        ]
        mock_logger.warning.assert_has_calls(expected_calls)

    @patch('app.backtesting.strategy_factory.logger')
    def test_strategy_creation_warning_deduplication(self, mock_logger):
        """Test that creating multiple strategies with the same parameters only logs warnings once."""
        # Create multiple RSI strategies with the same warning-triggering parameters
        for _ in range(3):
            create_strategy('rsi',
                            rsi_period=7,
                            lower_threshold=30,
                            upper_threshold=70,
                            rollover=False,
                            trailing=None,
                            slippage_ticks=0,
                            symbol=None)

        # Count how many times warnings about RSI period 7 were logged
        rsi_period_warnings = [call for call in mock_logger.warning.call_args_list
                               if 'RSI period 7' in str(call)]

        # Should only be logged once despite creating 3 strategies
        self.assertEqual(len(rsi_period_warnings), 1)

    @patch('app.backtesting.strategy_factory.logger')
    def test_mixed_strategy_creation_warning_deduplication(self, mock_logger):
        """Test warning deduplication across different strategy types."""
        # Create strategies with parameters that trigger warnings
        create_strategy('rsi',
                        rsi_period=7,
                        lower_threshold=30,
                        upper_threshold=70,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        create_strategy('ema',
                        short_ema_period=5,
                        long_ema_period=18,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        create_strategy('rsi',
                        rsi_period=7,
                        lower_threshold=30,
                        upper_threshold=70,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)  # Same as first
        create_strategy('bollinger',
                        period=10,
                        number_of_standard_deviations=3,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        create_strategy('ema',
                        short_ema_period=5,
                        long_ema_period=18,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)  # Same as second

        # Count warnings for each type
        rsi_warnings = [call for call in mock_logger.warning.call_args_list
                        if 'rsi Strategy Parameter Guidance' in str(call)]
        ema_warnings = [call for call in mock_logger.warning.call_args_list
                        if 'ema Strategy Parameter Guidance' in str(call)]
        bb_warnings = [call for call in mock_logger.warning.call_args_list
                       if 'bollinger Strategy Parameter Guidance' in str(call)]

        # Each strategy type should have its warnings logged only once
        # RSI should have warnings about period 7
        rsi_period_warnings = [call for call in rsi_warnings if 'RSI period 7' in str(call)]
        self.assertEqual(len(rsi_period_warnings), 1)

        # EMA should have warnings about a ratio
        ema_ratio_warnings = [call for call in ema_warnings if 'ratio' in str(call)]
        self.assertEqual(len(ema_ratio_warnings), 1)

        # Bollinger Bands should have warnings
        self.assertGreater(len(bb_warnings), 0)

    def test_logged_warnings_set_persistence(self):
        """Test that the logged warnings set persists across function calls."""
        # Initially empty
        self.assertEqual(len(_logged_warnings), 0)

        # Add some warnings
        _log_warnings_once(["Test warning 1"], "TEST")
        self.assertEqual(len(_logged_warnings), 1)

        # Add more warnings
        _log_warnings_once(["Test warning 2", "Test warning 3"], "TEST")
        self.assertEqual(len(_logged_warnings), 3)

        # Try to add duplicate
        _log_warnings_once(["Test warning 1"], "TEST")
        self.assertEqual(len(_logged_warnings), 3)  # Should not increase


class TestWarningConfiguration(unittest.TestCase):
    """Tests for warning enable/disable functionality."""

    def setUp(self):
        """Store the original warning state and clear logged warnings before each test."""
        self.original_warnings_enabled = strategy_factory._log_warnings_enabled
        _logged_warnings.clear()

    def tearDown(self):
        """Restore the original warning state and clear logged warnings after each test."""
        strategy_factory._log_warnings_enabled = self.original_warnings_enabled
        _logged_warnings.clear()

    def test_warnings_enabled_by_default(self):
        """Test that warnings are enabled by default."""
        # The default state should be True (warnings enabled)
        self.assertTrue(strategy_factory._log_warnings_enabled)

    @patch('app.backtesting.strategy_factory.logger')
    def test_warnings_disabled_no_logging(self, mock_logger):
        """Test that no warnings are logged when warnings are disabled."""
        # Ensure warnings are disabled
        strategy_factory._log_warnings_enabled = False

        # Create strategies with parameters that would normally trigger warnings
        create_strategy('rsi',
                        rsi_period=7,
                        lower_threshold=30,
                        upper_threshold=70,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        create_strategy('ema',
                        short_ema_period=5,
                        long_ema_period=18,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        create_strategy('bollinger',
                        period=10,
                        number_of_standard_deviations=3,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)

        # Verify no warnings were logged
        self.assertEqual(mock_logger.warning.call_count, 0)

    @patch('app.backtesting.strategy_factory.logger')
    def test_warnings_enabled_logging_works(self, mock_logger):
        """Test that warnings are logged when warnings are enabled."""
        # Enable warnings
        strategy_factory._log_warnings_enabled = True

        # Create a strategy with parameters that trigger warnings
        create_strategy('rsi',
                        rsi_period=7,
                        lower_threshold=30,
                        upper_threshold=70,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)

        # Verify warnings were logged
        self.assertGreater(mock_logger.warning.call_count, 0)

        # Check that at least one warning contains the expected content
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        rsi_warnings = [call for call in warning_calls if 'RSI period 7' in call]
        self.assertGreater(len(rsi_warnings), 0)

    @patch('app.backtesting.strategy_factory.logger')
    def test_toggle_warnings_during_execution(self, mock_logger):
        """Test toggling warnings on and off during execution."""
        # Start with warnings enabled
        strategy_factory._log_warnings_enabled = True

        # Create strategy - should log warnings
        create_strategy('rsi',
                        rsi_period=7,
                        lower_threshold=30,
                        upper_threshold=70,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        initial_warning_count = mock_logger.warning.call_count
        self.assertGreater(initial_warning_count, 0)

        # Disable warnings
        strategy_factory._log_warnings_enabled = False

        # Create another strategy - should not log warnings
        create_strategy('ema',
                        short_ema_period=5,
                        long_ema_period=18,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        self.assertEqual(mock_logger.warning.call_count, initial_warning_count)

        # Re-enable warnings
        strategy_factory._log_warnings_enabled = True

        # Create another strategy - should log warnings again
        create_strategy('bollinger',
                        period=10,
                        number_of_standard_deviations=3,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        self.assertGreater(mock_logger.warning.call_count, initial_warning_count)

    @patch('app.backtesting.strategy_factory.logger')
    def test_log_warnings_once_respects_disabled_state(self, mock_logger):
        """Test that _log_warnings_once respects the disabled state."""
        # Disable warnings
        strategy_factory._log_warnings_enabled = False

        # Call _log_warnings_once directly
        warnings = ["Test warning message"]
        _log_warnings_once(warnings, "TEST")

        # Verify no warnings were logged
        self.assertEqual(mock_logger.warning.call_count, 0)

        # Enable warnings and try again
        strategy_factory._log_warnings_enabled = True
        _log_warnings_once(warnings, "TEST")

        # Verify warning was logged
        self.assertEqual(mock_logger.warning.call_count, 1)
        mock_logger.warning.assert_called_with("TEST Strategy Parameter Guidance: Test warning message")

    @patch('app.backtesting.strategy_factory.logger')
    def test_warning_deduplication_with_enabled_disabled_cycle(self, mock_logger):
        """Test that warning deduplication works correctly when toggling the enabled / disabled state."""
        # Enable warnings and create a strategy
        strategy_factory._log_warnings_enabled = True
        create_strategy('rsi',
                        rsi_period=7,
                        lower_threshold=30,
                        upper_threshold=70,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        initial_warning_count = mock_logger.warning.call_count
        self.assertGreater(initial_warning_count, 0)

        # Disable warnings and create the same strategy - no new warnings
        strategy_factory._log_warnings_enabled = False
        create_strategy('rsi',
                        rsi_period=7,
                        lower_threshold=30,
                        upper_threshold=70,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        self.assertEqual(mock_logger.warning.call_count, initial_warning_count)

        # Re-enable warnings and create the same strategy - still no new warnings (deduplication)
        strategy_factory._log_warnings_enabled = True
        create_strategy('rsi',
                        rsi_period=7,
                        lower_threshold=30,
                        upper_threshold=70,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        self.assertEqual(mock_logger.warning.call_count, initial_warning_count)

        # Create a different strategy - should log new warnings
        create_strategy('ema',
                        short_ema_period=5,
                        long_ema_period=18,
                        rollover=False,
                        trailing=None,
                        slippage_ticks=0,
                        symbol=None)
        self.assertGreater(mock_logger.warning.call_count, initial_warning_count)


if __name__ == '__main__':
    unittest.main()
