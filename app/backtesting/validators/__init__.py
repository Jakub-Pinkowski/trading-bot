"""
Validators Module

This module provides parameter validation for all trading strategies.
It exports both validator classes and backward-compatible validation functions.
"""

from app.backtesting.validators.base import Validator
from app.backtesting.validators.rsi_validator import RSIValidator
from app.backtesting.validators.ema_validator import EMAValidator
from app.backtesting.validators.macd_validator import MACDValidator
from app.backtesting.validators.bollinger_validator import BollingerValidator
from app.backtesting.validators.ichimoku_validator import IchimokuValidator
from app.backtesting.validators.common_validator import CommonValidator


# ==================== Backward Compatibility Functions ====================

def validate_rsi_parameters(rsi_period, lower, upper):
    """
    Validate RSI parameters (backward compatible function).

    Args:
        rsi_period: RSI calculation period
        lower: Lower oversold threshold
        upper: Upper overbought threshold

    Returns:
        List of warning messages
    """
    validator = RSIValidator()
    return validator.validate(rsi_period=rsi_period, lower=lower, upper=upper)


def validate_ema_parameters(ema_short, ema_long):
    """
    Validate EMA crossover parameters (backward compatible function).

    Args:
        ema_short: Short EMA period
        ema_long: Long EMA period

    Returns:
        List of warning messages
    """
    validator = EMAValidator()
    return validator.validate(ema_short=ema_short, ema_long=ema_long)


def validate_macd_parameters(fast_period, slow_period, signal_period):
    """
    Validate MACD parameters (backward compatible function).

    Args:
        fast_period: MACD fast EMA period
        slow_period: MACD slow EMA period
        signal_period: MACD signal line period

    Returns:
        List of warning messages
    """
    validator = MACDValidator()
    return validator.validate(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)


def validate_bollinger_parameters(period, num_std):
    """
    Validate Bollinger Bands parameters (backward compatible function).

    Args:
        period: Moving average period for Bollinger Bands
        num_std: Number of standard deviations for bands

    Returns:
        List of warning messages
    """
    validator = BollingerValidator()
    return validator.validate(period=period, num_std=num_std)


def validate_ichimoku_parameters(tenkan_period, kijun_period, senkou_span_b_period, displacement):
    """
    Validate Ichimoku Cloud parameters (backward compatible function).

    Args:
        tenkan_period: Tenkan-sen (conversion line) period
        kijun_period: Kijun-sen (base line) period
        senkou_span_b_period: Senkou Span B (leading span B) period
        displacement: Cloud displacement/offset

    Returns:
        List of warning messages
    """
    validator = IchimokuValidator()
    return validator.validate(
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        senkou_span_b_period=senkou_span_b_period,
        displacement=displacement
    )


def validate_common_parameters(rollover, trailing, slippage):
    """
    Validate common strategy parameters (backward compatible function).

    Args:
        rollover: Whether to use contract rollover
        trailing: Trailing stop percentage (or None)
        slippage: Slippage percentage (or None)

    Returns:
        List of warning messages

    Raises:
        ValueError: If rollover is not a boolean
    """
    validator = CommonValidator()
    return validator.validate(rollover=rollover, trailing=trailing, slippage=slippage)


# ==================== Exports ====================

__all__ = [
    # Base class
    'Validator',
    # Validator classes
    'RSIValidator',
    'EMAValidator',
    'MACDValidator',
    'BollingerValidator',
    'IchimokuValidator',
    'CommonValidator',
    # Backward compatible functions
    'validate_rsi_parameters',
    'validate_ema_parameters',
    'validate_macd_parameters',
    'validate_bollinger_parameters',
    'validate_ichimoku_parameters',
    'validate_common_parameters',
]
