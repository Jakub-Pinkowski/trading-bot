"""
Strategy Parameter Validation Functions

This module contains all validation functions for strategy parameters.
Each validation function checks parameter ranges and returns a list of warnings
based on common trading practices and backtesting experience.

When adding new strategies, add their validation function here following the same pattern.
"""

from app.backtesting.validation_constants import *


def validate_rsi_parameters(rsi_period, lower, upper):
    """
    Enhanced validation for RSI parameters with guidance on reasonable ranges.

    Reasonable ranges based on common trading practices:
    - RSI Period: 10-30 (14 is the most common, shorter periods = more sensitive)
    - Lower threshold: 20-40 (30 is standard, lower = more aggressive)
    - Upper threshold: 60-80 (70 is standard, higher = more conservative)
    - Threshold gap: Should be at least 20 points to avoid excessive signals
    """
    warnings = []

    # RSI Period validation
    if rsi_period < RSI_PERIOD_MIN_RECOMMENDED:
        warnings.append(f"RSI period {rsi_period} is quite short and may generate excessive noise. "
                        f"Consider using {RSI_PERIOD_MIN_RECOMMENDED}-{RSI_PERIOD_MAX_RECOMMENDED} range "
                        f"({RSI_PERIOD_STANDARD} is most common).")
    elif rsi_period > RSI_PERIOD_MAX_RECOMMENDED:
        warnings.append(f"RSI period {rsi_period} is quite long and may be too slow to catch trends. "
                        f"Consider using {RSI_PERIOD_MIN_RECOMMENDED}-{RSI_PERIOD_MAX_RECOMMENDED} range "
                        f"({RSI_PERIOD_STANDARD} is most common).")

    # Lower threshold validation
    if lower < RSI_LOWER_MIN_AGGRESSIVE:
        warnings.append(f"RSI lower threshold {lower} is very aggressive and may generate many false signals. "
                        f"Consider using {RSI_LOWER_MIN_AGGRESSIVE}-{RSI_LOWER_MAX_CONSERVATIVE} range "
                        f"({RSI_LOWER_STANDARD} is standard).")
    elif lower > RSI_LOWER_MAX_CONSERVATIVE:
        warnings.append(f"RSI lower threshold {lower} is very conservative and may miss opportunities. "
                        f"Consider using {RSI_LOWER_MIN_AGGRESSIVE}-{RSI_LOWER_MAX_CONSERVATIVE} range "
                        f"({RSI_LOWER_STANDARD} is standard).")

    # Upper threshold validation
    if upper < RSI_UPPER_MIN_AGGRESSIVE:
        warnings.append(f"RSI upper threshold {upper} is very aggressive and may generate many false signals. "
                        f"Consider using {RSI_UPPER_MIN_AGGRESSIVE}-{RSI_UPPER_MAX_CONSERVATIVE} range "
                        f"({RSI_UPPER_STANDARD} is standard).")
    elif upper > RSI_UPPER_MAX_CONSERVATIVE:
        warnings.append(f"RSI upper threshold {upper} is very conservative and may miss opportunities. "
                        f"Consider using {RSI_UPPER_MIN_AGGRESSIVE}-{RSI_UPPER_MAX_CONSERVATIVE} range "
                        f"({RSI_UPPER_STANDARD} is standard).")

    # Threshold gap validation
    gap = upper - lower
    if gap < RSI_GAP_MIN:
        warnings.append(f"RSI threshold gap ({gap}) is quite narrow and may generate excessive signals. "
                        f"Consider using a gap of at least {RSI_GAP_MIN} points (e.g., {RSI_LOWER_STANDARD}/{RSI_UPPER_STANDARD}).")
    elif gap > RSI_GAP_MAX:
        warnings.append(f"RSI threshold gap ({gap}) is very wide and may miss many opportunities. "
                        f"Consider using a gap of {RSI_GAP_MIN}-{RSI_GAP_MAX} points.")

    return warnings


def validate_ema_parameters(ema_short, ema_long):
    """
    Enhanced validation for EMA crossover parameters with guidance on reasonable ranges.

    Reasonable ranges based on common trading practices:
    - Short EMA: 5-21 (9-12 are most common for short-term signals)
    - Long EMA: 15-50 (21-26 are most common for trend confirmation)
    - Ratio: Long should be 1.5-3x the short period for good separation
    """
    warnings = []

    # Short EMA validation
    if ema_short < EMA_SHORT_MIN:
        warnings.append(f"Short EMA period {ema_short} is very sensitive and may generate excessive noise. "
                        f"Consider using {EMA_SHORT_MIN}-{EMA_SHORT_MAX} range "
                        f"({EMA_SHORT_COMMON_MIN}-{EMA_SHORT_COMMON_MAX} are most common).")
    elif ema_short > EMA_SHORT_MAX:
        warnings.append(f"Short EMA period {ema_short} may be too slow for crossover signals. "
                        f"Consider using {EMA_SHORT_MIN}-{EMA_SHORT_MAX} range "
                        f"({EMA_SHORT_COMMON_MIN}-{EMA_SHORT_COMMON_MAX} are most common).")

    # Long EMA validation
    if ema_long < EMA_LONG_MIN:
        warnings.append(f"Long EMA period {ema_long} may be too short for trend confirmation. "
                        f"Consider using {EMA_LONG_MIN}-{EMA_LONG_MAX} range "
                        f"({EMA_LONG_COMMON_MIN}-{EMA_LONG_COMMON_MAX} are most common).")
    elif ema_long > EMA_LONG_MAX:
        warnings.append(f"Long EMA period {ema_long} may be too slow and miss trend changes. "
                        f"Consider using {EMA_LONG_MIN}-{EMA_LONG_MAX} range "
                        f"({EMA_LONG_COMMON_MIN}-{EMA_LONG_COMMON_MAX} are most common).")

    # Ratio validation
    ratio = ema_long / ema_short
    if ratio < EMA_RATIO_MIN:
        warnings.append(f"EMA ratio ({ratio:.1f}) is too close - periods {ema_short}/{ema_long} may generate false signals. "
                        f"Consider using a ratio of {EMA_RATIO_MIN}-{EMA_RATIO_MAX}x (e.g., 9/21, 12/26).")
    elif ratio > EMA_RATIO_MAX:
        warnings.append(f"EMA ratio ({ratio:.1f}) is very wide - periods {ema_short}/{ema_long} may be too slow. "
                        f"Consider using a ratio of {EMA_RATIO_MIN}-{EMA_RATIO_MAX}x (e.g., 9/21, 12/26).")

    return warnings


def validate_macd_parameters(fast_period, slow_period, signal_period):
    """
    Enhanced validation for MACD parameters with guidance on reasonable ranges.

    Reasonable ranges based on common trading practices:
    - Fast period: 8-15 (12 is standard)
    - Slow period: 20-30 (26 is standard)
    - Signal period: 7-12 (9 is standard)
    - Standard MACD: 12/26/9 is most widely used
    """
    warnings = []

    # Fast period validation
    if fast_period < MACD_FAST_MIN:
        warnings.append(f"MACD fast period {fast_period} is very short and may generate excessive noise. "
                        f"Consider using {MACD_FAST_MIN}-{MACD_FAST_MAX} range ({MACD_FAST_STANDARD} is standard).")
    elif fast_period > MACD_FAST_MAX:
        warnings.append(f"MACD fast period {fast_period} may be too slow for responsive signals. "
                        f"Consider using {MACD_FAST_MIN}-{MACD_FAST_MAX} range ({MACD_FAST_STANDARD} is standard).")

    # Slow period validation
    if slow_period < MACD_SLOW_MIN:
        warnings.append(f"MACD slow period {slow_period} may be too short for trend confirmation. "
                        f"Consider using {MACD_SLOW_MIN}-{MACD_SLOW_MAX} range ({MACD_SLOW_STANDARD} is standard).")
    elif slow_period > MACD_SLOW_MAX:
        warnings.append(f"MACD slow period {slow_period} may be too slow and miss trend changes. "
                        f"Consider using {MACD_SLOW_MIN}-{MACD_SLOW_MAX} range ({MACD_SLOW_STANDARD} is standard).")

    # Signal period validation
    if signal_period < MACD_SIGNAL_MIN:
        warnings.append(f"MACD signal period {signal_period} is very short and may generate false signals. "
                        f"Consider using {MACD_SIGNAL_MIN}-{MACD_SIGNAL_MAX} range ({MACD_SIGNAL_STANDARD} is standard).")
    elif signal_period > MACD_SIGNAL_MAX:
        warnings.append(f"MACD signal period {signal_period} may be too slow for timely signals. "
                        f"Consider using {MACD_SIGNAL_MIN}-{MACD_SIGNAL_MAX} range ({MACD_SIGNAL_STANDARD} is standard).")

    # Standard combination check
    if (fast_period, slow_period, signal_period) == (MACD_FAST_STANDARD, MACD_SLOW_STANDARD, MACD_SIGNAL_STANDARD):
        warnings.append(f"Using standard MACD parameters ({MACD_FAST_STANDARD}/{MACD_SLOW_STANDARD}/{MACD_SIGNAL_STANDARD}) - widely tested and reliable.")

    return warnings


def validate_bollinger_parameters(period, num_std):
    """
    Enhanced validation for Bollinger Bands parameters with guidance on reasonable ranges.

    Reasonable ranges based on common trading practices:
    - Period: 15-25 (20 is standard)
    - Standard deviations: 1.5-2.5 (2.0 is standard)
    - Standard BB: 20/2.0 captures ~95% of price action
    """
    warnings = []

    # Period validation
    if period < BB_PERIOD_MIN:
        warnings.append(f"Bollinger Bands period {period} is quite short and may be too sensitive. "
                        f"Consider using {BB_PERIOD_MIN}-{BB_PERIOD_MAX} range ({BB_PERIOD_STANDARD} is standard).")
    elif period > BB_PERIOD_MAX:
        warnings.append(f"Bollinger Bands period {period} is quite long and may be too slow. "
                        f"Consider using {BB_PERIOD_MIN}-{BB_PERIOD_MAX} range ({BB_PERIOD_STANDARD} is standard).")

    # Standard deviation validation
    if num_std < BB_STD_MIN:
        warnings.append(f"Bollinger Bands standard deviation {num_std} is quite narrow and may generate excessive signals. "
                        f"Consider using {BB_STD_MIN}-{BB_STD_MAX} range ({BB_STD_STANDARD} is standard).")
    elif num_std > BB_STD_MAX:
        warnings.append(f"Bollinger Bands standard deviation {num_std} is quite wide and may miss opportunities. "
                        f"Consider using {BB_STD_MIN}-{BB_STD_MAX} range ({BB_STD_STANDARD} is standard).")

    # Standard combination check
    if (period, num_std) == (BB_PERIOD_STANDARD, BB_STD_STANDARD):
        warnings.append(f"Using standard Bollinger Bands parameters ({BB_PERIOD_STANDARD}/{BB_STD_STANDARD}) - captures ~95% of price action.")

    return warnings


def validate_ichimoku_parameters(tenkan_period, kijun_period, senkou_span_b_period, displacement):
    """
    Enhanced validation for Ichimoku parameters with guidance on reasonable ranges.

    Reasonable ranges based on traditional Ichimoku settings:
    - Tenkan-sen: 7-12 (9 is traditional)
    - Kijun-sen: 22-30 (26 is traditional)
    - Senkou Span B: 44-60 (52 is traditional)
    - Displacement: 22-30 (26 is traditional, should match Kijun-sen)
    - Traditional Ichimoku: 9/26/52/26 based on Japanese market cycles
    """
    warnings = []

    # Tenkan period validation
    if tenkan_period < ICHIMOKU_TENKAN_MIN:
        warnings.append(f"Ichimoku Tenkan period {tenkan_period} is quite short and may be too sensitive. "
                        f"Consider using {ICHIMOKU_TENKAN_MIN}-{ICHIMOKU_TENKAN_MAX} range ({ICHIMOKU_TENKAN_STANDARD} is traditional).")
    elif tenkan_period > ICHIMOKU_TENKAN_MAX:
        warnings.append(f"Ichimoku Tenkan period {tenkan_period} may be too slow for conversion line. "
                        f"Consider using {ICHIMOKU_TENKAN_MIN}-{ICHIMOKU_TENKAN_MAX} range ({ICHIMOKU_TENKAN_STANDARD} is traditional).")

    # Kijun period validation
    if kijun_period < ICHIMOKU_KIJUN_MIN:
        warnings.append(f"Ichimoku Kijun period {kijun_period} may be too short for baseline. "
                        f"Consider using {ICHIMOKU_KIJUN_MIN}-{ICHIMOKU_KIJUN_MAX} range ({ICHIMOKU_KIJUN_STANDARD} is traditional).")
    elif kijun_period > ICHIMOKU_KIJUN_MAX:
        warnings.append(f"Ichimoku Kijun period {kijun_period} may be too slow for trend confirmation. "
                        f"Consider using {ICHIMOKU_KIJUN_MIN}-{ICHIMOKU_KIJUN_MAX} range ({ICHIMOKU_KIJUN_STANDARD} is traditional).")

    # Senkou Span B period validation
    if senkou_span_b_period < ICHIMOKU_SENKOU_B_MIN:
        warnings.append(f"Ichimoku Senkou Span B period {senkou_span_b_period} may be too short for cloud formation. "
                        f"Consider using {ICHIMOKU_SENKOU_B_MIN}-{ICHIMOKU_SENKOU_B_MAX} range ({ICHIMOKU_SENKOU_B_STANDARD} is traditional).")
    elif senkou_span_b_period > ICHIMOKU_SENKOU_B_MAX:
        warnings.append(f"Ichimoku Senkou Span B period {senkou_span_b_period} may be too slow. "
                        f"Consider using {ICHIMOKU_SENKOU_B_MIN}-{ICHIMOKU_SENKOU_B_MAX} range ({ICHIMOKU_SENKOU_B_STANDARD} is traditional).")

    # Displacement validation
    if displacement < ICHIMOKU_DISPLACEMENT_MIN:
        warnings.append(f"Ichimoku displacement {displacement} may be too short for proper cloud projection. "
                        f"Consider using {ICHIMOKU_DISPLACEMENT_MIN}-{ICHIMOKU_DISPLACEMENT_MAX} range ({ICHIMOKU_DISPLACEMENT_STANDARD} is traditional).")
    elif displacement > ICHIMOKU_DISPLACEMENT_MAX:
        warnings.append(f"Ichimoku displacement {displacement} may project too far into future. "
                        f"Consider using {ICHIMOKU_DISPLACEMENT_MIN}-{ICHIMOKU_DISPLACEMENT_MAX} range ({ICHIMOKU_DISPLACEMENT_STANDARD} is traditional).")

    # Traditional ratios validation
    tenkan_kijun_ratio = kijun_period / tenkan_period
    if tenkan_kijun_ratio < ICHIMOKU_TENKAN_KIJUN_RATIO_MIN or tenkan_kijun_ratio > ICHIMOKU_TENKAN_KIJUN_RATIO_MAX:
        warnings.append(f"Ichimoku Tenkan/Kijun ratio ({tenkan_kijun_ratio:.1f}) deviates from traditional ~3:1 ratio. "
                        f"Consider maintaining traditional proportions (e.g., {ICHIMOKU_TENKAN_STANDARD}/{ICHIMOKU_KIJUN_STANDARD}).")

    kijun_senkou_ratio = senkou_span_b_period / kijun_period
    if kijun_senkou_ratio < ICHIMOKU_KIJUN_SENKOU_RATIO_MIN or kijun_senkou_ratio > ICHIMOKU_KIJUN_SENKOU_RATIO_MAX:
        warnings.append(f"Ichimoku Kijun/Senkou B ratio ({kijun_senkou_ratio:.1f}) deviates from traditional 2:1 ratio. "
                        f"Consider maintaining traditional proportions (e.g., {ICHIMOKU_KIJUN_STANDARD}/{ICHIMOKU_SENKOU_B_STANDARD}).")

    # Displacement vs Kijun check
    if displacement != kijun_period:
        warnings.append(f"Ichimoku displacement ({displacement}) differs from Kijun period ({kijun_period}). "
                        f"Traditional Ichimoku uses same value for both (typically {ICHIMOKU_DISPLACEMENT_STANDARD}).")

    # Traditional combination check
    if (tenkan_period, kijun_period, senkou_span_b_period, displacement) == (
            ICHIMOKU_TENKAN_STANDARD,
            ICHIMOKU_KIJUN_STANDARD,
            ICHIMOKU_SENKOU_B_STANDARD,
            ICHIMOKU_DISPLACEMENT_STANDARD
    ):
        warnings.append(f"Using traditional Ichimoku parameters ({ICHIMOKU_TENKAN_STANDARD}/{ICHIMOKU_KIJUN_STANDARD}/{ICHIMOKU_SENKOU_B_STANDARD}/{ICHIMOKU_DISPLACEMENT_STANDARD}) - based on Japanese market cycles.")

    return warnings


def validate_common_parameters(rollover, trailing, slippage):
    """
    Enhanced validation for common strategy parameters with guidance.

    Reasonable ranges:
    - Rollover: Boolean (True for continuous contracts, False for single contract)
    - Trailing: None or 1-5% (2-3% is common for futures)
    - Slippage: 0-0.5% (0.1-0.2% is typical for liquid futures)
    """
    warnings = []

    # Rollover validation
    if not isinstance(rollover, bool):
        raise ValueError(f"rollover must be a boolean (True or False), got {type(rollover).__name__}")

    # Trailing stop validation
    if trailing is not None:
        if trailing < TRAILING_STOP_MIN:
            warnings.append(f"Trailing stop {trailing}% is very tight and may be stopped out frequently. "
                            f"Consider using {TRAILING_STOP_MIN}-{TRAILING_STOP_MAX}% range "
                            f"({TRAILING_STOP_COMMON_MIN}-{TRAILING_STOP_COMMON_MAX}% is common for futures).")
        elif trailing > TRAILING_STOP_MAX:
            warnings.append(f"Trailing stop {trailing}% is very wide and may give back large profits. "
                            f"Consider using {TRAILING_STOP_MIN}-{TRAILING_STOP_MAX}% range "
                            f"({TRAILING_STOP_COMMON_MIN}-{TRAILING_STOP_COMMON_MAX}% is common for futures).")

    # Slippage validation
    if slippage is not None:
        if slippage > SLIPPAGE_MAX:
            warnings.append(f"Slippage {slippage}% is very high and may significantly impact returns. "
                            f"Consider using 0-{SLIPPAGE_MAX}% range "
                            f"({SLIPPAGE_TYPICAL_MIN}-{SLIPPAGE_TYPICAL_MAX}% is typical for liquid futures).")
        elif slippage == 0:
            warnings.append(f"Slippage {slippage}% is unrealistic - all orders experience some slippage. "
                            f"Consider using {SLIPPAGE_TYPICAL_MIN}-{SLIPPAGE_TYPICAL_MAX}% for liquid futures.")

    return warnings
