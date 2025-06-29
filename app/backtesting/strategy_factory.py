from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy
from app.backtesting.strategies.ichimoku_cloud import IchimokuCloudStrategy
from app.backtesting.strategies.macd import MACDStrategy
from app.backtesting.strategies.rsi import RSIStrategy
from app.utils.logger import get_logger

logger = get_logger('backtesting/strategy_factory')

# Define strategy types
STRATEGY_TYPES = ['rsi', 'ema', 'macd', 'bollinger', 'ichimoku']


def _extract_common_params(**params):
    """Extract common parameters used by all strategies."""
    # Extract parameters with defaults
    rollover = params.get('rollover', False)
    trailing = params.get('trailing', None)
    slippage = params.get('slippage', None)

    # Validate rollover (should be boolean)
    if not isinstance(rollover, bool):
        logger.error(f"Invalid rollover: {rollover}")
        raise ValueError("rollover must be a boolean (True or False)")

    # Validate trailing (should be None or positive number)
    if trailing is not None and (not isinstance(trailing, (int, float)) or trailing <= 0):
        logger.error(f"Invalid trailing: {trailing}")
        raise ValueError("trailing must be None or a positive number")

    # Validate slippage (should be None or non-negative number)
    if slippage is not None and (not isinstance(slippage, (int, float)) or slippage < 0):
        logger.error(f"Invalid slippage: {slippage}")
        raise ValueError("slippage must be None or a non-negative number")

    return {
        'rollover': rollover,
        'trailing': trailing,
        'slippage': slippage
    }


def _validate_positive_integer(value, param_name):
    """Validate that a parameter is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        logger.error(f"Invalid {param_name}: {value}")
        raise ValueError(f"{param_name} must be a positive integer")


def _validate_positive_number(value, param_name):
    """Validate that a parameter is a positive number (int or float)."""
    if not isinstance(value, (int, float)) or value <= 0:
        logger.error(f"Invalid {param_name}: {value}")
        raise ValueError(f"{param_name} must be positive")


def _validate_range(value, param_name, min_val, max_val):
    """Validate that a parameter is within a specified range."""
    if not isinstance(value, (int, float)) or value < min_val or value > max_val:
        logger.error(f"Invalid {param_name}: {value}")
        raise ValueError(f"{param_name} must be between {min_val} and {max_val}")


def _validate_rsi_parameters(rsi_period, lower, upper):
    """
    Enhanced validation for RSI parameters with guidance on reasonable ranges.

    Reasonable ranges based on common trading practices:
    - RSI Period: 10-30 (14 is most common, shorter periods = more sensitive)
    - Lower threshold: 20-40 (30 is standard, lower = more aggressive)
    - Upper threshold: 60-80 (70 is standard, higher = more conservative)
    - Threshold gap: Should be at least 20 points to avoid excessive signals
    """
    warnings = []

    # RSI Period validation
    if rsi_period < 10:
        warnings.append(f"RSI period {rsi_period} is quite short and may generate excessive noise. "
                        f"Consider using 10-30 range (14 is most common).")
    elif rsi_period > 30:
        warnings.append(f"RSI period {rsi_period} is quite long and may be too slow to catch trends. "
                        f"Consider using 10-30 range (14 is most common).")

    # Lower threshold validation
    if lower < 20:
        warnings.append(f"RSI lower threshold {lower} is very aggressive and may generate many false signals. "
                        f"Consider using 20-40 range (30 is standard).")
    elif lower > 40:
        warnings.append(f"RSI lower threshold {lower} is very conservative and may miss opportunities. "
                        f"Consider using 20-40 range (30 is standard).")

    # Upper threshold validation
    if upper < 60:
        warnings.append(f"RSI upper threshold {upper} is very aggressive and may generate many false signals. "
                        f"Consider using 60-80 range (70 is standard).")
    elif upper > 80:
        warnings.append(f"RSI upper threshold {upper} is very conservative and may miss opportunities. "
                        f"Consider using 60-80 range (70 is standard).")

    # Threshold gap validation
    gap = upper - lower
    if gap < 20:
        warnings.append(f"RSI threshold gap ({gap}) is quite narrow and may generate excessive signals. "
                        f"Consider using a gap of at least 20 points (e.g., 30/70).")
    elif gap > 50:
        warnings.append(f"RSI threshold gap ({gap}) is very wide and may miss many opportunities. "
                        f"Consider using a gap of 20-50 points.")

    return warnings


def _validate_ema_parameters(ema_short, ema_long):
    """
    Enhanced validation for EMA crossover parameters with guidance on reasonable ranges.

    Reasonable ranges based on common trading practices:
    - Short EMA: 5-21 (9-12 are most common for short-term signals)
    - Long EMA: 15-50 (21-26 are most common for trend confirmation)
    - Ratio: Long should be 1.5-3x the short period for good separation
    """
    warnings = []

    # Short EMA validation
    if ema_short < 5:
        warnings.append(f"Short EMA period {ema_short} is very sensitive and may generate excessive noise. "
                        f"Consider using 5-21 range (9-12 are most common).")
    elif ema_short > 21:
        warnings.append(f"Short EMA period {ema_short} may be too slow for crossover signals. "
                        f"Consider using 5-21 range (9-12 are most common).")

    # Long EMA validation
    if ema_long < 15:
        warnings.append(f"Long EMA period {ema_long} may be too short for trend confirmation. "
                        f"Consider using 15-50 range (21-26 are most common).")
    elif ema_long > 50:
        warnings.append(f"Long EMA period {ema_long} may be too slow and miss trend changes. "
                        f"Consider using 15-50 range (21-26 are most common).")

    # Ratio validation
    ratio = ema_long / ema_short
    if ratio < 1.5:
        warnings.append(f"EMA ratio ({ratio:.1f}) is too close - periods {ema_short}/{ema_long} may generate false signals. "
                        f"Consider using a ratio of 1.5-3x (e.g., 9/21, 12/26).")
    elif ratio > 3:
        warnings.append(f"EMA ratio ({ratio:.1f}) is very wide - periods {ema_short}/{ema_long} may be too slow. "
                        f"Consider using a ratio of 1.5-3x (e.g., 9/21, 12/26).")

    return warnings


def _validate_macd_parameters(fast_period, slow_period, signal_period):
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
    if fast_period < 8:
        warnings.append(f"MACD fast period {fast_period} is very short and may generate excessive noise. "
                        f"Consider using 8-15 range (12 is standard).")
    elif fast_period > 15:
        warnings.append(f"MACD fast period {fast_period} may be too slow for responsive signals. "
                        f"Consider using 8-15 range (12 is standard).")

    # Slow period validation
    if slow_period < 20:
        warnings.append(f"MACD slow period {slow_period} may be too short for trend confirmation. "
                        f"Consider using 20-30 range (26 is standard).")
    elif slow_period > 30:
        warnings.append(f"MACD slow period {slow_period} may be too slow and miss trend changes. "
                        f"Consider using 20-30 range (26 is standard).")

    # Signal period validation
    if signal_period < 7:
        warnings.append(f"MACD signal period {signal_period} is very short and may generate false signals. "
                        f"Consider using 7-12 range (9 is standard).")
    elif signal_period > 12:
        warnings.append(f"MACD signal period {signal_period} may be too slow for timely signals. "
                        f"Consider using 7-12 range (9 is standard).")

    # Standard combination check
    if (fast_period, slow_period, signal_period) == (12, 26, 9):
        warnings.append("Using standard MACD parameters (12/26/9) - widely tested and reliable.")

    return warnings


def _validate_bollinger_parameters(period, num_std):
    """
    Enhanced validation for Bollinger Bands parameters with guidance on reasonable ranges.

    Reasonable ranges based on common trading practices:
    - Period: 15-25 (20 is standard)
    - Standard deviations: 1.5-2.5 (2.0 is standard)
    - Standard BB: 20/2.0 captures ~95% of price action
    """
    warnings = []

    # Period validation
    if period < 15:
        warnings.append(f"Bollinger Bands period {period} is quite short and may be too sensitive. "
                        f"Consider using 15-25 range (20 is standard).")
    elif period > 25:
        warnings.append(f"Bollinger Bands period {period} is quite long and may be too slow. "
                        f"Consider using 15-25 range (20 is standard).")

    # Standard deviation validation
    if num_std < 1.5:
        warnings.append(f"Bollinger Bands standard deviation {num_std} is quite narrow and may generate excessive signals. "
                        f"Consider using 1.5-2.5 range (2.0 is standard).")
    elif num_std > 2.5:
        warnings.append(f"Bollinger Bands standard deviation {num_std} is quite wide and may miss opportunities. "
                        f"Consider using 1.5-2.5 range (2.0 is standard).")

    # Standard combination check
    if (period, num_std) == (20, 2.0):
        warnings.append("Using standard Bollinger Bands parameters (20/2.0) - captures ~95% of price action.")

    return warnings


def _validate_ichimoku_parameters(tenkan_period, kijun_period, senkou_span_b_period, displacement):
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
    if tenkan_period < 7:
        warnings.append(f"Ichimoku Tenkan period {tenkan_period} is quite short and may be too sensitive. "
                        f"Consider using 7-12 range (9 is traditional).")
    elif tenkan_period > 12:
        warnings.append(f"Ichimoku Tenkan period {tenkan_period} may be too slow for conversion line. "
                        f"Consider using 7-12 range (9 is traditional).")

    # Kijun period validation
    if kijun_period < 22:
        warnings.append(f"Ichimoku Kijun period {kijun_period} may be too short for baseline. "
                        f"Consider using 22-30 range (26 is traditional).")
    elif kijun_period > 30:
        warnings.append(f"Ichimoku Kijun period {kijun_period} may be too slow for trend confirmation. "
                        f"Consider using 22-30 range (26 is traditional).")

    # Senkou Span B period validation
    if senkou_span_b_period < 44:
        warnings.append(f"Ichimoku Senkou Span B period {senkou_span_b_period} may be too short for cloud formation. "
                        f"Consider using 44-60 range (52 is traditional).")
    elif senkou_span_b_period > 60:
        warnings.append(f"Ichimoku Senkou Span B period {senkou_span_b_period} may be too slow. "
                        f"Consider using 44-60 range (52 is traditional).")

    # Displacement validation
    if displacement < 22:
        warnings.append(f"Ichimoku displacement {displacement} may be too short for proper cloud projection. "
                        f"Consider using 22-30 range (26 is traditional).")
    elif displacement > 30:
        warnings.append(f"Ichimoku displacement {displacement} may project too far into future. "
                        f"Consider using 22-30 range (26 is traditional).")

    # Traditional ratios validation
    tenkan_kijun_ratio = kijun_period / tenkan_period
    if tenkan_kijun_ratio < 2.5 or tenkan_kijun_ratio > 3.5:
        warnings.append(f"Ichimoku Tenkan/Kijun ratio ({tenkan_kijun_ratio:.1f}) deviates from traditional ~3:1 ratio. "
                        f"Consider maintaining traditional proportions (e.g., 9/26).")

    kijun_senkou_ratio = senkou_span_b_period / kijun_period
    if kijun_senkou_ratio < 1.8 or kijun_senkou_ratio > 2.2:
        warnings.append(f"Ichimoku Kijun/Senkou B ratio ({kijun_senkou_ratio:.1f}) deviates from traditional 2:1 ratio. "
                        f"Consider maintaining traditional proportions (e.g., 26/52).")

    # Displacement vs Kijun check
    if displacement != kijun_period:
        warnings.append(f"Ichimoku displacement ({displacement}) differs from Kijun period ({kijun_period}). "
                        f"Traditional Ichimoku uses same value for both (typically 26).")

    # Traditional combination check
    if (tenkan_period, kijun_period, senkou_span_b_period, displacement) == (9, 26, 52, 26):
        warnings.append("Using traditional Ichimoku parameters (9/26/52/26) - based on Japanese market cycles.")

    return warnings


def _validate_common_parameters(rollover, trailing, slippage):
    """
    Enhanced validation for common strategy parameters with guidance.

    Reasonable ranges:
    - Rollover: Boolean (True for continuous contracts, False for single contract)
    - Trailing: None or 1-5% (2-3% is common for futures)
    - Slippage: 0-0.5% (0.1-0.2% is typical for liquid futures)
    """
    warnings = []

    # Trailing stop validation
    if trailing is not None:
        if trailing < 1:
            warnings.append(f"Trailing stop {trailing}% is very tight and may be stopped out frequently. "
                            f"Consider using 1-5% range (2-3% is common for futures).")
        elif trailing > 5:
            warnings.append(f"Trailing stop {trailing}% is very wide and may give back large profits. "
                            f"Consider using 1-5% range (2-3% is common for futures).")

    # Slippage validation
    if slippage is not None:
        if slippage > 0.5:
            warnings.append(f"Slippage {slippage}% is very high and may significantly impact returns. "
                            f"Consider using 0-0.5% range (0.1-0.2% is typical for liquid futures).")
        elif slippage == 0:
            warnings.append("Slippage is set to 0% - this may be unrealistic for backtesting. "
                            "Consider using 0.1-0.2% for more realistic results.")

    return warnings


def create_strategy(strategy_type, **params):
    """ Create a strategy instance based on a strategy type and parameters. """
    # Validate strategy type
    if strategy_type.lower() not in STRATEGY_TYPES:
        logger.error(f"Unknown strategy type: {strategy_type}")
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    # Create a strategy based on type
    if strategy_type.lower() == 'rsi':
        return _create_rsi_strategy(**params)
    elif strategy_type.lower() == 'ema':
        return _create_ema_strategy(**params)
    elif strategy_type.lower() == 'macd':
        return _create_macd_strategy(**params)
    elif strategy_type.lower() == 'bollinger':
        return _create_bollinger_strategy(**params)
    elif strategy_type.lower() == 'ichimoku':
        return _create_ichimoku_strategy(**params)
    return None


def _create_rsi_strategy(**params):
    """  Create an RSI strategy instance. """
    # Extract parameters with defaults
    rsi_period = params.get('rsi_period', 14)
    lower = params.get('lower', 30)
    upper = params.get('upper', 70)
    common_params = _extract_common_params(**params)

    # Validate parameters
    _validate_positive_integer(rsi_period, "rsi period")
    _validate_range(lower, "lower threshold", 0, 100)
    _validate_range(upper, "upper threshold", 0, 100)

    if lower >= upper:
        logger.error(f"Lower threshold ({lower}) must be less than upper threshold ({upper})")
        raise ValueError(f"Lower threshold must be less than upper threshold")

    # Enhanced parameter validation with guidance
    rsi_warnings = _validate_rsi_parameters(rsi_period, lower, upper)
    common_warnings = _validate_common_parameters(common_params['rollover'],
                                                  common_params['trailing'],
                                                  common_params['slippage'])

    # Log all warnings
    all_warnings = rsi_warnings + common_warnings
    for warning in all_warnings:
        logger.warning(f"RSI Strategy Parameter Guidance: {warning}")

    # Create and return strategy
    return RSIStrategy(
        rsi_period=rsi_period,
        lower=lower,
        upper=upper,
        **common_params
    )


def _create_ema_strategy(**params):
    """ Create an EMA Crossover strategy instance. """
    # Extract parameters with defaults
    ema_short = params.get('ema_short', 9)
    ema_long = params.get('ema_long', 21)
    common_params = _extract_common_params(**params)

    # Validate parameters
    _validate_positive_integer(ema_short, "short EMA period")
    _validate_positive_integer(ema_long, "long EMA period")

    if ema_short >= ema_long:
        logger.error(f"Short EMA period ({ema_short}) must be less than long EMA period ({ema_long})")
        raise ValueError(f"Short EMA period must be less than long EMA period")

    # Enhanced parameter validation with guidance
    ema_warnings = _validate_ema_parameters(ema_short, ema_long)
    common_warnings = _validate_common_parameters(common_params['rollover'],
                                                  common_params['trailing'],
                                                  common_params['slippage'])

    # Log all warnings
    all_warnings = ema_warnings + common_warnings
    for warning in all_warnings:
        logger.warning(f"EMA Strategy Parameter Guidance: {warning}")

    # Create and return strategy
    return EMACrossoverStrategy(
        ema_short=ema_short,
        ema_long=ema_long,
        **common_params
    )


def _create_macd_strategy(**params):
    """ Create a MACD strategy instance. """
    # Extract parameters with defaults
    fast_period = params.get('fast_period', 12)
    slow_period = params.get('slow_period', 26)
    signal_period = params.get('signal_period', 9)
    common_params = _extract_common_params(**params)

    # Validate parameters
    _validate_positive_integer(fast_period, "fast period")
    _validate_positive_integer(slow_period, "slow period")
    _validate_positive_integer(signal_period, "signal period")

    if fast_period >= slow_period:
        logger.error(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")
        raise ValueError(f"Fast period must be less than slow period")

    # Enhanced parameter validation with guidance
    macd_warnings = _validate_macd_parameters(fast_period, slow_period, signal_period)
    common_warnings = _validate_common_parameters(common_params['rollover'],
                                                  common_params['trailing'],
                                                  common_params['slippage'])

    # Log all warnings
    all_warnings = macd_warnings + common_warnings
    for warning in all_warnings:
        logger.warning(f"MACD Strategy Parameter Guidance: {warning}")

    # Create and return strategy
    return MACDStrategy(
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
        **common_params
    )


def _create_bollinger_strategy(**params):
    """  Create a Bollinger Bands strategy instance. """
    # Extract parameters with defaults
    period = params.get('period', 20)
    num_std = params.get('num_std', 2)
    common_params = _extract_common_params(**params)

    # Validate parameters
    _validate_positive_integer(period, "period")
    _validate_positive_number(num_std, "number of standard deviations")

    # Enhanced parameter validation with guidance
    bollinger_warnings = _validate_bollinger_parameters(period, num_std)
    common_warnings = _validate_common_parameters(common_params['rollover'],
                                                  common_params['trailing'],
                                                  common_params['slippage'])

    # Log all warnings
    all_warnings = bollinger_warnings + common_warnings
    for warning in all_warnings:
        logger.warning(f"Bollinger Bands Strategy Parameter Guidance: {warning}")

    # Create and return strategy
    return BollingerBandsStrategy(
        period=period,
        num_std=num_std,
        **common_params
    )


def _create_ichimoku_strategy(**params):
    """ Create an Ichimoku Cloud strategy instance. """
    # Extract parameters with defaults
    tenkan_period = params.get('tenkan_period', 9)
    kijun_period = params.get('kijun_period', 26)
    senkou_span_b_period = params.get('senkou_span_b_period', 52)
    displacement = params.get('displacement', 26)
    common_params = _extract_common_params(**params)

    # Validate parameters
    _validate_positive_integer(tenkan_period, "tenkan period")
    _validate_positive_integer(kijun_period, "kijun period")
    _validate_positive_integer(senkou_span_b_period, "senkou span B period")
    _validate_positive_integer(displacement, "displacement")

    # Enhanced parameter validation with guidance
    ichimoku_warnings = _validate_ichimoku_parameters(tenkan_period, kijun_period, senkou_span_b_period, displacement)
    common_warnings = _validate_common_parameters(common_params['rollover'],
                                                  common_params['trailing'],
                                                  common_params['slippage'])

    # Log all warnings
    all_warnings = ichimoku_warnings + common_warnings
    for warning in all_warnings:
        logger.warning(f"Ichimoku Strategy Parameter Guidance: {warning}")

    # Create and return strategy
    return IchimokuCloudStrategy(
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        senkou_span_b_period=senkou_span_b_period,
        displacement=displacement,
        **common_params
    )


def _format_common_params(**params):
    """Format common parameters for the strategy name."""
    common_params = _extract_common_params(**params)
    return f"rollover={common_params['rollover']},trailing={common_params['trailing']},slippage={common_params['slippage']}"


def get_strategy_name(strategy_type, **params):
    """ Get a standardized name for a strategy with the given parameters. """
    common_params_str = _format_common_params(**params)

    if strategy_type.lower() == 'rsi':
        rsi_period = params.get('rsi_period', 14)
        lower = params.get('lower', 30)
        upper = params.get('upper', 70)
        return f'RSI(period={rsi_period},lower={lower},upper={upper},{common_params_str})'

    elif strategy_type.lower() == 'ema':
        ema_short = params.get('ema_short', 9)
        ema_long = params.get('ema_long', 21)
        return f'EMA(short={ema_short},long={ema_long},{common_params_str})'

    elif strategy_type.lower() == 'macd':
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        return f'MACD(fast={fast_period},slow={slow_period},signal={signal_period},{common_params_str})'

    elif strategy_type.lower() == 'bollinger':
        period = params.get('period', 20)
        num_std = params.get('num_std', 2)
        return f'BB(period={period},std={num_std},{common_params_str})'

    elif strategy_type.lower() == 'ichimoku':
        tenkan_period = params.get('tenkan_period', 9)
        kijun_period = params.get('kijun_period', 26)
        senkou_span_b_period = params.get('senkou_span_b_period', 52)
        displacement = params.get('displacement', 26)
        return f'Ichimoku(tenkan={tenkan_period},kijun={kijun_period},senkou_b={senkou_span_b_period},displacement={displacement},{common_params_str})'

    else:
        return f'Unknown({strategy_type})'
