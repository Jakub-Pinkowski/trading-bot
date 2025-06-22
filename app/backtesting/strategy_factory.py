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
