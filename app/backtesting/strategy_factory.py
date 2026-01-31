from app.backtesting.strategies.base.registry import (
    get_strategy_class,
    get_validator_class,
    list_strategies
)
from app.backtesting.validators import CommonValidator
from app.utils.logger import get_logger

# Import all strategies to trigger registration
from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy
from app.backtesting.strategies.ichimoku_cloud import IchimokuCloudStrategy
from app.backtesting.strategies.macd import MACDStrategy
from app.backtesting.strategies.rsi import RSIStrategy

logger = get_logger('backtesting/strategy_factory')

# ==================== Module Configuration ====================

# Set to track already logged warnings to prevent duplicates
_logged_warnings = set()

# Configuration variable to control whether warnings should be logged
_log_warnings_enabled = True


# ==================== Parameter Extraction & Validation ====================

# Strategy name mappings for logging
STRATEGY_DISPLAY_NAMES = {
    'rsi': 'RSI',
    'ema': 'EMA',
    'macd': 'MACD',
    'bollinger': 'Bollinger Bands',
    'ichimoku': 'Ichimoku'
}

def _log_warnings_once(warnings, strategy_type):
    """Log warnings only if they haven't been logged before and warnings are enabled."""
    if not _log_warnings_enabled:
        return

    display_name = STRATEGY_DISPLAY_NAMES.get(strategy_type, strategy_type)
    for warning in warnings:
        warning_key = f"{strategy_type}: {warning}"
        if warning_key not in _logged_warnings:
            logger.warning(f"{display_name} Strategy Parameter Guidance: {warning}")
            _logged_warnings.add(warning_key)


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


def _validate_strategy_params(strategy_type, params):
    """
    Validate strategy-specific parameters.
    
    Arguments:
        strategy_type: Strategy identifier
        params: Parameters to validate (merged with defaults)
    
    Raises:
        ValueError: If parameters are invalid
    """
    if strategy_type == 'rsi':
        rsi_period = params.get('rsi_period', 14)
        lower = params.get('lower', 30)
        upper = params.get('upper', 70)
        
        if not isinstance(rsi_period, int) or rsi_period <= 0:
            logger.error(f"Invalid rsi_period: {rsi_period}")
            raise ValueError("rsi period must be a positive integer")
        
        if not isinstance(lower, (int, float)) or lower < 0 or lower > 100:
            logger.error(f"Invalid lower: {lower}")
            raise ValueError("lower threshold must be between 0 and 100")
        
        if not isinstance(upper, (int, float)) or upper < 0 or upper > 100:
            logger.error(f"Invalid upper: {upper}")
            raise ValueError("upper threshold must be between 0 and 100")
        
        if lower >= upper:
            logger.error(f"Lower threshold ({lower}) must be less than upper threshold ({upper})")
            raise ValueError("Lower threshold must be less than upper threshold")
    
    elif strategy_type == 'ema':
        ema_short = params.get('ema_short', 9)
        ema_long = params.get('ema_long', 21)
        
        if not isinstance(ema_short, int) or ema_short <= 0:
            logger.error(f"Invalid ema_short: {ema_short}")
            raise ValueError("short EMA period must be a positive integer")
        
        if not isinstance(ema_long, int) or ema_long <= 0:
            logger.error(f"Invalid ema_long: {ema_long}")
            raise ValueError("long EMA period must be a positive integer")
        
        if ema_short >= ema_long:
            logger.error(f"Short EMA period ({ema_short}) must be less than long EMA period ({ema_long})")
            raise ValueError("Short EMA period must be less than long EMA period")
    
    elif strategy_type == 'macd':
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        
        if not isinstance(fast_period, int) or fast_period <= 0:
            logger.error(f"Invalid fast_period: {fast_period}")
            raise ValueError("fast period must be a positive integer")
        
        if not isinstance(slow_period, int) or slow_period <= 0:
            logger.error(f"Invalid slow_period: {slow_period}")
            raise ValueError("slow period must be a positive integer")
        
        if not isinstance(signal_period, int) or signal_period <= 0:
            logger.error(f"Invalid signal_period: {signal_period}")
            raise ValueError("signal period must be a positive integer")
        
        if fast_period >= slow_period:
            logger.error(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")
            raise ValueError("Fast period must be less than slow period")
    
    elif strategy_type == 'bollinger':
        period = params.get('period', 20)
        num_std = params.get('num_std', 2)
        
        if not isinstance(period, int) or period <= 0:
            logger.error(f"Invalid period: {period}")
            raise ValueError("period must be a positive integer")
        
        if not isinstance(num_std, (int, float)) or num_std <= 0:
            logger.error(f"Invalid num_std: {num_std}")
            raise ValueError("number of standard deviations must be positive")
    
    elif strategy_type == 'ichimoku':
        tenkan_period = params.get('tenkan_period', 9)
        kijun_period = params.get('kijun_period', 26)
        senkou_span_b_period = params.get('senkou_span_b_period', 52)
        displacement = params.get('displacement', 26)
        
        if not isinstance(tenkan_period, int) or tenkan_period <= 0:
            logger.error(f"Invalid tenkan_period: {tenkan_period}")
            raise ValueError("tenkan period must be a positive integer")
        
        if not isinstance(kijun_period, int) or kijun_period <= 0:
            logger.error(f"Invalid kijun_period: {kijun_period}")
            raise ValueError("kijun period must be a positive integer")
        
        if not isinstance(senkou_span_b_period, int) or senkou_span_b_period <= 0:
            logger.error(f"Invalid senkou_span_b_period: {senkou_span_b_period}")
            raise ValueError("senkou span B period must be a positive integer")
        
        if not isinstance(displacement, int) or displacement <= 0:
            logger.error(f"Invalid displacement: {displacement}")
            raise ValueError("displacement must be a positive integer")


def _validate_common_params(common_params):
    """
    Validate common parameters and return warnings.
    
    Arguments:
        common_params: Dictionary with 'rollover', 'trailing', and 'slippage' keys
        
    Returns:
        List of warning messages from CommonValidator
    """
    common_validator = CommonValidator()
    return common_validator.validate(
        rollover=common_params['rollover'],
        trailing=common_params['trailing'],
        slippage=common_params['slippage']
    )


# ==================== Strategy Creation ====================

# Parameter name mappings for validators
VALIDATOR_PARAM_MAPPING = {
    'ema': {
        'ema_short': 'short_ema_period',
        'ema_long': 'long_ema_period'
    },
    'bollinger': {
        'num_std': 'number_of_standard_deviations'
    }
}

def _get_strategy_defaults(strategy_class):
    """
    Extract default parameter values from a strategy class's __init__ method.
    
    Arguments:
        strategy_class: The strategy class to inspect
    
    Returns:
        Dictionary of parameter names to their default values
    """
    import inspect
    sig = inspect.signature(strategy_class.__init__)
    defaults = {}
    for param_name, param in sig.parameters.items():
        if param_name in ['self', 'rollover', 'trailing', 'slippage', 'symbol']:
            continue
        if param.default != inspect.Parameter.empty:
            defaults[param_name] = param.default
    return defaults

def _map_params_for_validator(strategy_type, params):
    """Map strategy parameter names to validator parameter names."""
    if strategy_type not in VALIDATOR_PARAM_MAPPING:
        return params
    
    mapping = VALIDATOR_PARAM_MAPPING[strategy_type]
    mapped_params = params.copy()
    
    for strategy_param, validator_param in mapping.items():
        if strategy_param in mapped_params:
            mapped_params[validator_param] = mapped_params[strategy_param]
    
    return mapped_params

def create_strategy(strategy_type, **params):
    """
    Create a strategy instance based on type and parameters.
    
    Uses registry pattern - no if/elif chains needed!
    
    Arguments:
        strategy_type: Strategy identifier (e.g., 'rsi', 'ema')
        **params: Strategy-specific and common parameters
    
    Returns:
        Strategy instance configured with the given parameters
    
    Raises:
        ValueError: If strategy type is unknown or parameters are invalid
    """
    strategy_type = strategy_type.lower()
    
    # Get strategy class from registry
    strategy_class = get_strategy_class(strategy_type)
    if not strategy_class:
        available = ', '.join(list_strategies())
        logger.error(f"Unknown strategy: {strategy_type}. Available: {available}")
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # Extract and validate common parameters
    common_params = _extract_common_params(**params)
    
    # Remove common params from strategy params to avoid duplication
    strategy_params = {k: v for k, v in params.items() if k not in ['rollover', 'trailing', 'slippage']}
    
    # Get defaults from strategy class for validation purposes
    strategy_defaults = _get_strategy_defaults(strategy_class)
    params_with_defaults = {**strategy_defaults, **params}
    
    # Validate strategy-specific parameters
    _validate_strategy_params(strategy_type, params_with_defaults)
    
    # Get and run validator for warnings
    validator_class = get_validator_class(strategy_type)
    # Always validate common parameters, regardless of strategy-specific validator
    common_warnings = _validate_common_params(common_params)
    all_warnings = common_warnings
    if validator_class:
        validator = validator_class()
        # Map parameter names for validator
        mapped_params = _map_params_for_validator(strategy_type, params_with_defaults)
        strategy_warnings = validator.validate(**mapped_params)
        all_warnings = strategy_warnings + common_warnings
    if all_warnings:
        _log_warnings_once(all_warnings, strategy_type)
    
    # Create strategy instance
    return strategy_class(**strategy_params, **common_params)


# ==================== Utility Functions ====================

def _format_common_params(**params):
    """Format common parameters for the strategy name."""
    common_params = _extract_common_params(**params)
    return f"rollover={common_params['rollover']},trailing={common_params['trailing']},slippage={common_params['slippage']}"


def get_strategy_name(strategy_type, **params):
    """ Get a standardized name for a strategy with the given parameters. """
    common_params_str = _format_common_params(**params)

    if strategy_type.lower() == 'bollinger':
        period = params.get('period', 20)
        number_of_standard_deviations = params.get('num_std', 2)
        return f'BB(period={period},std={number_of_standard_deviations},{common_params_str})'

    elif strategy_type.lower() == 'ema':
        short_ema_period = params.get('ema_short', 9)
        long_ema_period = params.get('ema_long', 21)
        return f'EMA(short={short_ema_period},long={long_ema_period},{common_params_str})'

    elif strategy_type.lower() == 'ichimoku':
        tenkan_period = params.get('tenkan_period', 9)
        kijun_period = params.get('kijun_period', 26)
        senkou_span_b_period = params.get('senkou_span_b_period', 52)
        displacement = params.get('displacement', 26)
        return f'Ichimoku(tenkan={tenkan_period},kijun={kijun_period},senkou_b={senkou_span_b_period},displacement={displacement},{common_params_str})'

    elif strategy_type.lower() == 'macd':
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        return f'MACD(fast={fast_period},slow={slow_period},signal={signal_period},{common_params_str})'

    elif strategy_type.lower() == 'rsi':
        rsi_period = params.get('rsi_period', 14)
        lower = params.get('lower', 30)
        upper = params.get('upper', 70)
        return f'RSI(period={rsi_period},lower={lower},upper={upper},{common_params_str})'

    else:
        return f'Unknown({strategy_type})'
