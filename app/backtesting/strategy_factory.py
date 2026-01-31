from app.backtesting.strategies.base.registry import (
    get_strategy_class,
    get_validator_class,
    list_strategies
)
from app.backtesting.validators import CommonValidator
from app.utils.logger import get_logger

# Import all strategies to trigger registration

logger = get_logger('backtesting/strategy_factory')

# ==================== Module Configuration ====================

# Set to track already logged warnings to prevent duplicates
_logged_warnings = set()

# Configuration variable to control whether warnings should be logged
_log_warnings_enabled = True

# ==================== Parameter Extraction & Validation ====================

# Strategy name mappings for logging
STRATEGY_DISPLAY_NAMES = {
    'bollinger': 'Bollinger Bands',
    'ema': 'EMA',
    'ichimoku': 'Ichimoku',
    'macd': 'MACD',
    'rsi': 'RSI'
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
    return {
        'rollover': params.get('rollover', False),
        'trailing': params.get('trailing', None),
        'slippage': params.get('slippage', None)
    }


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
    'bollinger': {
        'num_std': 'number_of_standard_deviations'
    },
    'ema': {
        'ema_short': 'short_ema_period',
        'ema_long': 'long_ema_period'
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

    # Extract common parameters
    common_params = _extract_common_params(**params)

    # Remove common params from strategy params to avoid duplication
    strategy_params = {param_name: param_value for param_name, param_value in params.items() if
                       param_name not in ['rollover', 'trailing', 'slippage']}

    # Get defaults from strategy class for validation purposes
    strategy_defaults = _get_strategy_defaults(strategy_class)
    params_with_defaults = {**strategy_defaults, **params}

    # Get and run validators (which will raise ValueError for critical issues and return warnings)
    validator_class = get_validator_class(strategy_type)

    # Always validate common parameters
    common_warnings = _validate_common_params(common_params)
    all_warnings = common_warnings

    # Validate strategy-specific parameters
    if validator_class:
        validator = validator_class()
        # Map parameter names for validator
        mapped_params = _map_params_for_validator(strategy_type, params_with_defaults)
        strategy_warnings = validator.validate(**mapped_params)
        all_warnings = strategy_warnings + common_warnings

    # Log all warnings
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
