from app.backtesting.strategies import (
    BollingerBandsStrategy,
    EMACrossoverStrategy,
    IchimokuCloudStrategy,
    MACDStrategy,
    RSIStrategy,
)
from app.backtesting.validators import (
    BollingerValidator,
    CommonValidator,
    EMAValidator,
    IchimokuValidator,
    MACDValidator,
    RSIValidator
)
from app.utils.logger import get_logger

logger = get_logger('backtesting/strategy_factory')

# ==================== Module Configuration ====================

# Define strategy types
STRATEGY_TYPES = ['bollinger', 'ema', 'ichimoku', 'macd', 'rsi']

# Set to track already logged warnings to prevent duplicates
_logged_warnings = set()

# Configuration variable to control whether warnings should be logged
_log_warnings_enabled = True

# Create singleton validator instances
_BOLLINGER_VALIDATOR = BollingerValidator()
_COMMON_VALIDATOR = CommonValidator()
_EMA_VALIDATOR = EMAValidator()
_ICHIMOKU_VALIDATOR = IchimokuValidator()
_MACD_VALIDATOR = MACDValidator()
_RSI_VALIDATOR = RSIValidator()


# ==================== Utility Functions ====================

def _log_warnings_once(warnings, strategy_type):
    """Log warnings only if they haven't been logged before and warnings are enabled."""
    if not _log_warnings_enabled:
        return

    for warning in warnings:
        warning_key = f"{strategy_type}: {warning}"
        if warning_key not in _logged_warnings:
            logger.warning(f"{strategy_type} Strategy Parameter Guidance: {warning}")
            _logged_warnings.add(warning_key)


def _extract_common_params(**params):
    """Extract common parameters used by all strategies. All parameters are required."""
    return {
        'rollover': params['rollover'],
        'trailing': params['trailing'],
        'slippage': params['slippage'],
        'symbol': params['symbol']
    }


# ==================== Strategy Creation ====================

def create_strategy(strategy_type, **params):
    """ Create a strategy instance based on a strategy type and parameters. """
    # Validate strategy type
    if strategy_type.lower() not in STRATEGY_TYPES:
        logger.error(f"Unknown strategy type: {strategy_type}")
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    # Create a strategy based on type
    if strategy_type.lower() == 'bollinger':
        return _create_bollinger_strategy(**params)
    elif strategy_type.lower() == 'ema':
        return _create_ema_strategy(**params)
    elif strategy_type.lower() == 'ichimoku':
        return _create_ichimoku_strategy(**params)
    elif strategy_type.lower() == 'macd':
        return _create_macd_strategy(**params)
    elif strategy_type.lower() == 'rsi':
        return _create_rsi_strategy(**params)
    return None


def _create_bollinger_strategy(**params):
    """Create a Bollinger Bands strategy instance."""
    # Extract parameters
    period = params['period']
    number_of_standard_deviations = params['number_of_standard_deviations']
    common_params = _extract_common_params(**params)

    # Validate all parameters using singleton validators
    bollinger_warnings = _BOLLINGER_VALIDATOR.validate(
        period=period,
        number_of_standard_deviations=number_of_standard_deviations
    )
    common_warnings = _COMMON_VALIDATOR.validate(**common_params)

    # Log all warnings (only once per unique warning)
    _log_warnings_once(bollinger_warnings + common_warnings, "Bollinger Bands")

    # Create and return strategy
    return BollingerBandsStrategy(
        period=period,
        number_of_standard_deviations=number_of_standard_deviations,
        **common_params
    )


def _create_ema_strategy(**params):
    """Create an EMA Crossover strategy instance."""
    # Extract parameters
    short_ema_period = params['short_ema_period']
    long_ema_period = params['long_ema_period']
    common_params = _extract_common_params(**params)

    # Validate all parameters using singleton validators
    ema_warnings = _EMA_VALIDATOR.validate(
        short_ema_period=short_ema_period,
        long_ema_period=long_ema_period
    )
    common_warnings = _COMMON_VALIDATOR.validate(**common_params)

    # Log all warnings (only once per unique warning)
    _log_warnings_once(ema_warnings + common_warnings, "EMA")

    # Create and return strategy
    return EMACrossoverStrategy(
        short_ema_period=short_ema_period,
        long_ema_period=long_ema_period,
        **common_params
    )


def _create_ichimoku_strategy(**params):
    """Create an Ichimoku Cloud strategy instance."""
    # Extract parameters (all required)
    tenkan_period = params['tenkan_period']
    kijun_period = params['kijun_period']
    senkou_span_b_period = params['senkou_span_b_period']
    displacement = params['displacement']
    common_params = _extract_common_params(**params)

    # Validate all parameters using singleton validators
    ichimoku_warnings = _ICHIMOKU_VALIDATOR.validate(
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        senkou_span_b_period=senkou_span_b_period,
        displacement=displacement
    )
    common_warnings = _COMMON_VALIDATOR.validate(**common_params)

    # Log all warnings (only once per unique warning)
    _log_warnings_once(ichimoku_warnings + common_warnings, "Ichimoku")

    # Create and return strategy
    return IchimokuCloudStrategy(
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        senkou_span_b_period=senkou_span_b_period,
        displacement=displacement,
        **common_params
    )


def _create_macd_strategy(**params):
    """Create a MACD strategy instance."""
    # Extract parameters
    fast_period = params['fast_period']
    slow_period = params['slow_period']
    signal_period = params['signal_period']
    common_params = _extract_common_params(**params)

    # Validate all parameters using singleton validators
    macd_warnings = _MACD_VALIDATOR.validate(
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period
    )
    common_warnings = _COMMON_VALIDATOR.validate(**common_params)

    # Log all warnings (only once per unique warning)
    _log_warnings_once(macd_warnings + common_warnings, "MACD")

    # Create and return strategy
    return MACDStrategy(
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
        **common_params
    )


def _create_rsi_strategy(**params):
    """Create an RSI strategy instance."""
    # Extract parameters
    rsi_period = params['rsi_period']
    lower_threshold = params['lower_threshold']
    upper_threshold = params['upper_threshold']
    common_params = _extract_common_params(**params)

    # Validate all parameters using singleton validators
    rsi_warnings = _RSI_VALIDATOR.validate(
        rsi_period=rsi_period,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold
    )
    common_warnings = _COMMON_VALIDATOR.validate(**common_params)

    # Log all warnings (only once per unique warning)
    _log_warnings_once(rsi_warnings + common_warnings, "RSI")

    # Create and return strategy
    return RSIStrategy(
        rsi_period=rsi_period,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        **common_params
    )


# ==================== Utility Functions ====================

def _format_common_params(**params):
    """Format common parameters for the strategy name."""
    common_params = _extract_common_params(**params)
    return f"rollover={common_params['rollover']},trailing={common_params['trailing']},slippage={common_params['slippage']}"


def get_strategy_name(strategy_type, **params):
    """ Get a standardized name for a strategy with the given parameters. All parameters are required. """
    common_params_str = _format_common_params(**params)

    if strategy_type.lower() == 'bollinger':
        period = params['period']
        number_of_standard_deviations = params['number_of_standard_deviations']
        return f'BB(period={period},std={number_of_standard_deviations},{common_params_str})'

    elif strategy_type.lower() == 'ema':
        short_ema_period = params['short_ema_period']
        long_ema_period = params['long_ema_period']
        return f'EMA(short={short_ema_period},long={long_ema_period},{common_params_str})'

    elif strategy_type.lower() == 'ichimoku':
        tenkan_period = params['tenkan_period']
        kijun_period = params['kijun_period']
        senkou_span_b_period = params['senkou_span_b_period']
        displacement = params['displacement']
        return f'Ichimoku(tenkan={tenkan_period},kijun={kijun_period},senkou_b={senkou_span_b_period},displacement={displacement},{common_params_str})'

    elif strategy_type.lower() == 'macd':
        fast_period = params['fast_period']
        slow_period = params['slow_period']
        signal_period = params['signal_period']
        return f'MACD(fast={fast_period},slow={slow_period},signal={signal_period},{common_params_str})'

    elif strategy_type.lower() == 'rsi':
        rsi_period = params['rsi_period']
        lower_threshold = params['lower_threshold']
        upper_threshold = params['upper_threshold']
        return f'RSI(period={rsi_period},lower={lower_threshold},upper={upper_threshold},{common_params_str})'

    else:
        return f'Unknown({strategy_type})'
