from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema_crossover import EMACrossoverStrategy
from app.backtesting.strategies.ichimoku_cloud import IchimokuCloudStrategy
from app.backtesting.strategies.macd import MACDStrategy
from app.backtesting.strategies.rsi import RSIStrategy
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
    """Extract common parameters used by all strategies."""
    return {
        'rollover': params.get('rollover', False),
        'trailing': params.get('trailing', None),
        'slippage': params.get('slippage', None)
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
    # Extract parameters with defaults
    period = params.get('period', 20)
    number_of_standard_deviations = params.get('num_std', 2)
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
        num_std=number_of_standard_deviations,
        **common_params
    )


def _create_ema_strategy(**params):
    """Create an EMA Crossover strategy instance."""
    # Extract parameters with defaults
    short_ema_period = params.get('ema_short', 9)
    long_ema_period = params.get('ema_long', 21)
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
        ema_short=short_ema_period,
        ema_long=long_ema_period,
        **common_params
    )


def _create_ichimoku_strategy(**params):
    """Create an Ichimoku Cloud strategy instance."""
    # Extract parameters with defaults
    tenkan_period = params.get('tenkan_period', 9)
    kijun_period = params.get('kijun_period', 26)
    senkou_span_b_period = params.get('senkou_span_b_period', 52)
    displacement = params.get('displacement', 26)
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
    # Extract parameters with defaults
    fast_period = params.get('fast_period', 12)
    slow_period = params.get('slow_period', 26)
    signal_period = params.get('signal_period', 9)
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
    # Extract parameters with defaults
    rsi_period = params.get('rsi_period', 14)
    lower = params.get('lower', 30)
    upper = params.get('upper', 70)
    common_params = _extract_common_params(**params)

    # Validate all parameters using singleton validators
    rsi_warnings = _RSI_VALIDATOR.validate(
        rsi_period=rsi_period,
        lower=lower,
        upper=upper
    )
    common_warnings = _COMMON_VALIDATOR.validate(**common_params)

    # Log all warnings (only once per unique warning)
    _log_warnings_once(rsi_warnings + common_warnings, "RSI")

    # Create and return strategy
    return RSIStrategy(
        rsi_period=rsi_period,
        lower=lower,
        upper=upper,
        **common_params
    )


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
