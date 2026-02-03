"""
Strategy Factory

Simple dict-based factory for creating strategy instances.
Maps strategy types to (class, validator, param_names).
"""

from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema import EMACrossoverStrategy
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

# ==================== Configuration ====================

# Strategy mapping: type -> (class, validator, param_names)
STRATEGY_MAP = {
    'bollinger': (
        BollingerBandsStrategy,
        BollingerValidator(),
        ['period', 'number_of_standard_deviations']
    ),
    'ema': (
        EMACrossoverStrategy,
        EMAValidator(),
        ['short_ema_period', 'long_ema_period']
    ),
    'ichimoku': (
        IchimokuCloudStrategy,
        IchimokuValidator(),
        ['tenkan_period', 'kijun_period', 'senkou_span_b_period', 'displacement']
    ),
    'macd': (
        MACDStrategy,
        MACDValidator(),
        ['fast_period', 'slow_period', 'signal_period']
    ),
    'rsi': (
        RSIStrategy,
        RSIValidator(),
        ['rsi_period', 'lower_threshold', 'upper_threshold']
    ),
}

COMMON_PARAMS = ['rollover', 'trailing', 'slippage_ticks', 'symbol']
COMMON_VALIDATOR = CommonValidator()

_logged_warnings = set()
_log_warnings_enabled = True


# ==================== Helper Functions ====================

def _log_warnings_once(warnings, strategy_type):
    """Log warnings only once per unique warning."""
    if not _log_warnings_enabled:
        return

    for warning in warnings:
        warning_key = f"{strategy_type}: {warning}"
        if warning_key not in _logged_warnings:
            logger.warning(f"{strategy_type} Strategy Parameter Guidance: {warning}")
            _logged_warnings.add(warning_key)


def _extract_params(param_names, all_params):
    """Extract specific parameters from params dict."""
    extracted = {}
    missing = []

    for name in param_names:
        if name in all_params:
            extracted[name] = all_params[name]
        else:
            missing.append(name)

    return extracted, missing


# ==================== Public API ====================

def create_strategy(strategy_type, **params):
    """
    Create a strategy instance based on type and parameters.

    Args:
        strategy_type: Type of strategy to create (e.g., 'bollinger', 'ema')
        **params: Strategy parameters (strategy-specific + common params)

    Returns:
        Instantiated strategy object

    Raises:
        ValueError: If strategy_type is unknown
        KeyError: If required parameters are missing
    """
    strategy_key = strategy_type.lower()

    # Check if strategy exists
    if strategy_key not in STRATEGY_MAP:
        available = ', '.join(STRATEGY_MAP.keys())
        logger.error(f"Unknown strategy type: {strategy_type}. Available: {available}")
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    strategy_class, validator, param_names = STRATEGY_MAP[strategy_key]

    # Extract strategy-specific parameters
    strategy_params, missing_strategy = _extract_params(param_names, params)
    if missing_strategy:
        logger.error(f"Missing parameters for {strategy_type}: {missing_strategy}")
        raise KeyError(f"Missing required parameters for {strategy_type}: {missing_strategy}")

    # Extract common parameters
    common_params, missing_common = _extract_params(COMMON_PARAMS, params)
    if missing_common:
        logger.error(f"Missing common parameters: {missing_common}")
        raise KeyError(f"Missing required common parameters: {missing_common}")

    # Validate parameters
    strategy_warnings = validator.validate(**strategy_params)
    common_warnings = COMMON_VALIDATOR.validate(**common_params)

    # Log warnings
    all_warnings = strategy_warnings + common_warnings
    if all_warnings:
        _log_warnings_once(all_warnings, strategy_type)

    return strategy_class(**strategy_params, **common_params)


def get_strategy_name(strategy_type, **params):
    """
    Get standardized name for a strategy.

    Args:
        strategy_type: Type of strategy
        **params: Strategy parameters

    Returns:
        Formatted strategy name string
    """
    strategy_key = strategy_type.lower()

    if strategy_key not in STRATEGY_MAP:
        logger.error(f"Unknown strategy type: {strategy_type}")
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    strategy_class, _, _ = STRATEGY_MAP[strategy_key]

    return strategy_class.format_name(**params)


def get_available_strategies():
    """Get list of all available strategy types."""
    return sorted(STRATEGY_MAP.keys())


def get_strategy_params(strategy_type):
    """
    Get required parameters for a strategy type.

    Args:
        strategy_type: Type of strategy

    Returns:
        Dict with strategy_params and common_params lists

    Raises:
        ValueError: If strategy_type is unknown
    """
    strategy_key = strategy_type.lower()

    if strategy_key not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    _, _, param_names = STRATEGY_MAP[strategy_key]

    return {
        'strategy_params': param_names,
        'common_params': COMMON_PARAMS
    }
