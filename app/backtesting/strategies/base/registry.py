"""
Strategy Registry - Automatic strategy registration and discovery.

Strategies self-register using @register_strategy decorator.
Factory uses registry for O(1) lookup instead of if/elif chains.
"""

from typing import Dict, Type, Optional

from app.utils.logger import get_logger

logger = get_logger('backtesting/strategies/registry')

# ==================== Global Registry ====================

STRATEGY_REGISTRY: Dict[str, Type] = {}
VALIDATOR_REGISTRY: Dict[str, Type] = {}


# ==================== Registration Decorators ====================

def register_strategy(name: str, validator_class: Optional[Type] = None):
    """
    Decorator to register a strategy class.
    
    Usage:
        @register_strategy('rsi', RSIValidator)
        class RSIStrategy(BaseStrategy):
            ...
    
    Arguments:
        name: Strategy identifier (e.g., 'rsi', 'ema')
        validator_class: Optional validator class for this strategy
    """

    def decorator(cls):
        strategy_name = name.lower()
        if strategy_name in STRATEGY_REGISTRY:
            logger.warning(f"Strategy '{strategy_name}' already registered, overwriting")

        STRATEGY_REGISTRY[strategy_name] = cls

        if validator_class:
            VALIDATOR_REGISTRY[strategy_name] = validator_class

        logger.debug(f"Registered strategy: {strategy_name} -> {cls.__name__}")
        return cls

    return decorator


# ==================== Registry Access Functions ====================

def get_strategy_class(name: str) -> Optional[Type]:
    """Get strategy class by name from registry."""
    return STRATEGY_REGISTRY.get(name.lower())


def get_validator_class(name: str) -> Optional[Type]:
    """Get validator class by name from registry."""
    return VALIDATOR_REGISTRY.get(name.lower())


def list_strategies() -> list:
    """Get list of all registered strategy names."""
    return sorted(STRATEGY_REGISTRY.keys())


def is_strategy_registered(name: str) -> bool:
    """Check if a strategy is registered."""
    return name.lower() in STRATEGY_REGISTRY
