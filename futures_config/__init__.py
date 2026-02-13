"""
Futures Configuration Package.

This package contains all futures contract specifications, symbol mappings,
and helper functions for working with futures symbols across TradingView
and IBKR platforms.
"""

# Import categories
from futures_config.categories import (
    GRAINS,
    SOFTS,
    ENERGY,
    METALS,
    CRYPTO,
    INDEX,
    FOREX,
    CATEGORIES,
)

# Import helper functions
from futures_config.helpers import (
    get_exchange_for_symbol,
    get_category_for_symbol,
    get_tick_size,
    get_contract_multiplier,
    get_margin_requirement,
    is_tradingview_compatible,
)

# Import symbol mapping
from futures_config.symbol_mapping import (
    TV_TO_IBKR_MAPPING,
    IBKR_TO_TV_MAPPING,
    map_tv_to_ibkr,
    map_ibkr_to_tv,
)

# Import symbol specifications
from futures_config.symbol_specs import SYMBOL_SPECS, DEFAULT_TICK_SIZE

__all__ = [
    # Symbol specifications
    'SYMBOL_SPECS',
    'DEFAULT_TICK_SIZE',

    # Symbol mapping
    'TV_TO_IBKR_MAPPING',
    'IBKR_TO_TV_MAPPING',
    'map_tv_to_ibkr',
    'map_ibkr_to_tv',

    # Categories
    'GRAINS',
    'SOFTS',
    'ENERGY',
    'METALS',
    'CRYPTO',
    'INDEX',
    'FOREX',
    'CATEGORIES',

    # Helper functions
    'get_exchange_for_symbol',
    'get_category_for_symbol',
    'get_tick_size',
    'get_contract_multiplier',
    'get_margin_requirement',
    'is_tradingview_compatible',
]
