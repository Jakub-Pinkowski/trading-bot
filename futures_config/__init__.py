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
)
# Import symbol groups
from futures_config.symbol_groups import (
    SYMBOL_GROUPS,
    SYMBOL_TO_GROUP,
    get_group_for_symbol,
    get_symbols_in_group,
    are_symbols_correlated,
    get_representative_symbol,
    filter_to_one_per_group,
)
# Import symbol mapping
from futures_config.symbol_mapping import (
    TV_TO_IBKR_MAPPING,
    IBKR_TO_TV_MAPPING,
    map_tv_to_ibkr,
    map_ibkr_to_tv,
)
# Import symbol specifications
from futures_config.symbol_specs import SYMBOL_SPECS

__all__ = [
    # Symbol specifications
    'SYMBOL_SPECS',

    # Symbol mapping
    'TV_TO_IBKR_MAPPING',
    'IBKR_TO_TV_MAPPING',
    'map_tv_to_ibkr',
    'map_ibkr_to_tv',

    # Symbol groups
    'SYMBOL_GROUPS',
    'SYMBOL_TO_GROUP',
    'get_group_for_symbol',
    'get_symbols_in_group',
    'are_symbols_correlated',
    'get_representative_symbol',
    'filter_to_one_per_group',

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
]
