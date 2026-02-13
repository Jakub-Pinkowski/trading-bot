"""
Auto-generated Category Lists.

All category lists are dynamically generated from SYMBOL_SPECS.
Only TradingView-compatible symbols are included in these lists.

Categories:
- GRAINS: Corn, Wheat, Soybeans, etc.
- SOFTS: Sugar, Coffee, Cocoa
- ENERGY: Crude Oil, Natural Gas
- METALS: Gold, Silver, Copper, Platinum
- CRYPTO: Bitcoin, Ethereum
- INDEX: Stock indices (S&P 500, NASDAQ, Dow, etc.)
- FOREX: Currency pairs
"""

from futures_config.symbol_specs import SYMBOL_SPECS

# Auto-generated category lists - only TradingView-compatible symbols
GRAINS = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Grains' and v['tv_compatible']])
SOFTS = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Softs' and v['tv_compatible']])
ENERGY = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Energy' and v['tv_compatible']])
METALS = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Metals' and v['tv_compatible']])
CRYPTO = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Crypto' and v['tv_compatible']])
INDEX = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Index' and v['tv_compatible']])
FOREX = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Forex' and v['tv_compatible']])

# Dictionary mapping category names to their symbol lists for dynamic access
CATEGORIES = {
    'Grains': GRAINS,
    'Softs': SOFTS,
    'Energy': ENERGY,
    'Metals': METALS,
    'Crypto': CRYPTO,
    'Index': INDEX,
    'Forex': FOREX,
}
