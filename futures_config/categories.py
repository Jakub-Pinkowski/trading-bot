"""
Auto-generated Category Lists.

All category lists are dynamically generated from SYMBOL_SPECS.

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

# Auto-generated category lists
GRAINS = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Grains'])
SOFTS = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Softs'])
ENERGY = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Energy'])
METALS = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Metals'])
CRYPTO = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Crypto'])
INDEX = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Index'])
FOREX = sorted([k for k, v in SYMBOL_SPECS.items() if v['category'] == 'Forex'])

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
