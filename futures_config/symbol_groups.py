"""
Symbol Groups for Correlated Instruments.

Groups symbols that track the same underlying market (e.g., standard, mini, micro).
Used to avoid pseudo-replication in strategy analysis where correlated symbols
would artificially inflate symbol counts and skew aggregated results.

Example:
    ZC (standard corn), XC (mini corn), and MZC (micro corn) all track the same
    corn futures market with nearly identical candle patterns. Including all three
    in the analysis would give 3x weight to corn strategies versus other markets.

Usage:
    - For strategy analysis: Use only one symbol per group to avoid correlation bias
    - For production trading: Use appropriate contract size based on account size
"""

# ==================== Symbol Groups ====================

SYMBOL_GROUPS = {
    # --- Grains ---
    'Corn': ['ZC', 'XC', 'MZC'],  # Standard, Mini, Micro Corn
    'Wheat': ['ZW', 'XW', 'MZW'],  # Standard, Mini, Micro Wheat
    'Soybeans': ['ZS', 'XK', 'MZS'],  # Standard, Mini (XK), Micro Soybeans
    'Soybean_Oil': ['ZL', 'MZL'],  # Standard, Micro Soybean Oil

    # --- Softs ---
    'Sugar': ['SB'],
    'Coffee': ['KC'],
    'Cocoa': ['CC'],

    # --- Energy ---
    'Crude_Oil': ['CL', 'MCL'],  # Standard, Micro Crude Oil
    'Natural_Gas': ['NG', 'MNG'],  # Standard, Micro Natural Gas

    # --- Metals ---
    'Gold': ['GC', 'MGC'],  # Standard, Micro Gold
    'Silver': ['SI', 'SIL'],  # Standard, Micro Silver (SIL maps to QI in IBKR)
    'Copper': ['HG', 'MHG'],  # Standard, Micro Copper
    'Platinum': ['PL'],

    # --- Index ---
    'SP500': ['ES', 'MES'],  # E-mini, Micro E-mini S&P 500
    'NASDAQ': ['NQ', 'MNQ'],  # E-mini, Micro E-mini NASDAQ-100
    'Dow': ['YM', 'MYM'],  # E-mini, Micro E-mini Dow
    'Russell': ['RTY', 'M2K'],  # E-mini, Micro E-mini Russell 2000
    'T_Bond': ['ZB'],

    # --- Crypto ---
    'Bitcoin': ['BTC', 'MBT'],  # Standard, Micro Bitcoin
    'Ethereum': ['ETH', 'MET'],  # Standard, Micro Ethereum

    # --- Forex ---
    'Euro': ['6E', 'M6E'],  # Standard, Micro Euro FX
    'Japanese_Yen': ['6J'],
    'British_Pound': ['6B', 'M6B'],  # Standard, Micro British Pound
    'Australian_Dollar': ['6A', 'M6A'],  # Standard, Micro Australian Dollar
    'Canadian_Dollar': ['6C'],
    'Swiss_Franc': ['6S'],
}

# Reverse mapping: symbol -> group name
SYMBOL_TO_GROUP = {}
for group_name, symbols in SYMBOL_GROUPS.items():
    for symbol in symbols:
        SYMBOL_TO_GROUP[symbol] = group_name


# ==================== Helper Functions ====================

def get_group_for_symbol(symbol):
    """
    Get the group name for a given symbol.

    Args:
        symbol: Trading symbol (e.g., 'ZC', 'XC', 'MZC')

    Returns:
        Group name (e.g., 'Corn') or None if symbol is not in any group

    Example:
        get_group_for_symbol('ZC')
        'Corn'

        get_group_for_symbol('MZC')
        'Corn'
    """
    return SYMBOL_TO_GROUP.get(symbol)


def get_symbols_in_group(group_name):
    """
    Get all symbols in a specific group.

    Args:
        group_name: Name of the symbol group (e.g., 'Corn', 'Gold')

    Returns:
        List of symbols in the group, or empty list if group doesn't exist

    Example:
        get_symbols_in_group('Corn')
        ['ZC', 'XC', 'MZC']
    """
    return SYMBOL_GROUPS.get(group_name, [])


def are_symbols_correlated(symbol1, symbol2):
    """
    Check if two symbols are in the same group (correlated).

    Args:
        symbol1: First trading symbol
        symbol2: Second trading symbol

    Returns:
        True if symbols are in the same group, False otherwise

    Example:
        are_symbols_correlated('ZC', 'MZC')
        True

        are_symbols_correlated('ZC', 'CL')
        False
    """
    group1 = get_group_for_symbol(symbol1)
    group2 = get_group_for_symbol(symbol2)

    # Both must be in groups and in the same group
    return group1 is not None and group1 == group2


def get_representative_symbol(group_name):
    """
    Get the first (typically standard size) symbol from a group.

    Useful for selecting one representative symbol per market when avoiding correlation.

    Args:
        group_name: Name of the symbol group

    Returns:
        First symbol in the group, or None if a group doesn't exist

    Example:
        get_representative_symbol('Corn')
        'ZC'
    """
    symbols = get_symbols_in_group(group_name)
    return symbols[0] if symbols else None


def filter_to_one_per_group(symbols):
    """
    Filter the symbol list to include only one symbol per correlated group.

    For each group, always keeps the representative (standard size) symbol and excludes
    mini/micro variants. Symbols not in any group are kept as-is.

    Args:
        symbols: List of symbol strings

    Returns:
        Filtered list with only one representative symbol per group

    Example:
        filter_to_one_per_group(['ZC', 'CL', 'MZC', 'ES', 'MES', 'GC'])
        ['ZC', 'CL', 'ES', 'GC'] # Keeps ZC (not MZC), ES (not MES)

        filter_to_one_per_group(['MZC', 'XC', 'ZC', 'CL'])
        ['ZC', 'CL'] # Always keeps ZC (standard) even if MZC/XC appear first
    """
    filtered_symbols = []
    ungrouped_symbols = []

    # First pass: collect ungrouped symbols and identify groups present
    groups_to_add = set()
    for symbol in symbols:
        group = get_group_for_symbol(symbol)
        if group is None:
            ungrouped_symbols.append(symbol)
        else:
            groups_to_add.add(group)

    # Second pass: add a representative symbol for each group
    for group in sorted(groups_to_add):
        representative = get_representative_symbol(group)
        if representative:
            filtered_symbols.append(representative)

    # Add ungrouped symbols
    filtered_symbols.extend(ungrouped_symbols)

    return filtered_symbols
