import re


def parse_symbol(symbol):
    match = re.match(r'^([A-Za-z]+)', symbol)
    if not match:
        raise ValueError(f"Invalid symbol format: {symbol}")
    return match.group(1)
