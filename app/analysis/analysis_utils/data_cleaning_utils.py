import json
from fractions import Fraction

import pandas as pd


def parse_description(description):
    try:
        if pd.notna(description):  # Ensure description is not null
            return json.loads(description)  # Convert JSON to a Python dict
        return {}
    except json.JSONDecodeError:
        return {}


def fractional_to_decimal(price_str):
    if isinstance(price_str, str) and ' ' in price_str:
        whole, frac = price_str.split(' ', 1)
        return float(whole) + float(Fraction(frac))
    elif isinstance(price_str, str) and '/' in price_str:
        return float(Fraction(price_str))
    else:
        return float(price_str)
