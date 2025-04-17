import re

from app.utils.logger import get_logger

logger = get_logger()


def parse_symbol(symbol):
    match = re.match(r'^([A-Za-z]+)', symbol)
    if not match:
        logger.error(f"Invalid symbol format: {symbol}")
        raise ValueError(f"Invalid symbol format: {symbol}")
    return match.group(1)
