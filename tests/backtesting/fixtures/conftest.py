"""
Conftest for fixtures directory.

This file makes fixtures from data_fixtures.py, strategy_fixtures.py, and
mock_fixtures.py available to pytest. It's separate from the main backtesting
conftest.py to keep new fixture architecture isolated from legacy fixtures.
"""

# Import all fixtures to make them discoverable by pytest
from tests.backtesting.fixtures.data_fixtures import *  # noqa: F401, F403
from tests.backtesting.fixtures.mock_fixtures import *  # noqa: F401, F403


# TODO: Uncomment when this is implemented
# from tests.backtesting.fixtures.strategy_fixtures import *  # noqa: F401, F403


# ==================== pytest Configuration ====================

def pytest_configure(config):
    """
    Register custom markers for fixture-based tests.
    """
    config.addinivalue_line(
        "markers", "real_data: tests that require real historical data files"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take significant time to execute"
    )
