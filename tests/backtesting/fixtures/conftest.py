"""
Conftest for fixtures directory.

This file makes fixtures from data_fixtures.py and strategy_fixtures.py
available to pytest.
"""

# ==================== Fixture Modules Registration ====================

# Register fixture modules as pytest plugins
from tests.backtesting.fixtures.data_fixtures import *  # noqa: F401, F403
from tests.backtesting.fixtures.strategy_fixtures import *  # noqa: F401, F403


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
