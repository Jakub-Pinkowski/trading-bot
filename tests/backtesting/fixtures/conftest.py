"""
Conftest for fixtures directory.

This file makes fixtures from data_fixtures.py, strategy_fixtures.py,
mock_fixtures.py, and indicator_test_data.py available to pytest.
"""

# ==================== Fixture Modules Registration ====================

# Register fixture modules as pytest plugins instead of using wildcard imports
from tests.backtesting.fixtures.data_fixtures import *  # noqa: F401, F403
from tests.backtesting.fixtures.indicator_test_data import *  # noqa: F401, F403
from tests.backtesting.fixtures.mock_fixtures import *  # noqa: F401, F403
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
