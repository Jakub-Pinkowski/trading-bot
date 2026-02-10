"""
Conftest for fixtures directory.

This file makes fixtures from data_fixtures.py and strategy_fixtures.py
available to pytest.
"""

# ==================== Fixture Modules Registration ====================

# Register fixture modules as pytest plugins
from tests.backtesting.fixtures.data_fixtures import *  # noqa: F401, F403
from tests.backtesting.fixtures.strategy_fixtures import *  # noqa: F401, F403
