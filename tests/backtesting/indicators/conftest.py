"""
Conftest for indicators tests.

Imports all shared fixtures from the fixtures directory to make them
available to indicator tests.
"""

# Import all fixtures to make them discoverable by pytest
from tests.backtesting.fixtures.data_fixtures import *  # noqa: F401, F403
from tests.backtesting.fixtures.indicator_test_data import *  # noqa: F401, F403
from tests.backtesting.fixtures.mock_fixtures import *  # noqa: F401, F403
