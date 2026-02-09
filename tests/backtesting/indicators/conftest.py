"""
Conftest for indicator tests.
Imports shared fixtures from the fixtures directory for PyCharm test runner compatibility.
"""
# Import fixtures for PyCharm test runner
from tests.backtesting.fixtures.data_fixtures import *  # noqa: F401, F403
from tests.backtesting.fixtures.indicator_test_data import *  # noqa: F401, F403
