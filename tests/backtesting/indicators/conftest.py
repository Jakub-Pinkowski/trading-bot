"""
Conftest for indicator tests.

Imports shared data fixtures from fixtures directory and local indicator test data.
"""

# Import fixtures for PyCharm test runner
from tests.backtesting.fixtures.data_fixtures import *  # noqa: F401, F403
from tests.backtesting.indicators.indicator_test_data import *  # noqa: F401, F403
