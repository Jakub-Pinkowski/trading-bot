"""
Conftest for indicators tests.

Imports all shared fixtures from the fixtures directory to make them
available to indicator tests.
"""

# Register fixture modules as pytest plugins
pytest_plugins = (
    "tests.backtesting.fixtures.data_fixtures",
    "tests.backtesting.fixtures.indicator_test_data",
    "tests.backtesting.fixtures.mock_fixtures",
)
