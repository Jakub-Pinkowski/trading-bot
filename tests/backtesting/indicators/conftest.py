"""
Conftest for indicators tests.

Imports all shared fixtures from the fixtures directory to make them
available to indicator tests.
"""

# ==================== Fixture Modules Registration ====================

# Register fixture modules as pytest plugins instead of using wildcard imports
pytest_plugins = [
    "tests.backtesting.fixtures.data_fixtures",
    "tests.backtesting.fixtures.indicator_test_data",
    "tests.backtesting.fixtures.mock_fixtures",
]
