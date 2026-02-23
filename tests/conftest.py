# ==================== Backtesting Fixtures Plugin Registration ====================
# Import all backtesting fixtures for availability across all test modules
pytest_plugins = [
    'tests.backtesting.fixtures.data_fixtures',
    'tests.backtesting.fixtures.strategy_fixtures',
]
