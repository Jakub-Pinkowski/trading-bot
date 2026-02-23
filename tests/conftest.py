import os

# Mark the process as running under pytest so child processes (ProcessPoolExecutor
# workers) also skip file log handlers - sys.modules check alone is insufficient
# because spawned subprocesses start fresh without pytest in sys.modules
os.environ['PYTEST_RUNNING'] = '1'

# ==================== Backtesting Fixtures Plugin Registration ====================
# Import all backtesting fixtures for availability across all test modules
pytest_plugins = [
    'tests.backtesting.fixtures.data_fixtures',
    'tests.backtesting.fixtures.strategy_fixtures',
]
