"""
Conftest for strategy tests.

Provides standard strategy fixtures with default parameters for common testing scenarios.
Imports shared fixtures from fixtures directory for PyCharm test runner compatibility.
"""

# Import fixtures for PyCharm test runner
from tests.backtesting.fixtures.data_fixtures import *  # noqa: F401, F403
from tests.backtesting.fixtures.strategy_fixtures import *  # noqa: F401, F403


# ==================== Standard Strategy Fixtures ====================
@pytest.fixture
def standard_rsi_strategy():
    """Standard RSI strategy with default parameters."""
    return RSIStrategy(
        rsi_period=14,
        lower_threshold=30,
        upper_threshold=70,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def standard_ema_strategy():
    """Standard EMA crossover strategy with default parameters."""
    return EMACrossoverStrategy(
        short_ema_period=9,
        long_ema_period=21,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def standard_macd_strategy():
    """Standard MACD strategy with default parameters."""
    return MACDStrategy(
        fast_period=12,
        slow_period=26,
        signal_period=9,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def standard_bollinger_strategy():
    """Standard Bollinger Bands strategy with default parameters."""
    return BollingerBandsStrategy(
        period=20,
        number_of_standard_deviations=2.0,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def standard_ichimoku_strategy():
    """Standard Ichimoku Cloud strategy with default parameters."""
    return IchimokuCloudStrategy(
        tenkan_period=9,
        kijun_period=26,
        senkou_span_b_period=52,
        displacement=26,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )
