"""
Strategy factory fixture for backtesting tests.
Provides a flexible factory fixture for creating strategy instances with custom parameters.
"""
import pytest

from app.backtesting.strategies import (
    BollingerBandsStrategy,
    EMACrossoverStrategy,
    MACDStrategy,
    IchimokuCloudStrategy,
    RSIStrategy
)


# ==================== Strategy Factory Fixture ====================
@pytest.fixture
def strategy_factory():
    """
    Factory fixture to create any strategy with custom parameters.
    Provides a flexible way to create strategy instances in tests without
    pre-defining every possible configuration. Supports all strategy types
    with sensible defaults and custom parameter overrides.
    Returns:
        Function that creates strategy instances
    Example:
        def test_custom_rsi(strategy_factory):
            strategy = strategy_factory('RSI', period=10, lower=25, upper=75)
            assert strategy.rsi_period == 10
        def test_custom_ema(strategy_factory):
            strategy = strategy_factory('EMA', short=5, long=13, symbol='CL')
            assert strategy.symbol == 'CL'
    Supported strategy types:
        - 'RSI': RSI oscillator strategy
        - 'EMA': EMA crossover strategy
        - 'MACD': MACD strategy
        - 'Bollinger': Bollinger Bands strategy
        - 'Ichimoku': Ichimoku Cloud strategy
    Common parameters (all strategies):
        - rollover: bool (default: False)
        - trailing: float or None (default: None)
        - slippage_ticks: int (default: 1)
        - symbol: str (default: 'ZS')
    Strategy-specific parameters:
        RSI:
            - period: int (default: 14)
            - lower: int (default: 30)
            - upper: int (default: 70)
        EMA:
            - short: int (default: 9)
            - long: int (default: 21)
        MACD:
            - fast: int (default: 12)
            - slow: int (default: 26)
            - signal: int (default: 9)
        Bollinger:
            - period: int (default: 20)
            - std_dev: float (default: 2.0)
        Ichimoku:
            - tenkan: int (default: 9)
            - kijun: int (default: 26)
            - senkou_b: int (default: 52)
            - displacement: int (default: 26)
    """

    def _create_strategy(strategy_type, **kwargs):
        # Set common defaults
        defaults = {
            'rollover': False,
            'trailing': None,
            'slippage_ticks': 1,
            'symbol': 'ZS'
        }
        # Merge with provided kwargs
        params = {**defaults, **kwargs}
        if strategy_type == 'RSI':
            return RSIStrategy(
                rsi_period=params.get('period', 14),
                lower_threshold=params.get('lower', 30),
                upper_threshold=params.get('upper', 70),
                rollover=params['rollover'],
                trailing=params['trailing'],
                slippage_ticks=params['slippage_ticks'],
                symbol=params['symbol']
            )
        elif strategy_type == 'EMA':
            return EMACrossoverStrategy(
                short_ema_period=params.get('short', 9),
                long_ema_period=params.get('long', 21),
                rollover=params['rollover'],
                trailing=params['trailing'],
                slippage_ticks=params['slippage_ticks'],
                symbol=params['symbol']
            )
        elif strategy_type == 'MACD':
            return MACDStrategy(
                fast_period=params.get('fast', 12),
                slow_period=params.get('slow', 26),
                signal_period=params.get('signal', 9),
                rollover=params['rollover'],
                trailing=params['trailing'],
                slippage_ticks=params['slippage_ticks'],
                symbol=params['symbol']
            )
        elif strategy_type == 'Bollinger':
            return BollingerBandsStrategy(
                period=params.get('period', 20),
                number_of_standard_deviations=params.get('std_dev', 2.0),
                rollover=params['rollover'],
                trailing=params['trailing'],
                slippage_ticks=params['slippage_ticks'],
                symbol=params['symbol']
            )
        elif strategy_type == 'Ichimoku':
            return IchimokuCloudStrategy(
                tenkan_period=params.get('tenkan', 9),
                kijun_period=params.get('kijun', 26),
                senkou_span_b_period=params.get('senkou_b', 52),
                displacement=params.get('displacement', 26),
                rollover=params['rollover'],
                trailing=params['trailing'],
                slippage_ticks=params['slippage_ticks'],
                symbol=params['symbol']
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    return _create_strategy
