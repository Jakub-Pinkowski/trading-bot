"""
Pre-configured strategy fixtures for backtesting tests.

Provides fixtures for strategy instances with standard, conservative,
and aggressive parameter configurations.
"""
import pytest

from app.backtesting.strategies import (
    BollingerBandsStrategy,
    EMACrossoverStrategy,
    MACDStrategy,
    IchimokuCloudStrategy,
    RSIStrategy

)


__all__ = [
    'bollinger_strategy_default',
    'bollinger_strategy_conservative',
    'bollinger_strategy_aggressive',
    'bollinger_strategy_with_rollover',
    'bollinger_strategy_with_trailing_stop',
    'ema_strategy_default',
    'ema_strategy_conservative',
    'ema_strategy_aggressive',
    'ema_strategy_with_rollover',
    'ema_strategy_with_trailing_stop',
    'ichimoku_strategy_default',
    'ichimoku_strategy_conservative',
    'ichimoku_strategy_aggressive',
    'ichimoku_strategy_with_rollover',
    'ichimoku_strategy_with_trailing_stop',
    'macd_strategy_default',
    'macd_strategy_conservative',
    'macd_strategy_aggressive',
    'macd_strategy_with_rollover',
    'macd_strategy_with_trailing_stop',
    'rsi_strategy_default',
    'rsi_strategy_conservative',
    'rsi_strategy_aggressive',
    'rsi_strategy_with_rollover',
    'rsi_strategy_with_trailing_stop',
    'rsi_strategy_fast',
    'rsi_strategy_slow',
    'all_strategies_default',
    'all_strategies_conservative',
    'all_strategies_aggressive',
]


# ==================== Bollinger Bands Strategy Fixtures ====================

@pytest.fixture
def bollinger_strategy_default():
    """
    Bollinger Bands strategy with default parameters.

    Standard 20-period with 2 standard deviations.

    Returns:
        BollingerBandsStrategy instance with standard configuration
    """
    return BollingerBandsStrategy(
        period=20,
        number_of_standard_deviations=2.0,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def bollinger_strategy_conservative():
    """
    Bollinger Bands strategy with conservative parameters.

    Wider bands (2.5 std dev) for fewer but stronger signals.

    Returns:
        BollingerBandsStrategy instance with conservative configuration
    """
    return BollingerBandsStrategy(
        period=20,
        number_of_standard_deviations=2.5,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def bollinger_strategy_aggressive():
    """
    Bollinger Bands strategy with aggressive parameters.

    Tighter bands (1.5 std dev) for more frequent signals.

    Returns:
        BollingerBandsStrategy instance with aggressive configuration
    """
    return BollingerBandsStrategy(
        period=20,
        number_of_standard_deviations=1.5,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def bollinger_strategy_with_rollover():
    """
    Bollinger Bands strategy configured for contract rollover.

    Returns:
        BollingerBandsStrategy instance with rollover enabled
    """
    return BollingerBandsStrategy(
        period=20,
        number_of_standard_deviations=2.0,
        rollover=True,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def bollinger_strategy_with_trailing_stop():
    """
    Bollinger Bands strategy with trailing stop.

    Returns:
        BollingerBandsStrategy instance with trailing stop at 2 ATR
    """
    return BollingerBandsStrategy(
        period=20,
        number_of_standard_deviations=2.0,
        rollover=False,
        trailing=2.0,
        slippage_ticks=1,
        symbol='ZS'
    )


# ==================== EMA Crossover Strategy Fixtures ====================

@pytest.fixture
def ema_strategy_default():
    """
    EMA crossover strategy with default parameters.

    Classic 9/21 crossover for balanced signals.

    Returns:
        EMACrossoverStrategy instance with standard configuration
    """
    return EMACrossoverStrategy(
        short_ema_period=9,
        long_ema_period=21,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def ema_strategy_conservative():
    """
    EMA crossover strategy with conservative parameters.

    Slower crossover (21/50) for stronger trend confirmation.

    Returns:
        EMACrossoverStrategy instance with conservative configuration
    """
    return EMACrossoverStrategy(
        short_ema_period=21,
        long_ema_period=50,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def ema_strategy_aggressive():
    """
    EMA crossover strategy with aggressive parameters.

    Faster crossover (5/13) for quick entries.

    Returns:
        EMACrossoverStrategy instance with aggressive configuration
    """
    return EMACrossoverStrategy(
        short_ema_period=5,
        long_ema_period=13,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def ema_strategy_with_rollover():
    """
    EMA crossover strategy configured for contract rollover.

    Returns:
        EMACrossoverStrategy instance with rollover enabled
    """
    return EMACrossoverStrategy(
        short_ema_period=9,
        long_ema_period=21,
        rollover=True,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def ema_strategy_with_trailing_stop():
    """
    EMA crossover strategy with trailing stop.

    Returns:
        EMACrossoverStrategy instance with trailing stop at 2 ATR
    """
    return EMACrossoverStrategy(
        short_ema_period=9,
        long_ema_period=21,
        rollover=False,
        trailing=2.0,
        slippage_ticks=1,
        symbol='ZS'
    )


# ==================== Ichimoku Cloud Strategy Fixtures ====================

@pytest.fixture
def ichimoku_strategy_default():
    """
    Ichimoku Cloud strategy with default parameters.

    Standard 9/26/52/26 configuration.

    Returns:
        IchimokuCloudStrategy instance with standard configuration
    """
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


@pytest.fixture
def ichimoku_strategy_conservative():
    """
    Ichimoku Cloud strategy with conservative parameters.

    Slower periods (12/30/60/30) for stronger signals.

    Returns:
        IchimokuCloudStrategy instance with conservative configuration
    """
    return IchimokuCloudStrategy(
        tenkan_period=12,
        kijun_period=30,
        senkou_span_b_period=60,
        displacement=30,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def ichimoku_strategy_aggressive():
    """
    Ichimoku Cloud strategy with aggressive parameters.

    Faster periods (7/22/44/22) for quicker signals.

    Returns:
        IchimokuCloudStrategy instance with aggressive configuration
    """
    return IchimokuCloudStrategy(
        tenkan_period=7,
        kijun_period=22,
        senkou_span_b_period=44,
        displacement=22,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def ichimoku_strategy_with_rollover():
    """
    Ichimoku Cloud strategy configured for contract rollover.

    Returns:
        IchimokuCloudStrategy instance with rollover enabled
    """
    return IchimokuCloudStrategy(
        tenkan_period=9,
        kijun_period=26,
        senkou_span_b_period=52,
        displacement=26,
        rollover=True,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def ichimoku_strategy_with_trailing_stop():
    """
    Ichimoku Cloud strategy with trailing stop.

    Returns:
        IchimokuCloudStrategy instance with trailing stop at 2 ATR
    """
    return IchimokuCloudStrategy(
        tenkan_period=9,
        kijun_period=26,
        senkou_span_b_period=52,
        displacement=26,
        rollover=False,
        trailing=2.0,
        slippage_ticks=1,
        symbol='ZS'
    )


# ==================== MACD Strategy Fixtures ====================

@pytest.fixture
def macd_strategy_default():
    """
    MACD strategy with default parameters.

    Standard 12/26/9 configuration.

    Returns:
        MACDStrategy instance with standard configuration
    """
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
def macd_strategy_conservative():
    """
    MACD strategy with conservative parameters.

    Slower periods (19/39/9) for stronger signals.

    Returns:
        MACDStrategy instance with conservative configuration
    """
    return MACDStrategy(
        fast_period=19,
        slow_period=39,
        signal_period=9,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def macd_strategy_aggressive():
    """
    MACD strategy with aggressive parameters.

    Faster periods (8/17/9) for quicker signals.

    Returns:
        MACDStrategy instance with aggressive configuration
    """
    return MACDStrategy(
        fast_period=8,
        slow_period=17,
        signal_period=9,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def macd_strategy_with_rollover():
    """
    MACD strategy configured for contract rollover.

    Returns:
        MACDStrategy instance with rollover enabled
    """
    return MACDStrategy(
        fast_period=12,
        slow_period=26,
        signal_period=9,
        rollover=True,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def macd_strategy_with_trailing_stop():
    """
    MACD strategy with trailing stop.

    Returns:
        MACDStrategy instance with trailing stop at 2 ATR
    """
    return MACDStrategy(
        fast_period=12,
        slow_period=26,
        signal_period=9,
        rollover=False,
        trailing=2.0,
        slippage_ticks=1,
        symbol='ZS'
    )


# ==================== RSI Strategy Fixtures ====================

@pytest.fixture
def rsi_strategy_default():
    """
    RSI strategy with default/standard parameters.

    Conservative threshold levels (30/70) with standard period.
    Good for most testing scenarios.

    Returns:
        RSIStrategy instance with standard configuration
    """
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
def rsi_strategy_conservative():
    """
    RSI strategy with conservative parameters.

    Wider thresholds (20/80) to reduce false signals.
    Fewer trades but higher quality entries.

    Returns:
        RSIStrategy instance with conservative configuration
    """
    return RSIStrategy(
        rsi_period=14,
        lower_threshold=20,
        upper_threshold=80,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def rsi_strategy_aggressive():
    """
    RSI strategy with aggressive parameters.

    Tighter thresholds (40/60) for more frequent trading.
    More signals but potentially more whipsaws.

    Returns:
        RSIStrategy instance with aggressive configuration
    """
    return RSIStrategy(
        rsi_period=14,
        lower_threshold=40,
        upper_threshold=60,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def rsi_strategy_with_rollover():
    """
    RSI strategy configured for contract rollover testing.

    Returns:
        RSIStrategy instance with rollover enabled
    """
    return RSIStrategy(
        rsi_period=14,
        lower_threshold=30,
        upper_threshold=70,
        rollover=True,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def rsi_strategy_with_trailing_stop():
    """
    RSI strategy with trailing stop loss.

    Returns:
        RSIStrategy instance with trailing stop at 2 ATR
    """
    return RSIStrategy(
        rsi_period=14,
        lower_threshold=30,
        upper_threshold=70,
        rollover=False,
        trailing=2.0,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def rsi_strategy_fast():
    """
    RSI strategy with faster period for quicker signals.

    Returns:
        RSIStrategy instance with 7-period RSI
    """
    return RSIStrategy(
        rsi_period=7,
        lower_threshold=30,
        upper_threshold=70,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


@pytest.fixture
def rsi_strategy_slow():
    """
    RSI strategy with slower period for smoother signals.

    Returns:
        RSIStrategy instance with 21-period RSI
    """
    return RSIStrategy(
        rsi_period=21,
        lower_threshold=30,
        upper_threshold=70,
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


# ==================== Multi-Strategy Fixtures ====================

@pytest.fixture
def all_strategies_default():
    """
    Collection of all strategies with default parameters.

    Returns:
        Dict mapping strategy names to strategy instances

    Example:
        def test_all_strategies(all_strategies_default, zs_1h_data):
            for name, strategy in all_strategies_default.items():
                trades = strategy.backtest(zs_1h_data, contract_info)
                assert len(trades) >= 0
    """
    return {
        'RSI': RSIStrategy(14, 30, 70, False, None, 1, 'ZS'),
        'EMA': EMACrossoverStrategy(9, 21, False, None, 1, 'ZS'),
        'MACD': MACDStrategy(12, 26, 9, False, None, 1, 'ZS'),
        'Bollinger': BollingerBandsStrategy(20, 2.0, False, None, 1, 'ZS'),
        'Ichimoku': IchimokuCloudStrategy(9, 26, 52, 26, False, None, 1, 'ZS')
    }


@pytest.fixture
def all_strategies_conservative():
    """
    Collection of all strategies with conservative parameters.

    Returns:
        Dict mapping strategy names to conservative strategy instances
    """
    return {
        'RSI': RSIStrategy(14, 20, 80, False, None, 1, 'ZS'),
        'EMA': EMACrossoverStrategy(21, 50, False, None, 1, 'ZS'),
        'MACD': MACDStrategy(19, 39, 9, False, None, 1, 'ZS'),
        'Bollinger': BollingerBandsStrategy(20, 2.5, False, None, 1, 'ZS'),
        'Ichimoku': IchimokuCloudStrategy(12, 30, 60, 30, False, None, 1, 'ZS')
    }


@pytest.fixture
def all_strategies_aggressive():
    """
    Collection of all strategies with aggressive parameters.

    Returns:
        Dict mapping strategy names to aggressive strategy instances
    """
    return {
        'RSI': RSIStrategy(14, 40, 60, False, None, 1, 'ZS'),
        'EMA': EMACrossoverStrategy(5, 13, False, None, 1, 'ZS'),
        'MACD': MACDStrategy(8, 17, 9, False, None, 1, 'ZS'),
        'Bollinger': BollingerBandsStrategy(20, 1.5, False, None, 1, 'ZS'),
        'Ichimoku': IchimokuCloudStrategy(7, 22, 44, 22, False, None, 1, 'ZS')
    }


# ==================== Strategy Factory Fixtures ====================

@pytest.fixture
def strategy_factory():
    """
    Factory fixture to create any strategy with custom parameters.

    Returns:
        Function that creates strategy instances

    Example:
        def test_custom_strategy(strategy_factory):
            strategy = strategy_factory('RSI', period=10, lower=25, upper=75)
            assert strategy.rsi_period == 10
    """

    def _create_strategy(strategy_type, **kwargs):
        # Set defaults
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


@pytest.fixture
def strategy_for_symbol():
    """
    Factory fixture to create strategies for different symbols.

    Returns:
        Function that creates strategy for specific symbol

    Example:
        def test_cl_strategy(strategy_for_symbol):
            cl_strategy = strategy_for_symbol('RSI', 'CL')
            assert cl_strategy.symbol == 'CL'
    """

    def _create_for_symbol(strategy_type, symbol):
        if strategy_type == 'RSI':
            return RSIStrategy(14, 30, 70, False, None, 1, symbol)
        elif strategy_type == 'EMA':
            return EMACrossoverStrategy(9, 21, False, None, 1, symbol)
        elif strategy_type == 'MACD':
            return MACDStrategy(12, 26, 9, False, None, 1, symbol)
        elif strategy_type == 'Bollinger':
            return BollingerBandsStrategy(20, 2.0, False, None, 1, symbol)
        elif strategy_type == 'Ichimoku':
            return IchimokuCloudStrategy(9, 26, 52, 26, False, None, 1, symbol)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    return _create_for_symbol
