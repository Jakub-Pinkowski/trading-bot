"""
Tests for strategy_factory module.

Tests cover:
- Strategy creation for all strategy types
- Parameter validation and extraction
- Strategy name generation
- Available strategies listing
- Error handling for invalid inputs
- Warning system behavior
"""
from unittest.mock import patch

import pytest

from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema import EMACrossoverStrategy
from app.backtesting.strategies.ichimoku_cloud import IchimokuCloudStrategy
from app.backtesting.strategies.macd import MACDStrategy
from app.backtesting.strategies.rsi import RSIStrategy
from app.backtesting.strategies.strategy_factory import (
    create_strategy,
    get_strategy_name,
    get_available_strategies,
    get_strategy_params,
    STRATEGY_MAP,
    COMMON_PARAMS
)


# ==================== Create Strategy Tests ====================

class TestCreateStrategyRSI:
    """Test RSI strategy creation."""

    def test_create_rsi_strategy_with_valid_params(self):
        """Test creating RSI strategy with all valid parameters."""
        strategy = create_strategy(
            'rsi',
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert isinstance(strategy, RSIStrategy)
        assert strategy.rsi_period == 14
        assert strategy.lower_threshold == 30
        assert strategy.upper_threshold == 70
        assert strategy.rollover is True
        assert strategy.trailing is None
        assert strategy.position_manager.slippage_ticks == 1

    def test_create_rsi_strategy_with_trailing_stop(self):
        """Test creating RSI strategy with trailing stop."""
        strategy = create_strategy(
            'rsi',
            rsi_period=21,
            lower_threshold=25,
            upper_threshold=75,
            rollover=False,
            trailing=2.5,
            slippage_ticks=2,
            symbol='CL'
        )

        assert strategy.rsi_period == 21
        assert strategy.trailing == 2.5
        assert strategy.rollover is False

    def test_create_rsi_strategy_case_insensitive(self):
        """Test that strategy type is case-insensitive."""
        strategy_lower = create_strategy(
            'rsi',
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_upper = create_strategy(
            'RSI',
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert type(strategy_lower) == type(strategy_upper)


class TestCreateStrategyEMA:
    """Test EMA strategy creation."""

    def test_create_ema_strategy_with_valid_params(self):
        """Test creating EMA strategy with all valid parameters."""
        strategy = create_strategy(
            'ema',
            short_ema_period=9,
            long_ema_period=21,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert isinstance(strategy, EMACrossoverStrategy)
        assert strategy.short_ema_period == 9
        assert strategy.long_ema_period == 21

    def test_create_ema_strategy_with_different_periods(self):
        """Test EMA strategy with different period combinations."""
        strategy = create_strategy(
            'ema',
            short_ema_period=12,
            long_ema_period=26,
            rollover=False,
            trailing=1.5,
            slippage_ticks=2,
            symbol='GC'
        )

        assert strategy.short_ema_period == 12
        assert strategy.long_ema_period == 26


class TestCreateStrategyMACD:
    """Test MACD strategy creation."""

    def test_create_macd_strategy_with_valid_params(self):
        """Test creating MACD strategy with all valid parameters."""
        strategy = create_strategy(
            'macd',
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert isinstance(strategy, MACDStrategy)
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.signal_period == 9

    def test_create_macd_strategy_with_custom_periods(self):
        """Test MACD strategy with custom period settings."""
        strategy = create_strategy(
            'macd',
            fast_period=8,
            slow_period=20,
            signal_period=6,
            rollover=False,
            trailing=2.0,
            slippage_ticks=1,
            symbol='ES'
        )

        assert strategy.fast_period == 8
        assert strategy.slow_period == 20
        assert strategy.signal_period == 6


class TestCreateStrategyBollinger:
    """Test Bollinger Bands strategy creation."""

    def test_create_bollinger_strategy_with_valid_params(self):
        """Test creating Bollinger Bands strategy with all valid parameters."""
        strategy = create_strategy(
            'bollinger',
            period=20,
            number_of_standard_deviations=2.0,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert isinstance(strategy, BollingerBandsStrategy)
        assert strategy.period == 20
        assert strategy.number_of_standard_deviations == 2.0

    def test_create_bollinger_strategy_with_different_std_dev(self):
        """Test Bollinger Bands strategy with different standard deviation."""
        strategy = create_strategy(
            'bollinger',
            period=15,
            number_of_standard_deviations=2.5,
            rollover=False,
            trailing=3.0,
            slippage_ticks=2,
            symbol='CL'
        )

        assert strategy.period == 15
        assert strategy.number_of_standard_deviations == 2.5


class TestCreateStrategyIchimoku:
    """Test Ichimoku Cloud strategy creation."""

    def test_create_ichimoku_strategy_with_valid_params(self):
        """Test creating Ichimoku strategy with all valid parameters."""
        strategy = create_strategy(
            'ichimoku',
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert isinstance(strategy, IchimokuCloudStrategy)
        assert strategy.tenkan_period == 9
        assert strategy.kijun_period == 26
        assert strategy.senkou_span_b_period == 52
        assert strategy.displacement == 26

    def test_create_ichimoku_strategy_with_custom_periods(self):
        """Test Ichimoku strategy with custom period settings."""
        strategy = create_strategy(
            'ichimoku',
            tenkan_period=7,
            kijun_period=22,
            senkou_span_b_period=44,
            displacement=22,
            rollover=False,
            trailing=2.5,
            slippage_ticks=1,
            symbol='GC'
        )

        assert strategy.tenkan_period == 7
        assert strategy.kijun_period == 22


# ==================== Error Handling Tests ====================

class TestCreateStrategyErrors:
    """Test error handling in strategy creation."""

    def test_create_strategy_with_unknown_type(self):
        """Test that unknown strategy type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy type"):
            create_strategy(
                'unknown_strategy',
                rollover=True,
                trailing=None,
                slippage_ticks=1,
                symbol='ZS'
            )

    def test_create_strategy_missing_strategy_params(self):
        """Test that missing strategy-specific parameters raises KeyError."""
        with pytest.raises(KeyError, match="Missing required parameters"):
            create_strategy(
                'rsi',
                # Missing rsi_period, lower_threshold, upper_threshold
                rollover=True,
                trailing=None,
                slippage_ticks=1,
                symbol='ZS'
            )

    def test_create_strategy_missing_common_params(self):
        """Test that missing common parameters raises KeyError."""
        with pytest.raises(KeyError, match="Missing required common parameters"):
            create_strategy(
                'rsi',
                rsi_period=14,
                lower_threshold=30,
                upper_threshold=70
                # Missing rollover, trailing, slippage_ticks, symbol
            )

    def test_create_strategy_missing_single_strategy_param(self):
        """Test missing single strategy parameter."""
        with pytest.raises(KeyError, match="rsi_period"):
            create_strategy(
                'rsi',
                # Missing rsi_period
                lower_threshold=30,
                upper_threshold=70,
                rollover=True,
                trailing=None,
                slippage_ticks=1,
                symbol='ZS'
            )

    def test_create_strategy_missing_single_common_param(self):
        """Test missing single common parameter."""
        with pytest.raises(KeyError, match="symbol"):
            create_strategy(
                'rsi',
                rsi_period=14,
                lower_threshold=30,
                upper_threshold=70,
                rollover=True,
                trailing=None,
                slippage_ticks=1
                # Missing symbol
            )


# ==================== Get Strategy Name Tests ====================

class TestGetStrategyName:
    """Test strategy name generation."""

    def test_get_rsi_strategy_name(self):
        """Test RSI strategy name generation."""
        name = get_strategy_name(
            'rsi',
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=True,
            trailing=None,
            slippage_ticks=1
        )

        assert 'RSI' in name
        assert '14' in name
        assert '30' in name
        assert '70' in name

    def test_get_ema_strategy_name(self):
        """Test EMA strategy name generation."""
        name = get_strategy_name(
            'ema',
            short_ema_period=9,
            long_ema_period=21,
            rollover=True,
            trailing=None,
            slippage_ticks=1
        )

        assert 'EMA' in name
        assert '9' in name
        assert '21' in name

    def test_get_macd_strategy_name(self):
        """Test MACD strategy name generation."""
        name = get_strategy_name(
            'macd',
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=True,
            trailing=None,
            slippage_ticks=1
        )

        assert 'MACD' in name
        assert '12' in name
        assert '26' in name
        assert '9' in name

    def test_get_bollinger_strategy_name(self):
        """Test Bollinger Bands strategy name generation."""
        name = get_strategy_name(
            'bollinger',
            period=20,
            number_of_standard_deviations=2.0,
            rollover=True,
            trailing=None,
            slippage_ticks=1
        )

        assert 'BB' in name or 'Bollinger' in name
        assert '20' in name
        assert '2' in name

    def test_get_strategy_name_with_unknown_type(self):
        """Test that unknown strategy type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy type"):
            get_strategy_name('unknown_strategy')

    def test_get_strategy_name_case_insensitive(self):
        """Test that strategy name generation is case-insensitive."""
        name_lower = get_strategy_name(
            'rsi',
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=True,
            trailing=None,
            slippage_ticks=1
        )

        name_upper = get_strategy_name(
            'RSI',
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=True,
            trailing=None,
            slippage_ticks=1
        )

        assert name_lower == name_upper


# ==================== Get Available Strategies Tests ====================

class TestGetAvailableStrategies:
    """Test available strategies listing."""

    def test_get_available_strategies_returns_list(self):
        """Test that get_available_strategies returns a list."""
        strategies = get_available_strategies()

        assert isinstance(strategies, list)

    def test_get_available_strategies_contains_all_types(self):
        """Test that all strategy types are included."""
        strategies = get_available_strategies()

        assert 'bollinger' in strategies
        assert 'ema' in strategies
        assert 'ichimoku' in strategies
        assert 'macd' in strategies
        assert 'rsi' in strategies

    def test_get_available_strategies_is_sorted(self):
        """Test that strategies are returned in sorted order."""
        strategies = get_available_strategies()

        assert strategies == sorted(strategies)

    def test_get_available_strategies_matches_strategy_map(self):
        """Test that returned strategies match STRATEGY_MAP keys."""
        strategies = get_available_strategies()

        assert set(strategies) == set(STRATEGY_MAP.keys())


# ==================== Get Strategy Params Tests ====================

class TestGetStrategyParams:
    """Test strategy parameter information retrieval."""

    def test_get_rsi_strategy_params(self):
        """Test getting RSI strategy parameters."""
        params = get_strategy_params('rsi')

        assert 'strategy_params' in params
        assert 'common_params' in params
        assert 'rsi_period' in params['strategy_params']
        assert 'lower_threshold' in params['strategy_params']
        assert 'upper_threshold' in params['strategy_params']

    def test_get_ema_strategy_params(self):
        """Test getting EMA strategy parameters."""
        params = get_strategy_params('ema')

        assert 'short_ema_period' in params['strategy_params']
        assert 'long_ema_period' in params['strategy_params']

    def test_get_macd_strategy_params(self):
        """Test getting MACD strategy parameters."""
        params = get_strategy_params('macd')

        assert 'fast_period' in params['strategy_params']
        assert 'slow_period' in params['strategy_params']
        assert 'signal_period' in params['strategy_params']

    def test_get_bollinger_strategy_params(self):
        """Test getting Bollinger Bands strategy parameters."""
        params = get_strategy_params('bollinger')

        assert 'period' in params['strategy_params']
        assert 'number_of_standard_deviations' in params['strategy_params']

    def test_get_ichimoku_strategy_params(self):
        """Test getting Ichimoku strategy parameters."""
        params = get_strategy_params('ichimoku')

        assert 'tenkan_period' in params['strategy_params']
        assert 'kijun_period' in params['strategy_params']
        assert 'senkou_span_b_period' in params['strategy_params']
        assert 'displacement' in params['strategy_params']

    def test_get_strategy_params_includes_common_params(self):
        """Test that common parameters are included for all strategies."""
        for strategy_type in ['rsi', 'ema', 'macd', 'bollinger', 'ichimoku']:
            params = get_strategy_params(strategy_type)

            assert params['common_params'] == COMMON_PARAMS
            assert 'rollover' in params['common_params']
            assert 'trailing' in params['common_params']
            assert 'slippage_ticks' in params['common_params']
            assert 'symbol' in params['common_params']

    def test_get_strategy_params_with_unknown_type(self):
        """Test that unknown strategy type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy type"):
            get_strategy_params('unknown_strategy')

    def test_get_strategy_params_case_insensitive(self):
        """Test that parameter retrieval is case-insensitive."""
        params_lower = get_strategy_params('rsi')
        params_upper = get_strategy_params('RSI')

        assert params_lower == params_upper


# ==================== Integration Tests ====================

class TestStrategyFactoryIntegration:
    """Test strategy factory integration scenarios."""

    def test_create_all_strategy_types(self):
        """Test creating instances of all available strategy types."""
        strategies_created = []

        # RSI
        rsi = create_strategy(
            'rsi',
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )
        strategies_created.append(('rsi', rsi))

        # EMA
        ema = create_strategy(
            'ema',
            short_ema_period=9,
            long_ema_period=21,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )
        strategies_created.append(('ema', ema))

        # MACD
        macd = create_strategy(
            'macd',
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )
        strategies_created.append(('macd', macd))

        # Bollinger
        bollinger = create_strategy(
            'bollinger',
            period=20,
            number_of_standard_deviations=2.0,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )
        strategies_created.append(('bollinger', bollinger))

        # Ichimoku
        ichimoku = create_strategy(
            'ichimoku',
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )
        strategies_created.append(('ichimoku', ichimoku))

        # Verify all strategies were created
        assert len(strategies_created) == 5
        assert all(strategy is not None for _, strategy in strategies_created)

    def test_strategy_creation_with_all_common_param_variations(self):
        """Test strategy creation with different common parameter combinations."""
        test_cases = [
            {'rollover': True, 'trailing': None, 'slippage_ticks': 1},
            {'rollover': False, 'trailing': 2.0, 'slippage_ticks': 2},
            {'rollover': True, 'trailing': 3.5, 'slippage_ticks': 0},
            {'rollover': False, 'trailing': None, 'slippage_ticks': 3},
        ]

        for common_params in test_cases:
            strategy = create_strategy(
                'rsi',
                rsi_period=14,
                lower_threshold=30,
                upper_threshold=70,
                symbol='ZS',
                **common_params
            )

            assert strategy.rollover == common_params['rollover']
            assert strategy.trailing == common_params['trailing']
            assert strategy.position_manager.slippage_ticks == common_params['slippage_ticks']

    def test_get_params_and_create_strategy_consistency(self):
        """Test that get_strategy_params returns params that work with create_strategy."""
        for strategy_type in get_available_strategies():
            params_info = get_strategy_params(strategy_type)

            # Build parameter dict with default values
            params = {}

            # Add strategy-specific params
            for param in params_info['strategy_params']:
                if param == 'short_ema_period':
                    params[param] = 9
                elif param == 'long_ema_period':
                    params[param] = 21
                elif param == 'fast_period':
                    params[param] = 12
                elif param == 'slow_period':
                    params[param] = 26
                elif param == 'signal_period':
                    params[param] = 9
                elif 'period' in param.lower():
                    params[param] = 14
                elif 'threshold' in param.lower():
                    params[param] = 30 if 'lower' in param else 70
                elif 'displacement' in param.lower():
                    params[param] = 26
                elif 'standard_deviations' in param.lower():
                    params[param] = 2.0
                else:
                    params[param] = 10

            # Add common params
            for param in params_info['common_params']:
                if param == 'rollover':
                    params[param] = True
                elif param == 'trailing':
                    params[param] = None
                elif param == 'slippage_ticks':
                    params[param] = 1
                elif param == 'symbol':
                    params[param] = 'ZS'

            # Should create successfully
            strategy = create_strategy(strategy_type, **params)
            assert strategy is not None


# ==================== Strategy Map Configuration Tests ====================

class TestStrategyMapConfiguration:
    """Test STRATEGY_MAP configuration."""

    def test_strategy_map_has_all_required_strategies(self):
        """Test that STRATEGY_MAP contains all expected strategies."""
        expected_strategies = ['bollinger', 'ema', 'ichimoku', 'macd', 'rsi']

        for strategy_type in expected_strategies:
            assert strategy_type in STRATEGY_MAP

    def test_strategy_map_entries_have_correct_structure(self):
        """Test that each STRATEGY_MAP entry has (class, validator, params)."""
        for strategy_type, entry in STRATEGY_MAP.items():
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            strategy_class, validator, param_names = entry
            assert callable(strategy_class)
            assert hasattr(validator, 'validate')
            assert isinstance(param_names, list)

    def test_common_params_includes_all_required_params(self):
        """Test that COMMON_PARAMS includes all required common parameters."""
        assert 'rollover' in COMMON_PARAMS
        assert 'trailing' in COMMON_PARAMS
        assert 'slippage_ticks' in COMMON_PARAMS
        assert 'symbol' in COMMON_PARAMS


class TestWarningSystem:
    """Test warning logging system."""

    def test_warnings_disabled_skips_logging(self):
        """Test _log_warnings_enabled=False causes early return (line 172)."""
        import app.backtesting.strategies.strategy_factory as factory_module

        # Save original value
        original_enabled = factory_module._log_warnings_enabled

        try:
            # Disable warnings
            factory_module._log_warnings_enabled = False

            # Mock logger to verify it's not called
            with patch.object(factory_module.logger, 'warning') as mock_warning:
                # Call _log_warnings_once with some warnings
                factory_module._log_warnings_once(['Test warning'], 'RSI')

                # Logger.warning should NOT be called when warnings are disabled
                mock_warning.assert_not_called()

        finally:
            # Restore original value
            factory_module._log_warnings_enabled = original_enabled

    def test_warnings_enabled_logs_messages(self):
        """Test warnings are logged when enabled."""
        import app.backtesting.strategies.strategy_factory as factory_module

        # Save original value
        original_enabled = factory_module._log_warnings_enabled
        original_logged = factory_module._logged_warnings.copy()

        try:
            # Enable warnings
            factory_module._log_warnings_enabled = True
            factory_module._logged_warnings.clear()

            # Mock logger
            with patch.object(factory_module.logger, 'warning') as mock_warning:
                # Call _log_warnings_once with warnings
                factory_module._log_warnings_once(['Test warning message'], 'RSI')

                # Logger.warning SHOULD be called when enabled
                mock_warning.assert_called_once()
                assert 'Test warning message' in str(mock_warning.call_args)

        finally:
            # Restore original values
            factory_module._log_warnings_enabled = original_enabled
            factory_module._logged_warnings = original_logged
