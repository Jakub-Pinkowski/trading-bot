"""
Tests for mass_tester module.

Tests cover:
- MassTester class initialization
- Strategy addition methods (RSI, EMA, MACD, Bollinger, Ichimoku)
- Parameter combination generation
- Strategy instance creation
- Configuration loading (switch dates)
- Empty/edge case parameter handling
- Integration with strategy factory
"""
from unittest.mock import patch, MagicMock

from app.backtesting.testing.mass_tester import MassTester


# ==================== Initialization Tests ====================

class TestMassTesterInitialization:
    """Test MassTester initialization."""

    def test_initialization_with_valid_parameters(self):
        """Test that MassTester initializes correctly with valid parameters."""
        tested_months = ['1!', '2!']
        symbols = ['ZS', 'CL']
        intervals = ['15m', '1h']

        tester = MassTester(tested_months, symbols, intervals)

        assert tester.tested_months == tested_months
        assert tester.symbols == symbols
        assert tester.intervals == intervals
        assert tester.strategies == []
        assert tester.results == []
        assert isinstance(tester.switch_dates_dict, dict)

    def test_initialization_loads_switch_dates(self):
        """Test that switch dates are loaded from YAML file on initialization."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        # Verify switch_dates_dict is populated
        assert tester.switch_dates_dict is not None
        assert isinstance(tester.switch_dates_dict, dict)
        # Should contain symbol keys
        assert len(tester.switch_dates_dict) > 0

    def test_initialization_with_single_values(self):
        """Test initialization with single month, symbol, and interval."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        assert tester.tested_months == ['1!']
        assert tester.symbols == ['ZS']
        assert tester.intervals == ['1h']

    def test_initialization_with_multiple_values(self):
        """Test initialization with multiple months, symbols, and intervals."""
        tested_months = ['1!', '2!', '3!']
        symbols = ['ZS', 'CL', 'GC', 'ES']
        intervals = ['15m', '1h', '4h', '1d']

        tester = MassTester(tested_months, symbols, intervals)

        assert len(tester.tested_months) == 3
        assert len(tester.symbols) == 4
        assert len(tester.intervals) == 4


# ==================== RSI Strategy Tests ====================

class TestAddRSITests:
    """Test RSI strategy addition."""

    def test_add_rsi_tests_basic(self, simple_tester, mock_strategy_factory):
        """Test adding RSI strategies with basic parameters."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "RSI_14_30_70"

        simple_tester.add_rsi_tests(
            rsi_periods=[14],
            lower_thresholds=[30],
            upper_thresholds=[70],
            rollovers=[True],
            trailing_stops=[None],
            slippage_ticks_list=[1]
        )

        # Should create 1 strategy (1*1*1*1*1*1 = 1)
        assert len(simple_tester.strategies) == 1
        assert mock_create.call_count == 1

    def test_add_rsi_tests_multiple_parameters(self, simple_tester, mock_strategy_factory):
        """Test adding RSI strategies with multiple parameter variations."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "RSI_test"

        simple_tester.add_rsi_tests(
            rsi_periods=[14, 21],
            lower_thresholds=[30, 35],
            upper_thresholds=[70, 75],
            rollovers=[True, False],
            trailing_stops=[None, 2.0],
            slippage_ticks_list=[1, 2]
        )

        # Should create 2*2*2*2*2*2 = 64 strategies
        assert len(simple_tester.strategies) == 64
        assert mock_create.call_count == 64

    def test_add_rsi_tests_parameter_passing(self, simple_tester, mock_strategy_factory):
        """Test that RSI parameters are correctly passed to strategy factory."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "RSI_14_30_70"

        simple_tester.add_rsi_tests(
            rsi_periods=[14],
            lower_thresholds=[30],
            upper_thresholds=[70],
            rollovers=[True],
            trailing_stops=[2.5],
            slippage_ticks_list=[1]
        )

        # Verify create_strategy was called with correct parameters
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['rsi_period'] == 14
        assert call_kwargs['lower_threshold'] == 30
        assert call_kwargs['upper_threshold'] == 70
        assert call_kwargs['rollover'] is True
        assert call_kwargs['trailing'] == 2.5
        assert call_kwargs['slippage_ticks'] == 1


# ==================== EMA Strategy Tests ====================

class TestAddEMATests:
    """Test EMA crossover strategy addition."""

    def test_add_ema_tests_basic(self, simple_tester, mock_strategy_factory):
        """Test adding EMA strategies with basic parameters."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "EMA_9_21"

        simple_tester.add_ema_crossover_tests(
            short_ema_periods=[9],
            long_ema_periods=[21],
            rollovers=[True],
            trailing_stops=[None],
            slippage_ticks_list=[1]
        )

        assert len(simple_tester.strategies) == 1
        assert mock_create.call_count == 1

    def test_add_ema_tests_multiple_parameters(self, simple_tester, mock_strategy_factory):
        """Test adding EMA strategies with multiple parameter variations."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "EMA_test"

        simple_tester.add_ema_crossover_tests(
            short_ema_periods=[9, 12],
            long_ema_periods=[21, 26],
            rollovers=[True, False],
            trailing_stops=[None, 2.0, 3.0],
            slippage_ticks_list=[1, 2]
        )

        # Should create 2*2*2*3*2 = 48 strategies
        assert len(simple_tester.strategies) == 48
        assert mock_create.call_count == 48

    def test_add_ema_tests_parameter_passing(self, simple_tester, mock_strategy_factory):
        """Test that EMA parameters are correctly passed to strategy factory."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "EMA_12_26"

        simple_tester.add_ema_crossover_tests(
            short_ema_periods=[12],
            long_ema_periods=[26],
            rollovers=[False],
            trailing_stops=[1.5],
            slippage_ticks_list=[2]
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['short_ema_period'] == 12
        assert call_kwargs['long_ema_period'] == 26
        assert call_kwargs['rollover'] is False
        assert call_kwargs['trailing'] == 1.5
        assert call_kwargs['slippage_ticks'] == 2


# ==================== MACD Strategy Tests ====================

class TestAddMACDTests:
    """Test MACD strategy addition."""

    def test_add_macd_tests_basic(self, simple_tester, mock_strategy_factory):
        """Test adding MACD strategies with basic parameters."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "MACD_12_26_9"

        simple_tester.add_macd_tests(
            fast_periods=[12],
            slow_periods=[26],
            signal_periods=[9],
            rollovers=[True],
            trailing_stops=[None],
            slippage_ticks_list=[1]
        )

        assert len(simple_tester.strategies) == 1
        assert mock_create.call_count == 1

    def test_add_macd_tests_multiple_parameters(self, simple_tester, mock_strategy_factory):
        """Test adding MACD strategies with multiple parameter variations."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "MACD_test"

        simple_tester.add_macd_tests(
            fast_periods=[12, 10],
            slow_periods=[26, 24],
            signal_periods=[9, 7],
            rollovers=[True],
            trailing_stops=[None, 2.0],
            slippage_ticks_list=[1, 2]
        )

        # Should create 2*2*2*1*2*2 = 32 strategies
        assert len(simple_tester.strategies) == 32
        assert mock_create.call_count == 32

    def test_add_macd_tests_parameter_passing(self, simple_tester, mock_strategy_factory):
        """Test that MACD parameters are correctly passed to strategy factory."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "MACD_8_20_6"

        simple_tester.add_macd_tests(
            fast_periods=[8],
            slow_periods=[20],
            signal_periods=[6],
            rollovers=[True],
            trailing_stops=[3.0],
            slippage_ticks_list=[1]
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['fast_period'] == 8
        assert call_kwargs['slow_period'] == 20
        assert call_kwargs['signal_period'] == 6
        assert call_kwargs['rollover'] is True
        assert call_kwargs['trailing'] == 3.0
        assert call_kwargs['slippage_ticks'] == 1


# ==================== Bollinger Bands Strategy Tests ====================

class TestAddBollingerBandsTests:
    """Test Bollinger Bands strategy addition."""

    def test_add_bollinger_tests_basic(self, simple_tester, mock_strategy_factory):
        """Test adding Bollinger Bands strategies with basic parameters."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "BB_20_2.0"

        simple_tester.add_bollinger_bands_tests(
            periods=[20],
            number_of_standard_deviations_list=[2.0],
            rollovers=[True],
            trailing_stops=[None],
            slippage_ticks_list=[1]
        )

        assert len(simple_tester.strategies) == 1
        assert mock_create.call_count == 1

    def test_add_bollinger_tests_multiple_parameters(self, simple_tester, mock_strategy_factory):
        """Test adding Bollinger Bands strategies with multiple parameter variations."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "BB_test"

        simple_tester.add_bollinger_bands_tests(
            periods=[15, 20, 25],
            number_of_standard_deviations_list=[1.5, 2.0, 2.5],
            rollovers=[True, False],
            trailing_stops=[None, 2.0],
            slippage_ticks_list=[1]
        )

        # Should create 3*3*2*2*1 = 36 strategies
        assert len(simple_tester.strategies) == 36
        assert mock_create.call_count == 36

    def test_add_bollinger_tests_parameter_passing(self, simple_tester, mock_strategy_factory):
        """Test that Bollinger Bands parameters are correctly passed to strategy factory."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "BB_15_1.5"

        simple_tester.add_bollinger_bands_tests(
            periods=[15],
            number_of_standard_deviations_list=[1.5],
            rollovers=[False],
            trailing_stops=[2.5],
            slippage_ticks_list=[2]
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['period'] == 15
        assert call_kwargs['number_of_standard_deviations'] == 1.5
        assert call_kwargs['rollover'] is False
        assert call_kwargs['trailing'] == 2.5
        assert call_kwargs['slippage_ticks'] == 2


# ==================== Ichimoku Cloud Strategy Tests ====================

class TestAddIchimokuCloudTests:
    """Test Ichimoku Cloud strategy addition."""

    def test_add_ichimoku_tests_basic(self, simple_tester, mock_strategy_factory):
        """Test adding Ichimoku strategies with basic parameters."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "Ichimoku_9_26_52_26"

        simple_tester.add_ichimoku_cloud_tests(
            tenkan_periods=[9],
            kijun_periods=[26],
            senkou_span_b_periods=[52],
            displacements=[26],
            rollovers=[True],
            trailing_stops=[None],
            slippage_ticks_list=[1]
        )

        assert len(simple_tester.strategies) == 1
        assert mock_create.call_count == 1

    def test_add_ichimoku_tests_multiple_parameters(self, simple_tester, mock_strategy_factory):
        """Test adding Ichimoku strategies with multiple parameter variations."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "Ichimoku_test"

        simple_tester.add_ichimoku_cloud_tests(
            tenkan_periods=[9, 7],
            kijun_periods=[26, 22],
            senkou_span_b_periods=[52, 44],
            displacements=[26, 22],
            rollovers=[True],
            trailing_stops=[None, 2.0],
            slippage_ticks_list=[1, 2]
        )

        # Should create 2*2*2*2*1*2*2 = 64 strategies
        assert len(simple_tester.strategies) == 64
        assert mock_create.call_count == 64

    def test_add_ichimoku_tests_parameter_passing(self, simple_tester, mock_strategy_factory):
        """Test that Ichimoku parameters are correctly passed to strategy factory."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "Ichimoku_7_22_44_22"

        simple_tester.add_ichimoku_cloud_tests(
            tenkan_periods=[7],
            kijun_periods=[22],
            senkou_span_b_periods=[44],
            displacements=[22],
            rollovers=[False],
            trailing_stops=[3.0],
            slippage_ticks_list=[2]
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['tenkan_period'] == 7
        assert call_kwargs['kijun_period'] == 22
        assert call_kwargs['senkou_span_b_period'] == 44
        assert call_kwargs['displacement'] == 22
        assert call_kwargs['rollover'] is False
        assert call_kwargs['trailing'] == 3.0
        assert call_kwargs['slippage_ticks'] == 2


# ==================== Multiple Strategy Tests ====================

class TestMultipleStrategies:
    """Test adding multiple strategy types together."""

    def test_add_multiple_strategy_types(self, simple_tester, mock_strategy_factory):
        """Test that multiple strategy types can be added and accumulate correctly."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "Strategy_test"

        # Add RSI strategies
        simple_tester.add_rsi_tests(
            rsi_periods=[14],
            lower_thresholds=[30],
            upper_thresholds=[70],
            rollovers=[True],
            trailing_stops=[None],
            slippage_ticks_list=[1]
        )
        assert len(simple_tester.strategies) == 1

        # Add EMA strategies
        simple_tester.add_ema_crossover_tests(
            short_ema_periods=[9],
            long_ema_periods=[21],
            rollovers=[True],
            trailing_stops=[None],
            slippage_ticks_list=[1]
        )
        assert len(simple_tester.strategies) == 2

        # Add MACD strategies
        simple_tester.add_macd_tests(
            fast_periods=[12],
            slow_periods=[26],
            signal_periods=[9],
            rollovers=[True],
            trailing_stops=[None],
            slippage_ticks_list=[1]
        )
        assert len(simple_tester.strategies) == 3

    def test_strategies_list_structure(self, simple_tester, mock_strategy_factory):
        """Test that strategies are stored as (name, instance) tuples."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_strategy = MagicMock()
        mock_create.return_value = mock_strategy
        mock_get_name.return_value = "RSI_14_30_70"

        simple_tester.add_rsi_tests(
            rsi_periods=[14],
            lower_thresholds=[30],
            upper_thresholds=[70],
            rollovers=[True],
            trailing_stops=[None],
            slippage_ticks_list=[1]
        )

        # Verify structure
        assert len(simple_tester.strategies) == 1
        strategy_name, strategy_instance = simple_tester.strategies[0]
        assert strategy_name == "RSI_14_30_70"
        assert strategy_instance == mock_strategy


# ==================== Edge Cases Tests ====================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_parameter_lists_handled(self, simple_tester, mock_strategy_factory):
        """Test that empty parameter lists are replaced with [None]."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "RSI_test"

        # Add RSI with some empty lists
        simple_tester.add_rsi_tests(
            rsi_periods=[],  # Empty list
            lower_thresholds=[30],
            upper_thresholds=[],  # Empty list
            rollovers=[True],
            trailing_stops=[],  # Empty list
            slippage_ticks_list=[1]
        )

        # Should still create strategies with None values
        assert len(simple_tester.strategies) > 0
        assert mock_create.call_count > 0

    def test_single_parameter_combinations(self, simple_tester, mock_strategy_factory):
        """Test with all single-value parameters (no combinations)."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "EMA_9_21"

        simple_tester.add_ema_crossover_tests(
            short_ema_periods=[9],
            long_ema_periods=[21],
            rollovers=[True],
            trailing_stops=[None],
            slippage_ticks_list=[1]
        )

        # Should create exactly 1 strategy
        assert len(simple_tester.strategies) == 1
        assert mock_create.call_count == 1

    def test_large_parameter_combinations(self, simple_tester, mock_strategy_factory):
        """Test with many parameter combinations to verify Cartesian product works."""
        mock_create, mock_get_name = mock_strategy_factory
        mock_get_name.return_value = "RSI_test"

        # Create many combinations
        simple_tester.add_rsi_tests(
            rsi_periods=[7, 14, 21, 28],  # 4 values
            lower_thresholds=[20, 25, 30, 35],  # 4 values
            upper_thresholds=[65, 70, 75, 80],  # 4 values
            rollovers=[True, False],  # 2 values
            trailing_stops=[None, 1.0, 2.0],  # 3 values
            slippage_ticks_list=[1, 2]  # 2 values
        )

        # Should create 4*4*4*2*3*2 = 768 strategies
        assert len(simple_tester.strategies) == 768
        assert mock_create.call_count == 768


# ==================== Run Tests Method ====================

class TestRunTests:
    """Test the run_tests method."""

    def test_run_tests_calls_orchestrator(self):
        """Test that run_tests delegates to orchestrator with correct parameters."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        with patch('app.backtesting.testing.mass_tester.orchestrator_run_tests') as mock_orchestrator:
            mock_orchestrator.return_value = []

            result = tester.run_tests(
                verbose=True,
                max_workers=4,
                skip_existing=False
            )

            # Verify orchestrator was called correctly
            mock_orchestrator.assert_called_once_with(
                tester,
                verbose=True,
                max_workers=4,
                skip_existing=False
            )
            assert result == []

    def test_run_tests_returns_orchestrator_result(self):
        """Test that run_tests returns the orchestrator's result."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        expected_results = [
            {'symbol': 'ZS', 'strategy': 'RSI_14_30_70', 'profit_factor': 1.5},
            {'symbol': 'CL', 'strategy': 'EMA_9_21', 'profit_factor': 1.2}
        ]

        with patch('app.backtesting.testing.mass_tester.orchestrator_run_tests') as mock_orchestrator:
            mock_orchestrator.return_value = expected_results

            result = tester.run_tests(
                verbose=False,
                max_workers=None,
                skip_existing=True
            )

            assert result == expected_results

    def test_run_tests_with_different_parameters(self):
        """Test run_tests with various parameter combinations."""
        tester = MassTester(['1!', '2!'], ['ZS', 'CL'], ['1h', '4h'])

        test_cases = [
            {'verbose': True, 'max_workers': 1, 'skip_existing': True},
            {'verbose': False, 'max_workers': 8, 'skip_existing': False},
            {'verbose': True, 'max_workers': None, 'skip_existing': True},
        ]

        for params in test_cases:
            with patch('app.backtesting.testing.mass_tester.orchestrator_run_tests') as mock_orchestrator:
                mock_orchestrator.return_value = []

                tester.run_tests(**params)

                # Verify called with correct parameters
                call_args = mock_orchestrator.call_args
                assert call_args[0][0] == tester
                assert call_args[1]['verbose'] == params['verbose']
                assert call_args[1]['max_workers'] == params['max_workers']
                assert call_args[1]['skip_existing'] == params['skip_existing']


# ==================== Integration Tests ====================

class TestMassTesterIntegration:
    """Test MassTester integration scenarios."""

    def test_complete_workflow(self):
        """Test complete workflow: initialize, add strategies, run tests."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        with patch('app.backtesting.testing.mass_tester.create_strategy') as mock_create, \
                patch('app.backtesting.testing.mass_tester.get_strategy_name') as mock_get_name, \
                patch('app.backtesting.testing.mass_tester.orchestrator_run_tests') as mock_orchestrator:
            mock_create.return_value = MagicMock()
            mock_get_name.return_value = "Test_Strategy"
            mock_orchestrator.return_value = [{'result': 'success'}]

            # Add strategies
            tester.add_rsi_tests([14], [30], [70], [True], [None], [1])
            tester.add_ema_crossover_tests([9], [21], [True], [None], [1])

            # Verify strategies added
            assert len(tester.strategies) == 2

            # Run tests
            results = tester.run_tests(verbose=False, max_workers=1, skip_existing=False)

            # Verify orchestrator was called
            mock_orchestrator.assert_called_once()
            assert len(results) == 1

    def test_multiple_months_symbols_intervals(self):
        """Test initialization with multiple months, symbols, and intervals."""
        tested_months = ['1!', '2!', '3!']
        symbols = ['ZS', 'CL', 'GC']
        intervals = ['15m', '1h', '4h']

        tester = MassTester(tested_months, symbols, intervals)

        # Verify all parameters stored correctly
        assert tester.tested_months == tested_months
        assert tester.symbols == symbols
        assert tester.intervals == intervals

        # Verify these would be used in orchestrator
        with patch('app.backtesting.testing.mass_tester.orchestrator_run_tests') as mock_orchestrator:
            mock_orchestrator.return_value = []

            tester.run_tests(verbose=False, max_workers=1, skip_existing=False)

            # Orchestrator should receive tester with all these parameters
            call_tester = mock_orchestrator.call_args[0][0]
            assert call_tester.tested_months == tested_months
            assert call_tester.symbols == symbols
            assert call_tester.intervals == intervals

    def test_strategy_name_generation(self):
        """Test that strategy names are generated correctly through factory."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        with patch('app.backtesting.testing.mass_tester.create_strategy') as mock_create, \
                patch('app.backtesting.testing.mass_tester.get_strategy_name') as mock_get_name:
            mock_create.return_value = MagicMock()
            # Simulate real strategy name generation
            mock_get_name.side_effect = lambda strategy_type, **kwargs: (
                f"{strategy_type.upper()}_"
                f"{kwargs.get('rsi_period', '')}_"
                f"{kwargs.get('lower_threshold', '')}_"
                f"{kwargs.get('upper_threshold', '')}"
            )

            tester.add_rsi_tests(
                rsi_periods=[14, 21],
                lower_thresholds=[30],
                upper_thresholds=[70],
                rollovers=[True],
                trailing_stops=[None],
                slippage_ticks_list=[1]
            )

            # Verify names are generated correctly
            assert len(tester.strategies) == 2
            names = [name for name, _ in tester.strategies]
            assert "rsi_14_30_70" in names[0].lower()
            assert "rsi_21_30_70" in names[1].lower()


# ==================== Strategy Instance Validation Tests ====================

class TestStrategyInstanceValidation:
    """Test that actual strategy instances are created correctly (no mocking)."""

    def test_macd_strategy_instances_have_correct_attributes(self):
        """Test MACD strategies are created with correct parameters."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        # Don't mock - test real strategy creation
        tester.add_macd_tests(
            fast_periods=[12],
            slow_periods=[26],
            signal_periods=[9],
            rollovers=[True],
            trailing_stops=[2.0],
            slippage_ticks_list=[1]
        )

        # Verify strategy instance
        assert len(tester.strategies) == 1
        strategy_name, strategy_instance = tester.strategies[0]

        # Verify attributes
        assert strategy_instance.fast_period == 12
        assert strategy_instance.slow_period == 26
        assert strategy_instance.signal_period == 9
        assert strategy_instance.rollover is True
        assert strategy_instance.trailing == 2.0
        assert strategy_instance.position_manager.slippage_ticks == 1

    def test_bollinger_strategy_instances_have_correct_attributes(self):
        """Test Bollinger strategies are created with correct parameters."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        tester.add_bollinger_bands_tests(
            periods=[20],
            number_of_standard_deviations_list=[2.0],
            rollovers=[False],
            trailing_stops=[None],
            slippage_ticks_list=[2]
        )

        assert len(tester.strategies) == 1
        strategy_name, strategy_instance = tester.strategies[0]

        assert strategy_instance.period == 20
        assert strategy_instance.number_of_standard_deviations == 2.0
        assert strategy_instance.rollover is False
        assert strategy_instance.trailing is None
        assert strategy_instance.position_manager.slippage_ticks == 2

    def test_ichimoku_strategy_instances_have_correct_attributes(self):
        """Test Ichimoku strategies are created with correct parameters."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        tester.add_ichimoku_cloud_tests(
            tenkan_periods=[9],
            kijun_periods=[26],
            senkou_span_b_periods=[52],
            displacements=[26],
            rollovers=[True],
            trailing_stops=[3.0],
            slippage_ticks_list=[1]
        )

        assert len(tester.strategies) == 1
        strategy_name, strategy_instance = tester.strategies[0]

        assert strategy_instance.tenkan_period == 9
        assert strategy_instance.kijun_period == 26
        assert strategy_instance.senkou_span_b_period == 52
        assert strategy_instance.displacement == 26
        assert strategy_instance.rollover is True
        assert strategy_instance.trailing == 3.0
        assert strategy_instance.position_manager.slippage_ticks == 1

    def test_rsi_strategy_instances_have_correct_attributes(self):
        """Test RSI strategies are created with correct parameters."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        tester.add_rsi_tests(
            rsi_periods=[14],
            lower_thresholds=[30],
            upper_thresholds=[70],
            rollovers=[True],
            trailing_stops=[2.5],
            slippage_ticks_list=[1]
        )

        assert len(tester.strategies) == 1
        strategy_name, strategy_instance = tester.strategies[0]

        assert strategy_instance.rsi_period == 14
        assert strategy_instance.lower_threshold == 30
        assert strategy_instance.upper_threshold == 70
        assert strategy_instance.rollover is True
        assert strategy_instance.trailing == 2.5
        assert strategy_instance.position_manager.slippage_ticks == 1

    def test_ema_strategy_instances_have_correct_attributes(self):
        """Test EMA strategies are created with correct parameters."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        tester.add_ema_crossover_tests(
            short_ema_periods=[9],
            long_ema_periods=[21],
            rollovers=[False],
            trailing_stops=[1.5],
            slippage_ticks_list=[2]
        )

        assert len(tester.strategies) == 1
        strategy_name, strategy_instance = tester.strategies[0]

        assert strategy_instance.short_ema_period == 9
        assert strategy_instance.long_ema_period == 21
        assert strategy_instance.rollover is False
        assert strategy_instance.trailing == 1.5
        assert strategy_instance.position_manager.slippage_ticks == 2

    def test_multiple_strategy_instances_with_different_parameters(self):
        """Test that multiple strategies are created with varying parameters."""
        tester = MassTester(['1!'], ['ZS'], ['1h'])

        tester.add_rsi_tests(
            rsi_periods=[14, 21],
            lower_thresholds=[30],
            upper_thresholds=[70],
            rollovers=[True],
            trailing_stops=[None],
            slippage_ticks_list=[1]
        )

        # Should create 2 strategies (2 periods)
        assert len(tester.strategies) == 2

        # Verify each has correct period
        periods_found = []
        for strategy_name, strategy_instance in tester.strategies:
            periods_found.append(strategy_instance.rsi_period)

        assert 14 in periods_found
        assert 21 in periods_found
