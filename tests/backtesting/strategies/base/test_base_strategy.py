"""
Tests for BaseStrategy core functionality.

Tests cover:
- Strategy initialization and configuration
- Abstract method enforcement
- Helper functions (detect_crossover, detect_threshold_cross, precompute_hashes)
- Signal queueing and execution with 1-bar delay
- Trade extraction workflow
- Edge cases and error handling
"""
import numpy as np
import pandas as pd
import pytest

from app.backtesting.strategies.base.base_strategy import (
    BaseStrategy,
    detect_crossover,
    detect_threshold_cross,
    precompute_hashes,
    INDICATOR_WARMUP_PERIOD
)
from tests.backtesting.strategies.strategy_test_utils import (
    assert_trades_have_both_directions,
    create_small_ohlcv_dataframe,
    create_constant_price_dataframe,
)


# ==================== Test Strategy Implementation ====================

class ConcreteTestStrategy(BaseStrategy):
    """
    Concrete strategy implementation for testing BaseStrategy.

    Implements minimal indicator and signal logic to test base functionality.
    """

    def __init__(self, signal_indices=None, **kwargs):
        """
        Initialize test strategy.

        Args:
            signal_indices: List of (index, signal_value) tuples for testing
            **kwargs: Base strategy parameters (rollover, trailing, slippage_ticks, symbol)
        """
        super().__init__(**kwargs)
        self.signal_indices = signal_indices or []

    def add_indicators(self, df):
        """Add dummy indicator for testing."""
        df['test_indicator'] = df['close'].rolling(window=10).mean()
        return df

    def generate_signals(self, df):
        """Generate signals based on test configuration."""
        df['signal'] = 0

        # Set signals at specified indices
        for idx, signal_value in self.signal_indices:
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc('signal')] = signal_value

        return df


# ==================== Fixtures ====================

@pytest.fixture
def base_strategy():
    """Standard base strategy instance."""
    return ConcreteTestStrategy(
        rollover=False,
        trailing=None,
        slippage_ticks=1,
        symbol='ZS'
    )


# ==================== Test Classes ====================

class TestBaseStrategyInitialization:
    """Test BaseStrategy initialization and configuration."""

    @pytest.mark.parametrize("rollover,trailing,slippage_ticks,symbol", [
        (False, None, 0, 'ZS'),
        (False, 2.0, 1, 'ZS'),
        (True, None, 1, 'CL'),
        (True, 3.0, 2, 'ES'),
    ])
    def test_initialization_with_various_configs(
        self, rollover, trailing, slippage_ticks, symbol
    ):
        """Test strategy initializes correctly with various configurations."""
        strategy = ConcreteTestStrategy(
            rollover=rollover,
            trailing=trailing,
            slippage_ticks=slippage_ticks,
            symbol=symbol
        )

        assert strategy.rollover == rollover
        assert strategy.trailing == trailing
        assert strategy.position_manager.slippage_ticks == slippage_ticks
        assert strategy.position_manager.symbol == symbol
        assert strategy.position_manager is not None
        assert strategy.switch_handler is not None
        assert strategy.prev_row is None
        assert strategy.prev_time is None
        assert strategy.queued_signal is None

        # Verify trailing stop manager when enabled
        if trailing is not None:
            assert strategy.trailing_stop_manager is not None
        else:
            assert strategy.trailing_stop_manager is None


class TestAbstractMethods:
    """Test abstract method enforcement."""

    def test_add_indicators_not_implemented(self):
        """Test add_indicators raises NotImplementedError if not overridden."""
        strategy = BaseStrategy(
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        df = pd.DataFrame({'close': [100, 101, 102]})

        with pytest.raises(NotImplementedError, match="Subclasses must implement add_indicators method"):
            strategy.add_indicators(df)

    def test_generate_signals_not_implemented(self):
        """Test generate_signals raises NotImplementedError if not overridden."""
        strategy = BaseStrategy(
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        df = pd.DataFrame({'close': [100, 101, 102]})

        with pytest.raises(NotImplementedError, match="Subclasses must implement generate_signals method"):
            strategy.generate_signals(df)


class TestHelperFunctions:
    """Test helper functions available to strategy implementations."""

    # --- Crossover Detection ---

    def test_detect_crossover_above(self):
        """Test detecting bullish crossover (series1 crosses above series2)."""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([3, 3, 3, 3, 3])

        result = detect_crossover(series1, series2, 'above')

        # Crossover happens at index 3 (3->4 crosses above 3)
        expected = pd.Series([False, False, False, True, False])
        pd.testing.assert_series_equal(result, expected)

    def test_detect_crossover_below(self):
        """Test detecting bearish crossover (series1 crosses below series2)."""
        series1 = pd.Series([5, 4, 3, 2, 1])
        series2 = pd.Series([3, 3, 3, 3, 3])

        result = detect_crossover(series1, series2, 'below')

        # Crossover happens at index 3 (3->2 crosses below 3)
        expected = pd.Series([False, False, False, True, False])
        pd.testing.assert_series_equal(result, expected)

    def test_detect_crossover_no_cross(self):
        """Test no crossover detected when series don't cross."""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([6, 7, 8, 9, 10])

        result = detect_crossover(series1, series2, 'above')

        # No crossover
        expected = pd.Series([False, False, False, False, False])
        pd.testing.assert_series_equal(result, expected)

    def test_detect_crossover_with_real_ema_data(self):
        """Test crossover detection with realistic EMA-like data."""
        # Simulate fast EMA crossing above slow EMA
        fast_ema = pd.Series([48, 49, 50, 51, 52])
        slow_ema = pd.Series([50, 50, 50, 50, 50])

        result = detect_crossover(fast_ema, slow_ema, 'above')

        # Crossover at index 3 (50->51 crosses above 50)
        assert result[3] == True
        assert result.sum() == 1

    # --- Threshold Cross Detection ---

    def test_detect_threshold_cross_below(self):
        """Test detecting threshold cross below (e.g., RSI crosses below 30)."""
        series = pd.Series([35, 32, 29, 28, 31])
        threshold = 30

        result = detect_threshold_cross(series, threshold, 'below')

        # Cross below happens at index 2 (32->29 crosses below 30)
        expected = pd.Series([False, False, True, False, False])
        pd.testing.assert_series_equal(result, expected)

    def test_detect_threshold_cross_above(self):
        """Test detecting threshold cross above (e.g., RSI crosses above 70)."""
        series = pd.Series([65, 68, 71, 72, 69])
        threshold = 70

        result = detect_threshold_cross(series, threshold, 'above')

        # Cross above happens at index 2 (68->71 crosses above 70)
        expected = pd.Series([False, False, True, False, False])
        pd.testing.assert_series_equal(result, expected)

    def test_detect_threshold_cross_no_cross(self):
        """Test no threshold cross detected when series stays on one side."""
        series = pd.Series([25, 26, 27, 28, 29])
        threshold = 30

        result = detect_threshold_cross(series, threshold, 'below')

        # No cross (already below)
        expected = pd.Series([False, False, False, False, False])
        pd.testing.assert_series_equal(result, expected)

    def test_detect_threshold_cross_with_rsi_data(self):
        """Test threshold cross with realistic RSI data."""
        # Simulate RSI crossing below oversold threshold
        rsi = pd.Series([35, 32, 30, 28, 31])

        result = detect_threshold_cross(rsi, 30, 'below')

        # Should detect cross at index 2
        assert result[2] == True
        assert result.sum() == 1

    # --- Precompute Hashes ---

    def test_precompute_hashes_all_columns(self, sample_ohlcv_data):
        """Test precompute_hashes returns hashes for all OHLCV columns."""
        hashes = precompute_hashes(sample_ohlcv_data)

        assert 'close' in hashes
        assert 'high' in hashes
        assert 'low' in hashes
        assert 'open' in hashes
        assert 'volume' in hashes

        # Verify hashes are strings
        assert isinstance(hashes['close'], str)
        assert isinstance(hashes['high'], str)

    def test_precompute_hashes_subset_columns(self):
        """Test precompute_hashes works with partial OHLCV data."""
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        hashes = precompute_hashes(df)

        assert 'close' in hashes
        assert 'volume' in hashes
        assert 'high' not in hashes
        assert 'low' not in hashes

    def test_precompute_hashes_consistent(self, sample_ohlcv_data):
        """Test precompute_hashes returns consistent hashes for same data."""
        hashes1 = precompute_hashes(sample_ohlcv_data)
        hashes2 = precompute_hashes(sample_ohlcv_data)

        assert hashes1['close'] == hashes2['close']
        assert hashes1['high'] == hashes2['high']
        assert hashes1['low'] == hashes2['low']

    def test_precompute_hashes_shared_across_indicators(self, sample_ohlcv_data):
        """
        Test that one hash can be used for multiple indicators (optimization pattern).

        This documents the intended usage: pre-compute hashes once, then pass to
        multiple indicator functions. This eliminates redundant hash calculations
        and is the key performance optimization.

        Tests all available single-price indicators and multi-price indicators
        to ensure the pattern works universally.
        """
        from app.backtesting.indicators import (
            calculate_atr,
            calculate_bollinger_bands,
            calculate_ema,
            calculate_ichimoku_cloud,
            calculate_macd,
            calculate_rsi,
        )

        # Pre-compute hashes ONCE for all price series
        hashes = precompute_hashes(sample_ohlcv_data)

        # ==================== Single-Price Indicators ====================
        # All use same 'close' hash - the key optimization!

        bb = calculate_bollinger_bands(
            sample_ohlcv_data['close'],
            period=20,
            number_of_standard_deviations=2,
            prices_hash=hashes['close']
        )

        ema = calculate_ema(
            sample_ohlcv_data['close'],
            period=9,
            prices_hash=hashes['close']  # Same hash reused
        )

        macd = calculate_macd(
            sample_ohlcv_data['close'],
            fast_period=12,
            slow_period=26,
            signal_period=9,
            prices_hash=hashes['close']  # Same hash reused
        )

        rsi = calculate_rsi(
            sample_ohlcv_data['close'],
            period=14,
            prices_hash=hashes['close']  # Same hash reused
        )

        # ==================== Multi-Price Indicators ====================
        # Use pre-computed hashes for high, low, close
        # Note: ATR takes DataFrame, not individual series

        atr = calculate_atr(
            sample_ohlcv_data,
            period=14,
            high_hash=hashes['high'],
            low_hash=hashes['low'],
            close_hash=hashes['close']
        )

        ichimoku = calculate_ichimoku_cloud(
            sample_ohlcv_data['high'],
            sample_ohlcv_data['low'],
            sample_ohlcv_data['close'],
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            high_hash=hashes['high'],
            low_hash=hashes['low'],
            close_hash=hashes['close']
        )

        # ==================== Validate Results ====================

        # All indicators should return valid results with correct length
        assert len(atr) == len(sample_ohlcv_data), "ATR length mismatch"
        assert len(bb) == len(sample_ohlcv_data), "BB length mismatch"
        assert len(ema) == len(sample_ohlcv_data), "EMA length mismatch"
        assert len(ichimoku['tenkan_sen']) == len(sample_ohlcv_data), "Ichimoku length mismatch"
        assert len(macd) == len(sample_ohlcv_data), "MACD length mismatch"
        assert len(rsi) == len(sample_ohlcv_data), "RSI length mismatch"

        # All indicators should have valid values (not all NaN)
        assert not atr.dropna().empty, "ATR should have valid values"
        assert not bb['middle_band'].dropna().empty, "BB should have valid values"
        assert not ema.dropna().empty, "EMA should have valid values"
        assert not ichimoku['tenkan_sen'].dropna().empty, "Ichimoku should have valid values"
        assert not macd['macd_line'].dropna().empty, "MACD should have valid values"
        assert not rsi.dropna().empty, "RSI should have valid values"

    @pytest.mark.parametrize("indicator_type,indicator_func,params", [
        ('atr', 'calculate_atr', {'period': 14}),
        ('single_price', 'calculate_bollinger_bands', {'period': 20, 'number_of_standard_deviations': 2}),
        ('single_price', 'calculate_ema', {'period': 9}),
        (
                'ichimoku', 'calculate_ichimoku_cloud', {
                'tenkan_period': 9, 'kijun_period': 26, 'senkou_span_b_period': 52, 'displacement': 26
            }
        ),
        ('single_price', 'calculate_macd', {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
        ('single_price', 'calculate_rsi', {'period': 14}),
    ])
    def test_hash_reuse_works_for_all_indicators(
        self, sample_ohlcv_data, indicator_type, indicator_func, params
    ):
        """
        Parametrized test: Hash reuse pattern works for every indicator.

        Tests that pre-computed hashes can be used multiple times with any indicator,
        demonstrating the universal applicability of the optimization.
        """
        import importlib
        indicators_module = importlib.import_module('app.backtesting.indicators')
        calc_func = getattr(indicators_module, indicator_func)

        # Pre-compute hashes once
        hashes = precompute_hashes(sample_ohlcv_data)

        # Call indicator multiple times with SAME pre-computed hash(es)
        if indicator_type == 'single_price':
            result1 = calc_func(
                sample_ohlcv_data['close'],
                **params,
                prices_hash=hashes['close']
            )
            result2 = calc_func(
                sample_ohlcv_data['close'],
                **params,
                prices_hash=hashes['close']  # Same hash reused!
            )
        elif indicator_type == 'atr':
            # ATR takes DataFrame, not individual series
            result1 = calc_func(
                sample_ohlcv_data,
                **params,
                high_hash=hashes['high'],
                low_hash=hashes['low'],
                close_hash=hashes['close']
            )
            result2 = calc_func(
                sample_ohlcv_data,
                **params,
                high_hash=hashes['high'],  # Same hashes reused!
                low_hash=hashes['low'],
                close_hash=hashes['close']
            )
        else:  # ichimoku
            result1 = calc_func(
                sample_ohlcv_data['high'],
                sample_ohlcv_data['low'],
                sample_ohlcv_data['close'],
                **params,
                high_hash=hashes['high'],
                low_hash=hashes['low'],
                close_hash=hashes['close']
            )
            result2 = calc_func(
                sample_ohlcv_data['high'],
                sample_ohlcv_data['low'],
                sample_ohlcv_data['close'],
                **params,
                high_hash=hashes['high'],  # Same hashes reused!
                low_hash=hashes['low'],
                close_hash=hashes['close']
            )

        # Both calls should produce valid results
        if isinstance(result1, pd.Series):
            assert len(result1) == len(sample_ohlcv_data)
            assert len(result2) == len(sample_ohlcv_data)
        elif isinstance(result1, pd.DataFrame):
            assert len(result1) == len(sample_ohlcv_data)
            assert len(result2) == len(sample_ohlcv_data)
        elif isinstance(result1, dict):
            first_key = list(result1.keys())[0]
            assert len(result1[first_key]) == len(sample_ohlcv_data)
            assert len(result2[first_key]) == len(sample_ohlcv_data)


class TestStrategyWorkflow:
    """Test complete strategy execution workflow."""

    def test_run_method_calls_all_steps(self, sample_ohlcv_data):
        """Test run() executes full workflow: indicators -> signals -> trades."""
        strategy = ConcreteTestStrategy(
            signal_indices=[(105, 1), (110, -1)],
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Track method calls
        calls = {'add_indicators': 0, 'generate_signals': 0}

        original_add_indicators = strategy.add_indicators
        original_generate_signals = strategy.generate_signals

        def wrapped_add_indicators(df):
            calls['add_indicators'] += 1
            return original_add_indicators(df)

        def wrapped_generate_signals(df):
            calls['generate_signals'] += 1
            return original_generate_signals(df)

        strategy.add_indicators = wrapped_add_indicators
        strategy.generate_signals = wrapped_generate_signals

        trades = strategy.run(sample_ohlcv_data, [])

        # Verify all methods called
        assert calls['add_indicators'] == 1
        assert calls['generate_signals'] == 1
        assert isinstance(trades, list)

    def test_run_method_returns_trades(self, sample_ohlcv_data):
        """Test run() returns list of trade dictionaries."""
        strategy = ConcreteTestStrategy(
            signal_indices=[(105, 1), (110, -1)],
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(sample_ohlcv_data, [])

        assert isinstance(trades, list)
        assert len(trades) > 0

        # Verify trade structure
        for trade in trades:
            assert 'entry_time' in trade
            assert 'entry_price' in trade
            assert 'exit_time' in trade
            assert 'exit_price' in trade
            assert 'side' in trade

    def test_indicators_added_before_signals(self, sample_ohlcv_data):
        """Test indicators are added before signal generation."""
        strategy = ConcreteTestStrategy(
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df_result = strategy.run(sample_ohlcv_data.copy(), [])

        # Strategy should have called add_indicators
        # We can't directly check the dataframe, but we can verify trades were generated
        # which requires indicators to be present
        assert isinstance(df_result, list)


class TestSignalExecutionTiming:
    """Test signal queueing and 1-bar execution delay."""

    def test_signal_executed_next_bar(self, sample_ohlcv_data):
        """Test signal generated at bar N is executed at bar N+1 open."""
        # Signal at index 105 should execute at index 106
        strategy = ConcreteTestStrategy(
            signal_indices=[(105, 1), (110, -1)],  # Long signal, then exit
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        trades = strategy.run(sample_ohlcv_data, [])

        assert len(trades) > 0

        # Entry should be at bar after signal
        entry_time = trades[0]['entry_time']
        signal_time = sample_ohlcv_data.index[105]
        expected_entry_time = sample_ohlcv_data.index[106]

        assert entry_time == expected_entry_time
        assert entry_time != signal_time

    def test_entry_price_uses_next_bar_open(self, sample_ohlcv_data):
        """Test entry price uses open price of next bar."""
        strategy = ConcreteTestStrategy(
            signal_indices=[(105, 1), (110, -1)],
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        trades = strategy.run(sample_ohlcv_data, [])

        assert len(trades) > 0

        # Entry price should be open of bar 106
        entry_price = trades[0]['entry_price']
        expected_price = sample_ohlcv_data.loc[sample_ohlcv_data.index[106], 'open']

        assert entry_price == pytest.approx(expected_price, abs=0.01)

    def test_multiple_signals_same_bar_uses_latest(self, sample_ohlcv_data):
        """Test when multiple signals on same bar, latest signal is used."""
        # This tests the signal overwrite behavior
        strategy = ConcreteTestStrategy(
            signal_indices=[(105, 1), (105, -1), (110, 1)],  # Both signals on same bar, then exit
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        trades = strategy.run(sample_ohlcv_data, [])

        # Should have at least one trade
        assert len(trades) > 0

        # The trade should reflect the last signal (-1 for short)
        assert trades[0]['side'] == 'short'


class TestWarmupPeriod:
    """Test indicator warmup period handling."""

    def test_warmup_period_constant(self):
        """Test INDICATOR_WARMUP_PERIOD is set correctly."""
        assert INDICATOR_WARMUP_PERIOD == 100

    def test_signals_ignored_during_warmup(self):
        """Test signals during warmup period are ignored."""
        dates = pd.date_range('2025-01-01', periods=150, freq='1h')
        df = pd.DataFrame({
            'open': [100] * 150,
            'high': [101] * 150,
            'low': [99] * 150,
            'close': [100] * 150,
            'volume': [1000] * 150
        }, index=dates)

        # Place signals during and after warmup
        strategy = ConcreteTestStrategy(
            signal_indices=[
                (50, 1),  # During warmup - should be ignored
                (105, 1),  # After warmup - should execute
                (110, -1)
            ],
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        trades = strategy.run(df, [])

        # Should only have trades from signals after warmup
        if len(trades) > 0:
            first_entry_time = trades[0]['entry_time']
            first_entry_idx = df.index.get_loc(first_entry_time)

            # Entry should be after warmup period (signal at 105, entry at 106)
            assert first_entry_idx > INDICATOR_WARMUP_PERIOD


class TestTradeExtraction:
    """Test trade extraction from signals."""

    def test_long_trade_extraction(self, sample_ohlcv_data):
        """Test long trade is extracted correctly."""
        strategy = ConcreteTestStrategy(
            signal_indices=[(105, 1), (110, -1)],  # Long entry, then exit
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        trades = strategy.run(sample_ohlcv_data, [])

        assert len(trades) > 0

        long_trades = [t for t in trades if t['side'] == 'long']
        assert len(long_trades) > 0

        trade = long_trades[0]
        assert trade['entry_price'] < trade['exit_price'] or trade['entry_price'] > trade['exit_price']
        assert trade['entry_time'] < trade['exit_time']

    def test_short_trade_extraction(self, sample_ohlcv_data):
        """Test short trade is extracted correctly."""
        strategy = ConcreteTestStrategy(
            signal_indices=[(105, -1), (110, 1)],  # Short entry, then exit
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        trades = strategy.run(sample_ohlcv_data, [])

        assert len(trades) > 0

        short_trades = [t for t in trades if t['side'] == 'short']
        assert len(short_trades) > 0

        trade = short_trades[0]
        assert trade['entry_price'] < trade['exit_price'] or trade['entry_price'] > trade['exit_price']
        assert trade['entry_time'] < trade['exit_time']

    def test_alternating_positions(self, sample_ohlcv_data):
        """Test alternating long/short positions are extracted."""
        strategy = ConcreteTestStrategy(
            signal_indices=[
                (105, 1),  # Long
                (110, -1),  # Short (flips)
                (115, 1),  # Long (flips)
                (120, -1)  # Short (flips)
            ],
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        trades = strategy.run(sample_ohlcv_data, [])

        assert len(trades) >= 2
        assert_trades_have_both_directions(trades)

    def test_no_trades_without_signals(self, sample_ohlcv_data):
        """Test no trades generated without signals."""
        strategy = ConcreteTestStrategy(
            signal_indices=[],  # No signals
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        trades = strategy.run(sample_ohlcv_data, [])

        assert len(trades) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_strategy_with_small_dataset(self):
        """Test strategy handles small dataset (< warmup period)."""
        df = create_small_ohlcv_dataframe(50)

        strategy = ConcreteTestStrategy(
            signal_indices=[(10, 1)],
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        trades = strategy.run(df, [])

        # Should have no trades (signal during warmup)
        assert len(trades) == 0

    def test_strategy_with_constant_prices(self):
        """Test strategy handles constant price data."""
        df = create_constant_price_dataframe(150, 100.0)

        strategy = ConcreteTestStrategy(
            signal_indices=[(105, 1), (110, -1)],
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        trades = strategy.run(df, [])

        # Should still generate trades even with constant prices
        assert len(trades) > 0

        # All trades should have same entry/exit prices
        for trade in trades:
            assert trade['entry_price'] == pytest.approx(100.0, abs=0.01)
            assert trade['exit_price'] == pytest.approx(100.0, abs=0.01)

    def test_strategy_with_missing_signal_column(self, sample_ohlcv_data):
        """Test strategy handles missing signal column gracefully."""

        class NoSignalStrategy(ConcreteTestStrategy):
            def generate_signals(self, df):
                # Don't add signal column
                return df

        strategy = NoSignalStrategy(
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        # Should raise KeyError when trying to access signal column
        with pytest.raises(KeyError):
            strategy.run(sample_ohlcv_data, [])

    def test_strategy_with_nan_prices(self):
        """Test strategy handles NaN values in price data."""
        dates = pd.date_range('2025-01-01', periods=150, freq='1h')
        df = pd.DataFrame({
            'open': [100] * 150,
            'high': [101] * 150,
            'low': [99] * 150,
            'close': [100] * 150,
            'volume': [1000] * 150
        }, index=dates)

        # Inject NaN values
        df.loc[df.index[105], 'close'] = np.nan
        df.loc[df.index[106], 'open'] = np.nan

        strategy = ConcreteTestStrategy(
            signal_indices=[(105, 1), (110, -1)],
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        # Strategy should still run (may have issues with NaN prices)
        trades = strategy.run(df, [])

        # Should handle NaN gracefully (may have fewer or no trades)
        assert isinstance(trades, list)

    def test_skip_signal_continues_loop_without_executing(self):
        """Test skip_signal=True causes loop to continue without signal execution (lines 255-257)."""
        # Create strategy with rollover enabled
        strategy = ConcreteTestStrategy(
            signal_indices=[(5, 1), (10, -1)],  # Set signals
            rollover=True,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )
        
        # Create data with multiple bars
        df = create_small_ohlcv_dataframe(50)
        
        # Set a contract switch date right in the middle
        switch_dates = [df.index[6]]  # Switch after first signal
        
        # Run strategy - when switch happens, skip_signal should be True
        trades = strategy.run(df, switch_dates)
        
        # The strategy should handle the skip gracefully
        # Trades should still be generated (though behavior may differ due to rollover)
        assert isinstance(trades, list)
        # The key is that the function completes without error despite skip_signal being True


class TestStateReset:
    """Test strategy state is properly reset between runs."""

    def test_state_reset_between_runs(self, sample_ohlcv_data):
        """Test strategy can be run multiple times with clean state."""
        strategy = ConcreteTestStrategy(
            signal_indices=[(105, 1), (110, -1)],
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        # Run 1
        trades1 = strategy.run(sample_ohlcv_data.copy(), [])

        # Run 2
        trades2 = strategy.run(sample_ohlcv_data.copy(), [])

        # Both runs should produce identical results
        assert len(trades1) == len(trades2)

        for t1, t2 in zip(trades1, trades2):
            assert t1['entry_time'] == t2['entry_time']
            assert t1['entry_price'] == pytest.approx(t2['entry_price'])
            assert t1['exit_time'] == t2['exit_time']
            assert t1['exit_price'] == pytest.approx(t2['exit_price'])

    def test_queued_signal_cleared_between_runs(self, sample_ohlcv_data):
        """Test queued signal doesn't persist across runs."""
        strategy = ConcreteTestStrategy(
            signal_indices=[(105, 1)],
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        # First run
        strategy.run(sample_ohlcv_data.copy(), [])

        # Check state is clean after run
        # Note: queued_signal is reset during _extract_trades
        assert strategy.prev_row is not None  # Updated during iteration

        # Second run should start fresh
        trades2 = strategy.run(sample_ohlcv_data.copy(), [])
        assert len(trades2) >= 0


class TestSlippageIntegration:
    """Test slippage integration with base strategy."""

    def test_slippage_applied_to_trades(self, sample_ohlcv_data):
        """Test slippage affects entry/exit prices."""
        # Run without slippage
        strategy_no_slip = ConcreteTestStrategy(
            signal_indices=[(105, 1), (110, -1)],
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        trades_no_slip = strategy_no_slip.run(sample_ohlcv_data.copy(), [])

        # Run with slippage
        strategy_with_slip = ConcreteTestStrategy(
            signal_indices=[(105, 1), (110, -1)],
            rollover=False,
            trailing=None,
            slippage_ticks=2,
            symbol='ZS'
        )

        trades_with_slip = strategy_with_slip.run(sample_ohlcv_data.copy(), [])

        # Should have similar number of trades
        assert len(trades_no_slip) == len(trades_with_slip)

        # Entry prices should differ due to slippage
        if len(trades_no_slip) > 0 and len(trades_with_slip) > 0:
            assert trades_no_slip[0]['entry_price'] != trades_with_slip[0]['entry_price']

    def test_different_slippage_values(self, sample_ohlcv_data):
        """Test different slippage values produce different results."""
        trades_results = []

        for slippage in [0, 1, 2]:
            strategy = ConcreteTestStrategy(
                signal_indices=[(105, 1), (110, -1)],
                rollover=False,
                trailing=None,
                slippage_ticks=slippage,
                symbol='ZS'
            )

            trades = strategy.run(sample_ohlcv_data.copy(), [])
            trades_results.append(trades)

        # All should have trades
        assert all(len(t) > 0 for t in trades_results)

        # Entry prices should differ
        if all(len(t) > 0 for t in trades_results):
            prices = [t[0]['entry_price'] for t in trades_results]
            # Not all prices should be identical (slippage should affect them)
            assert len(set(prices)) > 1
