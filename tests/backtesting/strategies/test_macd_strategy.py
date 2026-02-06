"""
MACD Strategy Test Suite.

Tests MACD strategy implementation with real historical data:
- Strategy initialization and configuration
- Indicator calculation within strategy (MACD line, signal line, histogram)
- Signal generation logic (MACD/signal crossovers)
- Full strategy execution and trade generation
- Edge cases and error handling

Uses real market data (ZS, CL) from data/historical_data/.
"""
import pandas as pd
import pytest

from app.backtesting.strategies import MACDStrategy
from tests.backtesting.helpers.assertions import (
    assert_valid_indicator,
    assert_valid_signals,
    assert_valid_trades,
    assert_no_overlapping_trades
)
from tests.backtesting.strategies.strategy_test_utils import (
    assert_strategy_basic_attributes,
    assert_trailing_stop_configured,
    assert_rollover_configured,
    assert_strategy_name_contains,
    assert_trades_have_both_directions,
    assert_similar_trade_count,
    assert_signals_convert_to_trades,
    assert_both_signal_types_present,
    assert_minimal_warmup_signals,
    assert_faster_params_generate_more_trades,
    assert_different_indicator_patterns,
    assert_indicator_columns_exist,
    create_small_ohlcv_dataframe,
    create_constant_price_dataframe,
    create_gapped_dataframe,
    get_common_backtest_configs,
)


# ==================== Test Strategy Initialization ====================

class TestMACDStrategyInitialization:
    """Test MACD strategy initialization and configuration."""

    @pytest.mark.parametrize("fast,slow,signal,description", [
        (12, 26, 9, "standard"),
        (8, 17, 9, "faster"),
        (19, 39, 9, "slower"),
        (12, 26, 6, "faster_signal"),
    ])
    def test_initialization_with_various_parameters(self, fast, slow, signal, description):
        """Test MACD strategy initializes correctly with various period combinations."""
        strategy = MACDStrategy(
            fast_period=fast,
            slow_period=slow,
            signal_period=signal,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert strategy.fast_period == fast
        assert strategy.slow_period == slow
        assert strategy.signal_period == signal
        assert_strategy_basic_attributes(strategy, rollover=False, trailing=None, slippage_ticks=1)

    def test_initialization_with_trailing_stop(self):
        """Test MACD strategy with trailing stop enabled."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=2.0,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert_trailing_stop_configured(strategy, 2.0)

    def test_initialization_with_rollover_enabled(self):
        """Test MACD strategy with contract rollover handling."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert_rollover_configured(strategy)

    def test_format_name_generates_correct_string(self):
        """Test strategy name formatting for identification."""
        name = MACDStrategy.format_name(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1
        )

        assert_strategy_name_contains(name, 'MACD', 'fast=12', 'slow=26', 'signal=9', 'rollover=False')


# ==================== Test Indicator Calculation ====================

class TestMACDStrategyIndicators:
    """Test indicator calculation within MACD strategy."""

    def test_add_indicators_creates_macd_columns(self, zs_1h_data):
        """Test that add_indicators properly calculates MACD components on real data."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())

        # All MACD columns should be added
        assert_indicator_columns_exist(df, 'macd_line', 'signal_line', 'histogram')

        # Validate MACD values (can be negative)
        assert_valid_indicator(df['macd_line'], 'MACD_line', allow_nan=True)
        assert_valid_indicator(df['signal_line'], 'MACD_signal', allow_nan=True)
        assert_valid_indicator(df['histogram'], 'MACD_histogram', allow_nan=True)

    def test_histogram_equals_macd_minus_signal(self, zs_1h_data):
        """Test that histogram correctly represents MACD line - signal line."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())

        # Remove NaN values for comparison
        valid_data = df.dropna(subset=['macd_line', 'signal_line', 'histogram'])

        # Histogram should equal MACD line - signal line
        calculated_histogram = valid_data['macd_line'] - valid_data['signal_line']

        # Use allclose for floating point comparison
        assert pd.Series(calculated_histogram).equals(valid_data['histogram']) or \
               (calculated_histogram - valid_data['histogram']).abs().max() < 1e-10, \
            "Histogram should equal MACD line - signal line"

    def test_fast_vs_slow_macd_generates_different_patterns(self, zs_1h_data):
        """Test that different MACD parameters create different indicator patterns."""
        strategy_fast = MACDStrategy(
            fast_period=8,
            slow_period=17,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_slow = MACDStrategy(
            fast_period=19,
            slow_period=39,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df_fast = strategy_fast.add_indicators(zs_1h_data.copy())
        df_slow = strategy_slow.add_indicators(zs_1h_data.copy())

        # MACD values should be different
        assert_different_indicator_patterns(
            df_fast['macd_line'],
            df_slow['macd_line'],
            min_difference=0.1,
            indicator_name='MACD'
        )

    def test_macd_calculation_on_different_symbols(self, zs_1h_data, cl_15m_data):
        """Test MACD calculation works on different market data."""
        strategy_zs = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_cl = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='CL'
        )

        df_zs = strategy_zs.add_indicators(zs_1h_data.copy())
        df_cl = strategy_cl.add_indicators(cl_15m_data.copy())

        # Both should have valid MACD
        assert_valid_indicator(df_zs['macd_line'], 'MACD_ZS')
        assert_valid_indicator(df_zs['signal_line'], 'Signal_ZS')
        assert_valid_indicator(df_cl['macd_line'], 'MACD_CL')
        assert_valid_indicator(df_cl['signal_line'], 'Signal_CL')


# ==================== Test Signal Generation ====================

class TestMACDStrategySignals:
    """Test signal generation logic for MACD strategy."""

    def test_generate_signals_creates_signal_column(self, zs_1h_data):
        """Test that generate_signals creates signal column."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        assert 'signal' in df.columns
        assert_valid_signals(df)

    def test_long_entry_signal_on_bullish_crossover(self, zs_1h_data):
        """Test long signals occur when MACD line crosses above signal line."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Find long signal bars
        long_signal_bars = df[df['signal'] == 1]

        # Long signals should occur when MACD > signal (after crossover)
        assert len(long_signal_bars) > 0, "Expected long signals on 2-year real data"
        assert (long_signal_bars['macd_line'] > long_signal_bars['signal_line']).all(), \
            "Long signals should occur when MACD line > signal line"

    def test_short_entry_signal_on_bearish_crossover(self, zs_1h_data):
        """Test short signals occur when MACD line crosses below signal line."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Find short signal bars
        short_signal_bars = df[df['signal'] == -1]

        # Short signals should occur when MACD < signal (after crossover)
        assert len(short_signal_bars) > 0, "Expected short signals on 2-year real data"
        assert (short_signal_bars['macd_line'] < short_signal_bars['signal_line']).all(), \
            "Short signals should occur when MACD line < signal line"

    def test_histogram_sign_matches_crossover_direction(self, zs_1h_data):
        """Test that histogram sign indicates which line is on top."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # When MACD > signal, histogram should be positive
        macd_above = df[df['macd_line'] > df['signal_line']]
        if len(macd_above) > 0:
            assert (macd_above['histogram'] > 0).all() or \
                   (macd_above['histogram'].abs() < 1e-10).all(), \
                "Histogram should be positive when MACD > signal"

        # When MACD < signal, histogram should be negative
        macd_below = df[df['macd_line'] < df['signal_line']]
        if len(macd_below) > 0:
            assert (macd_below['histogram'] < 0).all() or \
                   (macd_below['histogram'].abs() < 1e-10).all(), \
                "Histogram should be negative when MACD < signal"

    def test_fast_vs_slow_macd_generate_different_signal_frequency(self, zs_1h_data):
        """Test that faster MACD generates more crossover signals."""
        strategy_fast = MACDStrategy(
            fast_period=8,
            slow_period=17,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_slow = MACDStrategy(
            fast_period=19,
            slow_period=39,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df_fast = strategy_fast.add_indicators(zs_1h_data.copy())
        df_fast = strategy_fast.generate_signals(df_fast)

        df_slow = strategy_slow.add_indicators(zs_1h_data.copy())
        df_slow = strategy_slow.generate_signals(df_slow)

        # Fast MACD should generate more signals
        fast_signal_count = (df_fast['signal'] != 0).sum()
        slow_signal_count = (df_slow['signal'] != 0).sum()

        assert fast_signal_count >= slow_signal_count, \
            f"Fast MACD should generate more signals ({fast_signal_count} vs {slow_signal_count})"

    def test_both_long_and_short_signals_present(self, zs_1h_data):
        """Test that strategy generates both long and short signals."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        assert_both_signal_types_present(df)

    def test_no_signals_generated_during_macd_warmup_period(self, zs_1h_data):
        """Test that no signals are generated while MACD is stabilizing."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # During warmup (first ~35 bars for slow + signal period), signals should be minimal
        assert_minimal_warmup_signals(df, warmup_bars=35, max_warmup_signals=2)


# ==================== Test Strategy Execution ====================

class TestMACDStrategyExecution:
    """Test full strategy execution with trade generation."""

    @pytest.mark.parametrize("symbol,interval,trailing,description", get_common_backtest_configs())
    def test_backtest_execution_variants(
        self, symbol, interval, trailing, description,
        load_real_data, contract_switch_dates
    ):
        """Test MACD strategy backtest with various configurations and data sources."""
        data = load_real_data('1!', symbol, interval)

        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=trailing,
            slippage_ticks=1,
            symbol=symbol
        )

        trades = strategy.run(data.copy(), contract_switch_dates.get(symbol, []))

        assert len(trades) > 0, f"Expected trades for {symbol} {interval} (config: {description})"
        assert_valid_trades(trades)

    def test_trades_have_both_long_and_short_positions(self, zs_1h_data, contract_switch_dates):
        """Test that backtest generates both long and short trades."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_trades_have_both_directions(trades)

    def test_trades_do_not_overlap(self, zs_1h_data, contract_switch_dates):
        """Test that trades don't overlap (proper position management)."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_no_overlapping_trades(trades)

    def test_backtest_with_slippage_affects_prices(self, zs_1h_data, contract_switch_dates):
        """Test that slippage is properly applied to trade prices."""
        strategy_no_slip = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        strategy_with_slip = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=2,
            symbol='ZS'
        )

        trades_no_slip = strategy_no_slip.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))
        trades_with_slip = strategy_with_slip.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_similar_trade_count(trades_no_slip, trades_with_slip, max_difference=5)

    def test_backtest_with_different_macd_periods(self, zs_1h_data, contract_switch_dates):
        """Test that different MACD periods produce different trade patterns."""
        strategy_fast = MACDStrategy(
            fast_period=8,
            slow_period=17,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_slow = MACDStrategy(
            fast_period=19,
            slow_period=39,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades_fast = strategy_fast.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))
        trades_slow = strategy_slow.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_faster_params_generate_more_trades(trades_fast, trades_slow, "MACD period")

    def test_signals_convert_to_actual_trades(self, zs_1h_data, contract_switch_dates):
        """Test that generated signals result in actual trades."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)
        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_signals_convert_to_trades(df, trades)


# ==================== Test Edge Cases ====================

class TestMACDStrategyEdgeCases:
    """Test edge cases and error handling."""

    def test_strategy_with_insufficient_data(self):
        """Test strategy behavior with insufficient data for MACD calculation."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create small dataset using utility
        small_data = create_small_ohlcv_dataframe(bars=3, base_price=100)

        df = strategy.add_indicators(small_data.copy())

        # MACD should be mostly NaN for insufficient data
        assert df['macd_line'].isna().all()
        assert df['signal_line'].isna().all()
        assert df['histogram'].isna().all()

    def test_strategy_with_constant_prices(self):
        """Test strategy with constant prices (no volatility)."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create constant price data using utility
        constant_data = create_constant_price_dataframe(bars=50, price=100)

        df = strategy.add_indicators(constant_data.copy())
        df = strategy.generate_signals(df)

        # With constant prices, MACD should be zero and no crossovers occur
        valid_macd = df['macd_line'].dropna()
        if len(valid_macd) > 0:
            assert (valid_macd.abs() < 1e-10).all(), "MACD should be zero for constant prices"

        assert (df['signal'] == 0).all()

    def test_strategy_with_extreme_volatility(self, volatile_market_data):
        """Test strategy handles extreme volatility."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(volatile_market_data.copy())
        df = strategy.generate_signals(df)

        # Should still produce valid MACD and signals
        assert_valid_indicator(df['macd_line'], 'MACD')
        assert_valid_indicator(df['signal_line'], 'Signal')
        assert_valid_signals(df)

    def test_strategy_with_trending_market(self, trending_market_data):
        """Test strategy in strong trending market."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(trending_market_data.copy())
        df = strategy.generate_signals(df)

        # In trending market, MACD histogram should show persistent direction
        assert_valid_signals(df)

    def test_strategy_handles_gaps_in_data(self, zs_1h_data):
        """Test strategy handles missing bars in data."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create data with gap using utility
        gapped_data = create_gapped_dataframe(zs_1h_data, gap_start=100, gap_end=150)

        df = strategy.add_indicators(gapped_data.copy())
        df = strategy.generate_signals(df)

        # Should still calculate MACD and signals
        assert_indicator_columns_exist(df, 'macd_line', 'signal_line', 'histogram', 'signal')
        assert_valid_signals(df)
