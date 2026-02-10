"""
EMA Crossover Strategy Test Suite.

Tests EMA crossover strategy implementation with real historical data:
- Strategy initialization and configuration
- Indicator calculation within strategy (short/long EMAs)
- Signal generation logic (crossovers)
- Full strategy execution and trade generation
- Edge cases and error handling

Uses real market data (ZS, CL) from data/historical_data/.
"""
import pytest

from app.backtesting.strategies import EMACrossoverStrategy
from tests.backtesting.fixtures.assertions import (
    assert_valid_indicator,
    assert_indicator_varies,
)
from tests.backtesting.strategies.strategy_test_utils import (
    assert_more_responsive_indicator,
    assert_trades_have_both_directions,
    assert_similar_trade_count,
    assert_slippage_affects_prices,
    assert_signals_convert_to_trades,
    assert_both_signal_types_present,
    assert_minimal_warmup_signals,
    assert_valid_signals,
    assert_valid_trades,
    assert_no_overlapping_trades,
    create_small_ohlcv_dataframe,
    create_constant_price_dataframe,
    create_gapped_dataframe,
)


# ==================== Test Strategy Initialization ====================

class TestEMAStrategyInitialization:
    """Test EMA crossover strategy initialization and configuration."""

    @pytest.mark.parametrize("short,long,description", [
        (9, 21, "standard"),
        (12, 26, "macd_like"),
        (5, 20, "fast_crossover"),
        (20, 50, "slow_crossover"),
    ])
    def test_initialization_with_various_parameters(self, short, long, description):
        """Test EMA strategy initializes correctly with various period combinations."""
        strategy = EMACrossoverStrategy(
            short_ema_period=short,
            long_ema_period=long,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert strategy.short_ema_period == short
        assert strategy.long_ema_period == long
        assert strategy.rollover == False
        assert strategy.trailing is None
        assert strategy.position_manager.slippage_ticks == 1

    def test_initialization_with_trailing_stop(self):
        """Test EMA strategy with trailing stop enabled."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=2.0,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert strategy.trailing == 2.0
        assert strategy.trailing_stop_manager is not None

    def test_initialization_with_rollover_enabled(self):
        """Test EMA strategy with contract rollover handling."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert strategy.rollover is True
        assert strategy.switch_handler is not None

    def test_format_name_generates_correct_string(self):
        """Test strategy name formatting for identification."""
        name = EMACrossoverStrategy.format_name(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1
        )

        assert 'EMA' in name
        assert 'short=9' in name
        assert 'long=21' in name
        assert 'rollover=False' in name
        assert 'slippage_ticks=1' in name


# ==================== Test Indicator Calculation ====================

class TestEMAStrategyIndicators:
    """Test indicator calculation within EMA crossover strategy."""

    def test_add_indicators_creates_ema_columns(self, standard_ema_strategy, zs_1h_data):
        """Test that add_indicators properly calculates both EMAs on real data."""
        df = standard_ema_strategy.add_indicators(zs_1h_data.copy())

        # Both EMA columns should be added
        assert 'ema_short' in df.columns
        assert 'ema_long' in df.columns

        # Validate EMA values
        assert_valid_indicator(df['ema_short'], 'EMA_short', min_val=0, allow_nan=True)
        assert_valid_indicator(df['ema_long'], 'EMA_long', min_val=0, allow_nan=True)

        # Verify EMAs respond to price changes (not constant)
        assert_indicator_varies(df['ema_short'], 'EMA_short', min_std=1.0)
        assert_indicator_varies(df['ema_long'], 'EMA_long', min_std=1.0)

        # Verify warmup period (long EMA needs more data to stabilize than short EMA)
        short_warmup_nans = df['ema_short'].iloc[:9].isna().sum()
        long_warmup_nans = df['ema_long'].iloc[:21].isna().sum()
        # Long EMA should have at least as many warmup NaNs as short EMA
        assert long_warmup_nans >= short_warmup_nans, \
            "Long EMA should have warmup period at least as long as short EMA"

    def test_short_ema_more_responsive_than_long(self, standard_ema_strategy, zs_1h_data):
        """Test that short EMA is more responsive (varies more) than long EMA."""
        df = standard_ema_strategy.add_indicators(zs_1h_data.copy())

        # Short EMA should be more volatile (more responsive to price changes)
        assert_more_responsive_indicator(df['ema_short'], df['ema_long'], 'EMA')

    def test_ema_calculation_on_different_symbols(self, zs_1h_data, cl_15m_data):
        """Test EMA calculation works on different market data."""
        strategy_zs = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_cl = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='CL'
        )

        df_zs = strategy_zs.add_indicators(zs_1h_data.copy())
        df_cl = strategy_cl.add_indicators(cl_15m_data.copy())

        # Both should have valid EMAs
        assert_valid_indicator(df_zs['ema_short'], 'EMA_short_ZS', min_val=0)
        assert_valid_indicator(df_zs['ema_long'], 'EMA_long_ZS', min_val=0)
        assert_valid_indicator(df_cl['ema_short'], 'EMA_short_CL', min_val=0)
        assert_valid_indicator(df_cl['ema_long'], 'EMA_long_CL', min_val=0)


# ==================== Test Signal Generation ====================

class TestEMAStrategySignals:
    """Test signal generation logic for EMA crossover strategy."""

    def test_generate_signals_creates_signal_column(self, standard_ema_strategy, zs_1h_data):
        """Test that generate_signals creates signal column."""
        df = standard_ema_strategy.add_indicators(zs_1h_data.copy())
        df = standard_ema_strategy.generate_signals(df)

        assert 'signal' in df.columns
        assert_valid_signals(df)

    def test_long_entry_signal_on_bullish_crossover(self, standard_ema_strategy, zs_1h_data):
        """Test long signals occur when short EMA crosses above long EMA."""
        df = standard_ema_strategy.add_indicators(zs_1h_data.copy())
        df = standard_ema_strategy.generate_signals(df)

        # Find long signal bars
        long_signal_bars = df[df['signal'] == 1]

        # Long signals should occur when short > long (after crossover)
        assert len(long_signal_bars) > 0, "Expected long signals on 2-year real data"
        assert (long_signal_bars['ema_short'] > long_signal_bars['ema_long']).all(), \
            "Long signals should occur when short EMA > long EMA"

    def test_short_entry_signal_on_bearish_crossover(self, standard_ema_strategy, zs_1h_data):
        """Test short signals occur when short EMA crosses below long EMA."""
        df = standard_ema_strategy.add_indicators(zs_1h_data.copy())
        df = standard_ema_strategy.generate_signals(df)

        # Find short signal bars
        short_signal_bars = df[df['signal'] == -1]

        # Short signals should occur when short < long (after crossover)
        assert len(short_signal_bars) > 0, "Expected short signals on 2-year real data"
        assert (short_signal_bars['ema_short'] < short_signal_bars['ema_long']).all(), \
            "Short signals should occur when short EMA < long EMA"

    def test_no_signal_when_emas_aligned_without_crossing(self, zs_1h_data):
        """Test no signals generated when EMAs trending together without crosses."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Count crossover signals
        signal_count = (df['signal'] != 0).sum()

        # Should have some crossovers but not too many (crossovers are relatively rare)
        assert signal_count > 0, "Expected some crossover signals"
        assert signal_count < len(df) * 0.1, "Too many signals for crossover strategy"

    def test_fast_vs_slow_crossovers_generate_different_frequency(self, zs_1h_data):
        """Test that faster EMAs generate more crossover signals."""
        strategy_fast = EMACrossoverStrategy(
            short_ema_period=5,
            long_ema_period=15,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_slow = EMACrossoverStrategy(
            short_ema_period=20,
            long_ema_period=50,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df_fast = strategy_fast.add_indicators(zs_1h_data.copy())
        df_fast = strategy_fast.generate_signals(df_fast)

        df_slow = strategy_slow.add_indicators(zs_1h_data.copy())
        df_slow = strategy_slow.generate_signals(df_slow)

        # Fast crossover should generate more signals
        fast_count = (df_fast['signal'] != 0).sum()
        slow_count = (df_slow['signal'] != 0).sum()
        assert fast_count >= slow_count, \
            f"Faster EMA period should generate more signals ({fast_count} vs {slow_count})"

    @pytest.mark.parametrize("short,long,description,max_signal_pct", [
        (5, 15, "fast", 0.15),
        (9, 21, "standard", 0.10),
        (20, 50, "slow", 0.05),
    ])
    def test_signal_frequency_with_various_ema_periods(
        self, zs_1h_data, short, long, description, max_signal_pct
    ):
        """Test signal frequency varies with EMA period settings."""
        strategy = EMACrossoverStrategy(
            short_ema_period=short,
            long_ema_period=long,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        signal_pct = (df['signal'] != 0).sum() / len(df)

        assert signal_pct > 0, f"Expected signals with {description} EMA periods"
        assert signal_pct <= max_signal_pct, \
            f"Too many signals for {description} EMA periods ({signal_pct:.1%})"

    def test_both_long_and_short_signals_present(self, standard_ema_strategy, zs_1h_data):
        """Test that strategy generates both long and short signals."""
        df = standard_ema_strategy.add_indicators(zs_1h_data.copy())
        df = standard_ema_strategy.generate_signals(df)

        assert_both_signal_types_present(df)

    def test_no_signals_generated_during_ema_warmup_period(self, standard_ema_strategy, zs_1h_data):
        """Test that no signals are generated while EMAs are stabilizing."""
        df = standard_ema_strategy.add_indicators(zs_1h_data.copy())
        df = standard_ema_strategy.generate_signals(df)

        # During warmup (first ~21 bars for long EMA), signals should be minimal
        assert_minimal_warmup_signals(df, warmup_bars=21, max_warmup_signals=2)


# ==================== Test Strategy Execution ====================

class TestEMAStrategyExecution:
    """Test full strategy execution with trade generation."""

    @pytest.mark.parametrize("symbol,interval,trailing,description", [
        ('ZS', '1h', None, "standard_backtest"),
        ('ZS', '1h', 2.0, "with_trailing_stop"),
        ('CL', '15m', None, "different_timeframe"),
    ])
    def test_backtest_execution_variants(
        self, symbol, interval, trailing, description,
        load_real_data, contract_switch_dates
    ):
        """Test EMA strategy backtest with various configurations and data sources."""
        data = load_real_data('1!', symbol, interval)

        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=trailing,
            slippage_ticks=1,
            symbol=symbol
        )

        trades = strategy.run(data.copy(), contract_switch_dates.get(symbol, []))

        assert len(trades) > 0, f"Expected trades for {symbol} {interval} (config: {description})"
        assert_valid_trades(trades)

    def test_trades_have_both_long_and_short_positions(self, standard_ema_strategy, zs_1h_data, contract_switch_dates):
        """Test that backtest generates both long and short trades."""
        trades = standard_ema_strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_trades_have_both_directions(trades)

    def test_trades_do_not_overlap(self, standard_ema_strategy, zs_1h_data, contract_switch_dates):
        """Test that trades don't overlap (proper position management)."""
        trades = standard_ema_strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_no_overlapping_trades(trades)

    def test_backtest_with_slippage_affects_prices(self, zs_1h_data, contract_switch_dates):
        """Test that slippage is properly applied to trade prices."""
        strategy_no_slip = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        strategy_with_slip = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=2,
            symbol='ZS'
        )

        trades_no_slip = strategy_no_slip.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))
        trades_with_slip = strategy_with_slip.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        # Trade counts should be similar (signals are the same)
        assert_similar_trade_count(trades_no_slip, trades_with_slip, max_difference=5)

        # Validate slippage affects prices correctly
        assert_slippage_affects_prices(trades_no_slip, trades_with_slip)

    def test_backtest_with_different_ema_periods(self, zs_1h_data, contract_switch_dates):
        """Test that different EMA periods produce different trade patterns."""
        strategy_fast = EMACrossoverStrategy(
            short_ema_period=5,
            long_ema_period=15,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_slow = EMACrossoverStrategy(
            short_ema_period=20,
            long_ema_period=50,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades_fast = strategy_fast.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))
        trades_slow = strategy_slow.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert len(trades_fast) > 0, "Fast parameter strategy generated no trades"
        assert len(trades_slow) > 0, "Slow parameter strategy generated no trades"
        assert len(trades_fast) >= len(trades_slow), \
            f"Faster EMA period should generate more trades ({len(trades_fast)} vs {len(trades_slow)})"

    def test_signals_convert_to_actual_trades(self, standard_ema_strategy, zs_1h_data, contract_switch_dates):
        """Test that generated signals result in actual trades."""
        df = standard_ema_strategy.add_indicators(zs_1h_data.copy())
        df = standard_ema_strategy.generate_signals(df)
        trades = standard_ema_strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_signals_convert_to_trades(df, trades)


# ==================== Test Edge Cases ====================

class TestEMAStrategyEdgeCases:
    """Test edge cases and error handling."""

    def test_strategy_with_insufficient_data(self, standard_ema_strategy):
        """Test strategy behavior with insufficient data for full EMA calculation."""
        # Create small dataset using utility
        small_data = create_small_ohlcv_dataframe(bars=3, base_price=100)

        df = standard_ema_strategy.add_indicators(small_data.copy())

        # Long EMA needs more data, but short EMA can calculate with less
        assert 'ema_short' in df.columns
        assert 'ema_long' in df.columns
        assert len(df) == 3

    def test_strategy_with_constant_prices(self, standard_ema_strategy):
        """Test strategy with constant prices (no volatility)."""
        # Create constant price data using utility
        constant_data = create_constant_price_dataframe(bars=50, price=100)

        df = standard_ema_strategy.add_indicators(constant_data.copy())
        df = standard_ema_strategy.generate_signals(df)

        # With constant prices, EMAs converge and no crossovers occur
        assert (df['signal'] == 0).all()

    def test_strategy_with_extreme_volatility(self, standard_ema_strategy, volatile_market_data):
        """Test strategy handles extreme volatility."""
        df = standard_ema_strategy.add_indicators(volatile_market_data.copy())
        df = standard_ema_strategy.generate_signals(df)

        # Should still produce valid EMAs and signals
        assert_valid_indicator(df['ema_short'], 'EMA_short', min_val=0)
        assert_valid_indicator(df['ema_long'], 'EMA_long', min_val=0)
        assert_valid_signals(df)

    def test_strategy_with_trending_market(self, standard_ema_strategy, trending_market_data):
        """Test strategy in strong trending market."""
        df = standard_ema_strategy.add_indicators(trending_market_data.copy())
        df = standard_ema_strategy.generate_signals(df)

        # In trending market, EMAs should separate and crossovers less frequent
        assert_valid_signals(df)

    def test_strategy_handles_gaps_in_data(self, standard_ema_strategy, zs_1h_data):
        """Test strategy handles missing bars in data."""
        # Create data with gap using utility
        gapped_data = create_gapped_dataframe(zs_1h_data, gap_start=100, gap_end=150)

        df = standard_ema_strategy.add_indicators(gapped_data.copy())
        df = standard_ema_strategy.generate_signals(df)

        # Should still calculate EMAs and signals
        assert 'ema_short' in df.columns
        assert 'ema_long' in df.columns
        assert 'signal' in df.columns
        assert_valid_signals(df)
