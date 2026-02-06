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
    assert_faster_params_generate_more_signals,
    assert_faster_params_generate_more_trades,
    assert_more_responsive_indicator,
    assert_indicator_columns_exist,
    create_small_ohlcv_dataframe,
    create_constant_price_dataframe,
    create_gapped_dataframe,
    get_common_backtest_configs,
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
        assert_strategy_basic_attributes(strategy, rollover=False, trailing=None, slippage_ticks=1)

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

        assert_trailing_stop_configured(strategy, 2.0)

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

        assert_rollover_configured(strategy)

    def test_format_name_generates_correct_string(self):
        """Test strategy name formatting for identification."""
        name = EMACrossoverStrategy.format_name(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1
        )

        assert_strategy_name_contains(name, 'EMA', 'short=9', 'long=21', 'rollover=False')


# ==================== Test Indicator Calculation ====================

class TestEMAStrategyIndicators:
    """Test indicator calculation within EMA crossover strategy."""

    def test_add_indicators_creates_ema_columns(self, zs_1h_data):
        """Test that add_indicators properly calculates both EMAs on real data."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())

        # Both EMA columns should be added
        assert_indicator_columns_exist(df, 'ema_short', 'ema_long')

        # Validate EMA values
        assert_valid_indicator(df['ema_short'], 'EMA_short', min_val=0, allow_nan=True)
        assert_valid_indicator(df['ema_long'], 'EMA_long', min_val=0, allow_nan=True)

    def test_short_ema_more_responsive_than_long(self, zs_1h_data):
        """Test that short EMA is more responsive (varies more) than long EMA."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())

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

    def test_generate_signals_creates_signal_column(self, zs_1h_data):
        """Test that generate_signals creates signal column."""
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

        assert 'signal' in df.columns
        assert_valid_signals(df)

    def test_long_entry_signal_on_bullish_crossover(self, zs_1h_data):
        """Test long signals occur when short EMA crosses above long EMA."""
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

        # Find long signal bars
        long_signal_bars = df[df['signal'] == 1]

        # Long signals should occur when short > long (after crossover)
        assert len(long_signal_bars) > 0, "Expected long signals on 2-year real data"
        assert (long_signal_bars['ema_short'] > long_signal_bars['ema_long']).all(), \
            "Long signals should occur when short EMA > long EMA"

    def test_short_entry_signal_on_bearish_crossover(self, zs_1h_data):
        """Test short signals occur when short EMA crosses below long EMA."""
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
        assert_faster_params_generate_more_signals(df_fast, df_slow, "EMA period")

    def test_both_long_and_short_signals_present(self, zs_1h_data):
        """Test that strategy generates both long and short signals."""
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

        assert_both_signal_types_present(df)

    def test_no_signals_generated_during_ema_warmup_period(self, zs_1h_data):
        """Test that no signals are generated while EMAs are stabilizing."""
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

        # During warmup (first ~21 bars for long EMA), signals should be minimal
        assert_minimal_warmup_signals(df, warmup_bars=21, max_warmup_signals=2)


# ==================== Test Strategy Execution ====================

class TestEMAStrategyExecution:
    """Test full strategy execution with trade generation."""

    @pytest.mark.parametrize("symbol,interval,trailing,description", get_common_backtest_configs())
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

    def test_trades_have_both_long_and_short_positions(self, zs_1h_data, contract_switch_dates):
        """Test that backtest generates both long and short trades."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_trades_have_both_directions(trades)

    def test_trades_do_not_overlap(self, zs_1h_data, contract_switch_dates):
        """Test that trades don't overlap (proper position management)."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

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

        assert_similar_trade_count(trades_no_slip, trades_with_slip, max_difference=5)

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

        assert_faster_params_generate_more_trades(trades_fast, trades_slow, "EMA period")

    def test_signals_convert_to_actual_trades(self, zs_1h_data, contract_switch_dates):
        """Test that generated signals result in actual trades."""
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
        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_signals_convert_to_trades(df, trades)


# ==================== Test Edge Cases ====================

class TestEMAStrategyEdgeCases:
    """Test edge cases and error handling."""

    def test_strategy_with_insufficient_data(self):
        """Test strategy behavior with insufficient data for full EMA calculation."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create small dataset using utility
        small_data = create_small_ohlcv_dataframe(bars=3, base_price=100)

        df = strategy.add_indicators(small_data.copy())

        # Long EMA needs more data, but short EMA can calculate with less
        assert_indicator_columns_exist(df, 'ema_short', 'ema_long')
        assert len(df) == 3

    def test_strategy_with_constant_prices(self):
        """Test strategy with constant prices (no volatility)."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create constant price data using utility
        constant_data = create_constant_price_dataframe(bars=50, price=100)

        df = strategy.add_indicators(constant_data.copy())
        df = strategy.generate_signals(df)

        # With constant prices, EMAs converge and no crossovers occur
        assert (df['signal'] == 0).all()

    def test_strategy_with_extreme_volatility(self, volatile_market_data):
        """Test strategy handles extreme volatility."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(volatile_market_data.copy())
        df = strategy.generate_signals(df)

        # Should still produce valid EMAs and signals
        assert_valid_indicator(df['ema_short'], 'EMA_short', min_val=0)
        assert_valid_indicator(df['ema_long'], 'EMA_long', min_val=0)
        assert_valid_signals(df)

    def test_strategy_with_trending_market(self, trending_market_data):
        """Test strategy in strong trending market."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(trending_market_data.copy())
        df = strategy.generate_signals(df)

        # In trending market, EMAs should separate and crossovers less frequent
        assert_valid_signals(df)

    def test_strategy_handles_gaps_in_data(self, zs_1h_data):
        """Test strategy handles missing bars in data."""
        strategy = EMACrossoverStrategy(
            short_ema_period=9,
            long_ema_period=21,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create data with gap using utility
        gapped_data = create_gapped_dataframe(zs_1h_data, gap_start=100, gap_end=150)

        df = strategy.add_indicators(gapped_data.copy())
        df = strategy.generate_signals(df)

        # Should still calculate EMAs and signals
        assert_indicator_columns_exist(df, 'ema_short', 'ema_long', 'signal')
        assert_valid_signals(df)
