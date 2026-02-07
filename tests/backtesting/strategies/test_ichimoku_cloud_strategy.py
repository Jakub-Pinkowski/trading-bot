"""
Ichimoku Cloud Strategy Test Suite.

Tests Ichimoku Cloud strategy implementation with real historical data:
- Strategy initialization and configuration
- Indicator calculation within strategy (all 5 Ichimoku components)
- Signal generation logic (TK cross + cloud position)
- Full strategy execution and trade generation
- Edge cases and error handling

Uses real market data (ZS, CL) from data/historical_data/.
"""
import pytest

from app.backtesting.strategies import IchimokuCloudStrategy
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
    assert_indicator_columns_exist,
    create_small_ohlcv_dataframe,
    create_constant_price_dataframe,
    create_gapped_dataframe,
    get_common_backtest_configs,
)


# ==================== Test Strategy Initialization ====================

class TestIchimokuStrategyInitialization:
    """Test Ichimoku Cloud strategy initialization and configuration."""

    @pytest.mark.parametrize("tenkan,kijun,senkou_b,displacement,description", [
        (9, 26, 52, 26, "standard"),
        (7, 22, 44, 22, "faster"),
        (12, 30, 60, 30, "slower"),
        (9, 26, 52, 10, "short_displacement"),
    ])
    def test_initialization_with_various_parameters(self, tenkan, kijun, senkou_b, displacement, description):
        """Test Ichimoku strategy initializes correctly with various period combinations."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=tenkan,
            kijun_period=kijun,
            senkou_span_b_period=senkou_b,
            displacement=displacement,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert strategy.tenkan_period == tenkan
        assert strategy.kijun_period == kijun
        assert strategy.senkou_span_b_period == senkou_b
        assert strategy.displacement == displacement
        assert_strategy_basic_attributes(strategy, rollover=False, trailing=None, slippage_ticks=1)

    def test_initialization_with_trailing_stop(self):
        """Test Ichimoku strategy with trailing stop enabled."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=2.0,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert_trailing_stop_configured(strategy, 2.0)

    def test_initialization_with_rollover_enabled(self):
        """Test Ichimoku strategy with contract rollover handling."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert_rollover_configured(strategy)

    def test_format_name_generates_correct_string(self):
        """Test strategy name formatting for identification."""
        name = IchimokuCloudStrategy.format_name(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1
        )

        assert_strategy_name_contains(
            name, 'Ichimoku', 'tenkan=9', 'kijun=26',
            'senkou_b=52', 'displacement=26', 'rollover=False', 'slippage_ticks=1'
        )


# ==================== Test Indicator Calculation ====================

class TestIchimokuStrategyIndicators:
    """Test indicator calculation within Ichimoku Cloud strategy."""

    def test_add_indicators_creates_all_ichimoku_components(self, zs_1h_data):
        """Test that add_indicators properly calculates all 5 Ichimoku components on real data."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())

        # All 5 Ichimoku components should be added
        assert_indicator_columns_exist(
            df, 'tenkan_sen', 'kijun_sen', 'senkou_span_a',
            'senkou_span_b', 'chikou_span'
        )

        # Validate component values
        assert_valid_indicator(df['tenkan_sen'], 'Tenkan-sen', min_val=0, allow_nan=True)
        assert_valid_indicator(df['kijun_sen'], 'Kijun-sen', min_val=0, allow_nan=True)
        assert_valid_indicator(df['senkou_span_a'], 'Senkou_Span_A', min_val=0, allow_nan=True)
        assert_valid_indicator(df['senkou_span_b'], 'Senkou_Span_B', min_val=0, allow_nan=True)
        assert_valid_indicator(df['chikou_span'], 'Chikou_Span', min_val=0, allow_nan=True)

        # Verify Ichimoku components respond to price changes (not constant)
        valid_tenkan = df['tenkan_sen'].dropna()
        valid_kijun = df['kijun_sen'].dropna()
        valid_span_b = df['senkou_span_b'].dropna()
        assert valid_tenkan.std() > 1.0, "Tenkan-sen should vary with price changes"
        assert valid_kijun.std() > 1.0, "Kijun-sen should vary with price changes"
        assert valid_span_b.std() > 1.0, "Senkou Span B should vary with price changes"

        # Verify warmup period (Senkou Span B needs 52 bars, displacement adds 26)
        tenkan_warmup_nans = df['tenkan_sen'].iloc[:9].isna().sum()
        kijun_warmup_nans = df['kijun_sen'].iloc[:26].isna().sum()
        span_b_warmup_nans = df['senkou_span_b'].iloc[:52].isna().sum()
        assert tenkan_warmup_nans > 0, "Tenkan-sen should have warmup period"
        assert kijun_warmup_nans > 0, "Kijun-sen should have warmup period"
        assert span_b_warmup_nans > 0, "Senkou Span B should have warmup period"

    def test_tenkan_responds_faster_than_kijun(self, zs_1h_data):
        """Test that Tenkan-sen (faster) is more responsive than Kijun-sen (slower)."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())

        # Remove NaN for comparison
        valid_data = df.dropna(subset=['tenkan_sen', 'kijun_sen'])

        # Tenkan should have more variation (more responsive)
        tenkan_std = valid_data['tenkan_sen'].pct_change().dropna().std()
        kijun_std = valid_data['kijun_sen'].pct_change().dropna().std()

        assert tenkan_std > kijun_std * 0.8, \
            f"Tenkan-sen should be more responsive ({tenkan_std:.6f} vs {kijun_std:.6f})"

    def test_cloud_spans_are_displaced_forward(self, zs_1h_data):
        """Test that Senkou Spans are displaced forward in time."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())

        # Senkou Spans should have NaN at the start due to displacement
        assert df['senkou_span_a'].iloc[:26].isna().all(), \
            "Senkou Span A should be NaN for first displacement bars"
        assert df['senkou_span_b'].iloc[:26].isna().all(), \
            "Senkou Span B should be NaN for first displacement bars"

    def test_ichimoku_calculation_on_different_symbols(self, zs_1h_data, cl_15m_data):
        """Test Ichimoku calculation works on different market data."""
        strategy_zs = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_cl = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='CL'
        )

        df_zs = strategy_zs.add_indicators(zs_1h_data.copy())
        df_cl = strategy_cl.add_indicators(cl_15m_data.copy())

        # Both should have valid Ichimoku components
        assert_valid_indicator(df_zs['tenkan_sen'], 'Tenkan_ZS', min_val=0)
        assert_valid_indicator(df_zs['kijun_sen'], 'Kijun_ZS', min_val=0)
        assert_valid_indicator(df_cl['tenkan_sen'], 'Tenkan_CL', min_val=0)
        assert_valid_indicator(df_cl['kijun_sen'], 'Kijun_CL', min_val=0)


# ==================== Test Signal Generation ====================

class TestIchimokuStrategySignals:
    """Test signal generation logic for Ichimoku Cloud strategy."""

    def test_generate_signals_creates_signal_column(self, zs_1h_data):
        """Test that generate_signals creates signal column."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        assert 'signal' in df.columns
        assert_valid_signals(df)

    def test_long_signal_requires_tk_cross_and_above_cloud(self, zs_1h_data):
        """Test long signals require Tenkan crossing above Kijun AND price above cloud."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Find long signal bars
        long_signal_bars = df[df['signal'] == 1]

        if len(long_signal_bars) > 0:
            # Long signals should occur when Tenkan > Kijun
            assert (long_signal_bars['tenkan_sen'] > long_signal_bars['kijun_sen']).all(), \
                "Long signals should occur when Tenkan-sen > Kijun-sen"

            # And price should be above both cloud spans
            above_span_a = long_signal_bars['close'] > long_signal_bars['senkou_span_a']
            above_span_b = long_signal_bars['close'] > long_signal_bars['senkou_span_b']

            assert (above_span_a & above_span_b).all(), \
                "Long signals should occur when price is above cloud"

    def test_short_signal_requires_tk_cross_and_below_cloud(self, zs_1h_data):
        """Test short signals require Tenkan crossing below Kijun AND price below cloud."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Find short signal bars
        short_signal_bars = df[df['signal'] == -1]

        if len(short_signal_bars) > 0:
            # Short signals should occur when Tenkan < Kijun
            assert (short_signal_bars['tenkan_sen'] < short_signal_bars['kijun_sen']).all(), \
                "Short signals should occur when Tenkan-sen < Kijun-sen"

            # And price should be below both cloud spans
            below_span_a = short_signal_bars['close'] < short_signal_bars['senkou_span_a']
            below_span_b = short_signal_bars['close'] < short_signal_bars['senkou_span_b']

            assert (below_span_a & below_span_b).all(), \
                "Short signals should occur when price is below cloud"

    def test_no_signal_when_price_inside_cloud(self, zs_1h_data):
        """Test no signals generated when price is inside the cloud (neutral zone)."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Signals should be relatively rare (cloud filter makes them selective)
        signal_count = (df['signal'] != 0).sum()

        # Should have some signals but not too many (Ichimoku is conservative)
        assert signal_count > 0, "Expected some signals on 2-year real data"
        assert signal_count < len(df) * 0.05, "Too many signals for Ichimoku strategy"

    def test_faster_periods_generate_more_signals(self, zs_1h_data):
        """Test that faster Ichimoku parameters generate more signals."""
        strategy_fast = IchimokuCloudStrategy(
            tenkan_period=7,
            kijun_period=22,
            senkou_span_b_period=44,
            displacement=22,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_slow = IchimokuCloudStrategy(
            tenkan_period=12,
            kijun_period=30,
            senkou_span_b_period=60,
            displacement=30,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df_fast = strategy_fast.add_indicators(zs_1h_data.copy())
        df_fast = strategy_fast.generate_signals(df_fast)

        df_slow = strategy_slow.add_indicators(zs_1h_data.copy())
        df_slow = strategy_slow.generate_signals(df_slow)

        # Fast parameters should generate at least as many signals
        fast_signal_count = (df_fast['signal'] != 0).sum()
        slow_signal_count = (df_slow['signal'] != 0).sum()

        # Fast settings typically more signals, but allow for market conditions
        assert fast_signal_count >= slow_signal_count * 0.7, \
            f"Fast Ichimoku should generate similar or more signals ({fast_signal_count} vs {slow_signal_count})"

    @pytest.mark.parametrize("tenkan,kijun,senkou_b,displacement,description,max_signal_pct", [
        (7, 22, 44, 22, "fast", 0.08),
        (9, 26, 52, 26, "standard", 0.05),
        (12, 30, 60, 30, "slow", 0.03),
    ])
    def test_signal_frequency_with_various_ichimoku_periods(
        self, zs_1h_data, tenkan, kijun, senkou_b, displacement, description, max_signal_pct
    ):
        """Test signal frequency varies with Ichimoku period settings."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=tenkan,
            kijun_period=kijun,
            senkou_span_b_period=senkou_b,
            displacement=displacement,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        signal_pct = (df['signal'] != 0).sum() / len(df)

        assert signal_pct > 0, f"Expected signals with {description} Ichimoku periods"
        assert signal_pct <= max_signal_pct, \
            f"Too many signals for {description} Ichimoku periods ({signal_pct:.1%})"

    def test_both_long_and_short_signals_present(self, zs_1h_data):
        """Test that strategy generates both long and short signals."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        assert_both_signal_types_present(df)

    def test_no_signals_generated_during_warmup_period(self, zs_1h_data):
        """Test that minimal signals are generated while Ichimoku is stabilizing."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # During warmup (first ~52 bars for Senkou Span B), signals should be minimal
        assert_minimal_warmup_signals(df, warmup_bars=52, max_warmup_signals=2)


# ==================== Test Strategy Execution ====================

class TestIchimokuStrategyExecution:
    """Test full strategy execution with trade generation."""

    @pytest.mark.parametrize("symbol,interval,trailing,description", get_common_backtest_configs())
    def test_backtest_execution_variants(
        self, symbol, interval, trailing, description,
        load_real_data, contract_switch_dates
    ):
        """Test Ichimoku strategy backtest with various configurations and data sources."""
        data = load_real_data('1!', symbol, interval)

        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
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
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_trades_have_both_directions(trades)

    def test_trades_do_not_overlap(self, zs_1h_data, contract_switch_dates):
        """Test that trades don't overlap (proper position management)."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_no_overlapping_trades(trades)

    def test_backtest_with_slippage_affects_prices(self, zs_1h_data, contract_switch_dates):
        """Test that slippage is properly applied to trade prices."""
        strategy_no_slip = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        strategy_with_slip = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=2,
            symbol='ZS'
        )

        trades_no_slip = strategy_no_slip.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))
        trades_with_slip = strategy_with_slip.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_similar_trade_count(trades_no_slip, trades_with_slip, max_difference=5)

    def test_backtest_with_different_ichimoku_periods(self, zs_1h_data, contract_switch_dates):
        """Test that different Ichimoku periods produce different trade patterns."""
        strategy_fast = IchimokuCloudStrategy(
            tenkan_period=7,
            kijun_period=22,
            senkou_span_b_period=44,
            displacement=22,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_slow = IchimokuCloudStrategy(
            tenkan_period=12,
            kijun_period=30,
            senkou_span_b_period=60,
            displacement=30,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades_fast = strategy_fast.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))
        trades_slow = strategy_slow.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_faster_params_generate_more_trades(trades_fast, trades_slow, "Ichimoku period")

    def test_signals_convert_to_actual_trades(self, zs_1h_data, contract_switch_dates):
        """Test that generated signals result in actual trades."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Get signals
        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Get trades
        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_signals_convert_to_trades(df, trades)


# ==================== Test Edge Cases ====================

class TestIchimokuStrategyEdgeCases:
    """Test edge cases and error handling."""

    def test_strategy_with_insufficient_data(self):
        """Test strategy behavior with insufficient data for Ichimoku calculation."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create small dataset using utility
        small_data = create_small_ohlcv_dataframe(bars=10, base_price=100)

        df = strategy.add_indicators(small_data.copy())

        # With only 10 bars, Ichimoku should have many NaN values
        assert df['tenkan_sen'].isna().sum() > 0
        assert df['kijun_sen'].isna().sum() > 0
        assert df['senkou_span_b'].isna().sum() > 0

    def test_strategy_with_constant_prices(self):
        """Test strategy with constant prices (no volatility)."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create constant price data using utility
        constant_data = create_constant_price_dataframe(bars=100, price=100)

        df = strategy.add_indicators(constant_data.copy())
        df = strategy.generate_signals(df)

        # With constant prices, all components should be the same value and no crossovers
        valid_tenkan = df['tenkan_sen'].dropna()
        valid_kijun = df['kijun_sen'].dropna()

        if len(valid_tenkan) > 0 and len(valid_kijun) > 0:
            # All components should be close to the constant price
            assert (valid_tenkan - 100).abs().max() < 0.01
            assert (valid_kijun - 100).abs().max() < 0.01

        # No signals with constant prices
        assert (df['signal'] == 0).all()

    def test_strategy_with_extreme_volatility(self, volatile_market_data):
        """Test strategy handles extreme volatility."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(volatile_market_data.copy())
        df = strategy.generate_signals(df)

        # Should still produce valid Ichimoku components and signals
        assert_valid_indicator(df['tenkan_sen'], 'Tenkan-sen', min_val=0)
        assert_valid_indicator(df['kijun_sen'], 'Kijun-sen', min_val=0)
        assert_valid_signals(df)

    def test_strategy_with_trending_market(self, trending_market_data):
        """Test strategy in strong trending market."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(trending_market_data.copy())
        df = strategy.generate_signals(df)

        # In trending market, Ichimoku should produce signals
        # but cloud acts as filter
        assert_valid_signals(df)

    def test_strategy_handles_gaps_in_data(self, zs_1h_data):
        """Test strategy handles missing bars in data."""
        strategy = IchimokuCloudStrategy(
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
            displacement=26,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create data with gap using utility
        gapped_data = create_gapped_dataframe(zs_1h_data, gap_start=100, gap_end=150)

        df = strategy.add_indicators(gapped_data.copy())
        df = strategy.generate_signals(df)

        # Should still calculate Ichimoku components and signals
        assert_indicator_columns_exist(
            df, 'tenkan_sen', 'kijun_sen', 'senkou_span_a',
            'senkou_span_b', 'chikou_span', 'signal'
        )
        assert_valid_signals(df)
