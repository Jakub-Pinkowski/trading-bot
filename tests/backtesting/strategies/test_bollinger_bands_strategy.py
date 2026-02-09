"""
Bollinger Bands Strategy Test Suite.

Tests Bollinger Bands strategy implementation with real historical data:
- Strategy initialization and configuration
- Indicator calculation within strategy (middle, upper, lower bands)
- Signal generation logic (band bounce/reversion)
- Full strategy execution and trade generation
- Edge cases and error handling

Uses real market data (ZS, CL) from data/historical_data/.
"""
import pytest

from app.backtesting.strategies import BollingerBandsStrategy
from tests.backtesting.fixtures.assertions import (
    assert_valid_indicator,

)
from tests.backtesting.strategies.strategy_test_utils import (
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

class TestBollingerBandsStrategyInitialization:
    """Test Bollinger Bands strategy initialization and configuration."""

    @pytest.mark.parametrize("period,std_dev,description", [
        (20, 2.0, "standard"),
        (15, 2.0, "shorter_period"),
        (30, 2.0, "longer_period"),
        (20, 1.5, "tighter_bands"),
        (20, 2.5, "wider_bands"),
    ])
    def test_initialization_with_various_parameters(self, period, std_dev, description):
        """Test Bollinger Bands strategy initializes correctly with various parameters."""
        strategy = BollingerBandsStrategy(
            period=period,
            number_of_standard_deviations=std_dev,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert strategy.period == period
        assert strategy.number_of_standard_deviations == std_dev
        assert strategy.rollover == False
        assert strategy.trailing is None
        assert strategy.position_manager.slippage_ticks == 1

    def test_initialization_with_trailing_stop(self):
        """Test Bollinger Bands strategy with trailing stop enabled."""
        strategy = BollingerBandsStrategy(
            period=20,
            number_of_standard_deviations=2.0,
            rollover=False,
            trailing=2.0,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert strategy.trailing == 2.0
        assert strategy.trailing_stop_manager is not None

    def test_initialization_with_rollover_enabled(self):
        """Test Bollinger Bands strategy with contract rollover handling."""
        strategy = BollingerBandsStrategy(
            period=20,
            number_of_standard_deviations=2.0,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert strategy.rollover is True
        assert strategy.switch_handler is not None

    def test_format_name_generates_correct_string(self):
        """Test strategy name formatting for identification."""
        name = BollingerBandsStrategy.format_name(
            period=20,
            number_of_standard_deviations=2.0,
            rollover=False,
            trailing=None,
            slippage_ticks=1
        )

        assert 'BB' in name
        assert 'period=20' in name
        assert 'std=2.0' in name
        assert 'rollover=False' in name
        assert 'slippage_ticks=1' in name


# ==================== Test Indicator Calculation ====================

class TestBollingerBandsStrategyIndicators:
    """Test indicator calculation within Bollinger Bands strategy."""

    def test_add_indicators_creates_band_columns(self, standard_bollinger_strategy, zs_1h_data):
        """Test that add_indicators properly calculates all Bollinger Band components."""
        df = standard_bollinger_strategy.add_indicators(zs_1h_data.copy())

        # All Bollinger Bands columns should be added
        assert 'middle_band' in df.columns
        assert 'upper_band' in df.columns
        assert 'lower_band' in df.columns

        # Validate band values
        assert_valid_indicator(df['middle_band'], 'Middle_Band', min_val=0, allow_nan=True)
        assert_valid_indicator(df['upper_band'], 'Upper_Band', min_val=0, allow_nan=True)
        assert_valid_indicator(df['lower_band'], 'Lower_Band', min_val=0, allow_nan=True)

        # Verify bands respond to price changes (not constant)
        valid_middle = df['middle_band'].dropna()
        valid_upper = df['upper_band'].dropna()
        valid_lower = df['lower_band'].dropna()
        assert valid_middle.std() > 1.0, "Middle band should vary with price changes"
        assert valid_upper.std() > 1.0, "Upper band should vary with price changes"
        assert valid_lower.std() > 1.0, "Lower band should vary with price changes"

        # Verify warmup period (period = 20 bars for standard Bollinger Bands)
        middle_warmup_nans = df['middle_band'].iloc[:20].isna().sum()
        upper_warmup_nans = df['upper_band'].iloc[:20].isna().sum()
        assert middle_warmup_nans > 0, "Middle band should have warmup period"
        assert upper_warmup_nans > 0, "Upper band should have warmup period"

    def test_upper_band_above_lower_band(self, standard_bollinger_strategy, zs_1h_data):
        """Test that upper band is always above lower band."""
        df = standard_bollinger_strategy.add_indicators(zs_1h_data.copy())

        # Remove NaN values for comparison
        valid_data = df.dropna(subset=['upper_band', 'lower_band'])

        # Upper band should always be >= lower band
        assert (valid_data['upper_band'] >= valid_data['lower_band']).all(), \
            "Upper band should always be above or equal to lower band"

    def test_middle_band_between_upper_and_lower(self, standard_bollinger_strategy, zs_1h_data):
        """Test that middle band is between upper and lower bands."""
        df = standard_bollinger_strategy.add_indicators(zs_1h_data.copy())

        # Remove NaN values for comparison
        valid_data = df.dropna(subset=['middle_band', 'upper_band', 'lower_band'])

        # Middle band should be between upper and lower
        assert (valid_data['middle_band'] >= valid_data['lower_band']).all(), \
            "Middle band should be above or equal to lower band"
        assert (valid_data['middle_band'] <= valid_data['upper_band']).all(), \
            "Middle band should be below or equal to upper band"

    def test_wider_std_dev_creates_wider_bands(self, zs_1h_data):
        """Test that higher standard deviation creates wider bands."""
        strategy_narrow = BollingerBandsStrategy(
            period=20,
            number_of_standard_deviations=1.5,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_wide = BollingerBandsStrategy(
            period=20,
            number_of_standard_deviations=2.5,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df_narrow = strategy_narrow.add_indicators(zs_1h_data.copy())
        df_wide = strategy_wide.add_indicators(zs_1h_data.copy())

        # Remove NaN for comparison
        valid_narrow = df_narrow.dropna(subset=['upper_band', 'lower_band'])
        valid_wide = df_wide.dropna(subset=['upper_band', 'lower_band'])

        # Calculate band widths
        narrow_width = (valid_narrow['upper_band'] - valid_narrow['lower_band']).mean()
        wide_width = (valid_wide['upper_band'] - valid_wide['lower_band']).mean()

        assert wide_width > narrow_width, \
            f"Wider std dev should create wider bands ({wide_width:.2f} vs {narrow_width:.2f})"

    def test_bollinger_calculation_on_different_symbols(self, zs_1h_data, cl_15m_data):
        """Test Bollinger Bands calculation works on different market data."""
        strategy_zs = BollingerBandsStrategy(
            period=20,
            number_of_standard_deviations=2.0,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_cl = BollingerBandsStrategy(
            period=20,
            number_of_standard_deviations=2.0,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='CL'
        )

        df_zs = strategy_zs.add_indicators(zs_1h_data.copy())
        df_cl = strategy_cl.add_indicators(cl_15m_data.copy())

        # Both should have valid Bollinger Bands
        assert_valid_indicator(df_zs['middle_band'], 'Middle_ZS', min_val=0)
        assert_valid_indicator(df_zs['upper_band'], 'Upper_ZS', min_val=0)
        assert_valid_indicator(df_cl['middle_band'], 'Middle_CL', min_val=0)
        assert_valid_indicator(df_cl['lower_band'], 'Lower_CL', min_val=0)


# ==================== Test Signal Generation ====================

class TestBollingerBandsStrategySignals:
    """Test signal generation logic for Bollinger Bands strategy."""

    def test_generate_signals_creates_signal_column(self, standard_bollinger_strategy, zs_1h_data):
        """Test that generate_signals creates signal column."""
        df = standard_bollinger_strategy.add_indicators(zs_1h_data.copy())
        df = standard_bollinger_strategy.generate_signals(df)

        assert 'signal' in df.columns
        assert_valid_signals(df)

    def test_long_signal_on_lower_band_bounce(self, standard_bollinger_strategy, zs_1h_data):
        """Test long signals occur when price bounces from lower band."""
        df = standard_bollinger_strategy.add_indicators(zs_1h_data.copy())
        df = standard_bollinger_strategy.generate_signals(df)

        # Find long signal bars
        long_signal_bars = df[df['signal'] == 1]

        # Long signals should occur when price crosses back above lower band
        assert len(long_signal_bars) > 0, "Expected long signals on 2-year real data"

        # At signal bars, price should be near or above lower band (mean reversion bounce)
        # Check that signals occur when price is close to lower band
        for idx in long_signal_bars.index[:5]:  # Check first 5 signals
            price_at_signal = df.loc[idx, 'close']
            lower_band_at_signal = df.loc[idx, 'lower_band']

            # Price should be near lower band (within reasonable distance)
            assert price_at_signal >= lower_band_at_signal * 0.97, \
                "Long signal should occur near or above lower band"

    def test_short_signal_on_upper_band_rejection(self, standard_bollinger_strategy, zs_1h_data):
        """Test short signals occur when price falls back from upper band."""
        df = standard_bollinger_strategy.add_indicators(zs_1h_data.copy())
        df = standard_bollinger_strategy.generate_signals(df)

        # Find short signal bars
        short_signal_bars = df[df['signal'] == -1]

        # Short signals should occur when price crosses back below upper band
        assert len(short_signal_bars) > 0, "Expected short signals on 2-year real data"

        # At signal bars, price should be near or below upper band (mean reversion rejection)
        # Check that signals occur when price is close to upper band
        for idx in short_signal_bars.index[:5]:  # Check first 5 signals
            price_at_signal = df.loc[idx, 'close']
            upper_band_at_signal = df.loc[idx, 'upper_band']

            # Price should be near upper band (within reasonable distance)
            assert price_at_signal <= upper_band_at_signal * 1.03, \
                "Short signal should occur near or below upper band"

    def test_no_signals_when_price_in_middle_of_bands(self, standard_bollinger_strategy, zs_1h_data):
        """Test fewer signals when price stays in middle of bands."""
        df = standard_bollinger_strategy.add_indicators(zs_1h_data.copy())
        df = standard_bollinger_strategy.generate_signals(df)

        # Bollinger Bands mean reversion should have selective signals
        signal_pct = (df['signal'] != 0).sum() / len(df)

        assert signal_pct > 0, "Expected some signals"
        assert signal_pct < 0.15, f"Too many signals for mean reversion strategy ({signal_pct:.1%})"

    @pytest.mark.parametrize("period,std_dev,description,max_signal_pct", [
        (15, 1.5, "tight_fast", 0.20),
        (20, 2.0, "standard", 0.15),
        (30, 2.5, "wide_slow", 0.10),
    ])
    def test_signal_frequency_with_various_parameters(
        self, zs_1h_data, period, std_dev, description, max_signal_pct
    ):
        """Test signal frequency varies with Bollinger Bands parameters."""
        strategy = BollingerBandsStrategy(
            period=period,
            number_of_standard_deviations=std_dev,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Validate signal structure
        assert_valid_signals(df)

        signal_pct = (df['signal'] != 0).sum() / len(df)

        assert signal_pct > 0, f"Expected signals with {description} parameters"
        assert signal_pct <= max_signal_pct, \
            f"Too many signals for {description} parameters ({signal_pct:.1%})"

    def test_both_long_and_short_signals_present(self, standard_bollinger_strategy, zs_1h_data):
        """Test that strategy generates both long and short signals."""
        df = standard_bollinger_strategy.add_indicators(zs_1h_data.copy())
        df = standard_bollinger_strategy.generate_signals(df)

        assert_both_signal_types_present(df)

    def test_no_signals_generated_during_warmup_period(self, standard_bollinger_strategy, zs_1h_data):
        """Test that minimal signals are generated while Bollinger Bands are stabilizing."""
        df = standard_bollinger_strategy.add_indicators(zs_1h_data.copy())
        df = standard_bollinger_strategy.generate_signals(df)

        # During warmup (first ~20 bars for period), signals should be minimal
        assert_minimal_warmup_signals(df, warmup_bars=20, max_warmup_signals=2)


# ==================== Test Strategy Execution ====================

class TestBollingerBandsStrategyExecution:
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
        """Test Bollinger Bands strategy backtest with various configurations."""
        data = load_real_data('1!', symbol, interval)

        strategy = BollingerBandsStrategy(
            period=20,
            number_of_standard_deviations=2.0,
            rollover=False,
            trailing=trailing,
            slippage_ticks=1,
            symbol=symbol
        )

        trades = strategy.run(data.copy(), contract_switch_dates.get(symbol, []))

        assert len(trades) > 0, f"Expected trades for {symbol} {interval} (config: {description})"
        assert_valid_trades(trades)

    def test_trades_have_both_long_and_short_positions(
        self,
        standard_bollinger_strategy,
        zs_1h_data,
        contract_switch_dates
    ):
        """Test that backtest generates both long and short trades."""
        trades = standard_bollinger_strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_trades_have_both_directions(trades)

    def test_trades_do_not_overlap(self, standard_bollinger_strategy, zs_1h_data, contract_switch_dates):
        """Test that trades don't overlap (proper position management)."""
        trades = standard_bollinger_strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_no_overlapping_trades(trades)

    def test_backtest_with_slippage_affects_prices(self, zs_1h_data, contract_switch_dates):
        """Test that slippage is properly applied to trade prices."""
        strategy_no_slip = BollingerBandsStrategy(
            period=20,
            number_of_standard_deviations=2.0,
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        strategy_with_slip = BollingerBandsStrategy(
            period=20,
            number_of_standard_deviations=2.0,
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

    def test_backtest_with_different_band_parameters(self, zs_1h_data, contract_switch_dates):
        """Test that different Bollinger Bands parameters produce different trade patterns."""
        strategy_tight = BollingerBandsStrategy(
            period=15,
            number_of_standard_deviations=1.5,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_wide = BollingerBandsStrategy(
            period=30,
            number_of_standard_deviations=2.5,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades_tight = strategy_tight.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))
        trades_wide = strategy_wide.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert len(trades_tight) > 0, "Tight parameter strategy generated no trades"
        assert len(trades_wide) > 0, "Wide parameter strategy generated no trades"
        assert len(trades_tight) >= len(trades_wide), \
            f"Tighter Bollinger Bands period/std should generate more trades ({len(trades_tight)} vs {len(trades_wide)})"

    def test_signals_convert_to_actual_trades(self, standard_bollinger_strategy, zs_1h_data, contract_switch_dates):
        """Test that generated signals result in actual trades."""
        df = standard_bollinger_strategy.add_indicators(zs_1h_data.copy())
        df = standard_bollinger_strategy.generate_signals(df)
        trades = standard_bollinger_strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_signals_convert_to_trades(df, trades)


# ==================== Test Edge Cases ====================

class TestBollingerBandsStrategyEdgeCases:
    """Test edge cases and error handling."""

    def test_strategy_with_insufficient_data(self, standard_bollinger_strategy):
        """Test strategy behavior with insufficient data for Bollinger Bands calculation."""
        # Create small dataset using utility
        small_data = create_small_ohlcv_dataframe(bars=5, base_price=100)

        df = standard_bollinger_strategy.add_indicators(small_data.copy())

        # Bollinger Bands need period bars, so should have many NaN
        assert df['middle_band'].isna().sum() > 0
        assert df['upper_band'].isna().sum() > 0
        assert df['lower_band'].isna().sum() > 0

    def test_strategy_with_constant_prices(self, standard_bollinger_strategy):
        """Test strategy with constant prices (no volatility)."""
        # Create constant price data using utility
        constant_data = create_constant_price_dataframe(bars=50, price=100)

        df = standard_bollinger_strategy.add_indicators(constant_data.copy())
        df = standard_bollinger_strategy.generate_signals(df)

        # With constant prices, bands should converge and no volatility-based signals
        valid_bands = df.dropna(subset=['upper_band', 'lower_band'])
        if len(valid_bands) > 0:
            band_width = (valid_bands['upper_band'] - valid_bands['lower_band']).mean()
            assert band_width < 1.0, "Band width should be minimal with constant prices"

        # Should not generate many signals with no volatility
        assert (df['signal'] == 0).sum() > len(df) * 0.8, \
            "Most signals should be 0 with constant prices"

    def test_strategy_with_extreme_volatility(self, standard_bollinger_strategy, volatile_market_data):
        """Test strategy handles extreme volatility."""
        df = standard_bollinger_strategy.add_indicators(volatile_market_data.copy())
        df = standard_bollinger_strategy.generate_signals(df)

        # Should still produce valid bands and signals
        assert_valid_indicator(df['middle_band'], 'Middle_Band', min_val=0)
        assert_valid_indicator(df['upper_band'], 'Upper_Band', min_val=0)
        assert_valid_indicator(df['lower_band'], 'Lower_Band', min_val=0)
        assert_valid_signals(df)

    def test_strategy_with_trending_market(self, standard_bollinger_strategy, trending_market_data):
        """Test strategy in strong trending market."""
        df = standard_bollinger_strategy.add_indicators(trending_market_data.copy())
        df = standard_bollinger_strategy.generate_signals(df)

        # In trending market, Bollinger Bands should widen
        valid_bands = df.dropna(subset=['upper_band', 'lower_band'])
        band_width = (valid_bands['upper_band'] - valid_bands['lower_band']).mean()

        assert band_width > 0, "Bands should have positive width"
        assert_valid_signals(df)

    def test_strategy_handles_gaps_in_data(self, standard_bollinger_strategy, zs_1h_data):
        """Test strategy handles missing bars in data."""
        # Create data with gap using utility
        gapped_data = create_gapped_dataframe(zs_1h_data, gap_start=100, gap_end=150)

        df = standard_bollinger_strategy.add_indicators(gapped_data.copy())
        df = standard_bollinger_strategy.generate_signals(df)

        # Should still calculate Bollinger Bands and signals
        assert 'middle_band' in df.columns
        assert 'upper_band' in df.columns
        assert 'lower_band' in df.columns
        assert 'signal' in df.columns
        assert_valid_signals(df)
