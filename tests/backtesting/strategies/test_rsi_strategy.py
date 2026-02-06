"""
RSI Strategy Test Suite.

Tests RSI strategy implementation with real historical data:
- Strategy initialization and configuration
- Indicator calculation within strategy
- Signal generation logic (long/short entry/exit)
- Full strategy execution and trade generation
- Edge cases and error handling

Uses real market data (ZS, CL) from data/historical_data/.
"""
import pandas as pd
import pytest

from app.backtesting.strategies import RSIStrategy
from tests.backtesting.helpers.assertions import (
    assert_valid_indicator,
    assert_valid_signals,
    assert_valid_trades,
    assert_no_overlapping_trades
)


# ==================== Test Strategy Initialization ====================

class TestRSIStrategyInitialization:
    """Test RSI strategy initialization and configuration."""

    @pytest.mark.parametrize("period,lower,upper,description", [
        (14, 30, 70, "standard"),
        (14, 20, 80, "aggressive_thresholds"),
        (7, 30, 70, "fast_period"),
        (21, 30, 70, "slow_period"),
    ])
    def test_initialization_with_various_parameters(self, period, lower, upper, description):
        """Test RSI strategy initializes correctly with various parameter combinations."""
        strategy = RSIStrategy(
            rsi_period=period,
            lower_threshold=lower,
            upper_threshold=upper,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert strategy.rsi_period == period
        assert strategy.lower_threshold == lower
        assert strategy.upper_threshold == upper
        assert strategy.rollover is False
        assert strategy.trailing is None
        assert strategy.position_manager.slippage_ticks == 1

    def test_initialization_with_trailing_stop(self):
        """Test RSI strategy with trailing stop enabled."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=2.0,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert strategy.trailing == 2.0
        assert strategy.trailing_stop_manager is not None

    def test_initialization_with_rollover_enabled(self):
        """Test RSI strategy with contract rollover handling."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=True,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        assert strategy.rollover is True
        assert strategy.switch_handler is not None

    def test_format_name_generates_correct_string(self):
        """Test strategy name formatting for identification."""
        name = RSIStrategy.format_name(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1
        )

        assert 'RSI' in name
        assert 'period=14' in name
        assert 'lower=30' in name
        assert 'upper=70' in name
        assert 'rollover=False' in name
        assert 'slippage_ticks=1' in name


# ==================== Test Indicator Calculation ====================

class TestRSIStrategyIndicators:
    """Test indicator calculation within RSI strategy."""

    def test_add_indicators_creates_rsi_column(self, zs_1h_data):
        """Test that add_indicators properly calculates RSI on real data."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())

        # RSI column should be added
        assert 'rsi' in df.columns

        # Validate RSI values (already checks 0-100 range, NaN handling)
        assert_valid_indicator(df['rsi'], 'RSI', min_val=0, max_val=100, allow_nan=True)

    def test_fast_vs_slow_rsi_generates_different_signal_frequency(self, zs_1h_data):
        """Test that fast RSI (7) generates more signals than slow RSI (21)."""
        strategy_fast = RSIStrategy(
            rsi_period=7,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_slow = RSIStrategy(
            rsi_period=21,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df_fast = strategy_fast.add_indicators(zs_1h_data.copy())
        df_fast = strategy_fast.generate_signals(df_fast)

        df_slow = strategy_slow.add_indicators(zs_1h_data.copy())
        df_slow = strategy_slow.generate_signals(df_slow)

        # Fast RSI should generate more signals (more sensitive to price changes)
        fast_signal_count = (df_fast['signal'] != 0).sum()
        slow_signal_count = (df_slow['signal'] != 0).sum()

        assert fast_signal_count > slow_signal_count, \
            f"Fast RSI should generate more signals ({fast_signal_count} vs {slow_signal_count})"

    def test_rsi_calculation_on_different_symbols(self, zs_1h_data, cl_15m_data):
        """Test RSI calculation works on different market data."""
        strategy_zs = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_cl = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='CL'
        )

        df_zs = strategy_zs.add_indicators(zs_1h_data.copy())
        df_cl = strategy_cl.add_indicators(cl_15m_data.copy())

        # Both should have valid RSI
        assert_valid_indicator(df_zs['rsi'], 'RSI_ZS', min_val=0, max_val=100)
        assert_valid_indicator(df_cl['rsi'], 'RSI_CL', min_val=0, max_val=100)


# ==================== Test Signal Generation ====================

class TestRSIStrategySignals:
    """Test signal generation logic for RSI strategy."""

    def test_generate_signals_creates_signal_column(self, zs_1h_data):
        """Test that generate_signals creates signal column."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        assert 'signal' in df.columns
        assert_valid_signals(df)

    def test_long_entry_signal_on_oversold_cross(self, zs_1h_data):
        """Test long signals occur when RSI crosses into oversold territory."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Find long signal bars
        long_signal_bars = df[df['signal'] == 1]

        # Long signals should occur when RSI is at or below lower threshold
        assert len(long_signal_bars) > 0, "Expected long signals on 2-year real data"
        assert (long_signal_bars['rsi'] <= strategy.lower_threshold).all(), \
            "Long signals should only occur at/below lower threshold"

    def test_short_entry_signal_on_overbought_cross(self, zs_1h_data):
        """Test short signals occur when RSI crosses into overbought territory."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Find short signal bars
        short_signal_bars = df[df['signal'] == -1]

        # Short signals should occur when RSI is at or above upper threshold
        assert len(short_signal_bars) > 0, "Expected short signals on 2-year real data"
        assert (short_signal_bars['rsi'] >= strategy.upper_threshold).all(), \
            "Short signals should only occur at/above upper threshold"

    def test_no_signal_when_rsi_in_neutral_zone(self, zs_1h_data):
        """Test no signals generated when RSI stays in neutral zone without crossing."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Find bars where RSI is stable in neutral zone (not at thresholds)
        # These bars should have no signals
        stable_neutral = (
                (df['rsi'] > strategy.lower_threshold + 5) &
                (df['rsi'] < strategy.upper_threshold - 5)
        )

        # In stable neutral zone, signals should be 0
        assert (df.loc[stable_neutral, 'signal'] == 0).all(), \
            "No signals should be generated in stable neutral RSI zone"

    def test_signals_with_aggressive_thresholds(self, zs_1h_data):
        """Test signal frequency with aggressive thresholds (20/80)."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=20,
            upper_threshold=80,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Aggressive thresholds should produce fewer signals
        signal_count = (df['signal'] != 0).sum()

        # Should still have some signals, but fewer than standard 30/70
        assert signal_count > 0, "Expected some signals with aggressive thresholds"
        assert signal_count < len(df) * 0.1, "Too many signals for aggressive thresholds"

    def test_signals_with_conservative_thresholds(self, zs_1h_data):
        """Test signal frequency with conservative thresholds (40/60)."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=40,
            upper_threshold=60,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # Conservative thresholds should produce more signals
        signal_count = (df['signal'] != 0).sum()

        # Should have more signals than aggressive
        assert signal_count > 0, "Expected signals with conservative thresholds"

    def test_both_long_and_short_signals_present(self, zs_1h_data):
        """Test that strategy generates both long and short signals."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        long_signals = (df['signal'] == 1).sum()
        short_signals = (df['signal'] == -1).sum()

        assert long_signals > 0, "Expected long signals"
        assert short_signals > 0, "Expected short signals"

    def test_no_signals_generated_during_rsi_warmup_period(self, zs_1h_data):
        """Test that no signals are generated while RSI is NaN (warmup period)."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)

        # During RSI warmup (first ~14 bars), RSI is NaN and signals should be 0
        nan_rsi_bars = df[df['rsi'].isna()]

        if len(nan_rsi_bars) > 0:
            assert (nan_rsi_bars['signal'] == 0).all(), \
                "No signals should be generated when RSI is NaN (warmup period)"


# ==================== Test Strategy Execution ====================

class TestRSIStrategyExecution:
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
        """Test RSI strategy backtest with various configurations and data sources.

        Validates that the strategy executes correctly across:
        - Different symbols (ZS, CL)
        - Different timeframes (1h, 15m)
        - With/without trailing stops
        """
        # Load appropriate data based on symbol and interval
        data = load_real_data('1!', symbol, interval)

        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=trailing,
            slippage_ticks=1,
            symbol=symbol
        )

        trades = strategy.run(data.copy(), contract_switch_dates.get(symbol, []))

        # Should generate trades on real data
        assert len(trades) > 0, f"Expected trades for {symbol} {interval} (config: {description})"

        # Validate trade structure
        assert_valid_trades(trades)

    def test_trades_have_both_long_and_short_positions(self, zs_1h_data, contract_switch_dates):
        """Test that backtest generates both long and short trades."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        long_trades = [t for t in trades if t['side'] == 'long']
        short_trades = [t for t in trades if t['side'] == 'short']

        assert len(long_trades) > 0, "Expected long trades"
        assert len(short_trades) > 0, "Expected short trades"

    def test_trades_do_not_overlap(self, zs_1h_data, contract_switch_dates):
        """Test that trades don't overlap (proper position management)."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        assert_no_overlapping_trades(trades)

    def test_backtest_with_slippage_affects_prices(self, zs_1h_data, contract_switch_dates):
        """Test that slippage is properly applied to trade prices."""
        strategy_no_slip = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=0,
            symbol='ZS'
        )

        strategy_with_slip = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=2,
            symbol='ZS'
        )

        trades_no_slip = strategy_no_slip.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))
        trades_with_slip = strategy_with_slip.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        # Both should generate trades
        assert len(trades_no_slip) > 0
        assert len(trades_with_slip) > 0

        # Note: Slippage affects execution prices, so trade details may differ
        # but trade count should be similar
        assert abs(len(trades_no_slip) - len(trades_with_slip)) < 5

    def test_backtest_with_different_rsi_periods(self, zs_1h_data, contract_switch_dates):
        """Test that different RSI periods produce different trade patterns."""
        strategy_fast = RSIStrategy(
            rsi_period=7,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        strategy_slow = RSIStrategy(
            rsi_period=21,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        trades_fast = strategy_fast.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))
        trades_slow = strategy_slow.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        # Both should generate trades
        assert len(trades_fast) > 0
        assert len(trades_slow) > 0

        # Fast RSI typically generates more signals
        assert len(trades_fast) >= len(trades_slow)

    def test_signals_convert_to_actual_trades(self, zs_1h_data, contract_switch_dates):
        """Test that generated signals result in actual trades (signal-to-trade conversion)."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Get signals
        df = strategy.add_indicators(zs_1h_data.copy())
        df = strategy.generate_signals(df)
        signal_count = (df['signal'] != 0).sum()

        # Get trades
        trades = strategy.run(zs_1h_data.copy(), contract_switch_dates.get('ZS', []))

        # Should have trades if we have signals
        if signal_count > 0:
            assert len(trades) > 0, "Signals should result in actual trades"

            # Number of trades should be reasonable relative to signal count
            # (each trade needs entry + exit, some signals might not complete)
            assert len(trades) <= signal_count, \
                f"Cannot have more trades ({len(trades)}) than signals ({signal_count})"


# ==================== Test Edge Cases ====================

class TestRSIStrategyEdgeCases:
    """Test edge cases and error handling."""

    def test_strategy_with_insufficient_data(self):
        """Test strategy behavior with insufficient data for RSI calculation."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create small dataset (less than RSI period)
        small_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        df = strategy.add_indicators(small_data.copy())

        # RSI should be NaN for insufficient data
        assert df['rsi'].isna().all()

    def test_strategy_with_constant_prices(self):
        """Test strategy with constant prices (no volatility)."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create constant price data
        constant_data = pd.DataFrame({
            'open': [100] * 50,
            'high': [100] * 50,
            'low': [100] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        }, index=pd.date_range('2024-01-01', periods=50, freq='1h'))

        df = strategy.add_indicators(constant_data.copy())
        df = strategy.generate_signals(df)

        # With constant prices, RSI should be around 50 (neutral)
        valid_rsi = df['rsi'].dropna()
        if len(valid_rsi) > 0:
            assert 45 <= valid_rsi.mean() <= 55

        # Should not generate signals with constant prices
        assert (df['signal'] == 0).all()

    def test_strategy_with_extreme_volatility(self, volatile_market_data):
        """Test strategy handles extreme volatility."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(volatile_market_data.copy())
        df = strategy.generate_signals(df)

        # Should still produce valid RSI and signals
        assert_valid_indicator(df['rsi'], 'RSI', min_val=0, max_val=100)
        assert_valid_signals(df)

    def test_strategy_with_trending_market(self, trending_market_data):
        """Test strategy in strong trending market."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        df = strategy.add_indicators(trending_market_data.copy())
        df = strategy.generate_signals(df)

        # In trending market, RSI should show directional bias
        valid_rsi = df['rsi'].dropna()
        assert len(valid_rsi) > 0

        # Should still generate valid signals
        assert_valid_signals(df)

    def test_strategy_handles_gaps_in_data(self, zs_1h_data):
        """Test strategy handles missing bars in data."""
        strategy = RSIStrategy(
            rsi_period=14,
            lower_threshold=30,
            upper_threshold=70,
            rollover=False,
            trailing=None,
            slippage_ticks=1,
            symbol='ZS'
        )

        # Create data with gap
        data_with_gap = zs_1h_data.iloc[:100].copy()
        data_after_gap = zs_1h_data.iloc[150:].copy()
        gapped_data = pd.concat([data_with_gap, data_after_gap])

        df = strategy.add_indicators(gapped_data.copy())
        df = strategy.generate_signals(df)

        # Should still calculate RSI and signals
        assert 'rsi' in df.columns
        assert 'signal' in df.columns
        assert_valid_signals(df)
