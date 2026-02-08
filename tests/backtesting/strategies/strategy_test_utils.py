"""
Strategy Test Utilities.

Common test helpers and patterns for strategy tests.
Reduces code duplication across RSI, EMA, MACD, and other strategy tests.
"""
import pandas as pd


# ==================== Trade Validation Helpers ====================

def assert_trades_have_both_directions(trades):
    """
    Assert trades list contains both long and short positions.

    Args:
        trades: List of trade dictionaries

    Example:
        trades = strategy.run(data, switch_dates)
        assert_trades_have_both_directions(trades)
    """
    long_trades = [t for t in trades if t['side'] == 'long']
    short_trades = [t for t in trades if t['side'] == 'short']

    assert len(long_trades) > 0, "Expected long trades"
    assert len(short_trades) > 0, "Expected short trades"


def assert_similar_trade_count(trades1, trades2, max_difference=5):
    """
    Assert two trade lists have similar number of trades.

    Useful for comparing strategies with/without slippage or different parameters.

    Args:
        trades1: First list of trades
        trades2: Second list of trades
        max_difference: Maximum allowed difference in trade count

    Example:
        trades_no_slip = strategy_no_slip.run(...)
        trades_with_slip = strategy_with_slip.run(...)
        assert_similar_trade_count(trades_no_slip, trades_with_slip, max_difference=5)
    """
    assert len(trades1) > 0, "First trade list is empty"
    assert len(trades2) > 0, "Second trade list is empty"
    assert abs(len(trades1) - len(trades2)) <= max_difference, \
        f"Trade counts too different: {len(trades1)} vs {len(trades2)}"


def assert_signals_convert_to_trades(signals_df, trades, signal_col='signal'):
    """
    Assert that signal DataFrame with non-zero signals produces trades.

    Args:
        signals_df: DataFrame with signal column
        trades: List of trade dictionaries
        signal_col: Name of signal column (default: 'signal')

    Example:
        df = strategy.generate_signals(df)
        trades = strategy.run(data, switch_dates)
        assert_signals_convert_to_trades(df, trades)
    """
    signal_count = (signals_df[signal_col] != 0).sum()

    if signal_count > 0:
        assert len(trades) > 0, "Signals should result in actual trades"
        assert len(trades) <= signal_count, \
            f"Cannot have more trades ({len(trades)}) than signals ({signal_count})"


# ==================== Signal Validation Helpers ====================

def assert_both_signal_types_present(df, signal_col='signal'):
    """
    Assert DataFrame contains both long (1) and short (-1) signals.

    Args:
        df: DataFrame with signal column
        signal_col: Name of signal column (default: 'signal')

    Example:
        df = strategy.generate_signals(df)
        assert_both_signal_types_present(df)
    """
    long_signals = (df[signal_col] == 1).sum()
    short_signals = (df[signal_col] == -1).sum()

    assert long_signals > 0, "Expected long signals"
    assert short_signals > 0, "Expected short signals"


def assert_reasonable_signal_frequency(df, max_signal_pct=0.15, signal_col='signal'):
    """
    Assert signals are not too frequent (would indicate overtrading).

    Args:
        df: DataFrame with signal column
        max_signal_pct: Maximum percentage of bars that can have signals (default: 0.15 = 15%)
        signal_col: Name of signal column (default: 'signal')

    Example:
        df = strategy.generate_signals(df)
        assert_reasonable_signal_frequency(df, max_signal_pct=0.1)
    """
    signal_count = (df[signal_col] != 0).sum()
    signal_pct = signal_count / len(df)

    assert signal_count > 0, "Expected some signals"
    assert signal_pct <= max_signal_pct, \
        f"Too many signals ({signal_pct:.1%} of bars), expected <= {max_signal_pct:.1%}"


def assert_minimal_warmup_signals(df, warmup_bars, max_warmup_signals=2, signal_col='signal'):
    """
    Assert minimal signals generated during indicator warmup period.

    Args:
        df: DataFrame with signal column
        warmup_bars: Number of warmup bars to check
        max_warmup_signals: Maximum allowed signals during warmup (default: 2)
        signal_col: Name of signal column (default: 'signal')

    Example:
        df = strategy.generate_signals(df)
        assert_minimal_warmup_signals(df, warmup_bars=21, max_warmup_signals=2)
    """
    warmup_data = df.iloc[:warmup_bars]
    warmup_signal_count = (warmup_data[signal_col] != 0).sum()

    assert warmup_signal_count <= max_warmup_signals, \
        f"Too many signals during warmup period ({warmup_signal_count}, expected <= {max_warmup_signals})"


# ==================== Indicator Validation Helpers ====================


def assert_more_responsive_indicator(fast_series, slow_series, indicator_name="indicator"):
    """
    Assert that faster indicator is more responsive (varies more) than slower indicator.

    Uses percentage changes to compare volatility/responsiveness.

    Args:
        fast_series: Pandas Series with faster indicator values
        slow_series: Pandas Series with slower indicator values
        indicator_name: Name for error messages

    Example:
        assert_more_responsive_indicator(df['ema_short'], df['ema_long'], 'EMA')
    """
    fast_std = fast_series.pct_change().dropna().std()
    slow_std = slow_series.pct_change().dropna().std()

    assert fast_std > slow_std, \
        f"Fast {indicator_name} should be more responsive " \
        f"(std: {fast_std:.6f} vs {slow_std:.6f})"


def assert_different_indicator_patterns(series1, series2, min_difference=0.1, indicator_name="indicator"):
    """
    Assert that two indicator series produce different patterns.

    Compares mean values to ensure they're sufficiently different.

    Args:
        series1: First indicator series
        series2: Second indicator series
        min_difference: Minimum required difference in means
        indicator_name: Name for error messages

    Example:
        assert_different_indicator_patterns(
            df_fast['macd_line'],
            df_slow['macd_line'],
            min_difference=0.1,
            indicator_name='MACD'
        )
    """
    mean1 = series1.dropna().mean()
    mean2 = series2.dropna().mean()

    difference = abs(mean1 - mean2)
    assert difference > min_difference, \
        f"{indicator_name} patterns too similar " \
        f"(means: {mean1:.4f} vs {mean2:.4f}, diff: {difference:.4f})"


# ==================== Edge Case Test Helpers ====================

def create_small_ohlcv_dataframe(bars=3, base_price=100):
    """
    Create small OHLCV DataFrame for testing insufficient data scenarios.

    Args:
        bars: Number of bars to create (default: 3)
        base_price: Starting price (default: 100)

    Returns:
        DataFrame with OHLCV columns

    Example:
        small_data = create_small_ohlcv_dataframe(bars=5, base_price=50)
        df = strategy.add_indicators(small_data)
    """
    return pd.DataFrame({
        'open': [base_price + i for i in range(bars)],
        'high': [base_price + i + 1 for i in range(bars)],
        'low': [base_price + i - 1 for i in range(bars)],
        'close': [base_price + i + 0.5 for i in range(bars)],
        'volume': [1000 + i * 100 for i in range(bars)]
    }, index=pd.date_range('2024-01-01', periods=bars, freq='1h'))


def create_constant_price_dataframe(bars=50, price=100):
    """
    Create DataFrame with constant prices (no volatility).

    Args:
        bars: Number of bars to create (default: 50)
        price: Constant price for all bars (default: 100)

    Returns:
        DataFrame with constant OHLCV values

    Example:
        constant_data = create_constant_price_dataframe(bars=30, price=1000)
        df = strategy.generate_signals(strategy.add_indicators(constant_data))
        assert (df['signal'] == 0).all()  # No signals with constant prices
    """
    return pd.DataFrame({
        'open': [price] * bars,
        'high': [price] * bars,
        'low': [price] * bars,
        'close': [price] * bars,
        'volume': [1000] * bars
    }, index=pd.date_range('2024-01-01', periods=bars, freq='1h'))


def create_gapped_dataframe(original_df, gap_start, gap_end):
    """
    Create DataFrame with gap by removing bars between gap_start and gap_end.

    Args:
        original_df: Original DataFrame
        gap_start: Start index of gap (inclusive)
        gap_end: End index of gap (exclusive)

    Returns:
        DataFrame with gap

    Example:
        gapped_data = create_gapped_dataframe(zs_1h_data, 100, 150)
        df = strategy.generate_signals(strategy.add_indicators(gapped_data))
    """
    before_gap = original_df.iloc[:gap_start].copy()
    after_gap = original_df.iloc[gap_end:].copy()
    return pd.concat([before_gap, after_gap])
