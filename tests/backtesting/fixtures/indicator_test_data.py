"""
Basic test data fixtures for indicator tests.

Provides reusable synthetic data patterns for testing basic indicator logic
across all indicators (RSI, EMA, MACD, etc.). These fixtures create simple,
predictable price patterns that are useful for sanity checks.
"""
import pandas as pd
import pytest


# ==================== Basic Price Series Fixtures ====================

@pytest.fixture
def short_price_series():
    """
    Short price series (20 bars) with consistent upward trend.

    Useful for quick tests that don't need much data.
    Pattern: Strictly increasing by 1 unit per bar (100 to 119)

    Returns:
        Series of 20 prices increasing from 100 to 119
    """
    return pd.Series([
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119
    ])


@pytest.fixture
def volatile_price_series():
    """
    Volatile price series with large swings.

    Useful for testing indicator behavior during high volatility.
    Pattern: Large up/down swings

    Returns:
        Series of 30 prices with high volatility
    """
    return pd.Series([
        100, 105, 98, 107, 95, 110, 92, 112, 90, 115,
        88, 118, 85, 120, 83, 122, 80, 125, 78, 128,
        75, 130, 73, 132, 70, 135, 68, 138, 65, 140
    ])


@pytest.fixture
def rising_price_series():
    """
    Continuously rising prices (30 bars).

    Useful for testing indicator behavior during uptrends.
    Pattern: Steady increase from 100 to 129

    Returns:
        Series of 30 prices rising consecutively
    """
    return pd.Series(range(100, 130))


@pytest.fixture
def falling_price_series():
    """
    Continuously falling prices (30 bars).

    Useful for testing indicator behavior during downtrends.
    Pattern: Steady decrease from 130 to 101

    Returns:
        Series of 30 prices falling consecutively
    """
    return pd.Series(range(130, 100, -1))


@pytest.fixture
def oscillating_price_series():
    """
    Small oscillations around a price level.

    Useful for testing ranging/sideways market behavior.
    Pattern: Alternates between 100 and 101

    Returns:
        Series of 20 prices oscillating between 100-101
    """
    return pd.Series([
        100, 101, 100, 101, 100, 101, 100, 101, 100, 101,
        100, 101, 100, 101, 100, 101, 100, 101, 100, 101
    ])


@pytest.fixture
def constant_price_series():
    """
    Constant prices (no movement).

    Useful for testing zero-variance edge cases.
    Pattern: All prices are 100.0

    Returns:
        Series of 100 constant prices
    """
    return pd.Series([100.0] * 100)


@pytest.fixture
def medium_price_series():
    """
    Medium-length series (50 bars) with realistic movement.

    Useful for tests needing more data than short series.
    Pattern: General uptrend with some pullbacks

    Returns:
        Series of 50 prices with mixed movement
    """
    return pd.Series(range(100, 150))


@pytest.fixture
def minimal_price_series():
    """
    Minimal price series (5 bars).

    Useful for testing insufficient data handling.
    Pattern: Simple 5-bar sequence

    Returns:
        Series of 5 prices
    """
    return pd.Series([100, 101, 102, 103, 104])


@pytest.fixture
def exact_period_price_series():
    """
    Exactly 15 bars for period=14 tests.

    Useful for testing minimum data requirements.
    Pattern: 15 consecutive increases (14 warmup + 1 valid)

    Returns:
        Series of 15 prices
    """
    return pd.Series(range(100, 115))


# ==================== Price Level Comparison Fixtures ====================

@pytest.fixture
def low_price_level_series():
    """
    Price series at low absolute level with 10% increases.

    Useful for testing that indicators respond to % changes, not absolute levels.
    Pattern: 10-30 range with 10% steps

    Returns:
        Series of 20 prices at low level
    """
    return pd.Series([
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29
    ])


@pytest.fixture
def high_price_level_series():
    """
    Price series at high absolute level with 10% increases.

    Useful for testing that indicators respond to % changes, not absolute levels.
    Pattern: 100-290 range with 10% steps (same % as low_price_level_series)

    Returns:
        Series of 20 prices at high level
    """
    return pd.Series([
        100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
        200, 210, 220, 230, 240, 250, 260, 270, 280, 290
    ])


# ==================== Mixed Pattern Fixtures ====================

@pytest.fixture
def uptrend_then_downtrend_series():
    """
    Price series with clear uptrend followed by downtrend.

    Useful for testing indicator transitions between trends.
    Pattern: Rise from 100 to 124 (25 bars), then fall back to 101 (25 bars)

    Returns:
        Series of 50 prices with trend reversal (25 up + 25 down)
    """
    rising = list(range(100, 125))  # 25 bars rising
    falling = list(range(125, 100, -1))  # 25 bars falling
    return pd.Series(rising + falling)


@pytest.fixture
def flat_then_volatile_series():
    """
    Flat prices followed by volatile movement.

    Useful for testing indicator response to volatility changes.
    Pattern: 50 bars flat at 100, then 50 bars oscillating

    Returns:
        Series of 100 prices (flat then oscillating)
    """
    flat = [100.0] * 50
    oscillating = [100 + (i % 10 - 5) * 0.5 for i in range(50)]
    return pd.Series(flat + oscillating)


# ==================== Empty Data Fixtures ====================

@pytest.fixture
def empty_price_series():
    """
    Empty price series.

    Useful for testing empty data handling.

    Returns:
        Empty pandas Series
    """
    return pd.Series([], dtype=float)


# ==================== Common Test Periods ====================

@pytest.fixture
def standard_periods():
    """
    Standard indicator periods for parameterized tests.

    Returns:
        List of common periods: [7, 14, 21, 28]
    """
    return [7, 14, 21, 28]


@pytest.fixture
def extreme_periods():
    """
    Extreme indicator periods for edge case tests.

    Returns:
        Dict with 'short' and 'long' period values
    """
    return {'short': 3, 'long': 100}


# ==================== Bollinger Bands Specific Fixtures ====================

@pytest.fixture
def bb_breakout_price_series():
    """
    Controlled price series for testing BB breakout detection.

    Pattern designed to produce predictable breakouts on both sides:
    - Starts with stable prices around 100 (builds 20-period baseline)
    - Includes sharp upward spike (breaks upper band)
    - Returns to mean
    - Includes sharp downward spike (breaks lower band)
    - Returns to stable range

    Returns:
        Series of 80 prices with controlled breakout patterns
    """
    # Start with 30 stable bars around 100 to establish baseline (period=20)
    stable_start = [100 + (i % 3) * 0.5 for i in range(30)]

    # Sharp upward spike (5 bars) - should break upper band
    upward_spike = [100, 103, 107, 112, 115]

    # Return to mean (10 bars)
    return_to_mean = [112, 108, 104, 102, 101, 100, 100, 99, 100, 100]

    # Sharp downward spike (5 bars) - should break lower band
    downward_spike = [100, 96, 91, 86, 82]

    # Return to stable (20 bars)
    stable_end = [
        86, 90, 94, 97, 99, 100, 100, 101, 100, 100,
        100, 101, 100, 101, 100, 100, 101, 100, 100, 100
    ]

    return pd.Series(stable_start + upward_spike + return_to_mean +
                     downward_spike + stable_end)
