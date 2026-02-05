"""
Data manipulation utilities for backtesting tests.

Provides helper functions for:
- Extracting data subsets
- Finding market scenarios in real data
- Creating synthetic test scenarios
- Modifying data to trigger specific conditions
"""
import pandas as pd


# ==================== Data Subset Extraction ====================

def get_data_subset(df, start_date=None, end_date=None, rows=None):
    """
    Extract subset of data by date range or row count.

    Useful for creating smaller test datasets from large historical data.
    Can specify either date range OR number of rows, not both.

    Args:
        df: DataFrame with datetime index and OHLCV columns
        start_date: Start date for subset (inclusive)
        end_date: End date for subset (inclusive)
        rows: Number of rows to extract from start

    Returns:
        DataFrame subset matching criteria

    Raises:
        ValueError: If both date range and rows specified

    Example:
        # Get first 100 rows
        subset = get_data_subset(df, rows=100)

        # Get data for January 2025
        subset = get_data_subset(df, start_date='2025-01-01', end_date='2025-01-31')

        # Get data from specific date onwards
        subset = get_data_subset(df, start_date='2025-06-01')
    """
    if rows is not None and (start_date is not None or end_date is not None):
        raise ValueError("Cannot specify both date range and row count")

    if rows is not None:
        return df.head(rows).copy()

    if start_date is None and end_date is None:
        return df.copy()

    # Convert string dates to timestamps if needed
    if start_date is not None and isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if end_date is not None and isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)

    # Filter by date range
    if start_date is not None and end_date is not None:
        mask = (df.index >= start_date) & (df.index <= end_date)
    elif start_date is not None:
        mask = df.index >= start_date
    else:  # end_date is not None
        mask = df.index <= end_date

    return df[mask].copy()


def get_data_window(df, center_idx, window_before=50, window_after=50):
    """
    Extract data window around a specific point.

    Useful for examining context around a signal or event. Returns data
    before and after the specified index.

    Args:
        df: DataFrame with datetime index
        center_idx: Index position (int) or timestamp to center window on
        window_before: Number of bars to include before center
        window_after: Number of bars to include after center

    Returns:
        DataFrame with center_idx ± window size

    Example:
        # Get 50 bars before and after index 1000
        window = get_data_window(df, 1000, 50, 50)

        # Get context around a specific timestamp
        window = get_data_window(df, pd.Timestamp('2025-01-15 10:00'), 100, 100)
    """
    if isinstance(center_idx, (pd.Timestamp, str)):
        # Convert to integer position
        center_idx = df.index.get_loc(center_idx)

    start_idx = max(0, center_idx - window_before)
    end_idx = min(len(df), center_idx + window_after + 1)

    return df.iloc[start_idx:end_idx].copy()


# ==================== Market Scenario Detection ====================

def find_trending_period(df, min_length=50, direction='up', min_price_change_pct=5.0):
    """
    Find period in real data showing clear trend.

    Searches through data to identify sustained uptrend or downtrend.
    Useful for testing strategy behavior in trending markets.

    Args:
        df: DataFrame with OHLCV data
        min_length: Minimum number of bars in trending period
        direction: 'up' for uptrend, 'down' for downtrend
        min_price_change_pct: Minimum price change percentage required

    Returns:
        DataFrame subset showing trending period, or None if not found

    Example:
        # Find uptrend period in ZS data
        uptrend = find_trending_period(zs_data, min_length=100, direction='up')

        # Find downtrend with at least 10% move
        downtrend = find_trending_period(zs_data, direction='down', min_price_change_pct=10)
    """
    if len(df) < min_length:
        return None

    # Calculate rolling return over min_length period
    close_prices = df['close']

    for i in range(len(df) - min_length):
        window = close_prices.iloc[i:i + min_length]
        price_change_pct = ((window.iloc[-1] - window.iloc[0]) / window.iloc[0]) * 100

        # Check if trend matches criteria
        if direction == 'up' and price_change_pct >= min_price_change_pct:
            # Verify it's mostly upward movement (not just first and last)
            if _is_consistent_trend(window, 'up'):
                return df.iloc[i:i + min_length].copy()

        elif direction == 'down' and price_change_pct <= -min_price_change_pct:
            if _is_consistent_trend(window, 'down'):
                return df.iloc[i:i + min_length].copy()

    return None


def find_ranging_period(df, min_length=50, max_price_change_pct=3.0):
    """
    Find period in real data showing sideways/ranging movement.

    Searches for periods where price oscillates within narrow range.
    Useful for testing strategy behavior in consolidation.

    Args:
        df: DataFrame with OHLCV data
        min_length: Minimum number of bars in ranging period
        max_price_change_pct: Maximum net price change allowed

    Returns:
        DataFrame subset showing ranging period, or None if not found

    Example:
        # Find sideways period in ZS data
        ranging = find_ranging_period(zs_data, min_length=100, max_price_change_pct=2)
    """
    if len(df) < min_length:
        return None

    close_prices = df['close']

    for i in range(len(df) - min_length):
        window = close_prices.iloc[i:i + min_length]
        price_change_pct = abs(((window.iloc[-1] - window.iloc[0]) / window.iloc[0]) * 100)

        # Check if price stayed within tight range
        if price_change_pct <= max_price_change_pct:
            # Verify price didn't make large swings in between
            high_low_range_pct = ((window.max() - window.min()) / window.iloc[0]) * 100
            if high_low_range_pct <= max_price_change_pct * 3:  # Allow some oscillation
                return df.iloc[i:i + min_length].copy()

    return None


def find_volatile_period(df, min_length=50, min_volatility_factor=2.0):
    """
    Find period in real data with high volatility.

    Searches for periods where price movement is significantly higher
    than typical. Useful for stress-testing strategies.

    Args:
        df: DataFrame with OHLCV data
        min_length: Minimum number of bars in volatile period
        min_volatility_factor: Multiplier of average volatility required

    Returns:
        DataFrame subset showing volatile period, or None if not found

    Example:
        # Find period with 2x normal volatility
        volatile = find_volatile_period(cl_data, min_length=50, min_volatility_factor=2.0)
    """
    if len(df) < min_length * 2:  # Need baseline data
        return None

    # Calculate rolling standard deviation of returns
    returns = df['close'].pct_change()
    avg_volatility = returns.std()

    rolling_vol = returns.rolling(window=min_length).std()

    # Find periods where volatility exceeds threshold
    high_vol_mask = rolling_vol > (avg_volatility * min_volatility_factor)
    high_vol_indices = high_vol_mask[high_vol_mask].index

    if len(high_vol_indices) >= min_length:
        # Get first high volatility period
        start_idx = df.index.get_loc(high_vol_indices[0])
        end_idx = min(start_idx + min_length, len(df))
        return df.iloc[start_idx:end_idx].copy()

    return None


def _is_consistent_trend(prices, direction, consistency_threshold=0.6):
    """
    Check if price series shows consistent trend (internal use only).

    Validates that majority of moves are in trend direction.

    Args:
        prices: Series of prices
        direction: 'up' or 'down'
        consistency_threshold: Fraction of moves that must align with trend

    Returns:
        True if trend is consistent
    """
    returns = prices.pct_change().dropna()

    if direction == 'up':
        positive_moves = (returns > 0).sum()
        return (positive_moves / len(returns)) >= consistency_threshold
    else:
        negative_moves = (returns < 0).sum()
        return (negative_moves / len(returns)) >= consistency_threshold


# ==================== Indicator-Based Scenario Finding ====================

def find_rsi_oversold_period(df, rsi_period=14, oversold_threshold=30, context_bars=100):
    """
    Find period where RSI drops below oversold threshold.

    Locates actual oversold conditions in real data. Returns data window
    around the oversold event for testing long entry signals.

    Args:
        df: DataFrame with OHLCV data
        rsi_period: RSI calculation period
        oversold_threshold: RSI threshold for oversold (default 30)
        context_bars: Bars to include before and after oversold event

    Returns:
        DataFrame with context around oversold event, or None if not found

    Example:
        # Find RSI oversold in ZS data
        oversold_data = find_rsi_oversold_period(zs_data)

        # Test long entry strategy
        strategy.run(oversold_data)
        assert strategy.has_long_entry()
    """
    from app.backtesting.indicators import calculate_rsi

    rsi = calculate_rsi(df['close'], period=rsi_period)
    oversold_mask = rsi < oversold_threshold

    if not oversold_mask.any():
        return None

    # Get first oversold occurrence
    oversold_idx = oversold_mask[oversold_mask].index[0]
    return get_data_window(df, oversold_idx, context_bars, context_bars)


def find_rsi_overbought_period(df, rsi_period=14, overbought_threshold=70, context_bars=100):
    """
    Find period where RSI rises above overbought threshold.

    Locates actual overbought conditions in real data. Returns data window
    around the overbought event for testing short entry signals.

    Args:
        df: DataFrame with OHLCV data
        rsi_period: RSI calculation period
        overbought_threshold: RSI threshold for overbought (default 70)
        context_bars: Bars to include before and after overbought event

    Returns:
        DataFrame with context around overbought event, or None if not found

    Example:
        # Find RSI overbought in ZS data
        overbought_data = find_rsi_overbought_period(zs_data)

        # Test short entry strategy
        strategy.run(overbought_data)
        assert strategy.has_short_entry()
    """
    from app.backtesting.indicators import calculate_rsi

    rsi = calculate_rsi(df['close'], period=rsi_period)
    overbought_mask = rsi > overbought_threshold

    if not overbought_mask.any():
        return None

    # Get first overbought occurrence
    overbought_idx = overbought_mask[overbought_mask].index[0]
    return get_data_window(df, overbought_idx, context_bars, context_bars)


def find_ema_crossover_period(df, fast_period=9, slow_period=21, direction='bullish', context_bars=100):
    """
    Find period where EMA crossover occurs.

    Locates actual EMA crossover in real data. Useful for testing
    crossover-based entry signals.

    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        direction: 'bullish' for fast crossing above slow, 'bearish' for opposite
        context_bars: Bars to include before and after crossover

    Returns:
        DataFrame with context around crossover, or None if not found

    Example:
        # Find bullish EMA crossover
        crossover_data = find_ema_crossover_period(zs_data, direction='bullish')

        # Test long entry on crossover
        strategy.run(crossover_data)
        assert strategy.has_long_entry()
    """
    from app.backtesting.indicators import calculate_ema
    from app.backtesting.strategies.base.base_strategy import detect_crossover

    fast_ema = calculate_ema(df['close'], period=fast_period)
    slow_ema = calculate_ema(df['close'], period=slow_period)

    if direction == 'bullish':
        crossovers = detect_crossover(fast_ema, slow_ema, 'above')
    else:
        crossovers = detect_crossover(fast_ema, slow_ema, 'below')

    if not crossovers.any():
        return None

    # Get first crossover occurrence
    crossover_idx = crossovers[crossovers].index[0]
    return get_data_window(df, crossover_idx, context_bars, context_bars)


# ==================== Data Modification for Testing ====================

def inject_price_spike(df, index, spike_pct, direction='up'):
    """
    Inject artificial price spike at specific index.

    Useful for testing strategy behavior during extreme movements.
    Modifies high/low/close prices at specified bar.

    Args:
        df: DataFrame with OHLCV data
        index: Index position or timestamp where spike occurs
        spike_pct: Percentage size of spike
        direction: 'up' for spike up, 'down' for spike down

    Returns:
        Modified DataFrame with price spike

    Example:
        # Inject 5% spike upward at bar 100
        modified_df = inject_price_spike(df.copy(), 100, 5.0, 'up')

        # Test stop loss trigger
        strategy.run(modified_df)
        assert strategy.position_closed_by_stop()
    """
    df = df.copy()

    if isinstance(index, (pd.Timestamp, str)):
        index = df.index.get_loc(index)

    base_close = df.iloc[index]['close']
    spike_amount = base_close * (spike_pct / 100)

    if direction == 'up':
        df.iloc[index, df.columns.get_loc('high')] = base_close + spike_amount
        df.iloc[index, df.columns.get_loc('close')] = base_close + (spike_amount * 0.5)
    else:
        df.iloc[index, df.columns.get_loc('low')] = base_close - spike_amount
        df.iloc[index, df.columns.get_loc('close')] = base_close - (spike_amount * 0.5)

    return df


def inject_gap(df, index, gap_pct, direction='up'):
    """
    Inject price gap between bars.

    Creates gap between previous close and next open. Useful for testing
    gap-related logic.

    Args:
        df: DataFrame with OHLCV data
        index: Index where gap occurs (gap is between index-1 and index)
        gap_pct: Percentage size of gap
        direction: 'up' for gap up, 'down' for gap down

    Returns:
        Modified DataFrame with price gap

    Example:
        # Create 3% gap up at bar 50
        modified_df = inject_gap(df.copy(), 50, 3.0, 'up')

        # Test gap handling logic
        strategy.run(modified_df)
        assert strategy.position_adjusted_for_gap()
    """
    df = df.copy()

    if isinstance(index, (pd.Timestamp, str)):
        index = df.index.get_loc(index)

    if index == 0:
        raise ValueError("Cannot inject gap at first bar")

    prev_close = df.iloc[index - 1]['close']
    gap_amount = prev_close * (gap_pct / 100)

    if direction == 'up':
        new_open = prev_close + gap_amount
        # Adjust all OHLC values for the bar
        df.iloc[index, df.columns.get_loc('open')] = new_open
        df.iloc[index, df.columns.get_loc('high')] = new_open + (df.iloc[index]['high'] - df.iloc[index]['open'])
        df.iloc[index, df.columns.get_loc('low')] = new_open + (df.iloc[index]['low'] - df.iloc[index]['open'])
        df.iloc[index, df.columns.get_loc('close')] = new_open + (df.iloc[index]['close'] - df.iloc[index]['open'])
    else:
        new_open = prev_close - gap_amount
        df.iloc[index, df.columns.get_loc('open')] = new_open
        df.iloc[index, df.columns.get_loc('high')] = new_open + (df.iloc[index]['high'] - df.iloc[index]['open'])
        df.iloc[index, df.columns.get_loc('low')] = new_open + (df.iloc[index]['low'] - df.iloc[index]['open'])
        df.iloc[index, df.columns.get_loc('close')] = new_open + (df.iloc[index]['close'] - df.iloc[index]['open'])

    return df


def create_flat_period(df, start_index, length, flat_price=None):
    """
    Replace section of data with flat prices.

    Useful for testing indicator behavior with constant prices.

    Args:
        df: DataFrame with OHLCV data
        start_index: Where flat period begins
        length: Number of bars to make flat
        flat_price: Price to use (if None, uses price at start_index)

    Returns:
        Modified DataFrame with flat price period

    Example:
        # Create 50 bars of flat prices starting at bar 100
        modified_df = create_flat_period(df.copy(), 100, 50)

        # Test RSI behavior (should converge to 50)
        rsi = calculate_rsi(modified_df['close'], period=14)
        assert abs(rsi.iloc[-1] - 50) < 1
    """
    df = df.copy()

    if isinstance(start_index, (pd.Timestamp, str)):
        start_index = df.index.get_loc(start_index)

    if flat_price is None:
        flat_price = df.iloc[start_index]['close']

    end_index = min(start_index + length, len(df))

    df.iloc[start_index:end_index, df.columns.get_loc('open')] = flat_price
    df.iloc[start_index:end_index, df.columns.get_loc('high')] = flat_price
    df.iloc[start_index:end_index, df.columns.get_loc('low')] = flat_price
    df.iloc[start_index:end_index, df.columns.get_loc('close')] = flat_price

    return df


# ==================== Data Validation ====================

def validate_ohlcv_structure(df):
    """
    Validate DataFrame has proper OHLCV structure.

    Checks for required columns, datetime index, valid OHLC relationships,
    and reasonable values.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list_of_issues)

    Example:
        is_valid, issues = validate_ohlcv_structure(df)
        if not is_valid:
            for issue in issues:
                print(f"Data issue: {issue}")
    """
    issues = []

    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("Index is not DatetimeIndex")

    # Check for duplicate timestamps
    if df.index.duplicated().any():
        issues.append(f"Found {df.index.duplicated().sum()} duplicate timestamps")

    # Check OHLC relationships
    if 'high' in df.columns and 'low' in df.columns:
        if not (df['high'] >= df['low']).all():
            invalid_count = (~(df['high'] >= df['low'])).sum()
            issues.append(f"Found {invalid_count} bars where high < low")

    if all(col in df.columns for col in ['high', 'open', 'close']):
        if not (df['high'] >= df['open']).all():
            issues.append("Some bars have high < open")
        if not (df['high'] >= df['close']).all():
            issues.append("Some bars have high < close")

    if all(col in df.columns for col in ['low', 'open', 'close']):
        if not (df['low'] <= df['open']).all():
            issues.append("Some bars have low > open")
        if not (df['low'] <= df['close']).all():
            issues.append("Some bars have low > close")

    # Check for negative prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns and (df[col] <= 0).any():
            issues.append(f"Found negative or zero values in {col}")

    # Check for NaN values
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        issues.append(f"Found NaN values in columns: {nan_cols}")

    return len(issues) == 0, issues


def get_data_statistics(df):
    """
    Get summary statistics for OHLCV data.

    Provides quick overview of data characteristics. Useful for
    understanding data before testing.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary with statistics

    Example:
        stats = get_data_statistics(zs_data)
        print(f"Date range: {stats['start_date']} to {stats['end_date']}")
        print(f"Average daily range: {stats['avg_daily_range_pct']:.2f}%")
    """
    close_prices = df['close']
    returns = close_prices.pct_change().dropna()

    stats = {
        'rows': len(df),
        'start_date': df.index[0],
        'end_date': df.index[-1],
        'duration_days': (df.index[-1] - df.index[0]).days,
        'price_min': close_prices.min(),
        'price_max': close_prices.max(),
        'price_range_pct': ((close_prices.max() - close_prices.min()) / close_prices.min()) * 100,
        'avg_daily_range_pct': (((df['high'] - df['low']) / df['close']) * 100).mean(),
        'volatility': returns.std() * 100,
        'avg_volume': df['volume'].mean() if 'volume' in df.columns else None,
        'missing_data_pct': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
    }

    return stats
