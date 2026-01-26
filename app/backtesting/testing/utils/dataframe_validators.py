import pandas as pd

from app.utils.logger import get_logger

logger = get_logger('backtesting/testing/utils/dataframe_validators')

# ==================== Module Configuration ====================

# Minimum rows required for reliable backtesting (100 warm-up + 50 for indicators)
MIN_ROWS_FOR_BACKTEST = 150


# ==================== DataFrame Validation ====================

def validate_dataframe(df, filepath):
    """Comprehensive DataFrame validation.

    Args:
        df: DataFrame to validate
        filepath: Path to the source file (for logging)

    Returns:
        bool: True if DataFrame is valid, False otherwise
    """
    # Check if DataFrame exists and is not empty
    if df is None or df.empty:
        logger.error(f'Empty or None DataFrame: {filepath}')
        return False

    # Check required columns
    required_columns = ['open', 'high', 'low', 'close']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f'DataFrame missing required columns {missing}: {filepath}')
        return False

    # Check data types - all OHLC columns must be numeric
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f'Non-numeric column "{col}" (type: {df[col].dtype}): {filepath}')
            return False

    # Check for excessive NaN values
    for col in required_columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_pct = (nan_count / len(df)) * 100
            # More than 10% NaN is concerning
            if nan_pct > 10:
                logger.warning(f'Column "{col}" has {nan_pct:.1f}% NaN values ({nan_count}/{len(df)} rows): {filepath}')

    # Check index is DatetimeIndex (required for time-series operations)
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error(f'DataFrame index is not a DatetimeIndex (type: {type(df.index).__name__}): {filepath}')
        return False

    # Check index is sorted (critical for time-series data)
    if not df.index.is_monotonic_increasing:
        logger.error(f'DataFrame index is not sorted in ascending order: {filepath}')
        return False

    # Check for duplicate timestamps
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        logger.warning(f'DataFrame has {dup_count} duplicate timestamp(s): {filepath}')
        # Don't fail validation, just warn - duplicates might be intentional

    return True
