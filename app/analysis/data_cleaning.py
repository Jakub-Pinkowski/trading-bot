import pandas as pd

from app.utils.analysis_utils.data_cleaning_utils import fractional_to_decimal, parse_description
from app.utils.generic_utils import parse_symbol
from app.utils.logger import get_logger

logger = get_logger('analysis/data_cleaning')


def clean_ibkr_alerts_data(df):
    # Remove the dummy column and order columns if they exist
    df = df.drop(columns=[col for col in ['dummy', 'order'] if col in df.columns])

    # Sort the DataFrame by 'timestamp'
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Clean the symbol and change the name of the column to match the trade data
    df['symbol'] = df['symbol'].apply(parse_symbol)
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'trade_time'})

    # Drop rows with missing or invalid data
    df = df.dropna().reset_index(drop=True)

    # Remove consecutive orders on the same side for a given symbol
    mask = df.groupby('symbol')['side'].transform(lambda x: x != x.shift())
    df = df[mask]

    return df


def clean_tw_alerts_data(df):
    # Extract relevant fields from the 'Description' column
    df['description_parsed'] = df['Description'].apply(parse_description)
    df['symbol'] = df['description_parsed'].apply(lambda x: x.get('symbol'))
    df['side'] = df['description_parsed'].apply(lambda x: x.get('side'))
    df['price'] = df['description_parsed'].apply(lambda x: x.get('price'))

    # Rename the `Time` column to `timestamp` and convert to datetime
    df = df.rename(columns={'Time': 'timestamp'})
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        logger.error(f'Failed to convert \'timestamp\' to datetime: {e}')
        return pd.DataFrame()

    # Remove timezone information to keep dtype as datetime64[ns]
    if isinstance(df['timestamp'].dtype, pd.DatetimeTZDtype):
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    # Select only relevant columns
    df = df[['timestamp', 'symbol', 'side', 'price']]

    # Sort the DataFrame by 'timestamp'
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Clean the symbol and change the name of the column to match the trade data
    df['symbol'] = df['symbol'].apply(parse_symbol)
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'trade_time'})

    # Drop rows with missing or invalid data
    df = df.dropna().reset_index(drop=True)

    # Remove consecutive orders on the same side for a given symbol
    mask = df.groupby('symbol')['side'].transform(lambda x: x != x.shift())
    df = df[mask]

    return df


def clean_trades_data(trades_df):
    columns_needed = [
        'symbol',
        'side',
        'size',
        'price',
        'trade_time',
        'commission',
        'net_amount',
    ]

    cleaned_df = trades_df[columns_needed].copy()

    # Ensure correct datatypes explicitly
    cleaned_df['trade_time'] = pd.to_datetime(cleaned_df['trade_time'])
    cleaned_df['size'] = cleaned_df['size'].astype(float)
    cleaned_df['commission'] = cleaned_df['commission'].astype(float)
    cleaned_df['net_amount'] = cleaned_df['net_amount'].astype(float)

    # Convert fractional price strings properly
    cleaned_df['price'] = cleaned_df['price'].apply(fractional_to_decimal)

    # Reorder columns
    reordered_columns = [
        'trade_time',
        'symbol',
        'side',
        'price',
        'size',
        'commission',
        'net_amount',
    ]
    cleaned_df = cleaned_df[reordered_columns]

    return cleaned_df
