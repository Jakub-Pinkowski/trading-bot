from app.utils.generic_utils import parse_symbol

# TODO: It's all fucking WRONG!!!
def clean_alerts_data(df):
    # Remove the dummy column if it exists
    if "dummy" in df.columns:
        df = df.drop(columns=["dummy"])

    # Clean symbol
    df["symbol"] = df["symbol"].apply(parse_symbol)

    # Remove consecutive orders on the same side for a given symbol
    df['prev_symbol'] = df['symbol'].shift(1)
    df['prev_side'] = df['side'].shift(1)
    df = df[~((df['symbol'] == df['prev_symbol']) & (df['side'] == df['prev_side']))]
    df = df.drop(columns=['prev_symbol', 'prev_side'])

    return df


def clean_trades_data(trades_df):
    columns_needed = [
        "symbol",
        "side",
        "size",
        "price",
        "trade_time",
        "commission",
        "net_amount",
    ]

    cleaned_df = trades_df[columns_needed].copy()

    # Ensure correct datatypes explicitly
    cleaned_df["trade_time"] = pd.to_datetime(cleaned_df["trade_time"])
    cleaned_df["size"] = cleaned_df["size"].astype(float)
    cleaned_df["commission"] = cleaned_df["commission"].astype(float)
    cleaned_df["net_amount"] = cleaned_df["net_amount"].astype(float)

    # Convert fractional price strings properly
    cleaned_df["price"] = cleaned_df["price"].apply(fractional_to_decimal)

    # Reorder columns
    reordered_columns = [
        "trade_time",
        "symbol",
        "side",
        "size",
        "price",
        "commission",
        "net_amount",
    ]
    cleaned_df = cleaned_df[reordered_columns]

    return cleaned_df
