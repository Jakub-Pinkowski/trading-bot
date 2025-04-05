import pandas as pd


def clean_trade_data(trades_df):
    columns_needed = [
        "execution_id",
        "symbol",
        "side",
        "size",
        "price",
        "trade_time",
        "exchange",
        "commission",
        "net_amount",
        "company_name"
    ]

    cleaned_df = trades_df[columns_needed].copy()

    # Ensure correct datatypes explicitly
    cleaned_df["trade_time"] = pd.to_datetime(cleaned_df["trade_time"])
    cleaned_df["size"] = cleaned_df["size"].astype(float)
    cleaned_df["price"] = cleaned_df["price"].astype(float)
    cleaned_df["commission"] = cleaned_df["commission"].astype(float)
    cleaned_df["net_amount"] = cleaned_df["net_amount"].astype(float)

    return cleaned_df


def clean_alerts_data(df, default_value="NO"):
    if "dummy" not in df.columns:
        df["dummy"] = default_value
    else:
        df["dummy"] = df["dummy"].fillna(default_value)
    return df
