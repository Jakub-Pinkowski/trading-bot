import pandas as pd
import pytest

from app.analysis.data_cleaning import (
    pre_clean_tw_alerts_data,
    pre_clean_alerts_data,
    clean_alerts_data,
    clean_trades_data
)


@pytest.fixture
def sample_alerts_data():
    """Sample alerts data for testing."""
    return pd.DataFrame([
        {"symbol": "ZW1!", "side": "B", "price": "34.20", "dummy": "YES", "timestamp": "23-05-01 10:30:45"},
        {"symbol": "ZC1!", "side": "S", "price": "423.20", "timestamp": "23-05-01 11:45:30"}
    ])


@pytest.fixture
def sample_tw_alerts_data():
    """Sample TradingView alerts data for testing."""
    return pd.DataFrame([
        {
            "Alert ID": "2223583442",
            "Ticker": "NYMEX:MCL1!, 15m",
            "Name": "",
            "Description": '{"symbol":"MCL1!","side":"S","price":56.98}',
            "Time": "2025-05-05T14:07:00Z"
        },
        {
            "Alert ID": "2223584382",
            "Ticker": "NYMEX:MNG1!, 15m",
            "Name": "",
            "Description": '{"symbol":"MNG1!","side":"S","price":3.694}',
            "Time": "2025-05-05T12:06:09Z"
        },
        {
            "Alert ID": "2223584182",
            "Ticker": "NYMEX:MNG1!, 15m",
            "Name": "",
            "Description": '{"symbol":"MNG1!","side":"B","price":3.706}',
            "Time": "2025-05-05T10:12:34Z"
        }
    ])


@pytest.fixture
def sample_trades_data():
    """Sample trades data for testing."""
    return pd.DataFrame([
        {
            "execution_id": "0000e0d5.6576fd38.01.01",
            "symbol": "AAPL",
            "supports_tax_opt": "1",
            "side": "S",
            "order_description": "Sold 5 @ 192.26 on ISLAND",
            "trade_time": "20231211-18:00:49",
            "trade_time_r": 1702317649000,
            "size": 5,
            "price": "192.26",
            "order_ref": "Order123",
            "submitter": "user1234",
            "exchange": "ISLAND",
            "commission": "1.01",
            "net_amount": 961.3,
            "account": "U1234567",
            "accountCode": "U1234567",
            "account_allocation_name": "U1234567",
            "company_name": "APPLE INC",
            "contract_description_1": "AAPL",
            "sec_type": "STK",
            "listing_exchange": "NASDAQ.NMS",
            "conid": 265598,
            "conidEx": "265598",
            "clearing_id": "IB",
            "clearing_name": "IB",
            "liquidation_trade": "0",
            "is_event_trading": "0"
        }
    ])


def test_pre_clean_tw_alerts_data(sample_tw_alerts_data):
    """Test pre-cleaning of TradingView alerts data."""

    result = pre_clean_tw_alerts_data(sample_tw_alerts_data)

    # Check that the result has the expected columns
    assert set(result.columns) == {"timestamp", "symbol", "side", "price"}

    # Check that the data was parsed correctly
    assert result.iloc[0]["symbol"] == "MCL1!"
    assert result.iloc[0]["side"] == "S"
    assert result.iloc[0]["price"] == 56.98

    # Check that the timestamp was converted correctly
    assert isinstance(result.iloc[0]["timestamp"], pd.Timestamp)


def test_pre_clean_alerts_data(sample_alerts_data):
    """Test pre-cleaning of regular alerts data."""

    result = pre_clean_alerts_data(sample_alerts_data)

    # Check that the dummy column was removed
    assert "dummy" not in result.columns

    # Check that other columns are preserved
    assert "symbol" in result.columns
    assert "side" in result.columns
    assert "price" in result.columns
    assert "timestamp" in result.columns


def test_clean_alerts_data_regular(sample_alerts_data):
    """Test cleaning of regular alerts data."""

    result = clean_alerts_data(sample_alerts_data)

    # Check that the result has the expected columns
    assert "symbol" in result.columns
    assert "side" in result.columns
    assert "price" in result.columns
    assert "trade_time" in result.columns  # timestamp renamed to trade_time

    # Check that the dummy column was removed
    assert "dummy" not in result.columns


def test_clean_alerts_data_tw(sample_tw_alerts_data):
    """Test cleaning of TradingView alerts data."""

    result = clean_alerts_data(sample_tw_alerts_data, tw_alerts=True)

    # Check that the result has the expected columns
    assert "symbol" in result.columns
    assert "side" in result.columns
    assert "price" in result.columns
    assert "trade_time" in result.columns

    # Check that original columns are not present
    assert "Alert ID" not in result.columns
    assert "Ticker" not in result.columns
    assert "Description" not in result.columns
    assert "Time" not in result.columns


def test_clean_trades_data(sample_trades_data):
    """Test cleaning of trades data."""

    result = clean_trades_data(sample_trades_data)

    # Check that the result has the expected columns
    expected_columns = [
        "trade_time", "symbol", "side", "price", "size", "commission", "net_amount"
    ]
    assert list(result.columns) == expected_columns

    # Check that the data types are correct
    assert isinstance(result.iloc[0]["trade_time"], pd.Timestamp)
    assert isinstance(result.iloc[0]["size"], float)
    assert isinstance(result.iloc[0]["commission"], float)
    assert isinstance(result.iloc[0]["net_amount"], float)
    assert isinstance(result.iloc[0]["price"], float)
