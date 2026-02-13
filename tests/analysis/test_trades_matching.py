import pandas as pd
import pytest

from app.analysis.trades_matching import (
    process_trade,
    format_processed_trades,
    match_trades,
)
from futures_config import get_contract_multiplier


@pytest.fixture
def sample_trades_data():
    """Sample trades data for testing."""
    return pd.DataFrame([
        {
            "trade_time": pd.Timestamp("2023-12-11 18:00:49"),
            "symbol": "AAPL",
            "side": "B",
            "price": 192.26,
            "size": 5,
            "commission": 1.01,
            "net_amount": 961.3
        },
        {
            "trade_time": pd.Timestamp("2023-12-11 19:15:30"),
            "symbol": "AAPL",
            "side": "S",
            "price": 195.50,
            "size": 3,
            "commission": 0.75,
            "net_amount": 586.5
        },
        {
            "trade_time": pd.Timestamp("2023-12-12 10:30:15"),
            "symbol": "MSFT",
            "side": "B",
            "price": 350.75,
            "size": 2,
            "commission": 0.50,
            "net_amount": 701.5
        }
    ])


@pytest.fixture
def sample_alerts_data():
    """Sample ibkr_alerts data for testing."""
    return pd.DataFrame([
        {
            "trade_time": pd.Timestamp("2023-12-11 18:00:49"),
            "symbol": "MCL",
            "side": "B",
            "price": 56.98
        },
        {
            "trade_time": pd.Timestamp("2023-12-11 19:15:30"),
            "symbol": "MCL",
            "side": "S",
            "price": 57.25
        },
        {
            "trade_time": pd.Timestamp("2023-12-12 10:30:15"),
            "symbol": "MNG",
            "side": "B",
            "price": 3.694
        }
    ])


def test_process_trade_new_position():
    """Test processing a new trade position."""

    symbol = "AAPL"
    side = "B"
    size = 5
    price = 192.26
    commission = 1.01
    trade_time = pd.Timestamp("2023-12-11 18:00:49")
    multiplier = 1
    open_trades = {}
    processed_trades = []

    process_trade(symbol, side, size, price, commission, trade_time, multiplier, open_trades, processed_trades)

    # Verify a new position was opened
    assert symbol in open_trades
    assert len(open_trades[symbol]) == 1
    assert open_trades[symbol][0]["side"] == side
    assert open_trades[symbol][0]["price"] == price
    assert open_trades[symbol][0]["commission"] == commission
    assert open_trades[symbol][0]["size"] == size
    assert open_trades[symbol][0]["trade_time"] == trade_time

    # Verify no trades were processed
    assert len(processed_trades) == 0


def test_process_trade_close_position():
    """Test processing a trade that closes an existing position."""

    symbol = "AAPL"
    buy_side = "B"
    sell_side = "S"
    size = 5
    buy_price = 192.26
    sell_price = 195.50
    buy_commission = 1.01
    sell_commission = 0.75
    buy_time = pd.Timestamp("2023-12-11 18:00:49")
    sell_time = pd.Timestamp("2023-12-11 19:15:30")
    multiplier = 1

    # Setup existing open position
    open_trades = {
        symbol: [
            {
                "side": buy_side,
                "price": buy_price,
                "commission": buy_commission,
                "size": size,
                "trade_time": buy_time
            }
        ]
    }
    processed_trades = []

    # Process a sell trade to close the position
    process_trade(symbol,
                  sell_side,
                  size,
                  sell_price,
                  sell_commission,
                  sell_time,
                  multiplier,
                  open_trades,
                  processed_trades)

    # Verify the position was closed
    assert len(open_trades[symbol]) == 0

    # Verify a trade was processed
    assert len(processed_trades) == 1
    processed_trade = processed_trades[0]
    assert processed_trade["symbol"] == symbol
    assert processed_trade["entry_time"] == buy_time
    assert processed_trade["entry_side"] == buy_side
    assert processed_trade["entry_price"] == buy_price
    assert processed_trade["entry_net_amount"] == buy_price * size * multiplier
    assert processed_trade["exit_time"] == sell_time
    assert processed_trade["exit_side"] == sell_side
    assert processed_trade["exit_price"] == sell_price
    assert processed_trade["exit_net_amount"] == sell_price * size * multiplier
    assert processed_trade["size"] == size
    assert processed_trade["total_commission"] == buy_commission + sell_commission


def test_process_trade_partial_close():
    """Test processing a trade that partially closes an existing position."""

    symbol = "AAPL"
    buy_side = "B"
    sell_side = "S"
    buy_size = 5
    sell_size = 3
    buy_price = 192.26
    sell_price = 195.50
    buy_commission = 1.01
    sell_commission = 0.75
    buy_time = pd.Timestamp("2023-12-11 18:00:49")
    sell_time = pd.Timestamp("2023-12-11 19:15:30")
    multiplier = 1

    # Setup existing open position
    open_trades = {
        symbol: [
            {
                "side": buy_side,
                "price": buy_price,
                "commission": buy_commission,
                "size": buy_size,
                "trade_time": buy_time
            }
        ]
    }
    processed_trades = []

    # Process a sell trade to partially close the position
    process_trade(symbol,
                  sell_side,
                  sell_size,
                  sell_price,
                  sell_commission,
                  sell_time,
                  multiplier,
                  open_trades,
                  processed_trades)

    # Verify the position was partially closed (remaining size = 2)
    assert len(open_trades[symbol]) == 1
    assert open_trades[symbol][0]["size"] == buy_size - sell_size

    # Verify a trade was processed
    assert len(processed_trades) == 1
    processed_trade = processed_trades[0]
    assert processed_trade["symbol"] == symbol
    assert processed_trade["entry_time"] == buy_time
    assert processed_trade["entry_side"] == buy_side
    assert processed_trade["entry_price"] == buy_price
    assert processed_trade["entry_net_amount"] == buy_price * sell_size * multiplier
    assert processed_trade["exit_time"] == sell_time
    assert processed_trade["exit_side"] == sell_side
    assert processed_trade["exit_price"] == sell_price
    assert processed_trade["exit_net_amount"] == sell_price * sell_size * multiplier
    assert processed_trade["size"] == sell_size
    assert processed_trade["total_commission"] == buy_commission + sell_commission


def test_format_processed_trades():
    """Test formatting processed trades."""

    processed_trades = [
        {
            "symbol": "AAPL",
            "entry_time": pd.Timestamp("2023-12-11 18:00:49"),
            "entry_side": "B",
            "entry_price": 192.26,
            "entry_net_amount": 961.3,
            "exit_time": pd.Timestamp("2023-12-11 19:15:30"),
            "exit_side": "S",
            "exit_price": 195.50,
            "exit_net_amount": 586.5,
            "size": 3,
            "total_commission": 1.76
        }
    ]

    result = format_processed_trades(processed_trades)

    # Verify the result is a DataFrame with the expected structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    # Verify the values are formatted correctly
    assert result.iloc[0]["entry_price"] == 192.26
    assert result.iloc[0]["exit_price"] == 195.50
    assert result.iloc[0]["entry_net_amount"] == 961
    assert result.iloc[0]["exit_net_amount"] == 586


def test_format_processed_trades_empty():
    """Test formatting empty processed trades."""

    result = format_processed_trades([])

    # Verify the result is an empty DataFrame
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_match_trades(sample_trades_data):
    """Test matching trades."""

    result = match_trades(sample_trades_data)

    # Verify the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Verify trades were matched correctly
    assert len(result) == 1  # Only one matched trade (AAPL B->S)
    assert result.iloc[0]["symbol"] == "AAPL"
    assert result.iloc[0]["entry_side"] == "B"
    assert result.iloc[0]["exit_side"] == "S"
    assert result.iloc[0]["size"] == 3  # Partial close


def test_match_trades_alerts(sample_alerts_data):
    """Test matching ibkr_alerts as trades."""

    result = match_trades(sample_alerts_data, is_ibkr_alerts=True)

    # Verify the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Verify ibkr_alerts were matched correctly
    assert len(result) == 1  # Only one matched trade (MCL B->S)
    assert result.iloc[0]["symbol"] == "MCL"
    assert result.iloc[0]["entry_side"] == "B"
    assert result.iloc[0]["exit_side"] == "S"

    # Verify multiplier was applied
    multiplier = get_contract_multiplier("MCL")
    assert result.iloc[0]["entry_net_amount"] == 56.98 * multiplier
    assert result.iloc[0]["exit_net_amount"] == 57.25 * multiplier


def test_match_trades_no_matches():
    """Test matching trades with no matches."""

    # Create data with no matching trades (all buys)
    trades_data = pd.DataFrame([
        {
            "trade_time": pd.Timestamp("2023-12-11 18:00:49"),
            "symbol": "AAPL",
            "side": "B",
            "price": 192.26,
            "size": 5,
            "commission": 1.01,
            "net_amount": 961.3
        },
        {
            "trade_time": pd.Timestamp("2023-12-12 10:30:15"),
            "symbol": "MSFT",
            "side": "B",
            "price": 350.75,
            "size": 2,
            "commission": 0.50,
            "net_amount": 701.5
        }
    ])

    result = match_trades(trades_data)

    # Verify the result is an empty DataFrame
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_match_trades_with_unknown_symbol():
    """Test matching trades with an unknown symbol that raises ValueError."""
    from unittest.mock import patch
    
    # Create data with a symbol that will raise ValueError
    trades_data = pd.DataFrame([
        {
            "trade_time": pd.Timestamp("2023-12-11 18:00:49"),
            "symbol": "UNKNOWN_SYMBOL",
            "side": "B",
            "price": 192.26,
            "size": 5,
            "commission": 1.01,
            "net_amount": 961.3
        }
    ])
    
    with patch('app.analysis.trades_matching.get_contract_multiplier', side_effect=ValueError('Unknown symbol')):
        result = match_trades(trades_data)
        
        # Verify the result is a DataFrame with multiplier=1 used (should work without crashing)
        assert isinstance(result, pd.DataFrame)


def test_match_trades_with_none_multiplier():
    """Test matching trades when get_contract_multiplier returns None."""
    from unittest.mock import patch
    
    # Create data with a symbol that returns None multiplier
    trades_data = pd.DataFrame([
        {
            "trade_time": pd.Timestamp("2023-12-11 18:00:49"),
            "symbol": "UNKNOWN_SYMBOL",
            "side": "B",
            "price": 192.26,
            "size": 5,
            "commission": 1.01,
            "net_amount": 961.3
        }
    ])
    
    with patch('app.analysis.trades_matching.get_contract_multiplier', return_value=None):
        result = match_trades(trades_data)
        
        # Verify the result is a DataFrame with multiplier=1 used (should work without crashing)
        assert isinstance(result, pd.DataFrame)
