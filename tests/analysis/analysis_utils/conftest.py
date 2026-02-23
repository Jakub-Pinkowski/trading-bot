import pytest


@pytest.fixture
def sample_trades():
    return [
        {
            "execution_id": "1",
            "trade_time": "20250401-10:30:00",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "price": 150.0
        },
        {
            "execution_id": "2",
            "trade_time": "20250401-14:45:00",
            "symbol": "MSFT",
            "side": "SELL",
            "quantity": 50,
            "price": 300.0
        },
        {
            "execution_id": "3",
            "trade_time": "20250402-09:15:00",
            "symbol": "GOOGL",
            "side": "BUY",
            "quantity": 25,
            "price": 2500.0
        }
    ]
