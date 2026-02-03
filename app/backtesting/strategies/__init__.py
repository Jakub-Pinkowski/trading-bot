"""
Trading Strategies Module

This module provides concrete strategy implementations for backtesting.
All strategies inherit from BaseStrategy and implement the template methods:
- add_indicators(df): Add technical indicators to the dataframe
- generate_signals(df): Generate trading signals based on indicators
"""

from app.backtesting.strategies.bollinger_bands import BollingerBandsStrategy
from app.backtesting.strategies.ema import EMACrossoverStrategy
from app.backtesting.strategies.ichimoku_cloud import IchimokuCloudStrategy
from app.backtesting.strategies.macd import MACDStrategy
from app.backtesting.strategies.rsi import RSIStrategy

__all__ = [
    'BollingerBandsStrategy',
    'EMACrossoverStrategy',
    'IchimokuCloudStrategy',
    'MACDStrategy',
    'RSIStrategy',
]
