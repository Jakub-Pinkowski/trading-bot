"""
Backtesting module for trading strategy evaluation.

This module provides tools for backtesting trading strategies including
- MassTester: Run tests across multiple strategies, symbols, and timeframes
- StrategyAnalyzer: Analyze and compare strategy performance
- Strategy creation utilities: create_strategy, get_strategy_name
"""

from app.backtesting.mass_testing import MassTester
from app.backtesting.strategy_analysis import StrategyAnalyzer
from app.backtesting.strategy_factory import create_strategy, get_strategy_name

__all__ = [
    'MassTester',
    'StrategyAnalyzer',
    'create_strategy',
    'get_strategy_name',
]
