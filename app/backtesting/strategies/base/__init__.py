"""Base strategy components for backtesting."""

from app.backtesting.strategies.base.base_strategy import BaseStrategy
from app.backtesting.strategies.base.contract_switch_handler import ContractSwitchHandler
from app.backtesting.strategies.base.position_manager import PositionManager
from app.backtesting.strategies.base.trailing_stop_manager import TrailingStopManager

__all__ = [
    "BaseStrategy",
    "PositionManager",
    "TrailingStopManager",
    "ContractSwitchHandler",
]
