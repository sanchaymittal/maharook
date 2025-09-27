"""
ROOK Agent Module
----------------
Autonomous trading agents with modular Brain-Portfolio-Executor architecture.
"""

from .agent import RookAgent, RookConfig, RookState
from .brain import Brain, MarketFeatures, PortfolioState, TradingAction
from .executor import ExecutionResult, Executor
from .portfolio import PerformanceMetrics, Portfolio, Trade
from .strategy import Order, SimpleRebalanceStrategy, Strategy
from .swapper import SwapParams, SwapResult, UniswapV4Swapper

__all__ = [
    # Core components
    "Brain",
    "Portfolio",
    "Executor",
    "UniswapV4Swapper",

    # Main agent
    "RookAgent",
    "RookConfig",
    "RookState",

    # Data structures
    "TradingAction",
    "MarketFeatures",
    "PortfolioState",
    "Trade",
    "PerformanceMetrics",
    "ExecutionResult",
    "SwapParams",
    "SwapResult",
    "Order",

    # Strategies
    "Strategy",
    "SimpleRebalanceStrategy",
]
