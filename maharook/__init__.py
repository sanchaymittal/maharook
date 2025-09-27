"""
Maharook: Agentic Trading Arena for DeFi
========================================

A multi-agent trading system where autonomous agents (ROOKs) compete on Uniswap v4.
"""

from .core.config import settings
from .core.exceptions import MaharookError

__version__ = "0.1.0"
__author__ = "Maharook Team"

__all__ = [
    "settings",
    "MaharookError",
]
