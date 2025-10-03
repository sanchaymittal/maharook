"""
Trading Strategy Components
--------------------------
Basic strategy components and order definitions for ROOK agents.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    """Trading order specification."""
    token_in: str
    token_out: str
    amount_in: float
    slippage_tolerance: float = 0.005
    pool_fee: int = 3000  # 0.3% default
    deadline_minutes: int = 20


class Strategy:
    """Base strategy class for trading logic."""

    def __init__(self, name: str) -> None:
        self.name: str = name

    def generate_signal(self, market_data: dict[str, float]) -> Optional[Order]:
        """Generate trading signal based on market data."""
        raise NotImplementedError("Subclasses must implement generate_signal")


class SimpleRebalanceStrategy(Strategy):
    """Simple rebalancing strategy."""

    def __init__(self, target_allocation: float = 0.5, threshold: float = 0.05) -> None:
        super().__init__("SimpleRebalance")
        self.target_allocation: float = target_allocation
        self.threshold: float = threshold

    def generate_signal(self, market_data: dict[str, float]) -> Optional[Order]:
        """Generate rebalancing order if needed."""
        current_allocation = market_data.get("current_eth_allocation", 0.5)
        deviation = abs(current_allocation - self.target_allocation)

        if deviation > self.threshold:
            if current_allocation > self.target_allocation:
                # Sell ETH
                return Order(
                    token_in="ETH",
                    token_out="USDC",
                    amount_in=market_data.get("excess_eth", 0.0)
                )
            else:
                # Buy ETH
                return Order(
                    token_in="USDC",
                    token_out="ETH",
                    amount_in=market_data.get("usdc_to_spend", 0.0)
                )

        return None
