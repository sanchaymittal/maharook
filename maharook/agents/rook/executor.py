"""
Trading Executor for ROOK Agents
--------------------------------
Handles trade execution via Uniswap v4 swapper with risk management and monitoring.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from .brain import TradingAction
from .portfolio import Trade
from .swapper import SwapParams, SwapResult, UniswapV4Swapper


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    success: bool
    trade: Trade | None = None
    error_message: str | None = None
    swap_result: SwapResult | None = None


class Executor:
    """
    Trading executor that interfaces with Uniswap v4 swapper.

    Features:
    - Action execution with risk checks
    - Slippage and gas optimization
    - Trade result parsing and recording
    - Error handling and recovery
    """

    def __init__(
        self,
        swapper: UniswapV4Swapper,
        config: dict[str, Any] | None = None
    ):
        """Initialize executor.

        Args:
            swapper: Uniswap v4 swapper instance
            config: Executor configuration
        """
        self.swapper = swapper
        self.config = config or {}

        # Default execution parameters
        self.default_slippage = self.config.get("default_slippage", 0.005)  # 0.5%
        self.default_deadline = self.config.get("default_deadline", 20)  # 20 minutes
        self.min_trade_size = self.config.get("min_trade_size", 0.001)  # 0.001 ETH
        self.max_slippage = self.config.get("max_slippage", 0.05)  # 5%

        logger.info("Executor initialized with swapper: {}", type(swapper).__name__)

    def execute(self, action: TradingAction, market_price: float) -> ExecutionResult:
        """Execute a trading action.

        Args:
            action: Trading action from Brain
            market_price: Current market price for validation

        Returns:
            Execution result with trade details
        """
        try:
            # Validate action
            validation_result = self._validate_action(action, market_price)
            if not validation_result[0]:
                return ExecutionResult(
                    success=False,
                    error_message=f"Action validation failed: {validation_result[1]}"
                )

            # Handle HOLD action
            if action.side == "HOLD" or action.size <= 0:
                logger.info("HOLD action - no trade executed")
                return ExecutionResult(success=True)

            # Execute the trade
            swap_result = self._execute_swap(action, market_price)

            # Create trade record
            trade = self._create_trade_record(action, swap_result, market_price)

            logger.info(
                "Trade executed: {} {:.6f} ETH, success: {}, tx: {}",
                action.side,
                action.size,
                swap_result.success,
                swap_result.transaction_hash or "N/A"
            )

            return ExecutionResult(
                success=swap_result.success,
                trade=trade,
                swap_result=swap_result,
                error_message=swap_result.error_message if not swap_result.success else None
            )

        except Exception as e:
            logger.error("Execution failed: {}", e)
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )

    def _validate_action(self, action: TradingAction, market_price: float) -> tuple[bool, str]:
        """Validate trading action before execution."""
        # Check minimum trade size
        if action.side != "HOLD" and action.size < self.min_trade_size:
            return False, f"Trade size below minimum: {action.size} < {self.min_trade_size}"

        # Check maximum slippage
        if action.slippage > self.max_slippage:
            return False, f"Slippage too high: {action.slippage} > {self.max_slippage}"

        # Check confidence threshold
        min_confidence = self.config.get("min_confidence", 0.1)
        if action.confidence < min_confidence:
            return False, f"Confidence too low: {action.confidence} < {min_confidence}"

        # Validate deadline
        if action.deadline < 1 or action.deadline > 60:
            return False, f"Invalid deadline: {action.deadline} minutes"

        # Check side validity
        if action.side not in ["BUY", "SELL", "HOLD"]:
            return False, f"Invalid trade side: {action.side}"

        return True, "Action valid"

    def _execute_swap(self, action: TradingAction, market_price: float) -> SwapResult:
        """Execute the actual swap via Uniswap v4."""
        try:
            if action.side == "BUY":
                # Buy ETH with USDC
                return self._execute_buy(action, market_price)
            elif action.side == "SELL":
                # Sell ETH for USDC
                return self._execute_sell(action, market_price)
            else:
                raise ValueError(f"Cannot execute swap for side: {action.side}")

        except Exception as e:
            logger.error("Swap execution failed: {}", e)
            return SwapResult(
                success=False,
                amount_in=action.size,
                amount_out=0.0,
                error_message=str(e)
            )

    def _execute_buy(self, action: TradingAction, market_price: float) -> SwapResult:
        """Execute ETH buy (USDC -> ETH)."""
        # Calculate USDC amount needed
        usdc_amount = action.size * market_price

        swap_params = SwapParams(
            token_in="USDC",
            token_out="ETH",
            amount_in=usdc_amount,
            slippage_tolerance=action.slippage,
            deadline_minutes=action.deadline
        )

        logger.debug(
            "Executing BUY: {:.6f} ETH ({:.2f} USDC) with {:.1%} slippage",
            action.size,
            usdc_amount,
            action.slippage
        )

        return self.swapper.swap_exact_input_single(swap_params)

    def _execute_sell(self, action: TradingAction, market_price: float) -> SwapResult:
        """Execute ETH sell (ETH -> USDC)."""
        swap_params = SwapParams(
            token_in="ETH",
            token_out="USDC",
            amount_in=action.size,
            slippage_tolerance=action.slippage,
            deadline_minutes=action.deadline
        )

        logger.debug(
            "Executing SELL: {:.6f} ETH with {:.1%} slippage",
            action.size,
            action.slippage
        )

        return self.swapper.swap_exact_input_single(swap_params)

    def _create_trade_record(
        self,
        action: TradingAction,
        swap_result: SwapResult,
        market_price: float
    ) -> Trade:
        """Create trade record from execution results."""
        # Calculate effective price
        if swap_result.success and swap_result.amount_out > 0:
            if action.side == "BUY":
                # Bought ETH with USDC
                effective_price = swap_result.amount_in / swap_result.amount_out
            else:  # SELL
                # Sold ETH for USDC
                effective_price = swap_result.amount_out / swap_result.amount_in
        else:
            effective_price = market_price

        # Calculate actual slippage
        actual_slippage = 0.0
        if swap_result.success:
            if action.side == "BUY":
                expected_eth = swap_result.amount_in / market_price
                actual_slippage = (expected_eth - swap_result.amount_out) / expected_eth
            else:  # SELL
                expected_usdc = swap_result.amount_in * market_price
                actual_slippage = (expected_usdc - swap_result.amount_out) / expected_usdc

        # Estimate gas cost (would be parsed from receipt in production)
        gas_cost = 0.002  # Approximate gas cost in ETH

        return Trade(
            timestamp=datetime.now(),
            side=action.side,
            size=action.size,
            price=effective_price,
            slippage=actual_slippage,
            gas_cost=gas_cost,
            tx_hash=swap_result.transaction_hash or "",
            success=swap_result.success
        )

    def get_execution_stats(self) -> dict[str, Any]:
        """Get executor performance statistics."""
        # This would track execution metrics over time
        return {
            "total_executions": 0,  # Would track actual statistics
            "success_rate": 0.0,
            "avg_slippage": 0.0,
            "avg_gas_cost": 0.0
        }

    def update_config(self, new_config: dict[str, Any]):
        """Update executor configuration."""
        self.config.update(new_config)

        # Update derived parameters
        self.default_slippage = self.config.get("default_slippage", self.default_slippage)
        self.default_deadline = self.config.get("default_deadline", self.default_deadline)
        self.min_trade_size = self.config.get("min_trade_size", self.min_trade_size)
        self.max_slippage = self.config.get("max_slippage", self.max_slippage)

        logger.info("Executor configuration updated")

    def estimate_gas_cost(self, action: TradingAction) -> float:
        """Estimate gas cost for a trading action."""
        # Base gas estimates for different operations
        base_gas = {
            "BUY": 150000,   # USDC -> ETH
            "SELL": 120000,  # ETH -> USDC
            "HOLD": 0
        }

        gas_limit = base_gas.get(action.side, 150000)

        # Get current gas price (would integrate with gas oracle)
        gas_price_gwei = 20  # Placeholder
        gas_price_wei = gas_price_gwei * 10**9

        # Calculate cost in ETH
        gas_cost_wei = gas_limit * gas_price_wei
        gas_cost_eth = gas_cost_wei / 10**18

        return gas_cost_eth

    def dry_run(self, action: TradingAction, market_price: float) -> dict[str, Any]:
        """Perform dry run without executing trade."""
        validation_result = self._validate_action(action, market_price)
        gas_cost = self.estimate_gas_cost(action)

        result = {
            "valid": validation_result[0],
            "validation_message": validation_result[1],
            "estimated_gas_cost": gas_cost,
            "action": {
                "side": action.side,
                "size": action.size,
                "slippage": action.slippage,
                "confidence": action.confidence
            }
        }

        if action.side in ["BUY", "SELL"]:
            if action.side == "BUY":
                result["estimated_cost_usdc"] = action.size * market_price
            else:
                result["estimated_proceeds_usdc"] = action.size * market_price

        return result
