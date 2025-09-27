"""
Portfolio Management for ROOK Agents
------------------------------------
Handles portfolio tracking, allocation management, and performance metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from loguru import logger

from .brain import PortfolioState


@dataclass
class PortfolioSnapshot:
    """Immutable portfolio snapshot."""
    timestamp: datetime
    eth_balance: float
    usdc_balance: float
    eth_price: float
    total_value_usd: float
    target_allocation: float
    current_allocation: float
    unrealized_pnl: float


@dataclass
class Trade:
    """Record of a completed trade."""
    timestamp: datetime
    side: str  # BUY, SELL
    size: float  # Amount traded in ETH
    price: float  # Execution price
    slippage: float  # Actual slippage
    gas_cost: float  # Gas cost in ETH
    tx_hash: str
    success: bool


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    total_trades: int = 0
    total_volume: float = 0.0
    total_gas_costs: float = 0.0


class Portfolio:
    """
    Portfolio management for ROOK agents.

    Features:
    - Real-time balance tracking
    - Target allocation management
    - Trade history and performance metrics
    - P&L calculation
    - Risk monitoring
    """

    def __init__(
        self,
        target_eth_allocation: float = 0.5,
        initial_eth_balance: float = 0.0,
        initial_usdc_balance: float = 0.0,
        client=None
    ):
        """Initialize portfolio.

        Args:
            target_eth_allocation: Target ETH allocation (0-1)
            initial_eth_balance: Starting ETH balance
            initial_usdc_balance: Starting USDC balance
            client: Blockchain client for balance queries
        """
        self.target_eth_allocation = target_eth_allocation
        self.client = client

        # Current balances
        self.eth_balance = Decimal(str(initial_eth_balance))
        self.usdc_balance = Decimal(str(initial_usdc_balance))

        # Performance tracking
        self.initial_value = 0.0
        self.trade_history: list[Trade] = []
        self.snapshots: list[PortfolioSnapshot] = []

        # Risk parameters
        self.max_position_size = 0.1  # Max 10% of portfolio per trade
        self.max_daily_trades = 50
        self.max_drawdown_threshold = 0.2  # 20% max drawdown

        logger.info(
            "Portfolio initialized: target_allocation={:.1%}, ETH={:.6f}, USDC={:.2f}",
            target_eth_allocation,
            float(self.eth_balance),
            float(self.usdc_balance)
        )

    def update_balances(self, eth_balance: float, usdc_balance: float):
        """Update portfolio balances."""
        self.eth_balance = Decimal(str(eth_balance))
        self.usdc_balance = Decimal(str(usdc_balance))

        logger.debug(
            "Portfolio balances updated: ETH={:.6f}, USDC={:.2f}",
            float(self.eth_balance),
            float(self.usdc_balance)
        )

    def update_from_client(self):
        """Update balances from blockchain client."""
        if not self.client:
            logger.warning("No client configured for balance updates")
            return

        try:
            # Get ETH balance
            eth_balance = self.client.get_balance(self.client.address)

            # Get USDC balance (assuming USDC contract integration)
            # This would be implemented based on the actual client interface
            usdc_balance = float(self.usdc_balance)  # Placeholder

            self.update_balances(eth_balance, usdc_balance)

        except Exception as e:
            logger.error("Failed to update balances from client: {}", e)

    def snapshot(self, eth_price: float) -> PortfolioState:
        """Create current portfolio state snapshot."""
        eth_balance = float(self.eth_balance)
        usdc_balance = float(self.usdc_balance)
        total_value = eth_balance * eth_price + usdc_balance
        current_allocation = (eth_balance * eth_price) / total_value if total_value > 0 else 0

        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        if self.initial_value > 0:
            unrealized_pnl = (total_value - self.initial_value) / self.initial_value

        return PortfolioState(
            eth_balance=eth_balance,
            usdc_balance=usdc_balance,
            total_value_usd=total_value,
            target_allocation=self.target_eth_allocation,
            current_allocation=current_allocation,
            unrealized_pnl=unrealized_pnl
        )

    def record_trade(self, trade: Trade):
        """Record a completed trade."""
        self.trade_history.append(trade)

        # Update balances based on trade
        if trade.success:
            if trade.side == "BUY":
                # Bought ETH with USDC
                cost = trade.size * trade.price
                self.eth_balance += Decimal(str(trade.size))
                self.usdc_balance -= Decimal(str(cost))
            elif trade.side == "SELL":
                # Sold ETH for USDC
                proceeds = trade.size * trade.price
                self.eth_balance -= Decimal(str(trade.size))
                self.usdc_balance += Decimal(str(proceeds))

            # Subtract gas costs
            self.eth_balance -= Decimal(str(trade.gas_cost))

            logger.info(
                "Trade recorded: {} {:.6f} ETH @ ${:.2f} (tx: {})",
                trade.side,
                trade.size,
                trade.price,
                trade.tx_hash[:10] + "..." if len(trade.tx_hash) > 10 else trade.tx_hash
            )

    def save_snapshot(self, eth_price: float):
        """Save current portfolio snapshot."""
        state = self.snapshot(eth_price)
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            eth_balance=state.eth_balance,
            usdc_balance=state.usdc_balance,
            eth_price=eth_price,
            total_value_usd=state.total_value_usd,
            target_allocation=state.target_allocation,
            current_allocation=state.current_allocation,
            unrealized_pnl=state.unrealized_pnl
        )
        self.snapshots.append(snapshot)

        # Set initial value if first snapshot
        if len(self.snapshots) == 1:
            self.initial_value = state.total_value_usd

    def needs_rebalancing(self, eth_price: float, threshold: float = 0.05) -> tuple[bool, str, float]:
        """Check if portfolio needs rebalancing.

        Args:
            eth_price: Current ETH price
            threshold: Rebalancing threshold (default 5%)

        Returns:
            (needs_rebalancing, direction, amount)
        """
        state = self.snapshot(eth_price)

        deviation = abs(state.current_allocation - state.target_allocation)
        if deviation <= threshold:
            return False, "HOLD", 0.0

        if state.current_allocation > state.target_allocation:
            # Need to sell ETH
            excess_value = (state.current_allocation - state.target_allocation) * state.total_value_usd
            amount = excess_value / eth_price
            return True, "SELL", amount
        else:
            # Need to buy ETH
            needed_value = (state.target_allocation - state.current_allocation) * state.total_value_usd
            amount = min(needed_value / eth_price, state.usdc_balance / eth_price)
            return True, "BUY", amount

    def calculate_performance(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not self.trade_history:
            return PerformanceMetrics()

        successful_trades = [t for t in self.trade_history if t.success]

        # Basic metrics
        total_trades = len(successful_trades)
        total_volume = sum(t.size * t.price for t in successful_trades)
        total_gas_costs = sum(t.gas_cost for t in successful_trades)

        # P&L calculation
        trade_pnls = []
        for i, trade in enumerate(successful_trades):
            # Simple P&L approximation - would need more sophisticated calculation
            if trade.side == "SELL" and i > 0:
                # Find corresponding buy to calculate P&L
                pnl = 0.0  # Placeholder - implement actual P&L calculation
                trade_pnls.append(pnl)

        # Performance ratios
        avg_trade_pnl = sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0.0
        win_rate = len([p for p in trade_pnls if p > 0]) / len(trade_pnls) if trade_pnls else 0.0

        # Total return
        total_return = 0.0
        if self.initial_value > 0 and self.snapshots:
            current_value = self.snapshots[-1].total_value_usd
            total_return = (current_value - self.initial_value) / self.initial_value

        # Max drawdown
        max_drawdown = 0.0
        if len(self.snapshots) > 1:
            peak_value = self.initial_value
            for snapshot in self.snapshots:
                peak_value = max(peak_value, snapshot.total_value_usd)
                drawdown = (peak_value - snapshot.total_value_usd) / peak_value
                max_drawdown = max(max_drawdown, drawdown)

        # Sharpe ratio (simplified)
        sharpe_ratio = 0.0
        if trade_pnls and len(trade_pnls) > 1:
            import statistics
            mean_return = statistics.mean(trade_pnls)
            std_return = statistics.stdev(trade_pnls)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0

        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_trade_pnl=avg_trade_pnl,
            total_trades=total_trades,
            total_volume=total_volume,
            total_gas_costs=total_gas_costs
        )

    def check_risk_limits(self, proposed_trade_size: float, eth_price: float) -> tuple[bool, str]:
        """Check if proposed trade violates risk limits.

        Args:
            proposed_trade_size: Size of proposed trade in ETH
            eth_price: Current ETH price

        Returns:
            (is_allowed, reason)
        """
        state = self.snapshot(eth_price)
        trade_value = proposed_trade_size * eth_price

        # Check maximum position size
        max_trade_value = state.total_value_usd * self.max_position_size
        if trade_value > max_trade_value:
            return False, f"Trade size exceeds maximum position limit: ${trade_value:.2f} > ${max_trade_value:.2f}"

        # Check daily trade count
        today_trades = len([
            t for t in self.trade_history
            if t.timestamp.date() == datetime.now().date()
        ])
        if today_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached: {today_trades} >= {self.max_daily_trades}"

        # Check drawdown threshold
        performance = self.calculate_performance()
        if performance.max_drawdown > self.max_drawdown_threshold:
            return False, f"Maximum drawdown exceeded: {performance.max_drawdown:.1%} > {self.max_drawdown_threshold:.1%}"

        return True, "Trade allowed"

    def get_summary(self, eth_price: float) -> dict[str, any]:
        """Get comprehensive portfolio summary."""
        state = self.snapshot(eth_price)
        performance = self.calculate_performance()

        return {
            "balances": {
                "eth": state.eth_balance,
                "usdc": state.usdc_balance,
                "total_value_usd": state.total_value_usd
            },
            "allocation": {
                "target": state.target_allocation,
                "current": state.current_allocation,
                "deviation": abs(state.current_allocation - state.target_allocation)
            },
            "performance": {
                "total_return": performance.total_return,
                "unrealized_pnl": state.unrealized_pnl,
                "max_drawdown": performance.max_drawdown,
                "sharpe_ratio": performance.sharpe_ratio,
                "win_rate": performance.win_rate
            },
            "activity": {
                "total_trades": performance.total_trades,
                "total_volume": performance.total_volume,
                "total_gas_costs": performance.total_gas_costs
            }
        }
