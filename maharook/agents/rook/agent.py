"""
ROOK Agent - Autonomous Trading Agent
------------------------------------
Composition-based trading agent that combines Brain, Portfolio, and Executor components.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from maharook.core.config import settings
from .brain import Brain, MarketFeatures, TradingAction
from .executor import ExecutionResult, Executor
from .portfolio import Portfolio, PortfolioState
from .swapper import UniswapV4Swapper


@dataclass
class RookConfig:
    """Configuration for a ROOK agent."""
    # Trading pair
    pair: str = settings.trading.default_pair
    pool_id: str | None = None
    fee_tier: float = settings.trading.default_fee_tier

    # Model configuration
    model_name: str = settings.trading.default_model_name
    model_provider: str = settings.trading.default_model_provider
    adapter_path: str | None = None

    # Portfolio settings
    target_allocation: float = settings.trading.default_target_allocation
    initial_eth_balance: float = 0.0
    initial_usdc_balance: float = 0.0

    # Risk limits
    max_slippage: float = settings.trading.max_slippage
    max_position_size: float = settings.trading.max_position_size
    max_daily_trades: int = settings.trading.max_daily_trades
    min_confidence: float = settings.trading.min_confidence

    # Execution settings
    default_slippage: float = settings.trading.default_slippage
    default_deadline: int = settings.trading.default_deadline_minutes
    min_trade_size: float = settings.trading.min_trade_size_eth


@dataclass
class RookState:
    """Current state of a ROOK agent."""
    timestamp: datetime
    portfolio_state: PortfolioState
    market_features: MarketFeatures
    last_action: TradingAction | None = None
    last_execution: ExecutionResult | None = None
    performance_summary: dict[str, Any] | None = None


class RookAgent:
    """
    Autonomous trading agent with modular architecture.

    Components:
    - Brain: Decision-making (LLM or direct model inference)
    - Portfolio: Balance tracking and performance metrics
    - Executor: Trade execution via Uniswap v4
    """

    def __init__(
        self,
        config: RookConfig,
        client=None,
        swapper: UniswapV4Swapper | None = None
    ):
        """Initialize ROOK agent.

        Args:
            config: Agent configuration
            client: Blockchain client
            swapper: Uniswap swapper instance
        """
        self.config = config
        self.client = client
        self.agent_id = f"rook_{config.pair.replace('/', '_').lower()}_{int(time.time())}"

        # Initialize components
        self._initialize_brain()
        self._initialize_portfolio()
        self._initialize_executor(swapper)

        # State tracking
        self.current_state: RookState | None = None
        self.step_count = 0
        self.start_time = datetime.now()

        logger.info(
            "ROOK Agent initialized: {} (pair: {}, model: {})",
            self.agent_id,
            config.pair,
            config.model_name
        )

    def _initialize_brain(self):
        """Initialize the Brain component."""
        brain_config = {
            "openrouter_api_key": None,  # Would load from environment
            "ollama_url": "http://localhost:11434",
            "min_confidence": self.config.min_confidence
        }

        self.brain = Brain(
            model_name=self.config.model_name,
            model_provider=self.config.model_provider,
            adapter_path=self.config.adapter_path,
            config=brain_config
        )

    def _initialize_portfolio(self):
        """Initialize the Portfolio component."""
        self.portfolio = Portfolio(
            target_eth_allocation=self.config.target_allocation,
            initial_eth_balance=self.config.initial_eth_balance,
            initial_usdc_balance=self.config.initial_usdc_balance,
            client=self.client
        )

        # Set risk parameters from config
        self.portfolio.max_position_size = self.config.max_position_size
        self.portfolio.max_daily_trades = self.config.max_daily_trades

    def _initialize_executor(self, swapper: UniswapV4Swapper | None):
        """Initialize the Executor component."""
        if swapper is None:
            swapper = UniswapV4Swapper(client=self.client)

        executor_config = {
            "default_slippage": self.config.default_slippage,
            "default_deadline": self.config.default_deadline,
            "min_trade_size": self.config.min_trade_size,
            "max_slippage": self.config.max_slippage,
            "min_confidence": self.config.min_confidence
        }

        self.executor = Executor(swapper=swapper, config=executor_config)

    def step(self, market_features: MarketFeatures, market_context: dict[str, Any] | None = None) -> RookState:
        """Execute one trading step.

        Args:
            market_features: Current market features
            market_context: Additional market context

        Returns:
            Current ROOK state after step
        """
        self.step_count += 1
        step_start = time.time()

        try:
            logger.debug("ROOK step {} starting", self.step_count)

            # Update portfolio from blockchain if client available
            if self.client:
                self.portfolio.update_from_client()

            # Get current portfolio state
            portfolio_state = self.portfolio.snapshot(market_features.price)

            # Save portfolio snapshot
            self.portfolio.save_snapshot(market_features.price)

            # Check if trade is allowed by risk limits
            risk_check = self._check_risk_limits(market_features, portfolio_state)
            if not risk_check[0]:
                logger.warning("Trade blocked by risk limits: {}", risk_check[1])
                action = TradingAction(
                    side="HOLD",
                    size=0.0,
                    slippage=self.config.default_slippage,
                    deadline=self.config.default_deadline,
                    reasoning=f"Risk limit: {risk_check[1]}",
                    confidence=0.0
                )
                execution = ExecutionResult(success=True)
            else:
                # Make trading decision
                action = self.brain.decide(
                    features=market_features,
                    portfolio_state=portfolio_state,
                    market_state=market_context or {}
                )

                # Execute the action
                execution = self.executor.execute(action, market_features.price)

                # Record trade if successful
                if execution.success and execution.trade:
                    self.portfolio.record_trade(execution.trade)

            # Update current state
            self.current_state = RookState(
                timestamp=datetime.now(),
                portfolio_state=portfolio_state,
                market_features=market_features,
                last_action=action,
                last_execution=execution,
                performance_summary=self.portfolio.get_summary(market_features.price)
            )

            step_duration = time.time() - step_start
            logger.info(
                "ROOK step {} completed: {} {:.6f} ETH in {:.3f}s",
                self.step_count,
                action.side,
                action.size,
                step_duration
            )

            return self.current_state

        except Exception as e:
            logger.error("ROOK step {} failed: {}", self.step_count, e)
            raise

    def _check_risk_limits(self, market_features: MarketFeatures, portfolio_state: PortfolioState) -> tuple[bool, str]:
        """Check if trading is allowed by risk limits."""
        # Basic portfolio value check
        if portfolio_state.total_value_usd < 10:  # Minimum $10
            return False, "Portfolio value too low"

        # Check maximum drawdown
        performance = self.portfolio.calculate_performance()
        if performance.max_drawdown > self.portfolio.max_drawdown_threshold:
            return False, f"Maximum drawdown exceeded: {performance.max_drawdown:.1%}"

        # Check volatility limits
        if market_features.volatility > 0.1:  # 10% volatility threshold
            return False, f"Market volatility too high: {market_features.volatility:.1%}"

        return True, "Trading allowed"

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.current_state:
            return {"error": "No state available"}

        performance = self.portfolio.calculate_performance()
        portfolio_summary = self.current_state.performance_summary

        return {
            "agent_id": self.agent_id,
            "config": {
                "pair": self.config.pair,
                "model": self.config.model_name,
                "target_allocation": self.config.target_allocation
            },
            "runtime": {
                "start_time": self.start_time.isoformat(),
                "current_time": datetime.now().isoformat(),
                "total_steps": self.step_count,
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
            },
            "portfolio": portfolio_summary,
            "performance": {
                "total_return": performance.total_return,
                "sharpe_ratio": performance.sharpe_ratio,
                "max_drawdown": performance.max_drawdown,
                "win_rate": performance.win_rate,
                "total_trades": performance.total_trades,
                "total_volume": performance.total_volume
            },
            "last_action": {
                "side": self.current_state.last_action.side if self.current_state.last_action else None,
                "size": self.current_state.last_action.size if self.current_state.last_action else None,
                "confidence": self.current_state.last_action.confidence if self.current_state.last_action else None,
                "success": self.current_state.last_execution.success if self.current_state.last_execution else None
            }
        }

    def update_config(self, new_config: dict[str, Any]):
        """Update agent configuration."""
        # Update internal config
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Update component configs
        if "model_config" in new_config:
            self.brain.update_config(new_config["model_config"])

        if "executor_config" in new_config:
            self.executor.update_config(new_config["executor_config"])

        logger.info("ROOK configuration updated")

    def save_state(self, filepath: str):
        """Save agent state to file."""
        import json

        state_data = {
            "agent_id": self.agent_id,
            "config": self.config.__dict__,
            "step_count": self.step_count,
            "start_time": self.start_time.isoformat(),
            "performance_report": self.get_performance_report()
        }

        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)

        logger.info("ROOK state saved to {}", filepath)

    @classmethod
    def load_from_config_file(cls, config_path: str, client=None) -> 'RookAgent':
        """Load ROOK agent from configuration file."""
        import json

        with open(config_path) as f:
            config_data = json.load(f)

        config = RookConfig(**config_data)
        return cls(config=config, client=client)

    def run_autonomous(self, market_data_source, duration_hours: int = 24, step_interval: int = 30):
        """Run agent autonomously for specified duration.

        Args:
            market_data_source: Source of market data updates
            duration_hours: How long to run (hours)
            step_interval: Seconds between steps
        """
        end_time = datetime.now().timestamp() + (duration_hours * 3600)

        logger.info(
            "Starting autonomous trading for {} hours (interval: {}s)",
            duration_hours,
            step_interval
        )

        try:
            while time.time() < end_time:
                # Get current market data
                market_features = market_data_source.get_current_features()

                # Execute trading step
                state = self.step(market_features)

                # Log performance
                if self.step_count % 10 == 0:  # Every 10 steps
                    report = self.get_performance_report()
                    logger.info(
                        "Step {}: Return {:.2%}, Trades {}, Confidence {:.1%}",
                        self.step_count,
                        report["performance"]["total_return"],
                        report["performance"]["total_trades"],
                        state.last_action.confidence if state.last_action else 0
                    )

                # Wait for next step
                time.sleep(step_interval)

        except KeyboardInterrupt:
            logger.info("Autonomous trading stopped by user")
        except Exception as e:
            logger.error("Autonomous trading failed: {}", e)
            raise

        logger.info("Autonomous trading completed after {} steps", self.step_count)
