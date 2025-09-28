#!/usr/bin/env python3
"""
ROOK Multi-Model Runner - Test multiple trained models
------------------------------------------------------
Runs ROOK agents with different model configurations using the MAHAROOK framework.
"""

import argparse
import asyncio
import time
from pathlib import Path
from typing import Dict, List

import yaml
from loguru import logger

from maharook.agents.rook.agent import RookAgent, RookConfig
from maharook.agents.rook.brain import MarketFeatures
from maharook.blockchain.client import BaseClient
from maharook.core.config import settings


class RookModelRunner:
    """Runs ROOK agents with different model configurations."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.agent = None

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        logger.info("📋 Loaded configuration: {}", self.config_path.name)
        return config

    def create_rook_agent(self) -> RookAgent:
        """Create RookAgent from configuration."""
        agent_config = self.config.get("agent", {})
        model_config = self.config.get("model", {})
        portfolio_config = self.config.get("portfolio", {})
        risk_config = self.config.get("risk", {})
        execution_config = self.config.get("execution", {})
        blockchain_config = self.config.get("blockchain", {})

        # Create RookConfig
        rook_config = RookConfig(
            pair=agent_config.get("pair", "WETH_USDC"),
            pool_id=agent_config.get("pool_id"),
            fee_tier=agent_config.get("fee_tier", 0.0005),

            # Model configuration
            model_name=model_config.get("model_name", "qwen2.5:1.5b"),
            model_provider=model_config.get("model_provider", "ollama"),
            adapter_path=model_config.get("adapter_path"),

            # Portfolio settings
            target_allocation=portfolio_config.get("target_allocation", 0.5),
            initial_eth_balance=portfolio_config.get("initial_eth_balance", 10.0),
            initial_usdc_balance=portfolio_config.get("initial_usdc_balance", 10000.0),

            # Risk limits
            max_slippage=risk_config.get("max_slippage", 0.005),
            max_position_size=risk_config.get("max_position_size", 0.1),
            max_daily_trades=risk_config.get("max_daily_trades", 50),
            min_confidence=risk_config.get("min_confidence", 0.3),

            # Execution settings
            default_slippage=execution_config.get("default_slippage", 0.002),
            default_deadline=execution_config.get("default_deadline", 15),
            min_trade_size=execution_config.get("min_trade_size", 0.01)
        )

        # Create blockchain client from configuration
        client = BaseClient(
            rpc_url=blockchain_config.get("rpc_url", "http://localhost:8545"),
            private_key=blockchain_config.get("private_key", "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80")
        )

        # Create agent
        agent = RookAgent(config=rook_config, client=client)
        agent.name = agent_config.get("name", "ROOK-Agent")

        logger.success("✅ Created ROOK agent: {}", agent.name)
        return agent

    def simulate_market_data(self) -> MarketFeatures:
        """Generate simulated market data for testing."""
        import random

        # Simulate realistic ETH/USDC market data
        base_price = 3500.0
        price_volatility = random.gauss(0, 0.02)  # 2% volatility
        current_price = base_price * (1 + price_volatility)

        return MarketFeatures(
            price=current_price,
            price_change_10m=random.gauss(0, 0.01),  # 1% 10-min change
            volatility=abs(random.gauss(0.02, 0.005)),  # Around 2% volatility
            volume_10m=random.uniform(500, 2000),  # 500-2000 ETH volume
            liquidity_depth=current_price * 10000,  # Depth estimate
            spread_bps=random.uniform(3, 8)  # 3-8 bps spread
        )

    async def run_trading_session(self, duration_minutes: int = 60):
        """Run a trading session with the agent."""
        logger.info("🚀 Starting trading session for {} minutes", duration_minutes)

        # Initialize agent
        self.agent = self.create_rook_agent()

        # Run trading loop
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        step_count = 0

        try:
            while time.time() < end_time:
                step_count += 1

                # Get market data
                market_features = self.simulate_market_data()

                # Agent step
                state = self.agent.step(market_features)

                # Log progress
                if step_count % 10 == 0:
                    logger.info(
                        "Step {}: Price ${:.2f}, Action: {}, Portfolio: ${:.2f}",
                        step_count,
                        market_features.price,
                        state.last_action.side if state.last_action else "NONE",
                        state.portfolio_state.total_value_usd
                    )

                # Wait between steps
                await asyncio.sleep(5)  # 5 second intervals

        except KeyboardInterrupt:
            logger.info("Trading session stopped by user after {} steps", step_count)
        except Exception as e:
            logger.error("Trading session failed: {}", e)
            raise

        logger.success("✅ Trading session completed: {} steps", step_count)

        # Generate final report
        if self.agent:
            await self._generate_report()

    async def _generate_report(self):
        """Generate performance report."""
        try:
            report = self.agent.get_performance_report()

            logger.info("📊 Performance Report:")
            logger.info("Agent: {}", self.agent.name)

            if "portfolio" in report:
                portfolio = report["portfolio"]
                logger.info("Portfolio Value: ${:.2f}", portfolio.get("total_value_usd", 0))
                logger.info("ETH Balance: {:.6f}", portfolio.get("eth_balance", 0))
                logger.info("USDC Balance: {:.2f}", portfolio.get("usdc_balance", 0))

            if "performance" in report:
                performance = report["performance"]
                logger.info("Total Return: {:.2%}", performance.get("total_return", 0))
                logger.info("Total Trades: {}", performance.get("total_trades", 0))
                logger.info("Win Rate: {:.1%}", performance.get("win_rate", 0))

        except Exception as e:
            logger.warning("Failed to generate report: {}", e)


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run ROOK agents with different models")
    parser.add_argument("--config", required=True, help="Path to model configuration YAML")
    parser.add_argument("--duration", type=int, default=60, help="Trading session duration in minutes")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    # Run model
    runner = RookModelRunner(args.config)
    await runner.run_trading_session(args.duration)


if __name__ == "__main__":
    asyncio.run(main())