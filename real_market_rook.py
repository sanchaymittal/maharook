#!/usr/bin/env python3
"""
Real Market ROOK Agent - Uses actual ETH/USDC prices
"""

import asyncio
import os
import time
import requests
from pathlib import Path
from typing import Dict

import yaml
from loguru import logger

from maharook.agents.rook.agent import RookAgent, RookConfig
from maharook.agents.rook.brain import MarketFeatures
from maharook.blockchain.client import BaseClient
from maharook.core.agent_registry import get_agent_registry, AgentState


class RealMarketRook:
    """ROOK agent using real ETH/USDC market data."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.agent = None
        self.agent_id = None
        self.registry = get_agent_registry()
        self.total_steps = 0
        self.total_trades = 0

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        logger.info("📋 Loaded configuration: {}", self.config_path.name)
        return config

    def get_real_market_data(self) -> MarketFeatures:
        """Get real ETH/USDC market data from CoinGecko API."""
        try:
            # Get ETH price from CoinGecko
            response = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": "ethereum",
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_24hr_vol": "true"
                },
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            eth_data = data["ethereum"]
            current_price = eth_data["usd"]
            price_change_24h = eth_data.get("usd_24h_change", 0.0) / 100  # Convert to decimal
            volume_24h_usd = eth_data.get("usd_24h_vol", 0.0)

            # Estimate volume in ETH
            volume_24h_eth = volume_24h_usd / current_price if current_price > 0 else 0
            volume_10m = volume_24h_eth / 144  # Rough estimate (24h / 144 = 10min periods)

            # Calculate volatility based on recent price change
            volatility = abs(price_change_24h) if price_change_24h else 0.02

            logger.info("📊 Real market data: ETH/USD ${:.2f} ({:+.2%} 24h)",
                       current_price, price_change_24h)

            return MarketFeatures(
                price=current_price,
                price_change_10m=price_change_24h / 144,  # Rough 10min estimate
                volatility=volatility,
                volume_10m=volume_10m,
                liquidity_depth=current_price * 50000,  # Estimate based on major exchange depth
                spread_bps=5.0  # Typical ETH/USDC spread on major exchanges
            )

        except Exception as e:
            logger.warning("⚠️ Failed to get real market data, using fallback: {}", e)

            # Fallback to reasonable default if API fails
            return MarketFeatures(
                price=3500.0,  # Reasonable fallback price
                price_change_10m=0.0,
                volatility=0.02,
                volume_10m=1000.0,
                liquidity_depth=175000000.0,  # 50k ETH depth
                spread_bps=5.0
            )

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
        agent.name = f"{agent_config.get('name', 'ROOK-Agent')}-RealMarket"

        logger.success("✅ Created REAL MARKET ROOK agent: {}", agent.name)
        return agent

    async def run_trading_session(self, duration_minutes: int = 60):
        """Run a trading session with real market data."""
        logger.info("🌍 Starting REAL MARKET trading session for {} minutes", duration_minutes)

        # Initialize agent
        self.agent = self.create_rook_agent()

        # Generate unique agent ID and register
        self.agent_id = f"real_rook_{self.agent.config.pair.lower()}_{int(time.time())}"

        # Create agent state for registry
        agent_state = AgentState(
            agent_id=self.agent_id,
            name=self.agent.name,
            config_path=str(self.config_path),
            model_type=self.config.get("model", {}).get("model_provider", "ollama"),
            model_name=self.config.get("model", {}).get("model_name", "qwen2.5:1.5b"),
            pair=self.agent.config.pair,
            status="starting",
            pid=os.getpid(),
            start_time=time.time(),
            last_update=time.time(),
            total_steps=0,
            total_trades=0,
            current_nav=20000.0,  # Default starting capital
            total_pnl=0.0
        )

        # Register agent
        self.registry.register_agent(agent_state)

        # Update status to running
        self.registry.update_agent(self.agent_id, status="running")

        # Run trading loop
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        step_count = 0

        try:
            while time.time() < end_time:
                step_count += 1
                self.total_steps = step_count

                # Get REAL market data
                market_features = self.get_real_market_data()

                # Agent step with real data
                state = self.agent.step(market_features)

                # Update registry with current state
                updates = {
                    "total_steps": step_count,
                    "current_nav": state.portfolio_state.total_value_usd,
                    "total_pnl": state.portfolio_state.total_value_usd - 20000.0,
                    "last_update": time.time()
                }

                # Track trading actions
                if state.last_action:
                    updates["last_action"] = state.last_action.side
                    updates["last_amount"] = state.last_action.size
                    updates["last_price"] = market_features.price
                    updates["last_confidence"] = getattr(state.last_action, 'confidence', 0.0)
                    updates["last_reasoning"] = getattr(state.last_action, 'reasoning', '')

                    if state.last_action.side in ["BUY", "SELL"]:
                        self.total_trades += 1
                        updates["total_trades"] = self.total_trades

                self.registry.update_agent(self.agent_id, **updates)

                # Log progress with real market data
                if step_count % 5 == 0:
                    logger.info(
                        "🌍 Step {}: REAL ETH ${:.2f}, Action: {}, Portfolio: ${:.2f}",
                        step_count,
                        market_features.price,
                        state.last_action.side if state.last_action else "NONE",
                        state.portfolio_state.total_value_usd
                    )

                # Wait between steps (30 seconds for real market data)
                await asyncio.sleep(30)

        except KeyboardInterrupt:
            logger.info("Trading session stopped by user after {} steps", step_count)
            self.registry.update_agent(self.agent_id, status="stopped")
        except Exception as e:
            logger.error("Trading session failed: {}", e)
            self.registry.update_agent(self.agent_id, status="error")
            raise
        finally:
            # Mark as completed
            if self.agent_id:
                self.registry.update_agent(self.agent_id, status="completed", last_update=time.time())

        logger.success("✅ REAL MARKET trading session completed: {} steps", step_count)

        # Unregister after delay
        await asyncio.sleep(30)
        if self.agent_id:
            self.registry.unregister_agent(self.agent_id)


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ROOK agents with REAL market data")
    parser.add_argument("--config", required=True, help="Path to model configuration YAML")
    parser.add_argument("--duration", type=int, default=60, help="Trading session duration in minutes")

    args = parser.parse_args()

    # Run real market agent
    runner = RealMarketRook(args.config)
    await runner.run_trading_session(args.duration)


if __name__ == "__main__":
    asyncio.run(main())