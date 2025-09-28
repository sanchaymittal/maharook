#!/usr/bin/env python3
"""
ROOK Fork Tester - Test trained models against mainnet fork
----------------------------------------------------------
Simulates trading strategies using trained LoRA models on a local mainnet fork.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger
from web3 import Web3

from maharook.agents.rook.agent import RookAgent, RookConfig
from maharook.agents.rook.brain import MarketFeatures
from maharook.blockchain.client import BaseClient
from maharook.core.config import settings


@dataclass
class ForkTestConfig:
    """Configuration for fork testing."""
    # Fork settings
    fork_rpc_url: str = "http://localhost:8545"
    chain_id: int = 8453  # Base mainnet

    # Testing parameters
    test_duration_hours: int = 24
    step_interval_seconds: int = 60
    initial_eth_balance: float = 10.0
    initial_usdc_balance: float = 10000.0

    # Market simulation
    price_volatility: float = 0.02  # 2% volatility
    volume_base: float = 1000.0  # Base volume in ETH

    # Models to test
    models_to_test: List[str] = None

    def __post_init__(self):
        if self.models_to_test is None:
            self.models_to_test = [
                "training/models/finr1_trained",
                "training/models/eth_usdc_mlx_lora",
                "ollama/hf.co/Mungert/Fin-R1-GGUF:latest"
            ]


class ForkMarketSimulator:
    """Simulates realistic market conditions on the fork."""

    def __init__(self, web3: Web3, config: ForkTestConfig):
        self.web3 = web3
        self.config = config
        self.current_price = 3500.0  # Starting ETH price
        self.price_history = [self.current_price]
        self.volume_history = [self.config.volume_base]

    def get_current_market_features(self) -> MarketFeatures:
        """Generate realistic market features."""
        # Simulate price movement
        self._update_price()

        # Calculate features
        volatility = self._calculate_volatility()
        volume_10m = self._calculate_volume()
        liquidity_depth = self._estimate_liquidity()
        price_change_10m = self._calculate_price_change()

        return MarketFeatures(
            price=self.current_price,
            price_change_10m=price_change_10m,
            volatility=volatility,
            volume_10m=volume_10m,
            liquidity_depth=liquidity_depth,
            spread_bps=0.05  # Typical 5 bps spread for ETH/USDC
        )

    def _update_price(self):
        """Update price with realistic movement."""
        import random

        # Add some volatility
        price_change = random.gauss(0, self.config.price_volatility)
        self.current_price *= (1 + price_change)

        # Keep price in reasonable bounds
        self.current_price = max(1000, min(10000, self.current_price))
        self.price_history.append(self.current_price)

        # Keep history manageable
        if len(self.price_history) > 1440:  # 24 hours of minute data
            self.price_history = self.price_history[-1440:]

    def _calculate_volatility(self) -> float:
        """Calculate rolling volatility."""
        if len(self.price_history) < 10:
            return self.config.price_volatility

        returns = []
        for i in range(1, min(60, len(self.price_history))):
            returns.append((self.price_history[-i] - self.price_history[-i-1]) / self.price_history[-i-1])

        if not returns:
            return self.config.price_volatility

        import statistics
        return statistics.stdev(returns)

    def _calculate_volume(self) -> float:
        """Calculate current volume."""
        import random

        # Volume correlated with volatility
        volatility_factor = self._calculate_volatility() / self.config.price_volatility
        base_volume = self.config.volume_base * (0.5 + volatility_factor)

        # Add randomness
        volume = base_volume * random.uniform(0.5, 2.0)
        self.volume_history.append(volume)

        # Keep history manageable
        if len(self.volume_history) > 1440:
            self.volume_history = self.volume_history[-1440:]

        return volume

    def _estimate_liquidity(self) -> float:
        """Estimate liquidity depth."""
        # Rough estimate based on price and volume
        return self.current_price * self.config.volume_base * 10

    def _calculate_price_change(self) -> float:
        """Calculate 10-minute price change."""
        if len(self.price_history) < 10:
            return 0.0

        old_price = self.price_history[-10]
        return (self.current_price - old_price) / old_price


class RookForkTester:
    """Main testing orchestrator for ROOK agents on fork."""

    def __init__(self, config: ForkTestConfig):
        self.config = config
        self.web3 = Web3(Web3.HTTPProvider(config.fork_rpc_url))
        self.market_simulator = ForkMarketSimulator(self.web3, config)
        self.test_results = {}

        # Verify fork connection
        if not self.web3.is_connected():
            raise ConnectionError(f"Cannot connect to fork at {config.fork_rpc_url}")

        logger.info("Connected to fork: Chain ID {}, Block {}",
                   self.web3.eth.chain_id,
                   self.web3.eth.block_number)

    def create_test_agent(self, model_path: str, test_id: str) -> RookAgent:
        """Create a ROOK agent for testing."""
        # Extract model name from ollama path
        if "ollama" in model_path:
            model_name = model_path.replace("ollama/", "")
            model_provider = "ollama"
            adapter_path = None
        else:
            model_name = "local_model"
            model_provider = "local"
            adapter_path = model_path

        config = RookConfig(
            pair="WETH_USDC",
            model_name=model_name,
            model_provider=model_provider,
            adapter_path=adapter_path,
            initial_eth_balance=self.config.initial_eth_balance,
            initial_usdc_balance=self.config.initial_usdc_balance,
            target_allocation=0.5,
            max_slippage=0.005,
            max_position_size=0.1,
            min_confidence=0.3
        )

        # Create BaseClient for fork testing with first anvil account
        fork_private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # First anvil account
        fork_client = BaseClient(
            rpc_url=self.config.fork_rpc_url,
            private_key=fork_private_key
        )

        # Create agent with fork client
        agent = RookAgent(config=config, client=fork_client)

        # Monkey patch the executor to use WETH instead of ETH for fork testing
        original_execute_sell = agent.executor._execute_sell
        def patched_execute_sell(action, market_price):
            from maharook.agents.rook.executor import SwapParams
            from loguru import logger
            swap_params = SwapParams(
                token_in="WETH",  # Use WETH instead of ETH
                token_out="USDC",
                amount_in=action.size,
                slippage_tolerance=action.slippage,
                deadline_minutes=action.deadline
            )
            logger.debug(
                "Executing SELL: {:.6f} WETH with {:.1%} slippage",
                action.size,
                action.slippage
            )
            return agent.executor.swapper.swap_exact_input_single(swap_params)

        agent.executor._execute_sell = patched_execute_sell

        # Also patch execute_buy to use WETH
        original_execute_buy = agent.executor._execute_buy
        def patched_execute_buy(action, market_price):
            from maharook.agents.rook.executor import SwapParams
            from loguru import logger
            usdc_amount = action.size * market_price
            swap_params = SwapParams(
                token_in="USDC",
                token_out="WETH",  # Use WETH instead of ETH
                amount_in=usdc_amount,
                slippage_tolerance=action.slippage,
                deadline_minutes=action.deadline
            )
            logger.debug(
                "Executing BUY: {:.6f} USDC -> WETH with {:.1%} slippage",
                usdc_amount,
                action.slippage
            )
            return agent.executor.swapper.swap_exact_input_single(swap_params)

        agent.executor._execute_buy = patched_execute_buy
        agent.test_id = test_id

        return agent

    def run_test_suite(self):
        """Run comprehensive test suite."""
        logger.info("🧪 Starting ROOK fork test suite...")
        logger.info("Duration: {} hours, Interval: {}s",
                   self.config.test_duration_hours,
                   self.config.step_interval_seconds)

        # Create test agents
        agents = []
        for i, model_path in enumerate(self.config.models_to_test):
            test_id = f"test_{i}_{Path(model_path).name}"

            try:
                agent = self.create_test_agent(model_path, test_id)
                agents.append(agent)
                logger.info("✅ Created test agent: {} ({})", test_id, model_path)
            except Exception as e:
                logger.error("❌ Failed to create agent for {}: {}", model_path, e)

        if not agents:
            logger.error("No test agents created. Exiting.")
            return

        # Run parallel testing
        self._run_parallel_tests(agents)

        # Generate results
        self._generate_test_report()

    def _run_parallel_tests(self, agents: List[RookAgent]):
        """Run tests for all agents in parallel."""
        end_time = time.time() + (self.config.test_duration_hours * 3600)
        step_count = 0

        logger.info("🚀 Starting parallel testing with {} agents", len(agents))

        try:
            while time.time() < end_time:
                step_count += 1

                # Get current market features
                market_features = self.market_simulator.get_current_market_features()

                logger.debug("Step {}: Price ${:.2f}, Vol {:.1f}%, Volume {:.2f} ETH",
                           step_count,
                           market_features.price,
                           market_features.volatility * 100,
                           market_features.volume_10m)

                # Execute step for each agent
                for agent in agents:
                    try:
                        state = agent.step(market_features)

                        # Store results
                        if agent.test_id not in self.test_results:
                            self.test_results[agent.test_id] = {
                                "steps": [],
                                "trades": [],
                                "performance": []
                            }

                        self.test_results[agent.test_id]["steps"].append({
                            "step": step_count,
                            "timestamp": state.timestamp.isoformat(),
                            "price": market_features.price,
                            "action": state.last_action.side if state.last_action else "NONE",
                            "size": state.last_action.size if state.last_action else 0,
                            "confidence": state.last_action.confidence if state.last_action else 0,
                            "portfolio_value": state.portfolio_state.total_value_usd,
                            "eth_balance": state.portfolio_state.eth_balance,
                            "usdc_balance": state.portfolio_state.usdc_balance
                        })

                    except Exception as e:
                        logger.error("Agent {} step failed: {}", agent.test_id, e)

                # Log progress every 10 steps
                if step_count % 10 == 0:
                    self._log_progress(agents, step_count)

                # Save intermediate results every 100 steps
                if step_count % 100 == 0:
                    self._save_intermediate_results(step_count)

                # Wait for next step
                time.sleep(self.config.step_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Testing stopped by user after {} steps", step_count)
        except Exception as e:
            logger.error("Testing failed: {}", e)
            raise

        logger.info("✅ Testing completed after {} steps", step_count)

    def _log_progress(self, agents: List[RookAgent], step_count: int):
        """Log testing progress."""
        logger.info("Step {} Progress:", step_count)

        for agent in agents:
            try:
                report = agent.get_performance_report()
                performance = report.get("performance", {})

                logger.info("  {}: Return {:.2%}, Trades {}, Value ${:.2f}",
                           agent.test_id,
                           performance.get("total_return", 0),
                           performance.get("total_trades", 0),
                           report.get("portfolio", {}).get("total_value_usd", 0))

            except Exception as e:
                logger.warning("Failed to get progress for {}: {}", agent.test_id, e)

    def _save_intermediate_results(self, step_count: int):
        """Save intermediate results."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fork_test_intermediate_{timestamp}_step_{step_count}.json"

        with open(results_dir / filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        logger.debug("Intermediate results saved: {}", filename)

    def _generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("📊 Generating test report...")

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        with open(results_dir / f"fork_test_raw_{timestamp}.json", 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        # Generate summary report
        summary = self._create_summary_report()

        with open(results_dir / f"fork_test_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Generate CSV for analysis
        self._create_analysis_csv(results_dir / f"fork_test_analysis_{timestamp}.csv")

        # Print summary
        self._print_summary(summary)

        logger.success("✅ Test report generated in testing/results/")

    def _create_summary_report(self) -> Dict[str, Any]:
        """Create summary report of test results."""
        summary = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "duration_hours": self.config.test_duration_hours,
                "step_interval_seconds": self.config.step_interval_seconds,
                "initial_eth": self.config.initial_eth_balance,
                "initial_usdc": self.config.initial_usdc_balance
            },
            "models_tested": list(self.test_results.keys()),
            "performance_comparison": {}
        }

        for test_id, results in self.test_results.items():
            if not results["steps"]:
                continue

            steps = results["steps"]
            first_step = steps[0]
            last_step = steps[-1]

            initial_value = first_step["portfolio_value"]
            final_value = last_step["portfolio_value"]
            total_return = (final_value - initial_value) / initial_value

            # Calculate metrics
            trades = [s for s in steps if s["action"] != "NONE" and s["action"] != "HOLD"]
            total_trades = len(trades)

            # Win rate calculation
            profitable_trades = 0
            for i, trade in enumerate(trades):
                if i < len(trades) - 1:
                    current_value = trade["portfolio_value"]
                    next_value = trades[i + 1]["portfolio_value"]
                    if next_value > current_value:
                        profitable_trades += 1

            win_rate = profitable_trades / total_trades if total_trades > 0 else 0

            # Volatility of returns
            values = [s["portfolio_value"] for s in steps]
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]

            import statistics
            return_volatility = statistics.stdev(returns) if len(returns) > 1 else 0
            sharpe_ratio = (total_return / return_volatility) if return_volatility > 0 else 0

            # Max drawdown
            max_value = max(values)
            min_after_max = min(values[values.index(max_value):])
            max_drawdown = (max_value - min_after_max) / max_value

            summary["performance_comparison"][test_id] = {
                "total_return": total_return,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "final_portfolio_value": final_value,
                "total_steps": len(steps)
            }

        return summary

    def _create_analysis_csv(self, filepath: Path):
        """Create CSV file for detailed analysis."""
        all_data = []

        for test_id, results in self.test_results.items():
            for step in results["steps"]:
                row = {
                    "test_id": test_id,
                    **step
                }
                all_data.append(row)

        df = pd.DataFrame(all_data)
        df.to_csv(filepath, index=False)

        logger.info("Analysis CSV created: {}", filepath)

    def _print_summary(self, summary: Dict[str, Any]):
        """Print test summary to console."""
        print("\n" + "="*60)
        print("🏆 ROOK FORK TEST RESULTS")
        print("="*60)

        for test_id, performance in summary["performance_comparison"].items():
            print(f"\n📊 {test_id}:")
            print(f"   Total Return: {performance['total_return']:.2%}")
            print(f"   Total Trades: {performance['total_trades']}")
            print(f"   Win Rate: {performance['win_rate']:.1%}")
            print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {performance['max_drawdown']:.1%}")
            print(f"   Final Value: ${performance['final_portfolio_value']:.2f}")

        # Find best performer
        if summary["performance_comparison"]:
            best_model = max(summary["performance_comparison"].items(),
                           key=lambda x: x[1]["total_return"])

            print(f"\n🥇 Best Performer: {best_model[0]}")
            print(f"   Return: {best_model[1]['total_return']:.2%}")

        print("="*60)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test ROOK agents on mainnet fork")
    parser.add_argument("--fork-url", default="http://localhost:8545", help="Fork RPC URL")
    parser.add_argument("--duration", type=int, default=2, help="Test duration in hours")
    parser.add_argument("--interval", type=int, default=30, help="Step interval in seconds")
    parser.add_argument("--eth-balance", type=float, default=10.0, help="Initial ETH balance")
    parser.add_argument("--usdc-balance", type=float, default=10000.0, help="Initial USDC balance")
    parser.add_argument("--models", nargs="+", help="Model paths to test")

    args = parser.parse_args()

    # Create config
    config = ForkTestConfig(
        fork_rpc_url=args.fork_url,
        test_duration_hours=args.duration,
        step_interval_seconds=args.interval,
        initial_eth_balance=args.eth_balance,
        initial_usdc_balance=args.usdc_balance
    )

    if args.models:
        config.models_to_test = args.models

    # Run tests
    tester = RookForkTester(config)
    tester.run_test_suite()


if __name__ == "__main__":
    main()