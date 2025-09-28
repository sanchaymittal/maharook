#!/usr/bin/env python3
"""
Agent Registry - Shared state management for ROOK agents
--------------------------------------------------------
File-based registry for tracking running ROOK agents across processes.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class AgentState:
    """Agent state for registry tracking."""
    agent_id: str
    name: str
    config_path: str
    model_type: str
    model_name: str
    pair: str
    status: str  # "starting", "running", "stopped", "error"
    pid: int
    start_time: float
    last_update: float
    total_steps: int
    total_trades: int
    current_nav: float
    total_pnl: float
    last_action: Optional[str] = None
    last_amount: Optional[float] = None
    last_price: Optional[float] = None
    last_confidence: Optional[float] = None
    last_reasoning: Optional[str] = None


class AgentRegistry:
    """File-based registry for tracking ROOK agents."""

    def __init__(self, registry_dir: Optional[Path] = None):
        if registry_dir is None:
            registry_dir = Path.home() / ".maharook" / "agents"

        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Cleanup stale entries on init
        self._cleanup_stale_agents()

        logger.info("📋 Agent registry initialized: {}", self.registry_dir)

    def register_agent(self, agent_state: AgentState) -> bool:
        """Register a new agent in the registry."""
        try:
            agent_file = self.registry_dir / f"{agent_state.agent_id}.json"

            # Convert to dict and write
            agent_data = asdict(agent_state)
            with open(agent_file, 'w') as f:
                json.dump(agent_data, f, indent=2)

            logger.info("✅ Registered agent: {} ({})", agent_state.name, agent_state.agent_id)
            return True

        except Exception as e:
            logger.error("❌ Failed to register agent {}: {}", agent_state.agent_id, e)
            return False

    def update_agent(self, agent_id: str, **updates) -> bool:
        """Update agent state in registry."""
        try:
            agent_file = self.registry_dir / f"{agent_id}.json"

            if not agent_file.exists():
                logger.warning("Agent {} not found in registry", agent_id)
                return False

            # Read current state
            with open(agent_file, 'r') as f:
                agent_data = json.load(f)

            # Apply updates
            agent_data.update(updates)
            agent_data['last_update'] = time.time()

            # Write back
            with open(agent_file, 'w') as f:
                json.dump(agent_data, f, indent=2)

            return True

        except Exception as e:
            logger.error("❌ Failed to update agent {}: {}", agent_id, e)
            return False

    def unregister_agent(self, agent_id: str) -> bool:
        """Remove agent from registry."""
        try:
            agent_file = self.registry_dir / f"{agent_id}.json"

            if agent_file.exists():
                agent_file.unlink()
                logger.info("🗑️ Unregistered agent: {}", agent_id)
                return True
            else:
                logger.warning("Agent {} not found for unregistration", agent_id)
                return False

        except Exception as e:
            logger.error("❌ Failed to unregister agent {}: {}", agent_id, e)
            return False

    def get_agent(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state by ID."""
        try:
            agent_file = self.registry_dir / f"{agent_id}.json"

            if not agent_file.exists():
                return None

            with open(agent_file, 'r') as f:
                agent_data = json.load(f)

            return AgentState(**agent_data)

        except Exception as e:
            logger.error("❌ Failed to get agent {}: {}", agent_id, e)
            return None

    def list_agents(self, include_stale: bool = False) -> List[AgentState]:
        """List all registered agents."""
        agents = []
        current_time = time.time()

        try:
            for agent_file in self.registry_dir.glob("*.json"):
                try:
                    with open(agent_file, 'r') as f:
                        agent_data = json.load(f)

                    agent_state = AgentState(**agent_data)

                    # Check if agent is stale (no update in 30 seconds)
                    if not include_stale and (current_time - agent_state.last_update) > 30:
                        # Check if process is still running
                        if not self._is_process_running(agent_state.pid):
                            # Mark as stopped and optionally remove
                            agent_state.status = "stopped"
                        continue

                    agents.append(agent_state)

                except Exception as e:
                    logger.error("❌ Failed to load agent from {}: {}", agent_file, e)
                    continue

            return sorted(agents, key=lambda a: a.start_time, reverse=True)

        except Exception as e:
            logger.error("❌ Failed to list agents: {}", e)
            return []

    def _is_process_running(self, pid: int) -> bool:
        """Check if process is still running."""
        try:
            os.kill(pid, 0)  # Doesn't actually kill, just checks if PID exists
            return True
        except (OSError, ProcessLookupError):
            return False

    def _cleanup_stale_agents(self) -> None:
        """Clean up stale agent entries."""
        current_time = time.time()

        for agent_file in self.registry_dir.glob("*.json"):
            try:
                with open(agent_file, 'r') as f:
                    agent_data = json.load(f)

                # Remove entries older than 1 hour or from dead processes
                last_update = agent_data.get('last_update', 0)
                pid = agent_data.get('pid', 0)

                if (current_time - last_update) > 3600 or not self._is_process_running(pid):
                    agent_file.unlink()
                    logger.info("🧹 Cleaned up stale agent: {}", agent_data.get('agent_id', 'unknown'))

            except Exception as e:
                logger.error("❌ Failed to cleanup {}: {}", agent_file, e)
                # Remove corrupted files
                try:
                    agent_file.unlink()
                except:
                    pass


# Global registry instance
_registry_instance = None

def get_agent_registry() -> AgentRegistry:
    """Get global agent registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = AgentRegistry()
    return _registry_instance