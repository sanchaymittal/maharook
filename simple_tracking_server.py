#!/usr/bin/env python3
"""
Simple ROOK Trading Tracking Server - For WebSocket testing
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from run_rook_models import RookModelRunner


@dataclass
class TradingUpdate:
    """Structured trading update for frontend."""
    agent_id: str
    pair: str
    step: int
    action: str
    amount_eth: float
    price: float
    tx_hash: Optional[str]
    pnl: float
    nav: float
    timestamp: int
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentStatus:
    """Agent status tracking."""
    agent_id: str
    name: str
    config_path: str
    status: str  # "starting", "running", "paused", "stopped", "error"
    start_time: int
    last_update: int
    total_steps: int
    total_trades: int
    current_nav: float
    total_pnl: float


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected. Total: {}", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected. Total: {}", len(self.active_connections))

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning("Failed to send to client: {}", e)
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


class SimpleTrackingServer:
    """Simple tracking server for ROOK agents."""

    def __init__(self):
        self.app = FastAPI(title="ROOK Trading Tracker", version="1.0.0")
        self.connection_manager = ConnectionManager()
        self.agents: Dict[str, AgentStatus] = {}
        self.agent_runners: Dict[str, RookModelRunner] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}

        self._setup_routes()
        self._setup_cors()

    def _setup_cors(self):
        """Setup CORS for frontend integration."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connection_manager.connect(websocket)
            try:
                # Send current agent status on connect
                await self._send_agent_status(websocket)

                # Keep connection alive
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)

        @self.app.get("/agents")
        async def get_agents():
            """Get all agent statuses."""
            return {"agents": [asdict(agent) for agent in self.agents.values()]}

        @self.app.post("/agents/start")
        async def start_agent(request: dict, background_tasks: BackgroundTasks):
            """Start a new agent."""
            config_path = request.get("config_path")
            duration = request.get("duration", 60)

            if not config_path:
                return {"error": "config_path required"}

            agent_id = f"rook_{int(time.time())}_{uuid.uuid4().hex[:8]}"

            # Create agent runner
            try:
                runner = RookModelRunner(config_path)
                self.agent_runners[agent_id] = runner

                # Create agent status
                self.agents[agent_id] = AgentStatus(
                    agent_id=agent_id,
                    name=runner.config.get("agent", {}).get("name", "ROOK-Agent"),
                    config_path=config_path,
                    status="starting",
                    start_time=int(time.time()),
                    last_update=int(time.time()),
                    total_steps=0,
                    total_trades=0,
                    current_nav=10000.0,  # Initial value
                    total_pnl=0.0
                )

                # Start agent in background
                task = asyncio.create_task(self._run_agent_with_tracking(agent_id, duration))
                self.agent_tasks[agent_id] = task
                background_tasks.add_task(self._cleanup_agent, agent_id)

                await self._broadcast_agent_update(agent_id)
                return {"agent_id": agent_id, "status": "started"}

            except Exception as e:
                logger.error("Failed to start agent: {}", e)
                return {"error": str(e)}

        @self.app.post("/agents/{agent_id}/stop")
        async def stop_agent(agent_id: str):
            """Stop an agent."""
            if agent_id in self.agent_tasks:
                self.agent_tasks[agent_id].cancel()
                if agent_id in self.agents:
                    self.agents[agent_id].status = "stopped"
                await self._broadcast_agent_update(agent_id)
                return {"status": "stopped"}
            return {"error": "Agent not found"}

        @self.app.get("/")
        async def root():
            return {"message": "ROOK Trading Tracker API", "version": "1.0.0"}

    async def _run_agent_with_tracking(self, agent_id: str, duration: int):
        """Run agent with real-time tracking."""
        try:
            runner = self.agent_runners[agent_id]
            agent_status = self.agents[agent_id]

            # Update status
            agent_status.status = "running"
            await self._broadcast_agent_update(agent_id)

            # Initialize agent
            runner.agent = runner.create_rook_agent()

            # Run trading loop with tracking
            start_time = time.time()
            end_time = start_time + (duration * 60)
            step_count = 0

            while time.time() < end_time:
                step_count += 1

                # Get market data
                market_features = runner.simulate_market_data()

                # Agent step
                state = runner.agent.step(market_features)

                # Update tracking data
                agent_status.total_steps = step_count
                agent_status.last_update = int(time.time())

                # Extract trading information
                action = "HOLD"
                amount_eth = 0.0
                tx_hash = None
                confidence = 0.0
                reasoning = ""

                if state.last_action:
                    action = state.last_action.side
                    amount_eth = state.last_action.size
                    confidence = state.last_action.confidence
                    reasoning = getattr(state.last_action, 'reasoning', '')

                    if action in ["BUY", "SELL"]:
                        agent_status.total_trades += 1
                        # Mock transaction hash for demo
                        tx_hash = f"0x{uuid.uuid4().hex}"

                # Update portfolio values
                if state.portfolio_state:
                    agent_status.current_nav = state.portfolio_state.total_value_usd
                    agent_status.total_pnl = agent_status.current_nav - 10000.0  # Initial value

                # Create trading update
                update = TradingUpdate(
                    agent_id=agent_id,
                    pair=runner.config.get("agent", {}).get("pair", "WETH_USDC"),
                    step=step_count,
                    action=action,
                    amount_eth=amount_eth,
                    price=market_features.price,
                    tx_hash=tx_hash,
                    pnl=agent_status.total_pnl,
                    nav=agent_status.current_nav,
                    timestamp=int(time.time()),
                    confidence=confidence,
                    reasoning=reasoning
                )

                # Broadcast update
                await self.connection_manager.broadcast({
                    "type": "trading_update",
                    "data": update.to_dict()
                })

                # Broadcast agent status update every 10 steps
                if step_count % 10 == 0:
                    await self._broadcast_agent_update(agent_id)

                # Wait between steps
                await asyncio.sleep(5)

            # Mark as completed
            agent_status.status = "stopped"
            await self._broadcast_agent_update(agent_id)

        except asyncio.CancelledError:
            logger.info("Agent {} cancelled", agent_id)
            if agent_id in self.agents:
                self.agents[agent_id].status = "stopped"
            await self._broadcast_agent_update(agent_id)
        except Exception as e:
            logger.error("Agent {} failed: {}", agent_id, e)
            if agent_id in self.agents:
                self.agents[agent_id].status = "error"
            await self._broadcast_agent_update(agent_id)

    async def _broadcast_agent_update(self, agent_id: str):
        """Broadcast agent status update."""
        if agent_id in self.agents:
            await self.connection_manager.broadcast({
                "type": "agent_status",
                "data": asdict(self.agents[agent_id])
            })

    async def _send_agent_status(self, websocket: WebSocket):
        """Send current agent status to a specific client."""
        if self.agents:
            for agent in self.agents.values():
                await websocket.send_text(json.dumps({
                    "type": "agent_status",
                    "data": asdict(agent)
                }))

    async def _cleanup_agent(self, agent_id: str):
        """Cleanup agent resources."""
        # Wait for task to complete
        if agent_id in self.agent_tasks:
            try:
                await self.agent_tasks[agent_id]
            except asyncio.CancelledError:
                pass
            del self.agent_tasks[agent_id]

        # Keep agent status for history but clean up runner
        if agent_id in self.agent_runners:
            del self.agent_runners[agent_id]

    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the tracking server."""
        logger.info("🚀 Starting Simple ROOK Tracking Server on {}:{}", host, port)
        uvicorn.run(self.app, host=host, port=port, log_level="info")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple ROOK Trading Tracking Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")

    args = parser.parse_args()

    server = SimpleTrackingServer()
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()