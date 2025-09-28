#!/usr/bin/env python3
"""
Minimal ROOK Trading Tracking Server - Basic WebSocket functionality
"""

import asyncio
import json
import time
import uuid
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger


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


class MinimalTrackingServer:
    """Minimal tracking server for ROOK agents."""

    def __init__(self):
        self.app = FastAPI(title="ROOK Trading Tracker", version="1.0.0")
        self.connection_manager = ConnectionManager()
        self.agents: Dict[str, AgentStatus] = {}
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
            config_path = request.get("config_path", "demo_config.yaml")
            duration = request.get("duration", 5)  # 5 minutes default

            agent_id = f"rook_{int(time.time())}_{uuid.uuid4().hex[:8]}"

            # Create agent status
            self.agents[agent_id] = AgentStatus(
                agent_id=agent_id,
                name="ROOK-Demo-Agent",
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
            task = asyncio.create_task(self._run_demo_agent(agent_id, duration))
            self.agent_tasks[agent_id] = task
            background_tasks.add_task(self._cleanup_agent, agent_id)

            await self._broadcast_agent_update(agent_id)
            return {"agent_id": agent_id, "status": "started"}

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

    async def _run_demo_agent(self, agent_id: str, duration: int):
        """Run demo agent with simulated trading."""
        try:
            agent_status = self.agents[agent_id]

            # Update status to running
            agent_status.status = "running"
            await self._broadcast_agent_update(agent_id)

            # Run demo trading loop
            start_time = time.time()
            end_time = start_time + (duration * 60)
            step_count = 0
            base_price = 3500.0

            while time.time() < end_time:
                step_count += 1

                # Simulate market data
                price_change = random.gauss(0, 0.02)  # 2% volatility
                current_price = base_price * (1 + price_change)
                base_price = current_price  # Update base for next iteration

                # Simulate trading decisions
                action_prob = random.random()
                if action_prob < 0.1:  # 10% chance to buy
                    action = "BUY"
                    amount_eth = round(random.uniform(0.01, 0.1), 4)
                    agent_status.total_trades += 1
                    tx_hash = f"0x{uuid.uuid4().hex}"
                elif action_prob < 0.2:  # 10% chance to sell
                    action = "SELL"
                    amount_eth = round(random.uniform(0.01, 0.1), 4)
                    agent_status.total_trades += 1
                    tx_hash = f"0x{uuid.uuid4().hex}"
                else:  # 80% chance to hold
                    action = "HOLD"
                    amount_eth = 0.0
                    tx_hash = None

                # Update agent status
                agent_status.total_steps = step_count
                agent_status.last_update = int(time.time())

                # Simulate portfolio changes
                portfolio_change = random.gauss(0, 50)  # Random P&L
                agent_status.current_nav = max(5000.0, agent_status.current_nav + portfolio_change)
                agent_status.total_pnl = agent_status.current_nav - 10000.0

                # Create trading update
                update = TradingUpdate(
                    agent_id=agent_id,
                    pair="WETH_USDC",
                    step=step_count,
                    action=action,
                    amount_eth=amount_eth,
                    price=current_price,
                    tx_hash=tx_hash,
                    pnl=agent_status.total_pnl,
                    nav=agent_status.current_nav,
                    timestamp=int(time.time()),
                    confidence=random.uniform(0.5, 0.9),
                    reasoning=f"Market analysis suggests {action.lower()} at ${current_price:.2f}"
                )

                # Broadcast trading update
                await self.connection_manager.broadcast({
                    "type": "trading_update",
                    "data": update.to_dict()
                })

                # Broadcast agent status every 5 steps
                if step_count % 5 == 0:
                    await self._broadcast_agent_update(agent_id)

                # Wait between steps
                await asyncio.sleep(3)  # 3 second intervals

            # Mark as completed
            agent_status.status = "stopped"
            await self._broadcast_agent_update(agent_id)

        except asyncio.CancelledError:
            logger.info("Demo agent {} cancelled", agent_id)
            if agent_id in self.agents:
                self.agents[agent_id].status = "stopped"
            await self._broadcast_agent_update(agent_id)
        except Exception as e:
            logger.error("Demo agent {} failed: {}", agent_id, e)
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
        if agent_id in self.agent_tasks:
            try:
                await self.agent_tasks[agent_id]
            except asyncio.CancelledError:
                pass
            del self.agent_tasks[agent_id]

    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the tracking server."""
        logger.info("🚀 Starting Minimal ROOK Tracking Server on {}:{}", host, port)
        uvicorn.run(self.app, host=host, port=port, log_level="info")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Minimal ROOK Trading Tracking Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")

    args = parser.parse_args()

    server = MinimalTrackingServer()
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()