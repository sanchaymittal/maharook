#!/usr/bin/env python3
"""
Simple Log Broadcasting Server - Shows real ROOK agent logs on frontend
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from maharook.core.agent_registry import get_agent_registry


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


class SimpleLogServer:
    """Simple server that broadcasts real ROOK agent logs."""

    def __init__(self):
        self.app = FastAPI(title="ROOK Log Broadcaster", version="1.0.0")
        self.connection_manager = ConnectionManager()
        self.registry = get_agent_registry()
        self.known_agents = set()
        self.last_states = {}

        self._setup_routes()
        self._setup_cors()

    def _setup_cors(self):
        """Setup CORS for frontend integration."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
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
                # Keep connection alive
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)

        @self.app.get("/")
        async def root():
            return {"message": "ROOK Log Broadcaster", "status": "running"}

    async def _monitor_agents(self):
        """Monitor agent registry and broadcast real updates."""
        while True:
            try:
                # Get current agents from registry
                registry_agents = self.registry.list_agents()

                for agent_state in registry_agents:
                    agent_id = agent_state.agent_id

                    # Check if this is a new agent
                    if agent_id not in self.known_agents:
                        self.known_agents.add(agent_id)
                        logger.info("📡 New agent detected: {}", agent_state.name)

                        # Broadcast agent discovery
                        await self.connection_manager.broadcast({
                            "type": "agent_discovered",
                            "data": {
                                "agent_id": agent_id,
                                "name": agent_state.name,
                                "model": agent_state.model_name,
                                "pair": agent_state.pair,
                                "status": agent_state.status,
                                "timestamp": time.time()
                            }
                        })

                    # Check for state changes (new steps, trades, etc.)
                    last_state = self.last_states.get(agent_id)

                    if (last_state is None or
                        agent_state.total_steps > last_state.total_steps or
                        agent_state.last_update > last_state.last_update):

                        # Update last known state
                        self.last_states[agent_id] = agent_state

                        # Create log message based on what happened
                        log_message = self._create_log_message(agent_state, last_state)

                        if log_message:
                            await self.connection_manager.broadcast({
                                "type": "agent_log",
                                "data": log_message
                            })

                # Clean up removed agents
                current_agent_ids = {agent.agent_id for agent in registry_agents}
                removed_agents = self.known_agents - current_agent_ids
                for agent_id in removed_agents:
                    self.known_agents.discard(agent_id)
                    if agent_id in self.last_states:
                        del self.last_states[agent_id]

                    await self.connection_manager.broadcast({
                        "type": "agent_removed",
                        "data": {"agent_id": agent_id, "timestamp": time.time()}
                    })

            except Exception as e:
                logger.error("❌ Error monitoring agents: {}", e)

            # Check every second for real-time updates
            await asyncio.sleep(1)

    def _create_log_message(self, current_state, last_state):
        """Create a log message based on state changes."""
        agent_id = current_state.agent_id

        # If this is the first state, show initialization
        if last_state is None:
            return {
                "agent_id": agent_id,
                "agent_name": current_state.name,
                "level": "INFO",
                "message": f"🚀 Agent {current_state.name} started trading {current_state.pair}",
                "timestamp": current_state.start_time,
                "step": 0,
                "data": {
                    "model": current_state.model_name,
                    "pair": current_state.pair,
                    "initial_nav": current_state.current_nav
                }
            }

        # Check for new trading steps
        if current_state.total_steps > last_state.total_steps:
            step_message = f"📊 Step {current_state.total_steps}: "

            # Check if there was a trade
            if current_state.total_trades > last_state.total_trades:
                if current_state.last_action:
                    step_message += f"{current_state.last_action} {current_state.last_amount:.4f} ETH"
                    if current_state.last_price:
                        step_message += f" @ ${current_state.last_price:.2f}"
                else:
                    step_message += "Trade executed"
                level = "SUCCESS"
            else:
                step_message += f"HOLD - Portfolio: ${current_state.current_nav:.2f}"
                level = "INFO"

            return {
                "agent_id": agent_id,
                "agent_name": current_state.name,
                "level": level,
                "message": step_message,
                "timestamp": current_state.last_update,
                "step": current_state.total_steps,
                "data": {
                    "action": current_state.last_action or "HOLD",
                    "amount": current_state.last_amount or 0.0,
                    "price": current_state.last_price,
                    "nav": current_state.current_nav,
                    "pnl": current_state.total_pnl,
                    "confidence": current_state.last_confidence,
                    "reasoning": current_state.last_reasoning
                }
            }

        return None

    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the log broadcasting server."""
        logger.info("🚀 Starting Simple Log Server on {}:{}", host, port)

        @self.app.on_event("startup")
        async def startup_event():
            # Start agent monitoring
            asyncio.create_task(self._monitor_agents())
            logger.info("🔄 Agent monitoring started")

        uvicorn.run(self.app, host=host, port=port, log_level="info")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple ROOK Log Broadcasting Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")

    args = parser.parse_args()

    server = SimpleLogServer()
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()