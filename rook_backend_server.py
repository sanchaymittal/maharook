#!/usr/bin/env python3
"""
Unified ROOK Backend Server - Complete trading arena backend
Combines WebSocket streaming, agent management, and API endpoints
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
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
        return websocket

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected. Total: {}", len(self.active_connections))

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            logger.debug("No active connections for broadcast")
            return

        logger.info("📡 Broadcasting {} to {} clients", message.get("type", "unknown"), len(self.active_connections))

        # Validate message before broadcasting
        try:
            json_data = json.dumps(message, default=str)
            # Check payload size
            if len(json_data) > 32768:  # 32KB limit
                logger.warning("Message too large ({}), truncating", len(json_data))
                return
        except (TypeError, ValueError) as e:
            logger.error("Failed to serialize message: {}, message: {}", e, message)
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json_data)
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)  # 10ms delay between sends
            except Exception as e:
                logger.warning("Failed to send to client: {}", e)
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


class ROOKBackendServer:
    """Unified ROOK backend server with WebSocket streaming and API endpoints."""

    def __init__(self, port: int = 8001):
        self.port = port
        self.app = FastAPI(title="ROOK Trading Arena Backend", version="1.0.0")
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
        """Setup API routes and WebSocket endpoints."""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            websocket = await self.connection_manager.connect(websocket)

            # Send current agent states to the newly connected client
            try:
                registry_agents = self.registry.list_agents()
                for agent_state in registry_agents:
                    # Send agent status
                    await websocket.send_text(json.dumps({
                        "type": "agent_status",
                        "data": {
                            "agent_id": agent_state.agent_id,
                            "name": agent_state.name,
                            "config_path": agent_state.config_path,
                            "status": agent_state.status,
                            "start_time": int(agent_state.start_time) if agent_state.start_time else 0,
                            "last_update": int(agent_state.last_update) if agent_state.last_update else 0,
                            "total_steps": agent_state.total_steps if agent_state.total_steps else 0,
                            "total_trades": agent_state.total_trades if agent_state.total_trades else 0,
                            "current_nav": agent_state.current_nav if agent_state.current_nav else 10000.0,
                            "total_pnl": agent_state.total_pnl if agent_state.total_pnl else 0.0
                        }
                    }, default=str))
                    logger.info("🔄 Sent current agent state to new client: {}", agent_state.name)
            except Exception as e:
                logger.error("Failed to send initial agent states: {}", e)

            try:
                # Keep connection alive
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)

        @self.app.get("/")
        async def root():
            return {"message": "ROOK Trading Arena Backend", "status": "running", "agents": len(self.known_agents)}

        @self.app.get("/agents")
        async def list_agents():
            """List all known agents."""
            try:
                registry_agents = self.registry.list_agents()
                agents_data = []
                for agent_state in registry_agents:
                    agents_data.append({
                        "agent_id": agent_state.agent_id,
                        "name": agent_state.name,
                        "config_path": agent_state.config_path,
                        "status": agent_state.status,
                        "start_time": int(agent_state.start_time) if agent_state.start_time else 0,
                        "last_update": int(agent_state.last_update) if agent_state.last_update else 0,
                        "total_steps": agent_state.total_steps if agent_state.total_steps else 0,
                        "total_trades": agent_state.total_trades if agent_state.total_trades else 0,
                        "current_nav": agent_state.current_nav if agent_state.current_nav else 10000.0,
                        "total_pnl": agent_state.total_pnl if agent_state.total_pnl else 0.0,
                        "last_action": agent_state.last_action,
                        "last_amount": agent_state.last_amount,
                        "last_price": agent_state.last_price,
                        "last_confidence": agent_state.last_confidence,
                        "last_reasoning": agent_state.last_reasoning
                    })
                return {"agents": agents_data}
            except Exception as e:
                logger.error("Failed to list agents: {}", e)
                return {"agents": [], "error": str(e)}

        @self.app.post("/agents/start")
        async def start_agent(request: dict):
            """Start a new trading agent."""
            try:
                config_path = request.get("config_path")
                duration = request.get("duration", 10)  # Default 10 minutes

                if not config_path:
                    raise HTTPException(status_code=400, detail="config_path is required")

                # Start agent logic would go here
                # For now, return success message
                return {
                    "message": f"Agent start requested with config: {config_path} for {duration} minutes",
                    "config_path": config_path,
                    "duration": duration
                }
            except Exception as e:
                logger.error("Failed to start agent: {}", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/agents/{agent_id}/stop")
        async def stop_agent(agent_id: str):
            """Stop a running agent."""
            try:
                # Stop agent logic would go here
                return {"message": f"Agent {agent_id} stop requested"}
            except Exception as e:
                logger.error("Failed to stop agent {}: {}", agent_id, e)
                raise HTTPException(status_code=500, detail=str(e))

    async def _monitor_agents(self):
        """Monitor agent registry and broadcast real updates."""
        while True:
            try:
                # Get current agents from registry
                registry_agents = self.registry.list_agents()
                logger.debug("Monitoring: Found {} agents", len(registry_agents))

                for agent_state in registry_agents:
                    agent_id = agent_state.agent_id
                    logger.debug("Processing agent: {} - Steps: {}, Trades: {}, Last Update: {}",
                                agent_id, agent_state.total_steps, agent_state.total_trades, agent_state.last_update)

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

                        # Also broadcast as agent_status for the main UI
                        await self.connection_manager.broadcast({
                            "type": "agent_status",
                            "data": {
                                "agent_id": agent_id,
                                "name": agent_state.name,
                                "config_path": agent_state.config_path,
                                "status": agent_state.status,
                                "start_time": int(agent_state.start_time) if agent_state.start_time else 0,
                                "last_update": int(agent_state.last_update) if agent_state.last_update else 0,
                                "total_steps": agent_state.total_steps if agent_state.total_steps else 0,
                                "total_trades": agent_state.total_trades if agent_state.total_trades else 0,
                                "current_nav": agent_state.current_nav if agent_state.current_nav else 10000.0,
                                "total_pnl": agent_state.total_pnl if agent_state.total_pnl else 0.0
                            }
                        })

                    # Check for state changes (new steps, trades, etc.)
                    last_state = self.last_states.get(agent_id)

                    if last_state:
                        logger.debug("State comparison for {}: Current({}, {}, {}) vs Last({}, {}, {})",
                                    agent_id,
                                    agent_state.total_steps, agent_state.total_trades, agent_state.last_update,
                                    last_state.total_steps, last_state.total_trades, last_state.last_update)

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

                        # Also broadcast updated agent status
                        await self.connection_manager.broadcast({
                            "type": "agent_status",
                            "data": {
                                "agent_id": agent_id,
                                "name": agent_state.name,
                                "config_path": agent_state.config_path,
                                "status": agent_state.status,
                                "start_time": int(agent_state.start_time) if agent_state.start_time else 0,
                                "last_update": int(agent_state.last_update) if agent_state.last_update else 0,
                                "total_steps": agent_state.total_steps if agent_state.total_steps else 0,
                                "total_trades": agent_state.total_trades if agent_state.total_trades else 0,
                                "current_nav": agent_state.current_nav if agent_state.current_nav else 10000.0,
                                "total_pnl": agent_state.total_pnl if agent_state.total_pnl else 0.0
                            }
                        })

                        # If there was a trade, also broadcast it as trading_update
                        if (agent_state.total_trades > (last_state.total_trades if last_state else 0) and
                            agent_state.last_action and agent_state.last_action in ["BUY", "SELL"]):

                            await self.connection_manager.broadcast({
                                "type": "trading_update",
                                "data": {
                                    "agent_id": agent_id,
                                    "pair": agent_state.pair,
                                    "step": agent_state.total_steps,
                                    "action": agent_state.last_action,
                                    "amount_eth": agent_state.last_amount or 0.0,
                                    "price": agent_state.last_price or 0.0,
                                    "tx_hash": f"0x{agent_id[-16:]}...",  # Mock hash
                                    "pnl": agent_state.total_pnl,
                                    "nav": agent_state.current_nav,
                                    "timestamp": int(agent_state.last_update),
                                    "confidence": agent_state.last_confidence or 0.0,
                                    "reasoning": agent_state.last_reasoning or ""
                                }
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
                "timestamp": current_state.start_time if current_state.start_time else time.time(),
                "step": 0,
                "data": {
                    "model": current_state.model_name or "Unknown",
                    "pair": current_state.pair or "WETH/USDC",
                    "initial_nav": current_state.current_nav if current_state.current_nav else 10000.0
                }
            }

        # Check for new trading steps
        if current_state.total_steps > last_state.total_steps:
            step_message = f"📊 Step {current_state.total_steps}: "

            # Check if there was a trade
            if current_state.total_trades > last_state.total_trades:
                if current_state.last_action:
                    step_message += f"{current_state.last_action} {(current_state.last_amount or 0.0):.4f} ETH"
                    if current_state.last_price:
                        step_message += f" @ ${(current_state.last_price or 0.0):.2f}"
                else:
                    step_message += "Trade executed"
                level = "SUCCESS"
            else:
                step_message += f"HOLD - Portfolio: ${(current_state.current_nav or 10000.0):.2f}"
                level = "INFO"

            return {
                "agent_id": agent_id,
                "agent_name": current_state.name,
                "level": level,
                "message": step_message,
                "timestamp": current_state.last_update if current_state.last_update else time.time(),
                "step": current_state.total_steps,
                "data": {
                    "action": current_state.last_action or "HOLD",
                    "amount": current_state.last_amount or 0.0,
                    "price": current_state.last_price or 0.0,
                    "nav": current_state.current_nav if current_state.current_nav else 10000.0,
                    "pnl": current_state.total_pnl if current_state.total_pnl else 0.0,
                    "confidence": current_state.last_confidence or 0.0,
                    "reasoning": (current_state.last_reasoning or "")[:500]  # Limit reasoning to 500 chars
                }
            }

        return None

    def run(self, host: str = "0.0.0.0"):
        """Run the unified backend server."""
        logger.info("🚀 Starting ROOK Trading Arena Backend on {}:{}", host, self.port)

        @self.app.on_event("startup")
        async def startup_event():
            # Start agent monitoring
            asyncio.create_task(self._monitor_agents())
            logger.info("🔄 Agent monitoring started")

        uvicorn.run(self.app, host=host, port=self.port, log_level="info")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ROOK Trading Arena Backend Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")

    args = parser.parse_args()

    server = ROOKBackendServer(port=args.port)
    server.run(host=args.host)


if __name__ == "__main__":
    main()