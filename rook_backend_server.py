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
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from maharook.core.agent_registry import get_agent_registry
from maharook.core.logging import (
    log,
    set_trace_id,
    generate_trace_id,
    with_trace_id,
)
from maharook.core.metrics import get_metrics_collector


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> WebSocket:
        await websocket.accept()
        self.active_connections.append(websocket)
        trace_id = generate_trace_id()
        set_trace_id(trace_id)
        log.info("WebSocket client connected. Total: {}", len(self.active_connections))
        return websocket

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        trace_id = generate_trace_id()
        set_trace_id(trace_id)
        log.info("WebSocket client disconnected. Total: {}", len(self.active_connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients with fail-fast error handling."""
        if not self.active_connections:
            raise RuntimeError(
                "No active WebSocket connections available for broadcast. "
                "Ensure clients are connected before attempting to broadcast messages."
            )

        trace_id = generate_trace_id()
        set_trace_id(trace_id)
        log.info("📡 Broadcasting {} to {} clients", message.get("type", "unknown"), len(self.active_connections))

        # Validate message before broadcasting
        try:
            json_data = json.dumps(message, default=str)
            # Check payload size
            if len(json_data) > 32768:  # 32KB limit
                raise ValueError(
                    f"Message size ({len(json_data)} bytes) exceeds 32KB limit. "
                    f"Reduce message payload or implement chunking."
                )
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize broadcast message: {e}. Message data: {message}")

        disconnected = []
        failed_sends = []

        for connection in self.active_connections:
            try:
                await connection.send_text(json_data)
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)  # 10ms delay between sends
            except Exception as e:
                logger.error("Failed to send to WebSocket client: {}", e)
                disconnected.append(connection)
                failed_sends.append(str(e))

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

        # If all broadcasts failed, raise an error
        if failed_sends and len(failed_sends) == len(self.active_connections):
            raise RuntimeError(
                f"All WebSocket broadcasts failed. Errors: {'; '.join(failed_sends[:3])}. "
                f"Check network connectivity and WebSocket client health."
            )


class ROOKBackendServer:
    """Unified ROOK backend server with WebSocket streaming and API endpoints."""

    def __init__(self, port: int = 8001) -> None:
        self.port: int = port
        self.app: FastAPI = FastAPI(title="ROOK Trading Arena Backend", version="1.0.0")
        self.connection_manager: ConnectionManager = ConnectionManager()
        self.registry = get_agent_registry()
        self.known_agents: set[str] = set()
        self.last_states: Dict[str, Any] = {}

        self._setup_routes()
        self._setup_cors()

    def _setup_cors(self) -> None:
        """Setup CORS for frontend integration."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
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

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for monitoring and orchestration."""
            try:
                # Check agent registry connection
                registry_healthy = True
                try:
                    agents = self.registry.list_agents()
                    registry_healthy = True
                except Exception as e:
                    registry_healthy = False
                    log.error("Registry health check failed: {}", e)

                # Check WebSocket connections
                ws_healthy = len(self.connection_manager.active_connections) >= 0  # Always true, just checking it's accessible

                # Overall health status
                healthy = registry_healthy and ws_healthy

                return {
                    "status": "healthy" if healthy else "unhealthy",
                    "checks": {
                        "agent_registry": "up" if registry_healthy else "down",
                        "websocket_manager": "up" if ws_healthy else "down"
                    },
                    "metrics": {
                        "active_agents": len(self.known_agents),
                        "active_websocket_connections": len(self.connection_manager.active_connections)
                    },
                    "timestamp": time.time()
                }
            except Exception as e:
                log.error("Health check failed: {}", e)
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": time.time()
                }

        @self.app.get("/ready")
        async def readiness_check():
            """Readiness check - server is ready to accept requests."""
            try:
                # Server is ready if it can respond
                return {
                    "status": "ready",
                    "timestamp": time.time()
                }
            except Exception as e:
                log.error("Readiness check failed: {}", e)
                raise HTTPException(status_code=503, detail="Service not ready")

        @self.app.get("/metrics")
        async def get_metrics():
            """Get collected metrics."""
            try:
                metrics_collector = get_metrics_collector()
                return metrics_collector.get_all_metrics()
            except Exception as e:
                log.error("Failed to get metrics: {}", e)
                raise HTTPException(status_code=500, detail=str(e))

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

    @with_trace_id
    async def _monitor_agents(self) -> None:
        """Monitor agent registry and broadcast real updates."""
        while True:
            try:
                # Get current agents from registry
                registry_agents = self.registry.list_agents()
                log.debug("Monitoring: Found {} agents", len(registry_agents))

                for agent_state in registry_agents:
                    agent_id = agent_state.agent_id
                    log.debug("Processing agent: {} - Steps: {}, Trades: {}, Last Update: {}",
                                agent_id, agent_state.total_steps, agent_state.total_trades, agent_state.last_update)

                    # Check if this is a new agent
                    if agent_id not in self.known_agents:
                        self.known_agents.add(agent_id)
                        log.info("📡 New agent detected: {}", agent_state.name)

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
                        log.debug("State comparison for {}: Current({}, {}, {}) vs Last({}, {}, {})",
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

                # Periodic cleanup of completed agents (every 60 iterations = ~1 minute)
                if hasattr(self, '_cleanup_counter'):
                    self._cleanup_counter += 1
                else:
                    self._cleanup_counter = 0

                if self._cleanup_counter >= 60:
                    self._cleanup_counter = 0
                    cleaned = self.registry.cleanup_completed_agents(min_age_minutes=5.0)
                    if cleaned > 0:
                        log.info("🧹 Periodic cleanup: removed {} completed agent(s)", cleaned)

            except Exception as e:
                log.error("❌ Error monitoring agents: {}", e)

            # Check every second for real-time updates
            await asyncio.sleep(1)

    def _create_log_message(self, current_state: Any, last_state: Optional[Any]) -> Optional[dict[str, Any]]:
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

    def run(self, host: str = "0.0.0.0") -> None:
        """Run the unified backend server."""
        trace_id = generate_trace_id()
        set_trace_id(trace_id)
        log.info("🚀 Starting ROOK Trading Arena Backend on {}:{}", host, self.port)

        @self.app.on_event("startup")
        async def startup_event():
            # Start agent monitoring
            asyncio.create_task(self._monitor_agents())
            log.info("🔄 Agent monitoring started")

        uvicorn.run(self.app, host=host, port=self.port, log_level="info")


def main() -> None:
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