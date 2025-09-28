#!/usr/bin/env python3
"""
Simple WebSocket client to test ROOK tracking server.
"""

import asyncio
import json
import aiohttp
from loguru import logger


class TrackingWebSocketClient:
    """Simple WebSocket client for testing tracking server."""

    def __init__(self, url: str = "ws://localhost:8001/ws"):
        self.url = url
        self.session = None
        self.websocket = None

    async def connect(self):
        """Connect to WebSocket server."""
        self.session = aiohttp.ClientSession()
        try:
            self.websocket = await self.session.ws_connect(self.url)
            logger.info("🔗 Connected to tracking server at {}", self.url)
            return True
        except Exception as e:
            logger.error("❌ Failed to connect: {}", e)
            return False

    async def listen(self, duration: int = 30):
        """Listen for messages from tracking server."""
        if not self.websocket:
            logger.error("❌ Not connected to WebSocket")
            return

        logger.info("👂 Listening for messages for {} seconds...", duration)

        try:
            timeout = aiohttp.ClientTimeout(total=duration)
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_message(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error: {}", self.websocket.exception())
                    break
        except asyncio.TimeoutError:
            logger.info("⏰ Listening timeout reached")
        except Exception as e:
            logger.error("❌ Error while listening: {}", e)

    async def _handle_message(self, data: dict):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type", "unknown")

        if msg_type == "agent_status":
            agent_data = data.get("data", {})
            logger.info("📊 Agent Status: {} | {} | Steps: {} | NAV: ${:.2f}",
                       agent_data.get("name", "Unknown"),
                       agent_data.get("status", "Unknown"),
                       agent_data.get("total_steps", 0),
                       agent_data.get("current_nav", 0.0))

        elif msg_type == "trading_update":
            trade_data = data.get("data", {})
            logger.info("💱 Trading Update: {} {} {:.4f} ETH @ ${:.2f} | PnL: ${:.2f} | Confidence: {:.1%}",
                       trade_data.get("action", "UNKNOWN"),
                       trade_data.get("pair", "UNKNOWN"),
                       trade_data.get("amount_eth", 0.0),
                       trade_data.get("price", 0.0),
                       trade_data.get("pnl", 0.0),
                       trade_data.get("confidence", 0.0))

            if trade_data.get("reasoning"):
                logger.info("💭 Reasoning: {}", trade_data.get("reasoning"))
        else:
            logger.info("📨 Message [{}]: {}", msg_type, data)

    async def send_ping(self):
        """Send ping to keep connection alive."""
        if self.websocket:
            await self.websocket.send_str("ping")

    async def close(self):
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
        if self.session:
            await self.session.close()
        logger.info("🔌 Disconnected from tracking server")


async def start_agent_via_api(config_path: str, duration: int = 5):
    """Start agent via REST API."""
    async with aiohttp.ClientSession() as session:
        payload = {
            "config_path": config_path,
            "duration": duration  # minutes
        }

        try:
            async with session.post("http://localhost:8001/agents/start", json=payload) as response:
                result = await response.json()
                logger.info("🚀 Agent start response: {}", result)
                return result.get("agent_id")
        except Exception as e:
            logger.error("❌ Failed to start agent: {}", e)
            return None


async def main():
    """Main test function."""
    logger.info("🧪 Testing ROOK Tracking Server WebSocket")

    # Create WebSocket client
    client = TrackingWebSocketClient()

    # Connect to WebSocket
    if not await client.connect():
        return

    try:
        # Start listening in background
        listen_task = asyncio.create_task(client.listen(duration=60))

        # Wait a moment, then start an agent
        await asyncio.sleep(2)

        logger.info("🤖 Starting ROOK agent for testing...")
        agent_id = await start_agent_via_api("run_agent/configs/rook_qwen_ollama.yaml", duration=2)

        if agent_id:
            logger.info("✅ Agent {} started successfully", agent_id)
        else:
            logger.error("❌ Failed to start agent")

        # Wait for listen task to complete
        await listen_task

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())