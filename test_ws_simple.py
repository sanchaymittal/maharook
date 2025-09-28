#!/usr/bin/env python3
"""Simple WebSocket test to see live trading updates."""

import asyncio
import json
import websockets
from loguru import logger

async def test_websocket():
    uri = "ws://localhost:8001/ws"
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("🔗 Connected to WebSocket")

            # Listen for 30 seconds
            for i in range(30):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)

                    if data.get("type") == "trading_update":
                        trade = data["data"]
                        logger.info("💱 TRADE: {} {} {:.4f} ETH @ ${:.2f} | PnL: ${:.2f}",
                                  trade["action"], trade["pair"], trade["amount_eth"],
                                  trade["price"], trade["pnl"])
                    elif data.get("type") == "agent_status":
                        agent = data["data"]
                        logger.info("📊 AGENT: {} | {} | Steps: {} | NAV: ${:.2f}",
                                  agent["name"], agent["status"], agent["total_steps"], agent["current_nav"])
                    else:
                        logger.info("📨 Message: {}", data)

                except asyncio.TimeoutError:
                    logger.debug("Waiting for messages...")

            logger.info("✅ Test completed")

    except Exception as e:
        logger.error("❌ WebSocket error: {}", e)

if __name__ == "__main__":
    asyncio.run(test_websocket())