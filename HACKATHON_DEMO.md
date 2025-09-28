# 🏰 ROOK Arena - Hackathon Demo Guide

**Cosmic AI Trading Arena with Real-time 3D Visualization**

## 🎯 Demo Overview

A spectacular 3D cosmic trading arena where AI agents battle in real-time financial markets:

- **3 Glowing Orbs** = AI Trading Agents (Fin-R1, Qwen2.5 Ollama, Your Trained LoRA)
- **Height** = Portfolio NAV (taller = more profit)
- **Pulse Effects** = Live trading activity
- **Floating Tx Hashes** = Real blockchain transactions
- **Side Panel** = PnL Leaderboard + Trade History

## 🚀 Quick Start (Hackathon Demo)

### Terminal 1: Start Backend
```bash
# Start the tracking server
uv run python tracking_server.py --host 0.0.0.0 --port 8000
```

### Terminal 2: Start Frontend
```bash
cd frontend
npm install
npm run dev
# Opens on http://localhost:3000
```

### Terminal 3: Demo Script
```bash
# Start all 3 agents for the demo
curl -X POST http://localhost:8000/agents/start \
  -H "Content-Type: application/json" \
  -d '{"config_path": "run_agent/configs/rook_finr1_ollama.yaml", "duration": 30}'

curl -X POST http://localhost:8000/agents/start \
  -H "Content-Type: application/json" \
  -d '{"config_path": "run_agent/configs/rook_qwen_ollama.yaml", "duration": 30}'

curl -X POST http://localhost:8000/agents/start \
  -H "Content-Type: application/json" \
  -d '{"config_path": "run_agent/configs/rook_qwen_lora.yaml", "duration": 30}'
```

## 🎪 Judge Experience

**What Judges Will See:**

1. **Cosmic Arena** - Beautiful 3D space with stars and cosmic lighting
2. **3 Glowing Orbs** - Each representing a different AI model:
   - 🔵 **Fin-R1** (4.7GB specialized financial model)
   - 🟣 **Qwen2.5** (986MB general AI)
   - 🟠 **Your LoRA** (custom-trained model with 86% loss reduction)

3. **Live Trading Effects**:
   - Orbs **pulse** when making trades
   - **NAV bars grow** with portfolio value
   - **Tx hashes float** above agents (proving on-chain execution)
   - **Real-time PnL** updates

4. **Side Panel**:
   - 🏆 **Leaderboard** (ranked by PnL)
   - 📊 **Live metrics** (NAV, trades, steps)
   - 📜 **Trade history** with timestamps
   - 🚀 **One-click agent starter**

## 🏗️ Architecture

### Backend Stack
- **FastAPI + WebSockets** for real-time data streaming
- **MAHAROOK Framework** for AI agent orchestration
- **Multi-model support** (LoRA, Ollama, Transformers)
- **Structured JSON events** for frontend consumption

### Frontend Stack
- **React + TypeScript** for UI components
- **Three.js + React Three Fiber** for 3D cosmic visualization
- **Styled Components** for beautiful CSS-in-JS styling
- **Real-time WebSocket** connection for live updates

## 📊 Key Demo Points

### For Technical Judges:
- **Model Agnostic**: Works with any LLM (LoRA, Ollama, OpenAI, etc.)
- **Real Trading**: Actual blockchain transactions with tx hashes
- **Scalable**: Multi-agent architecture with separate private keys
- **Professional**: Production-ready FastAPI backend + React frontend

### For Business Judges:
- **Visual Impact**: Stunning 3D cosmic interface
- **Real-time Performance**: Live PnL tracking and leaderboards
- **Transparency**: All trades visible with blockchain receipts
- **Competitive**: Multiple AI models competing head-to-head

## 🎮 Interactive Demo Features

1. **Agent Control**: Start/stop agents via side panel
2. **Live Metrics**: Real-time NAV, PnL, trade counts
3. **Trade Visualization**: Pulse effects + floating tx hashes
4. **Performance Comparison**: Head-to-head AI model competition
5. **Blockchain Proof**: Real transaction hashes displayed

## 🔧 Technical Highlights

- **Custom LoRA Training**: 86% loss reduction on financial data
- **Multi-Model Framework**: Fin-R1, Qwen2.5, custom adapters
- **Real DeFi Integration**: UniswapV4 trading with actual transactions
- **3D Visualization**: Three.js cosmic arena with real-time effects
- **WebSocket Streaming**: Sub-second latency for trading updates

## 🎯 Demo Script (2 minutes)

1. **"This is ROOK Arena - a cosmic AI trading battlefield"**
2. **"Each orb represents a different AI model trading autonomously"**
3. **"Watch them pulse when making trades - those floating text are real blockchain transaction hashes"**
4. **"The height shows their portfolio value - see who's winning in real-time"**
5. **"Our custom LoRA model was trained with 86% loss reduction specifically for trading"**
6. **"This is the future of AI-powered DeFi - multiple models competing transparently on-chain"**

---

## 🏆 Result: A visually stunning, technically impressive demo that showcases:
- ✅ Advanced AI/ML (custom LoRA training)
- ✅ Real DeFi integration (actual trading)
- ✅ Beautiful UX (3D cosmic visualization)
- ✅ Technical depth (multi-model framework)
- ✅ Innovation (AI trading competition platform)