import React, { useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { Environment, OrbitControls, Stars } from '@react-three/drei'
import styled from 'styled-components'
import TradingArena from './components/TradingArena'
import SidePanel from './components/SidePanel'
import { useWebSocket } from './hooks/useWebSocket'
import { AgentData, TradingUpdate } from './types/trading'

const AppContainer = styled.div`
  width: 100vw;
  height: 100vh;
  position: relative;
  overflow: hidden;
  background: radial-gradient(ellipse at center, #1a1a2e 0%, #16213e 50%, #0f0f1a 100%);
`

const CanvasContainer = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 1;
`

const UIOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 10;
`

const Header = styled.header`
  position: absolute;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  text-align: center;
  pointer-events: auto;
  z-index: 20;
`

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: bold;
  background: linear-gradient(45deg, #00d4ff, #ff00ff, #ffaa00);
  background-size: 200% 200%;
  animation: gradientShift 3s ease infinite;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.5rem;
  text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);

  @keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
`

const Subtitle = styled.p`
  font-size: 1rem;
  opacity: 0.8;
  color: #ffffff;
  margin: 0;
`

const ConnectionStatus = styled.div<{ connected: boolean }>`
  position: absolute;
  top: 20px;
  right: 20px;
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: bold;
  pointer-events: auto;
  background: ${props => props.connected
    ? 'linear-gradient(45deg, #00ff88, #00cc66)'
    : 'linear-gradient(45deg, #ff4444, #cc0000)'};
  color: white;
  box-shadow: 0 0 20px ${props => props.connected ? 'rgba(0, 255, 136, 0.5)' : 'rgba(255, 68, 68, 0.5)'};
  animation: pulse 2s ease-in-out infinite;

  @keyframes pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
  }
`

function App() {
  const [agents, setAgents] = useState<AgentData[]>([])
  const [trades, setTrades] = useState<TradingUpdate[]>([])
  const { connected, latestMessage } = useWebSocket('ws://localhost:8001/ws')

  useEffect(() => {
    if (!latestMessage) return

    if (latestMessage.type === 'agent_status') {
      const agentData = latestMessage.data as AgentData
      setAgents(prev => {
        const existing = prev.find(a => a.agent_id === agentData.agent_id)
        if (existing) {
          return prev.map(a => a.agent_id === agentData.agent_id ? agentData : a)
        } else {
          return [...prev, agentData]
        }
      })
    } else if (latestMessage.type === 'trading_update') {
      const tradeData = latestMessage.data as TradingUpdate
      setTrades(prev => [tradeData, ...prev.slice(0, 49)]) // Keep last 50 trades
    }
  }, [latestMessage])

  return (
    <AppContainer>
      <CanvasContainer>
        <Canvas
          camera={{ position: [0, 5, 10], fov: 60 }}
          gl={{ antialias: true, alpha: true }}
        >
          <ambientLight intensity={0.2} />
          <pointLight position={[10, 10, 10]} intensity={0.5} />

          {/* Cosmic Background */}
          <Stars
            radius={300}
            depth={60}
            count={3000}
            factor={7}
            saturation={0}
            fade
            speed={0.5}
          />

          {/* Trading Arena */}
          <TradingArena agents={agents} trades={trades} />

          {/* Camera Controls */}
          <OrbitControls
            enablePan={false}
            enableZoom={true}
            enableRotate={true}
            minDistance={5}
            maxDistance={20}
            maxPolarAngle={Math.PI / 2}
          />

          {/* Environment */}
          <Environment preset="night" />
        </Canvas>
      </CanvasContainer>

      <UIOverlay>
        <Header>
          <Title>🏰 ROOK ARENA</Title>
          <Subtitle>Cosmic AI Trading Battlefield</Subtitle>
        </Header>

        <ConnectionStatus connected={connected}>
          {connected ? '🟢 LIVE' : '🔴 OFFLINE'}
        </ConnectionStatus>

        <SidePanel agents={agents} trades={trades} />
      </UIOverlay>
    </AppContainer>
  )
}

export default App