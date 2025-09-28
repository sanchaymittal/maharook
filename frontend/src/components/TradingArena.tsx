import React, { useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Group } from 'three'
import AgentOrb from './AgentOrb'
import TradingFloor from './TradingFloor'
import { AgentData, TradingUpdate, AgentPosition, AgentColors } from '../types/trading'

interface TradingArenaProps {
  agents: AgentData[]
  trades: TradingUpdate[]
}

const AGENT_POSITIONS: AgentPosition[] = [
  { x: -4, y: 0, z: 0 },   // Left
  { x: 4, y: 0, z: 0 },    // Right
  { x: 0, y: 0, z: -4 },   // Back
]

const AGENT_COLORS: AgentColors[] = [
  { primary: '#00d4ff', secondary: '#0088cc', glow: '#00d4ff' }, // Fin-R1 - Cyan
  { primary: '#ff00ff', secondary: '#cc00cc', glow: '#ff00ff' }, // Qwen2.5 Ollama - Magenta
  { primary: '#ffaa00', secondary: '#cc8800', glow: '#ffaa00' }, // Qwen2.5 LoRA - Orange
]

const TradingArena: React.FC<TradingArenaProps> = ({ agents, trades }) => {
  const groupRef = React.useRef<Group>(null)

  // Slowly rotate the entire arena
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.1) * 0.1
    }
  })

  // Get recent trades for each agent (last 10 seconds)
  const recentTrades = useMemo(() => {
    const now = Date.now() / 1000
    return trades.filter(trade => now - trade.timestamp < 10)
  }, [trades])

  return (
    <group ref={groupRef}>
      {/* Trading Floor */}
      <TradingFloor />

      {/* Agent Orbs */}
      {agents.slice(0, 3).map((agent, index) => {
        const position = AGENT_POSITIONS[index]
        const colors = AGENT_COLORS[index]
        const agentTrades = recentTrades.filter(trade => trade.agent_id === agent.agent_id)

        return (
          <AgentOrb
            key={agent.agent_id}
            agent={agent}
            position={position}
            colors={colors}
            recentTrades={agentTrades}
          />
        )
      })}

      {/* Ambient Cosmic Effects */}
      <pointLight position={[0, 10, 0]} intensity={0.3} color="#4a90e2" />
      <pointLight position={[-10, 5, 5]} intensity={0.2} color="#e24a4a" />
      <pointLight position={[10, 5, -5]} intensity={0.2} color="#4ae24a" />
    </group>
  )
}

export default TradingArena