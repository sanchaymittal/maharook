import React, { useMemo, useEffect } from 'react'
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

  // Debug logging with error handling
  React.useEffect(() => {
    try {
      console.log('TradingArena: Rendering update')
      console.log('TradingArena: agents=', agents.length, agents)
      console.log('TradingArena: trades=', trades.length, trades)

      // Check if agents data is valid
      agents.forEach((agent, idx) => {
        console.log(`Agent ${idx}:`, {
          id: agent.agent_id,
          name: agent.name,
          status: agent.status,
          nav: agent.current_nav,
          trades: agent.total_trades
        })
      })
    } catch (error) {
      console.error('TradingArena: Error in debug logging:', error)
    }
  }, [agents, trades])

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
        try {
          const position = AGENT_POSITIONS[index]
          const colors = AGENT_COLORS[index]
          const agentTrades = recentTrades.filter(trade => trade.agent_id === agent.agent_id)

          console.log(`Rendering agent ${index}:`, {
            id: agent.agent_id,
            position,
            colors,
            trades: agentTrades.length
          })

          return (
            <AgentOrb
              key={agent.agent_id}
              agent={agent}
              position={position}
              colors={colors}
              recentTrades={agentTrades}
            />
          )
        } catch (error) {
          console.error(`Error rendering agent ${index}:`, error, agent)
          return null
        }
      })}

      {/* Debug: Always visible orbs to ensure scene renders */}
      <mesh position={[0, 2, 0]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial
          color="#ff0000"
          emissive="#ff0000"
          emissiveIntensity={0.5}
        />
      </mesh>

      {/* Add a ground reference for visibility */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1, 0]}>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="#1a1a2e" />
      </mesh>

      {/* Ambient Cosmic Effects */}
      <pointLight position={[0, 10, 0]} intensity={0.3} color="#4a90e2" />
      <pointLight position={[-10, 5, 5]} intensity={0.2} color="#e24a4a" />
      <pointLight position={[10, 5, -5]} intensity={0.2} color="#4ae24a" />
    </group>
  )
}

export default TradingArena