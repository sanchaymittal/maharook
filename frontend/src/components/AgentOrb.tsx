import React, { useRef, useState, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { Mesh, Group } from 'three'
import { Text, Sphere } from '@react-three/drei'
import FloatingTxHash from './FloatingTxHash'
import { AgentData, TradingUpdate, AgentPosition, AgentColors } from '../types/trading'

interface AgentOrbProps {
  agent: AgentData
  position: AgentPosition
  colors: AgentColors
  recentTrades: TradingUpdate[]
}

const AgentOrb: React.FC<AgentOrbProps> = ({ agent, position, colors, recentTrades }) => {
  const meshRef = useRef<Mesh>(null)
  const groupRef = useRef<Group>(null)
  const glowRef = useRef<Mesh>(null)

  const [pulseIntensity, setPulseIntensity] = useState(1)
  const [isTrading, setIsTrading] = useState(false)

  // Calculate NAV-based height (normalize between 0.5 and 3)
  // Adjusted for larger NAV values (e.g., $361k)
  const navHeight = Math.max(0.5, Math.min(3, Math.log10(Math.max(1, agent.current_nav / 10000))))

  // Pulse effect when trading
  useEffect(() => {
    if (recentTrades.length > 0) {
      const latestTrade = recentTrades[0]
      if (latestTrade.action !== 'HOLD') {
        setIsTrading(true)
        setPulseIntensity(2)

        const timeout = setTimeout(() => {
          setIsTrading(false)
          setPulseIntensity(1)
        }, 2000)

        return () => clearTimeout(timeout)
      }
    }
  }, [recentTrades])

  useFrame((state) => {
    if (!meshRef.current || !glowRef.current) return

    const time = state.clock.elapsedTime

    // Floating animation
    meshRef.current.position.y = Math.sin(time * 2 + position.x) * 0.2

    // Pulsing glow effect
    const glowScale = 1 + Math.sin(time * 3) * 0.1 * pulseIntensity
    glowRef.current.scale.setScalar(glowScale)

    // Trading pulse
    if (isTrading) {
      const tradePulse = 1 + Math.sin(time * 10) * 0.3
      meshRef.current.scale.setScalar(tradePulse)
      glowRef.current.scale.setScalar(tradePulse * 1.5)
    } else {
      meshRef.current.scale.setScalar(1)
    }

    // Rotation
    if (groupRef.current) {
      groupRef.current.rotation.y = time * 0.5
    }
  })

  const getStatusColor = () => {
    switch (agent.status) {
      case 'running': return colors.primary
      case 'starting': return '#ffaa00'
      case 'stopped': return '#666666'
      case 'error': return '#ff4444'
      default: return colors.primary
    }
  }

  return (
    <group position={[position.x, position.y, position.z]} ref={groupRef}>
      {/* NAV Bar (grows with portfolio value) */}
      <mesh position={[0, navHeight / 2, 0]}>
        <cylinderGeometry args={[0.1, 0.1, navHeight, 8]} />
        <meshStandardMaterial
          color={colors.secondary}
          emissive={colors.primary}
          emissiveIntensity={0.2}
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Main Agent Orb */}
      <Sphere ref={meshRef} args={[0.5, 32, 32]} position={[0, navHeight + 0.5, 0]}>
        <meshStandardMaterial
          color={getStatusColor()}
          emissive={getStatusColor()}
          emissiveIntensity={isTrading ? 0.6 : 0.3}
          metalness={0.7}
          roughness={0.2}
        />
      </Sphere>

      {/* Glow Effect */}
      <Sphere ref={glowRef} args={[0.7, 16, 16]} position={[0, navHeight + 0.5, 0]}>
        <meshBasicMaterial
          color={colors.glow}
          transparent
          opacity={isTrading ? 0.4 : 0.2}
          depthWrite={false}
        />
      </Sphere>

      {/* Agent Name */}
      <Text
        position={[0, navHeight + 1.5, 0]}
        fontSize={0.3}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        font="/fonts/inter-bold.woff"
      >
        {agent.name.replace('ROOK-', '')}
      </Text>

      {/* Status and Metrics */}
      <Text
        position={[0, navHeight + 1.1, 0]}
        fontSize={0.15}
        color={agent.total_pnl >= 0 ? '#00ff88' : '#ff4444'}
        anchorX="center"
        anchorY="middle"
      >
        {agent.total_pnl >= 0 ? '+' : ''}${agent.total_pnl.toFixed(2)}
      </Text>

      <Text
        position={[0, navHeight + 0.9, 0]}
        fontSize={0.12}
        color="#aaaaaa"
        anchorX="center"
        anchorY="middle"
      >
        {agent.total_trades} trades • ${agent.current_nav.toFixed(0)}
      </Text>

      {/* Floating Transaction Hashes */}
      {recentTrades
        .filter(trade => trade.tx_hash && trade.action !== 'HOLD')
        .slice(0, 3)
        .map((trade, index) => (
          <FloatingTxHash
            key={`${trade.tx_hash}-${trade.timestamp}`}
            txHash={trade.tx_hash!}
            action={trade.action}
            position={[0, navHeight + 2 + index * 0.5, 0]}
            delay={index * 0.5}
          />
        ))}

      {/* Trading Action Indicator */}
      {isTrading && recentTrades.length > 0 && (
        <Text
          position={[0, navHeight + 2.5, 0]}
          fontSize={0.2}
          color={recentTrades[0].action === 'BUY' ? '#00ff88' : '#ff4444'}
          anchorX="center"
          anchorY="middle"
        >
          {recentTrades[0].action} {recentTrades[0].amount_eth.toFixed(4)} ETH
        </Text>
      )}
    </group>
  )
}

export default AgentOrb