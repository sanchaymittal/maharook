import React, { useRef, useEffect, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Text } from '@react-three/drei'
import { Group } from 'three'

interface FloatingTxHashProps {
  txHash: string
  action: string
  position: [number, number, number]
  delay?: number
}

const FloatingTxHash: React.FC<FloatingTxHashProps> = ({
  txHash,
  action,
  position,
  delay = 0
}) => {
  const groupRef = useRef<Group>(null)
  const [opacity, setOpacity] = useState(0)
  const [startTime, setStartTime] = useState<number | null>(null)

  useEffect(() => {
    const timeout = setTimeout(() => {
      setStartTime(Date.now())
      setOpacity(1)
    }, delay * 1000)

    const fadeOut = setTimeout(() => {
      setOpacity(0)
    }, (delay + 4) * 1000) // Fade out after 4 seconds

    return () => {
      clearTimeout(timeout)
      clearTimeout(fadeOut)
    }
  }, [delay])

  useFrame((state) => {
    if (!groupRef.current || !startTime) return

    const elapsed = (Date.now() - startTime) / 1000
    const maxTime = 5 // Total display time

    if (elapsed > maxTime) return

    // Float upward
    const floatY = elapsed * 0.5
    groupRef.current.position.y = position[1] + floatY

    // Fade out over time
    const fadeStart = 3
    if (elapsed > fadeStart) {
      const fadeProgress = (elapsed - fadeStart) / (maxTime - fadeStart)
      setOpacity(Math.max(0, 1 - fadeProgress))
    }

    // Slight wobble
    groupRef.current.position.x = position[0] + Math.sin(elapsed * 2) * 0.1
  })

  if (!startTime) return null

  const shortHash = `${txHash.substring(0, 6)}...${txHash.substring(-4)}`
  const color = action === 'BUY' ? '#00ff88' : '#ff4444'

  return (
    <group ref={groupRef} position={position}>
      <Text
        fontSize={0.08}
        color={color}
        anchorX="center"
        anchorY="middle"
        transparent
        opacity={opacity}
      >
        📋 {shortHash}
      </Text>
    </group>
  )
}

export default FloatingTxHash