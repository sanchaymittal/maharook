import React, { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Mesh } from 'three'

const TradingFloor: React.FC = () => {
  const meshRef = useRef<Mesh>(null)

  useFrame((state) => {
    if (!meshRef.current) return

    // Subtle glow animation
    const time = state.clock.elapsedTime
    const material = meshRef.current.material as any
    if (material.emissiveIntensity !== undefined) {
      material.emissiveIntensity = 0.1 + Math.sin(time * 0.5) * 0.05
    }
  })

  return (
    <group>
      {/* Main Floor */}
      <mesh ref={meshRef} position={[0, -1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[20, 20, 32, 32]} />
        <meshStandardMaterial
          color="#1a1a2e"
          emissive="#4a90e2"
          emissiveIntensity={0.1}
          metalness={0.8}
          roughness={0.3}
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Grid Lines */}
      <mesh position={[0, -0.99, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[20, 20]} />
        <meshBasicMaterial
          color="#4a90e2"
          transparent
          opacity={0.2}
          wireframe
        />
      </mesh>

      {/* Center Glow */}
      <mesh position={[0, -0.98, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[3, 32]} />
        <meshBasicMaterial
          color="#00d4ff"
          transparent
          opacity={0.3}
        />
      </mesh>

      {/* Perimeter Lights */}
      {Array.from({ length: 8 }, (_, i) => {
        const angle = (i / 8) * Math.PI * 2
        const x = Math.cos(angle) * 8
        const z = Math.sin(angle) * 8
        return (
          <pointLight
            key={i}
            position={[x, 2, z]}
            intensity={0.1}
            color="#4a90e2"
            distance={10}
          />
        )
      })}
    </group>
  )
}

export default TradingFloor