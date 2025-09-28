import { useState, useEffect, useRef } from 'react'
import { WebSocketMessage } from '../types/trading'

export const useWebSocket = (url: string) => {
  const [connected, setConnected] = useState(false)
  const [latestMessage, setLatestMessage] = useState<WebSocketMessage | null>(null)
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()

  const connect = () => {
    try {
      ws.current = new WebSocket(url)

      ws.current.onopen = () => {
        console.log('🚀 Connected to ROOK tracking server')
        setConnected(true)

        // Clear any pending reconnection
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
        }
      }

      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          setLatestMessage(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      ws.current.onclose = () => {
        console.log('🔌 Disconnected from server')
        setConnected(false)

        // Reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('🔄 Attempting to reconnect...')
          connect()
        }, 3000)
      }

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setConnected(false)
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setConnected(false)

      // Retry connection after 5 seconds
      reconnectTimeoutRef.current = setTimeout(connect, 5000)
    }
  }

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [url])

  const sendMessage = (message: any) => {
    if (ws.current && connected) {
      ws.current.send(JSON.stringify(message))
    }
  }

  return {
    connected,
    latestMessage,
    sendMessage
  }
}