import { useState, useEffect, useRef, useCallback } from 'react'
import { WebSocketMessage } from '../types/trading'

export const useWebSocket = (url: string) => {
  const [connected, setConnected] = useState(false)
  const [latestMessage, setLatestMessage] = useState<WebSocketMessage | null>(null)
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const reconnectAttemptsRef = useRef(0)

  const connect = useCallback(() => {
    try {
      // Close existing connection if any
      if (ws.current && ws.current.readyState !== WebSocket.CLOSED) {
        ws.current.close()
      }

      console.log('🔄 Connecting to WebSocket:', url)
      ws.current = new WebSocket(url)

      ws.current.onopen = () => {
        console.log('🚀 Connected to ROOK tracking server')
        setConnected(true)
        reconnectAttemptsRef.current = 0

        // Clear any pending reconnection
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
        }
      }

      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          console.log('📨 Received WebSocket message:', message.type, message.data)
          setLatestMessage(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error, event.data)
        }
      }

      ws.current.onclose = (event) => {
        console.log('🔌 Disconnected from server', event.code, event.reason)
        setConnected(false)

        // Only reconnect if not manually closed and attempts are reasonable
        if (event.code !== 1000 && reconnectAttemptsRef.current < 10) {
          const delay = Math.min(3000 * Math.pow(2, reconnectAttemptsRef.current), 30000)
          reconnectAttemptsRef.current++

          reconnectTimeoutRef.current = setTimeout(() => {
            console.log(`🔄 Attempting to reconnect... (attempt ${reconnectAttemptsRef.current})`)
            connect()
          }, delay)
        }
      }

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setConnected(false)
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setConnected(false)

      // Retry connection after 5 seconds
      if (reconnectAttemptsRef.current < 10) {
        reconnectTimeoutRef.current = setTimeout(connect, 5000)
      }
    }
  }, [url])

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
  }, [connect])

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