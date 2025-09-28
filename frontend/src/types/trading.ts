export interface AgentData {
  agent_id: string
  name: string
  config_path: string
  status: 'starting' | 'running' | 'paused' | 'stopped' | 'error'
  start_time: number
  last_update: number
  total_steps: number
  total_trades: number
  current_nav: number
  total_pnl: number
}

export interface TradingUpdate {
  agent_id: string
  pair: string
  step: number
  action: string
  amount_eth: number
  price: number
  tx_hash: string | null
  pnl: number
  nav: number
  timestamp: number
  confidence: number
  reasoning: string
}

export interface LogData {
  agent_id: string
  agent_name: string
  level: string
  message: string
  timestamp: number
  step?: number
  data?: any
}

export interface WebSocketMessage {
  type: 'agent_status' | 'trading_update' | 'agent_log' | 'agent_discovered' | 'agent_removed'
  data: AgentData | TradingUpdate | LogData | any
}

export interface AgentPosition {
  x: number
  y: number
  z: number
}

export interface AgentColors {
  primary: string
  secondary: string
  glow: string
}