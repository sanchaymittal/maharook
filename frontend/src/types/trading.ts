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

export interface WebSocketMessage {
  type: 'agent_status' | 'trading_update'
  data: AgentData | TradingUpdate
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