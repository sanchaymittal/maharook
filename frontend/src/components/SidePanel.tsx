import React, { useState } from 'react'
import styled from 'styled-components'
import { AgentData, TradingUpdate } from '../types/trading'

interface SidePanelProps {
  agents: AgentData[]
  trades: TradingUpdate[]
  logs?: any[]
}

const Panel = styled.div<{ isOpen: boolean }>`
  position: fixed;
  top: 0;
  right: ${props => props.isOpen ? '0' : '-400px'};
  width: 400px;
  height: 100vh;
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(20px);
  border-left: 1px solid rgba(255, 255, 255, 0.1);
  transition: right 0.3s ease;
  pointer-events: auto;
  overflow-y: auto;
  z-index: 100;
`

const ToggleButton = styled.button<{ isOpen: boolean }>`
  position: fixed;
  top: 50%;
  right: ${props => props.isOpen ? '400px' : '0'};
  transform: translateY(-50%);
  width: 40px;
  height: 100px;
  background: linear-gradient(45deg, #00d4ff, #ff00ff);
  border: none;
  border-radius: 20px 0 0 20px;
  color: white;
  font-size: 1.2rem;
  cursor: pointer;
  transition: right 0.3s ease;
  pointer-events: auto;
  z-index: 101;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    background: linear-gradient(45deg, #00a8cc, #cc00cc);
  }
`

const Header = styled.div`
  padding: 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  background: linear-gradient(45deg, #00d4ff20, #ff00ff20);
`

const Title = styled.h2`
  margin: 0;
  font-size: 1.5rem;
  background: linear-gradient(45deg, #00d4ff, #ff00ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
`

const Subtitle = styled.p`
  margin: 5px 0 0 0;
  opacity: 0.7;
  font-size: 0.9rem;
`

const Section = styled.div`
  padding: 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
`

const SectionTitle = styled.h3`
  margin: 0 0 15px 0;
  font-size: 1.1rem;
  color: #ffffff;
`

const AgentCard = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 10px;
`

const AgentName = styled.div`
  font-weight: bold;
  margin-bottom: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`

const StatusBadge = styled.span<{ status: string }>`
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.7rem;
  font-weight: bold;
  background: ${props => {
    switch (props.status) {
      case 'running': return '#065f46'
      case 'stopped': return '#7f1d1d'
      case 'starting': return '#92400e'
      case 'error': return '#7f1d1d'
      default: return '#374151'
    }
  }};
  color: ${props => {
    switch (props.status) {
      case 'running': return '#10b981'
      case 'stopped': return '#ef4444'
      case 'starting': return '#f59e0b'
      case 'error': return '#ef4444'
      default: return '#9ca3af'
    }
  }};
`

const Metric = styled.div`
  display: flex;
  justify-content: space-between;
  margin: 4px 0;
  font-size: 0.85rem;
`

const MetricValue = styled.span<{ positive?: boolean }>`
  font-weight: bold;
  color: ${props =>
    props.positive === true ? '#10b981' :
    props.positive === false ? '#ef4444' :
    '#ffffff'
  };
`

const TradeEntry = styled.div<{ action: string }>`
  background: ${props => {
    switch (props.action) {
      case 'BUY': return 'rgba(16, 185, 129, 0.1)'
      case 'SELL': return 'rgba(239, 68, 68, 0.1)'
      default: return 'rgba(55, 65, 81, 0.1)'
    }
  }};
  border-left: 3px solid ${props => {
    switch (props.action) {
      case 'BUY': return '#10b981'
      case 'SELL': return '#ef4444'
      default: return '#6b7280'
    }
  }};
  padding: 8px 12px;
  margin: 5px 0;
  border-radius: 4px;
  font-size: 0.8rem;
  font-family: monospace;
`

const TradeTime = styled.div`
  opacity: 0.6;
  font-size: 0.7rem;
  margin-bottom: 3px;
`

const ControlButton = styled.button`
  background: linear-gradient(45deg, #3b82f6, #1d4ed8);
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.8rem;
  margin: 5px 5px 5px 0;
  transition: all 0.2s;

  &:hover {
    background: linear-gradient(45deg, #2563eb, #1e40af);
    transform: translateY(-1px);
  }

  &:disabled {
    background: #6b7280;
    cursor: not-allowed;
    transform: none;
  }
`

const SidePanel: React.FC<SidePanelProps> = ({ agents, trades, logs = [] }) => {
  const [isOpen, setIsOpen] = useState(false)

  const startAgent = async (configPath: string) => {
    try {
      const response = await fetch('http://localhost:8001/agents/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_path: configPath, duration: 30 })
      })
      const result = await response.json()
      if (result.error) {
        alert('Error: ' + result.error)
      }
    } catch (error) {
      alert('Error starting agent: ' + error)
    }
  }

  const stopAgent = async (agentId: string) => {
    try {
      await fetch(`http://localhost:8001/agents/${agentId}/stop`, { method: 'POST' })
    } catch (error) {
      alert('Error stopping agent: ' + error)
    }
  }

  // Sort agents by PnL
  const sortedAgents = [...agents].sort((a, b) => b.total_pnl - a.total_pnl)

  return (
    <>
      <ToggleButton isOpen={isOpen} onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? '→' : '←'}
      </ToggleButton>

      <Panel isOpen={isOpen}>
        <Header>
          <Title>🏆 Leaderboard</Title>
          <Subtitle>Live PnL Rankings</Subtitle>
        </Header>

        <Section>
          <SectionTitle>Quick Start</SectionTitle>
          <ControlButton
            onClick={() => startAgent('run_agent/configs/rook_finr1_ollama.yaml')}
          >
            🤖 Start Fin-R1
          </ControlButton>
          <ControlButton
            onClick={() => startAgent('run_agent/configs/rook_qwen_ollama.yaml')}
          >
            🧠 Start Qwen2.5
          </ControlButton>
          <ControlButton
            onClick={() => startAgent('run_agent/configs/rook_qwen_lora.yaml')}
          >
            🎯 Start LoRA
          </ControlButton>
        </Section>

        <Section>
          <SectionTitle>Agents ({agents.length})</SectionTitle>
          {sortedAgents.map((agent, index) => (
            <AgentCard key={agent.agent_id}>
              <AgentName>
                <span>
                  {index === 0 && '🥇 '}
                  {index === 1 && '🥈 '}
                  {index === 2 && '🥉 '}
                  {agent.name.replace('ROOK-', '')}
                </span>
                <StatusBadge status={agent.status}>
                  {agent.status.toUpperCase()}
                </StatusBadge>
              </AgentName>

              <Metric>
                <span>Total PnL:</span>
                <MetricValue positive={agent.total_pnl >= 0}>
                  {agent.total_pnl >= 0 ? '+' : ''}${agent.total_pnl.toFixed(2)}
                </MetricValue>
              </Metric>

              <Metric>
                <span>NAV:</span>
                <MetricValue>${agent.current_nav.toFixed(2)}</MetricValue>
              </Metric>

              <Metric>
                <span>Trades:</span>
                <MetricValue>{agent.total_trades}</MetricValue>
              </Metric>

              <Metric>
                <span>Steps:</span>
                <MetricValue>{agent.total_steps}</MetricValue>
              </Metric>

              {agent.status === 'running' && (
                <ControlButton onClick={() => stopAgent(agent.agent_id)}>
                  Stop Agent
                </ControlButton>
              )}
            </AgentCard>
          ))}

          {agents.length === 0 && (
            <div style={{ textAlign: 'center', opacity: 0.6, padding: '20px' }}>
              No agents running. Start one above!
            </div>
          )}
        </Section>

        <Section>
          <SectionTitle>Recent Trades ({trades.length})</SectionTitle>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {trades.slice(0, 10).map((trade, index) => (
              <TradeEntry key={`${trade.agent_id}-${trade.step}-${index}`} action={trade.action}>
                <TradeTime>
                  {new Date(trade.timestamp * 1000).toLocaleTimeString()}
                </TradeTime>
                <div>
                  <strong>{trade.action}</strong> {trade.amount_eth.toFixed(4)} ETH
                  @ ${trade.price.toFixed(2)}
                </div>
                {trade.tx_hash && (
                  <div style={{ opacity: 0.7, fontSize: '0.7rem' }}>
                    📋 {trade.tx_hash.substring(0, 10)}...
                  </div>
                )}
              </TradeEntry>
            ))}

            {trades.length === 0 && (
              <div style={{ textAlign: 'center', opacity: 0.6, padding: '20px' }}>
                No trades yet. Waiting for agents...
              </div>
            )}
          </div>
        </Section>

        <Section>
          <SectionTitle>Live Logs ({logs.length})</SectionTitle>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {logs.slice(0, 20).map((log, index) => (
              <div
                key={`${log.agent_id || 'system'}-${log.timestamp}-${index}`}
                style={{
                  padding: '8px',
                  margin: '4px 0',
                  borderRadius: '4px',
                  borderLeft: `3px solid ${
                    log.level === 'SUCCESS' ? '#10b981' :
                    log.level === 'ERROR' ? '#ef4444' :
                    log.level === 'WARNING' ? '#f59e0b' :
                    '#6b7280'
                  }`,
                  background: `${
                    log.level === 'SUCCESS' ? 'rgba(16, 185, 129, 0.1)' :
                    log.level === 'ERROR' ? 'rgba(239, 68, 68, 0.1)' :
                    log.level === 'WARNING' ? 'rgba(245, 158, 11, 0.1)' :
                    'rgba(107, 114, 128, 0.1)'
                  }`,
                  fontSize: '0.8rem'
                }}
              >
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '4px'
                }}>
                  <span style={{ fontWeight: 'bold', opacity: 0.8 }}>
                    {log.agent_name || 'System'}
                  </span>
                  <span style={{ opacity: 0.6, fontSize: '0.7rem' }}>
                    {new Date(log.timestamp * 1000).toLocaleTimeString()}
                  </span>
                </div>
                <div>{log.message}</div>
                {log.data && log.data.reasoning && (
                  <div style={{
                    fontSize: '0.7rem',
                    opacity: 0.7,
                    marginTop: '4px',
                    fontStyle: 'italic'
                  }}>
                    💭 {log.data.reasoning}
                  </div>
                )}
              </div>
            ))}

            {logs.length === 0 && (
              <div style={{ textAlign: 'center', opacity: 0.6, padding: '20px' }}>
                No logs yet. Waiting for agents...
              </div>
            )}
          </div>
        </Section>
      </Panel>
    </>
  )
}

export default SidePanel