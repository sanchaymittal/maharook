# Structured Logging with Trace IDs - Usage Guide

## Overview

The MAHAROOK codebase now includes comprehensive structured logging with trace IDs for improved debugging and observability. This document explains how to use the new logging features.

## Features

- **Trace IDs**: Unique identifiers for request/operation tracking
- **Context Management**: Automatic injection of agent_id, step, session_id
- **JSON Format**: Optional structured JSON output for log aggregation
- **Decorators**: Easy-to-use decorators for automatic context injection
- **Helper Functions**: Convenience functions for common logging patterns

## Basic Usage

### Import the Logging Module

```python
from maharook.core.logging import (
    log,                    # Structured logger instance
    set_trace_id,          # Set trace ID manually
    set_agent_id,          # Set agent ID for context
    set_step,              # Set step number for context
    generate_trace_id,     # Generate new trace ID
    with_trace_id,         # Decorator for automatic trace ID
    log_trade_execution,   # Helper for trade logging
    log_agent_step,        # Helper for step logging
    configure_logging,     # Configure logging format
)
```

### Simple Logging with Context

```python
from maharook.core.logging import log, set_agent_id, set_trace_id, generate_trace_id

# Set context
trace_id = generate_trace_id()
set_trace_id(trace_id)
set_agent_id("rook_weth_usdc_12345")

# Log with automatic context injection
log.info("Starting trading operation")
log.debug("Market price: ${:.2f}", 3500.00)
log.warning("High volatility detected")
log.error("Trade execution failed")
```

**Output (Human-Readable Format):**
```
2025-01-03 10:30:45.123 | INFO     | trace=abc12345 | agent=rook_weth_usdc_12345 | Starting trading operation
2025-01-03 10:30:45.234 | DEBUG    | trace=abc12345 | agent=rook_weth_usdc_12345 | Market price: $3500.00
2025-01-03 10:30:45.345 | WARNING  | trace=abc12345 | agent=rook_weth_usdc_12345 | High volatility detected
2025-01-03 10:30:45.456 | ERROR    | trace=abc12345 | agent=rook_weth_usdc_12345 | Trade execution failed
```

**Output (JSON Format):**
```json
{"time": "2025-01-03 10:30:45.123", "level": "INFO", "trace_id": "abc12345", "agent_id": "rook_weth_usdc_12345", "message": "Starting trading operation", "module": "agent", "function": "step", "line": 42}
{"time": "2025-01-03 10:30:45.234", "level": "DEBUG", "trace_id": "abc12345", "agent_id": "rook_weth_usdc_12345", "message": "Market price: $3500.00", "module": "agent", "function": "step", "line": 43}
```

## Using Decorators

### Automatic Trace ID Injection

```python
from maharook.core.logging import with_trace_id, log

@with_trace_id
def process_trade(amount: float, price: float) -> None:
    """Trade processing with automatic trace ID."""
    log.info("Processing trade: {:.4f} ETH @ ${:.2f}", amount, price)
    # Trace ID is automatically generated and available in logs
    execute_swap(amount, price)
    log.success("Trade completed successfully")
```

### Agent Context Decorator

```python
from maharook.core.logging import with_agent_context, log

@with_agent_context("rook_weth_usdc_12345")
def analyze_market() -> dict:
    """Market analysis with agent context."""
    log.info("Analyzing market conditions")
    # Agent ID is automatically set in context
    return {"signal": "BUY", "confidence": 0.85}
```

## Helper Functions

### Log Trade Execution

```python
from maharook.core.logging import log_trade_execution

log_trade_execution(
    agent_id="rook_weth_usdc_12345",
    action="BUY",
    amount=1.5,
    price=3500.00,
    trace_id="abc12345",  # Optional
    confidence=0.85,
    reasoning="Strong upward trend detected"
)
```

**Output:**
```
2025-01-03 10:30:45.123 | INFO | trace=abc12345 | agent=rook_weth_usdc_12345 | Trade execution: BUY 1.500000 @ $3500.00
```

### Log Agent Step

```python
from maharook.core.logging import log_agent_step

log_agent_step(
    agent_id="rook_weth_usdc_12345",
    step=42,
    action="BUY",
    trace_id="abc12345",  # Optional
    amount=1.5,
    duration=0.234,
    portfolio_value=25000.00
)
```

**Output:**
```
2025-01-03 10:30:45.123 | INFO | trace=abc12345 | agent=rook_weth_usdc_12345 | step=42 | Agent step: BUY
```

## Configuration

### Configure Logging Format

```python
from maharook.core.logging import configure_logging

# Human-readable console output
configure_logging(level="INFO", format="human")

# JSON output to console
configure_logging(level="DEBUG", format="json")

# JSON output to file + console
configure_logging(
    level="INFO",
    format="human",
    log_file="/var/log/maharook/trading.log"
)
```

### Command-Line Configuration

The `run_rook_models.py` script now supports logging configuration via CLI:

```bash
# Human-readable output
python run_rook_models.py --config config.yaml --log-level INFO --log-format human

# JSON output
python run_rook_models.py --config config.yaml --log-level DEBUG --log-format json

# Write to file
python run_rook_models.py --config config.yaml --log-file /var/log/maharook/agent.log
```

## Integration Examples

### In RookAgent Class

```python
class RookAgent:
    def __init__(self, config: RookConfig, client: Optional[Any] = None):
        self.agent_id = f"rook_{config.pair}_{int(time.time())}"
        self.trace_id = generate_trace_id()

        # Set agent context
        set_agent_id(self.agent_id)
        set_trace_id(self.trace_id)

        log.info("ROOK Agent initialized: {} (pair: {})", self.agent_id, config.pair)

    @with_trace_id
    def step(self, market_features: MarketFeatures) -> RookState:
        """Execute trading step with trace ID."""
        self.step_count += 1

        # Update context
        set_agent_id(self.agent_id)
        set_step(self.step_count)

        log.debug("ROOK step {} starting", self.step_count)

        # ... trading logic ...

        log_agent_step(
            agent_id=self.agent_id,
            step=self.step_count,
            action=action.side,
            amount=action.size,
            duration=step_duration,
            portfolio_value=portfolio_state.total_value_usd
        )

        return state
```

### In Backend Server

```python
class ROOKBackendServer:
    @with_trace_id
    async def _monitor_agents(self) -> None:
        """Monitor agents with trace ID."""
        while True:
            try:
                agents = self.registry.list_agents()
                log.debug("Monitoring: Found {} agents", len(agents))

                for agent_state in agents:
                    # Set agent context for logs
                    set_agent_id(agent_state.agent_id)

                    if agent_state.agent_id not in self.known_agents:
                        log.info("New agent detected: {}", agent_state.name)

            except Exception as e:
                log.error("Error monitoring agents: {}", e)

            await asyncio.sleep(1)
```

## JSON Log Format

When using JSON format, logs are structured as:

```json
{
  "time": "2025-01-03 10:30:45.123",
  "level": "INFO",
  "message": "Trade execution: BUY 1.500000 @ $3500.00",
  "module": "agent",
  "function": "step",
  "line": 224,
  "trace_id": "abc12345",
  "agent_id": "rook_weth_usdc_12345",
  "step": 42,
  "session_id": "xyz67890"
}
```

This format is ideal for:
- Log aggregation systems (Elasticsearch, Splunk, etc.)
- Automated log analysis
- Debugging distributed systems
- Performance monitoring

## Context Variables

The logging system tracks these context variables:

| Variable | Type | Description | Set By |
|----------|------|-------------|--------|
| `trace_id` | str | Unique operation identifier | `set_trace_id()`, `@with_trace_id` |
| `agent_id` | str | Agent identifier | `set_agent_id()` |
| `step` | int | Step number in trading sequence | `set_step()` |
| `session_id` | str | Trading session identifier | `set_session_id()` |

## Best Practices

1. **Always use decorators for functions**: Use `@with_trace_id` for public APIs and entry points
2. **Set agent context early**: Call `set_agent_id()` in `__init__` methods
3. **Update step context**: Call `set_step()` at the beginning of each trading step
4. **Use helper functions**: Prefer `log_trade_execution()` and `log_agent_step()` for consistency
5. **Configure once**: Call `configure_logging()` at application startup
6. **Use JSON for production**: Enable JSON format for production systems
7. **Include context in exceptions**: Use helper functions to log errors with full context

## Trace ID Flow Example

```
Session Start
  └─ trace_id: abc12345 (session)
      ├─ Agent Init
      │   └─ agent_id: rook_weth_usdc_12345
      │       └─ log: "ROOK Agent initialized"
      │
      ├─ Trading Step 1
      │   └─ trace_id: def45678 (step operation)
      │       ├─ step: 1
      │       ├─ log: "ROOK step 1 starting"
      │       ├─ log: "Trade execution: BUY 1.5 @ $3500"
      │       └─ log: "Agent step: BUY"
      │
      └─ Trading Step 2
          └─ trace_id: ghi78901 (step operation)
              ├─ step: 2
              └─ log: "ROOK step 2 starting"
```

## Files Updated

The following files now use structured logging:

1. `/Users/sanchaymittal/github/maharook/maharook/core/logging.py` - **NEW** Logging module
2. `/Users/sanchaymittal/github/maharook/maharook/agents/rook/agent.py` - Updated to use structured logging
3. `/Users/sanchaymittal/github/maharook/rook_backend_server.py` - Updated to use structured logging
4. `/Users/sanchaymittal/github/maharook/run_rook_models.py` - Updated to use structured logging with CLI options

## Migration from Standard Logging

**Before:**
```python
logger.info("Agent {} completed step {}", agent_id, step)
```

**After:**
```python
set_agent_id(agent_id)
set_step(step)
log.info("Agent completed step")  # Context automatically included
```

Or use helper:
```python
log_agent_step(agent_id=agent_id, step=step, action="HOLD")
```

## Troubleshooting

### Trace IDs not appearing in logs

Make sure you've called `set_trace_id()` or used the `@with_trace_id` decorator:

```python
from maharook.core.logging import generate_trace_id, set_trace_id

trace_id = generate_trace_id()
set_trace_id(trace_id)
```

### Agent ID not in context

Set the agent ID explicitly:

```python
from maharook.core.logging import set_agent_id

set_agent_id(self.agent_id)
```

### JSON format not working

Configure logging explicitly:

```python
from maharook.core.logging import configure_logging

configure_logging(level="INFO", format="json")
```
