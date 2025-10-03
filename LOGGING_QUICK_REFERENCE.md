# Structured Logging - Quick Reference

## Import
```python
from maharook.core.logging import (
    log,                      # Use instead of logger
    set_trace_id,            # Set trace ID manually
    set_agent_id,            # Set agent ID
    set_step,                # Set step number
    generate_trace_id,       # Generate new trace ID
    with_trace_id,           # Decorator for auto trace ID
    log_trade_execution,     # Helper for trades
    log_agent_step,          # Helper for steps
    configure_logging,       # Configure format
)
```

## Quick Start

### Basic Logging
```python
trace_id = generate_trace_id()
set_trace_id(trace_id)
set_agent_id("rook_weth_usdc_12345")

log.info("Operation started")
log.debug("Debugging info")
log.warning("Warning message")
log.error("Error occurred")
log.success("Operation succeeded")
```

### With Decorator
```python
@with_trace_id
def my_function():
    log.info("Trace ID auto-generated")
```

### Log Trade
```python
log_trade_execution(
    agent_id="rook_001",
    action="BUY",
    amount=1.5,
    price=3500.00,
    confidence=0.85
)
```

### Log Step
```python
log_agent_step(
    agent_id="rook_001",
    step=42,
    action="BUY",
    amount=1.5,
    portfolio_value=25000.00
)
```

## Configuration

### Human-Readable Output
```python
configure_logging(level="INFO", format="human")
```

### JSON Output
```python
configure_logging(level="INFO", format="json")
```

### Write to File
```python
configure_logging(
    level="INFO",
    format="human",
    log_file="/var/log/maharook/app.log"
)
```

## CLI Usage

```bash
# Human-readable
python run_rook_models.py --config config.yaml --log-format human

# JSON output
python run_rook_models.py --config config.yaml --log-format json

# Write to file
python run_rook_models.py --config config.yaml --log-file logs/agent.log
```

## Context Variables

| Variable | Set By | Description |
|----------|--------|-------------|
| trace_id | `set_trace_id()` | Operation identifier |
| agent_id | `set_agent_id()` | Agent identifier |
| step | `set_step()` | Step number |
| session_id | `set_session_id()` | Session identifier |

## Log Formats

### Human-Readable
```
2025-01-03 10:30:45.123 | INFO | trace=abc12345 | agent=rook_001 | step=42 | Trade executed
```

### JSON
```json
{"time": "2025-01-03 10:30:45.123", "level": "INFO", "trace_id": "abc12345", "agent_id": "rook_001", "step": 42, "message": "Trade executed"}
```

## Best Practices

1. **Use decorators** for entry points: `@with_trace_id`
2. **Set context early** in `__init__` methods
3. **Update step context** at beginning of each step
4. **Use helpers** for consistency
5. **Configure once** at startup
6. **Use JSON in production** for log aggregation

## Migration

**Before:**
```python
logger.info("Agent {} step {}", agent_id, step)
```

**After:**
```python
set_agent_id(agent_id)
set_step(step)
log.info("Agent step")
```

## Demo
```bash
python examples/structured_logging_demo.py
```

## Documentation
- Usage Guide: `STRUCTURED_LOGGING_USAGE.md`
- Summary: `STRUCTURED_LOGGING_SUMMARY.md`
