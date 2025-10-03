#!/usr/bin/env python3
"""
Structured Logging Demo
-----------------------
Demonstrates the new structured logging features with trace IDs.
"""

import time
from maharook.core.logging import (
    log,
    set_agent_id,
    set_step,
    set_trace_id,
    set_session_id,
    generate_trace_id,
    with_trace_id,
    log_trade_execution,
    log_agent_step,
    configure_logging,
    get_context,
)


def demo_basic_logging():
    """Demonstrate basic structured logging with manual context."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Logging with Manual Context")
    print("=" * 60 + "\n")

    # Generate and set trace ID
    trace_id = generate_trace_id()
    set_trace_id(trace_id)
    set_agent_id("demo_agent_001")
    set_step(1)

    # Log with context
    log.info("Starting trading operation")
    log.debug("Market analysis in progress")
    log.success("Analysis completed successfully")

    # Show current context
    context = get_context()
    print(f"\nCurrent context: {context}")


@with_trace_id
def demo_decorator():
    """Demonstrate automatic trace ID injection via decorator."""
    print("\n" + "=" * 60)
    print("Demo 2: Automatic Trace ID with Decorator")
    print("=" * 60 + "\n")

    set_agent_id("demo_agent_002")

    log.info("Function called with automatic trace ID")
    log.debug("Trace ID was automatically generated")

    # Nested function call - trace ID persists
    process_trade()


def process_trade():
    """Nested function that inherits trace ID."""
    log.info("Processing trade - trace ID inherited from parent")


def demo_trade_execution_logging():
    """Demonstrate trade execution logging helper."""
    print("\n" + "=" * 60)
    print("Demo 3: Trade Execution Logging")
    print("=" * 60 + "\n")

    trace_id = generate_trace_id()

    # Log BUY trade
    log_trade_execution(
        agent_id="demo_agent_003",
        action="BUY",
        amount=1.5,
        price=3500.00,
        trace_id=trace_id,
        confidence=0.85,
        reasoning="Strong upward momentum detected"
    )

    # Log SELL trade
    log_trade_execution(
        agent_id="demo_agent_003",
        action="SELL",
        amount=0.75,
        price=3520.00,
        trace_id=trace_id,
        confidence=0.72,
        reasoning="Taking profit at resistance level"
    )


def demo_agent_step_logging():
    """Demonstrate agent step logging helper."""
    print("\n" + "=" * 60)
    print("Demo 4: Agent Step Logging")
    print("=" * 60 + "\n")

    trace_id = generate_trace_id()
    agent_id = "demo_agent_004"

    # Simulate 5 trading steps
    for step in range(1, 6):
        action = "BUY" if step % 2 == 1 else "SELL"
        amount = 0.5 + (step * 0.1)

        log_agent_step(
            agent_id=agent_id,
            step=step,
            action=action,
            trace_id=trace_id,
            amount=amount,
            duration=0.123 + (step * 0.01),
            portfolio_value=10000 + (step * 100)
        )

        time.sleep(0.1)


def demo_multi_agent_session():
    """Demonstrate multi-agent session with session ID."""
    print("\n" + "=" * 60)
    print("Demo 5: Multi-Agent Session")
    print("=" * 60 + "\n")

    # Create session
    session_id = generate_trace_id()
    set_session_id(session_id)

    log.info("Trading session started")

    # Simulate multiple agents in same session
    for agent_num in range(1, 4):
        agent_id = f"demo_agent_{agent_num:03d}"
        trace_id = generate_trace_id()

        set_agent_id(agent_id)
        set_trace_id(trace_id)

        log.info("Agent initialized in session")

        # Execute some steps
        for step in range(1, 3):
            set_step(step)
            log.debug("Executing step {}", step)

    log.info("Trading session completed")


def demo_error_logging():
    """Demonstrate error logging with context."""
    print("\n" + "=" * 60)
    print("Demo 6: Error Logging with Context")
    print("=" * 60 + "\n")

    trace_id = generate_trace_id()
    set_trace_id(trace_id)
    set_agent_id("demo_agent_005")
    set_step(10)

    try:
        # Simulate an error
        log.info("Attempting trade execution")
        raise ValueError("Insufficient liquidity for trade size")
    except Exception as e:
        log.error("Trade execution failed: {}", str(e))
        log.warning("Retrying with smaller trade size")


def demo_json_format():
    """Demonstrate JSON formatted logging."""
    print("\n" + "=" * 60)
    print("Demo 7: JSON Format Logging")
    print("=" * 60 + "\n")

    # Reconfigure for JSON output
    print("Switching to JSON format...\n")
    configure_logging(level="INFO", format="json")

    trace_id = generate_trace_id()
    set_trace_id(trace_id)
    set_agent_id("demo_agent_006")
    set_step(1)

    log.info("This is a JSON formatted log entry")
    log_trade_execution(
        agent_id="demo_agent_006",
        action="BUY",
        amount=2.0,
        price=3500.00,
        trace_id=trace_id,
    )

    # Switch back to human-readable
    print("\nSwitching back to human-readable format...\n")
    configure_logging(level="INFO", format="human")
    log.info("Back to human-readable format")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("MAHAROOK Structured Logging Demonstration")
    print("=" * 60)

    # Configure logging (human-readable format)
    configure_logging(level="DEBUG", format="human")

    # Run all demos
    demo_basic_logging()
    demo_decorator()
    demo_trade_execution_logging()
    demo_agent_step_logging()
    demo_multi_agent_session()
    demo_error_logging()
    demo_json_format()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
