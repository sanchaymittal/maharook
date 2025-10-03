"""
Structured Logging Module with Trace IDs
-----------------------------------------
Provides structured logging with trace IDs, context management, and decorators
for enhanced observability and debugging across the MAHAROOK system.

This module uses loguru's contextualization features to add trace IDs,
agent IDs, step numbers, and other contextual information to all log messages.
"""

import contextvars
import functools
import json
import sys
import uuid
from datetime import datetime
from typing import Any, Callable, Optional, TypeVar, cast

from loguru import logger

# Context variables for structured logging
trace_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)
agent_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "agent_id", default=None
)
step_ctx: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("step", default=None)
session_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id", default=None
)

F = TypeVar("F", bound=Callable[..., Any])


def generate_trace_id() -> str:
    """Generate a unique trace ID for request/operation tracking.

    Returns:
        str: UUID-based trace ID in short format (first 8 chars)
    """
    return str(uuid.uuid4())[:8]


def get_trace_id() -> Optional[str]:
    """Get the current trace ID from context.

    Returns:
        Optional[str]: Current trace ID or None if not set
    """
    return trace_id_ctx.get()


def set_trace_id(trace_id: str) -> None:
    """Set the trace ID in the current context.

    Args:
        trace_id: Trace ID to set
    """
    trace_id_ctx.set(trace_id)


def get_agent_id() -> Optional[str]:
    """Get the current agent ID from context.

    Returns:
        Optional[str]: Current agent ID or None if not set
    """
    return agent_id_ctx.get()


def set_agent_id(agent_id: str) -> None:
    """Set the agent ID in the current context.

    Args:
        agent_id: Agent ID to set
    """
    agent_id_ctx.set(agent_id)


def get_step() -> Optional[int]:
    """Get the current step number from context.

    Returns:
        Optional[int]: Current step number or None if not set
    """
    return step_ctx.get()


def set_step(step: int) -> None:
    """Set the step number in the current context.

    Args:
        step: Step number to set
    """
    step_ctx.set(step)


def get_session_id() -> Optional[str]:
    """Get the current session ID from context.

    Returns:
        Optional[str]: Current session ID or None if not set
    """
    return session_id_ctx.get()


def set_session_id(session_id: str) -> None:
    """Set the session ID in the current context.

    Args:
        session_id: Session ID to set
    """
    session_id_ctx.set(session_id)


def get_context() -> dict[str, Any]:
    """Get all current context values as a dictionary.

    Returns:
        dict[str, Any]: Dictionary of context values
    """
    return {
        "trace_id": get_trace_id(),
        "agent_id": get_agent_id(),
        "step": get_step(),
        "session_id": get_session_id(),
    }


class StructuredLogger:
    """
    Wrapper for loguru logger with automatic context injection.

    Provides logging methods that automatically include trace ID,
    agent ID, step number, and other contextual information.
    """

    def __init__(self) -> None:
        """Initialize the structured logger."""
        self._logger = logger

    def _bind_context(self) -> Any:
        """Bind current context to logger.

        Returns:
            Logger with bound context
        """
        context = {}
        if trace_id := get_trace_id():
            context["trace_id"] = trace_id
        if agent_id := get_agent_id():
            context["agent_id"] = agent_id
        if step := get_step():
            context["step"] = step
        if session_id := get_session_id():
            context["session_id"] = session_id

        return self._logger.bind(**context) if context else self._logger

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message with context.

        Args:
            message: Log message template
            *args: Positional arguments for message formatting
            **kwargs: Additional context data
        """
        self._bind_context().debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log info message with context.

        Args:
            message: Log message template
            *args: Positional arguments for message formatting
            **kwargs: Additional context data
        """
        self._bind_context().info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message with context.

        Args:
            message: Log message template
            *args: Positional arguments for message formatting
            **kwargs: Additional context data
        """
        self._bind_context().warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error message with context.

        Args:
            message: Log message template
            *args: Positional arguments for message formatting
            **kwargs: Additional context data
        """
        self._bind_context().error(message, *args, **kwargs)

    def success(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log success message with context.

        Args:
            message: Log message template
            *args: Positional arguments for message formatting
            **kwargs: Additional context data
        """
        self._bind_context().success(message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message with context.

        Args:
            message: Log message template
            *args: Positional arguments for message formatting
            **kwargs: Additional context data
        """
        self._bind_context().critical(message, *args, **kwargs)


def with_trace_id(func: F) -> F:
    """
    Decorator to automatically inject a trace ID for a function call.

    The trace ID persists throughout the function execution and all
    nested function calls, providing end-to-end tracing.

    Example:
        @with_trace_id
        def process_trade(amount: float) -> None:
            log.info("Processing trade for {}", amount)

    Args:
        func: Function to decorate

    Returns:
        Decorated function with trace ID injection
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Generate new trace ID if not already set
        if not get_trace_id():
            trace_id = generate_trace_id()
            set_trace_id(trace_id)

        try:
            return func(*args, **kwargs)
        finally:
            # Clear trace ID after function completes
            trace_id_ctx.set(None)

    return cast(F, wrapper)


def with_agent_context(agent_id: str) -> Callable[[F], F]:
    """
    Decorator to set agent context for a function.

    Example:
        @with_agent_context("rook_weth_usdc_12345")
        def trade_step(self) -> None:
            log.info("Executing trade step")

    Args:
        agent_id: Agent identifier to set in context

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            old_agent_id = get_agent_id()
            set_agent_id(agent_id)
            try:
                return func(*args, **kwargs)
            finally:
                if old_agent_id is not None:
                    set_agent_id(old_agent_id)
                else:
                    agent_id_ctx.set(None)

        return cast(F, wrapper)

    return decorator


def json_formatter(record: dict[str, Any]) -> str:
    """
    Format log record as JSON with structured fields.

    This formatter extracts all context variables and creates
    a structured JSON log entry suitable for log aggregation systems.

    Args:
        record: Loguru record dictionary

    Returns:
        str: JSON-formatted log line
    """
    # Extract base fields
    log_entry = {
        "time": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }

    # Add context fields if present
    extra = record.get("extra", {})
    if trace_id := extra.get("trace_id"):
        log_entry["trace_id"] = trace_id
    if agent_id := extra.get("agent_id"):
        log_entry["agent_id"] = agent_id
    if step := extra.get("step"):
        log_entry["step"] = step
    if session_id := extra.get("session_id"):
        log_entry["session_id"] = session_id

    # Add exception info if present
    if record.get("exception"):
        log_entry["exception"] = {
            "type": record["exception"].type.__name__ if record["exception"].type else None,
            "value": str(record["exception"].value) if record["exception"].value else None,
        }

    return json.dumps(log_entry)


def human_readable_formatter(record: dict[str, Any]) -> str:
    """
    Format log record in human-readable format with context.

    Args:
        record: Loguru record dictionary

    Returns:
        str: Formatted log line with color and context
    """
    # Base format
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level>"

    # Add context fields
    extra = record.get("extra", {})
    if trace_id := extra.get("trace_id"):
        fmt += f" | <cyan>trace={trace_id}</cyan>"
    if agent_id := extra.get("agent_id"):
        fmt += f" | <blue>agent={agent_id}</blue>"
    if step := extra.get("step"):
        fmt += f" | <yellow>step={step}</yellow>"

    # Add message
    fmt += " | <level>{message}</level>\n"

    # Add exception if present
    if record.get("exception"):
        fmt += "{exception}\n"

    return fmt


def configure_logging(
    level: str = "INFO",
    format: str = "human",
    log_file: Optional[str] = None,
    json_output: bool = False,
) -> None:
    """
    Configure structured logging for the application.

    This function sets up loguru with either human-readable or JSON
    formatting, optionally writing to a file in addition to stdout.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format - "human" or "json"
        log_file: Optional file path to write logs to
        json_output: If True, use JSON formatting (overrides format parameter)

    Example:
        # Human-readable console logging
        configure_logging(level="DEBUG", format="human")

        # JSON logging to file
        configure_logging(level="INFO", format="json", log_file="app.log")
    """
    # Remove default logger
    logger.remove()

    # Determine formatter
    if json_output or format == "json":
        formatter = json_formatter
    else:
        formatter = human_readable_formatter

    # Add console handler
    logger.add(
        sys.stderr,
        level=level,
        format=formatter if format == "json" else human_readable_formatter,
        colorize=format != "json",
        backtrace=True,
        diagnose=True,
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            format=json_formatter,  # Always use JSON for file output
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True,
        )


# Create global structured logger instance
log = StructuredLogger()


# Convenience functions for common logging patterns
def log_trade_execution(
    agent_id: str,
    action: str,
    amount: float,
    price: float,
    trace_id: Optional[str] = None,
    **extra: Any,
) -> None:
    """
    Log a trade execution with structured data.

    Args:
        agent_id: Agent executing the trade
        action: Trade action (BUY, SELL, HOLD)
        amount: Trade amount
        price: Execution price
        trace_id: Optional trace ID for correlation
        **extra: Additional context data
    """
    if trace_id:
        set_trace_id(trace_id)
    set_agent_id(agent_id)

    log.info(
        "Trade execution: {} {:.6f} @ ${:.2f}",
        action,
        amount,
        price,
        **extra,
    )


def log_agent_step(
    agent_id: str,
    step: int,
    action: str,
    trace_id: Optional[str] = None,
    **extra: Any,
) -> None:
    """
    Log an agent trading step with structured data.

    Args:
        agent_id: Agent executing the step
        step: Step number
        action: Action taken
        trace_id: Optional trace ID for correlation
        **extra: Additional context data
    """
    if trace_id:
        set_trace_id(trace_id)
    set_agent_id(agent_id)
    set_step(step)

    log.info(
        "Agent step: {}",
        action,
        **extra,
    )


def log_error_with_context(
    error: Exception,
    context: dict[str, Any],
    trace_id: Optional[str] = None,
) -> None:
    """
    Log an error with full context information.

    Args:
        error: Exception that occurred
        context: Context dictionary with relevant data
        trace_id: Optional trace ID for correlation
    """
    if trace_id:
        set_trace_id(trace_id)

    log.error(
        "Error occurred: {} | Context: {}",
        str(error),
        json.dumps(context),
        exc_info=error,
    )
