"""
Core Exception Classes
---------------------
Application-specific exceptions following fail-fast principles.
"""

from typing import Any


class MaharookError(Exception):
    """Base exception for all Maharook-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(MaharookError):
    """Raised when configuration is invalid or missing."""
    pass


class InsufficientFundsError(MaharookError):
    """Raised when account has insufficient funds for transaction."""
    pass


class TransactionError(MaharookError):
    """Raised when blockchain transaction fails."""

    def __init__(self, message: str, tx_hash: str | None = None, receipt: dict | None = None):
        details = {}
        if tx_hash:
            details["tx_hash"] = tx_hash
        if receipt:
            details["gas_used"] = receipt.get("gasUsed")
            details["status"] = receipt.get("status")
        super().__init__(message, details)
        self.tx_hash = tx_hash
        self.receipt = receipt


class NetworkError(MaharookError):
    """Raised when network/RPC operations fail."""
    pass


class ModelError(MaharookError):
    """Raised when ML model operations fail."""
    pass


class ValidationError(MaharookError):
    """Raised when data validation fails."""
    pass


class TradingError(MaharookError):
    """Raised when trading operations fail."""
    pass
