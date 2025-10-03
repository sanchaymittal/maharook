"""
Metrics Collection Module
-------------------------
Lightweight metrics collection for ROOK agents and trading operations.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Counter:
    """Simple counter metric."""
    name: str
    value: int = 0
    tags: Dict[str, str] = field(default_factory=dict)

    def increment(self, amount: int = 1) -> None:
        """Increment counter."""
        self.value += amount

    def reset(self) -> None:
        """Reset counter to zero."""
        self.value = 0


@dataclass
class Gauge:
    """Simple gauge metric."""
    name: str
    value: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)

    def set(self, value: float) -> None:
        """Set gauge value."""
        self.value = value

    def increment(self, amount: float = 1.0) -> None:
        """Increment gauge."""
        self.value += amount

    def decrement(self, amount: float = 1.0) -> None:
        """Decrement gauge."""
        self.value -= amount


@dataclass
class Histogram:
    """Simple histogram metric."""
    name: str
    values: List[float] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    max_size: int = 1000  # Keep last 1000 values

    def record(self, value: float) -> None:
        """Record a value."""
        self.values.append(value)
        # Keep only recent values
        if len(self.values) > self.max_size:
            self.values = self.values[-self.max_size:]

    def get_stats(self) -> Dict[str, float]:
        """Get histogram statistics."""
        if not self.values:
            return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

        sorted_values = sorted(self.values)
        count = len(sorted_values)

        return {
            "count": count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)] if count > 1 else sorted_values[0],
            "p99": sorted_values[int(count * 0.99)] if count > 1 else sorted_values[0],
        }


class MetricsCollector:
    """Lightweight metrics collector for ROOK agents."""

    def __init__(self) -> None:
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.start_time: float = time.time()

    def counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> Counter:
        """Get or create a counter metric."""
        key = self._make_key(name, tags)
        if key not in self.counters:
            self.counters[key] = Counter(name=name, tags=tags or {})
        return self.counters[key]

    def gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Gauge:
        """Get or create a gauge metric."""
        key = self._make_key(name, tags)
        if key not in self.gauges:
            self.gauges[key] = Gauge(name=name, tags=tags or {})
        return self.gauges[key]

    def histogram(self, name: str, tags: Optional[Dict[str, str]] = None) -> Histogram:
        """Get or create a histogram metric."""
        key = self._make_key(name, tags)
        if key not in self.histograms:
            self.histograms[key] = Histogram(name=name, tags=tags or {})
        return self.histograms[key]

    def increment_counter(self, name: str, amount: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        self.counter(name, tags).increment(amount)

    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value."""
        self.gauge(name, tags).set(value)

    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value."""
        self.histogram(name, tags).record(value)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        metrics = {
            "uptime_seconds": time.time() - self.start_time,
            "counters": {},
            "gauges": {},
            "histograms": {}
        }

        for key, counter in self.counters.items():
            metrics["counters"][key] = {
                "name": counter.name,
                "value": counter.value,
                "tags": counter.tags
            }

        for key, gauge in self.gauges.items():
            metrics["gauges"][key] = {
                "name": gauge.name,
                "value": gauge.value,
                "tags": gauge.tags
            }

        for key, histogram in self.histograms.items():
            metrics["histograms"][key] = {
                "name": histogram.name,
                "stats": histogram.get_stats(),
                "tags": histogram.tags
            }

        return metrics

    def reset_all(self) -> None:
        """Reset all metrics."""
        for counter in self.counters.values():
            counter.reset()
        self.gauges.clear()
        self.histograms.clear()
        logger.info("All metrics reset")

    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"


# Global metrics collector instance
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


# Convenience functions for common trading metrics
def record_trade(agent_id: str, side: str, amount: float, price: float) -> None:
    """Record a trade execution."""
    metrics = get_metrics_collector()
    metrics.increment_counter("trades_total", tags={"agent_id": agent_id, "side": side})
    metrics.record_histogram("trade_amount_eth", amount, tags={"agent_id": agent_id})
    metrics.record_histogram("trade_price_usd", price, tags={"agent_id": agent_id})


def record_step_duration(agent_id: str, duration: float) -> None:
    """Record agent step duration."""
    metrics = get_metrics_collector()
    metrics.record_histogram("step_duration_seconds", duration, tags={"agent_id": agent_id})


def update_portfolio_value(agent_id: str, value: float) -> None:
    """Update portfolio value gauge."""
    metrics = get_metrics_collector()
    metrics.set_gauge("portfolio_value_usd", value, tags={"agent_id": agent_id})


def record_error(agent_id: str, error_type: str) -> None:
    """Record an error occurrence."""
    metrics = get_metrics_collector()
    metrics.increment_counter("errors_total", tags={"agent_id": agent_id, "type": error_type})
