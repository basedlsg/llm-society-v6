"""
Structured Logging Configuration for LLM Society

Provides structured logging with JSON output, log levels, and correlation IDs
for distributed tracing of agent activities.
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

# Context variable for correlation ID (for tracing requests across components)
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
simulation_step: ContextVar[int] = ContextVar("simulation_step", default=0)


@dataclass
class LogContext:
    """Structured log context"""
    timestamp: str
    level: str
    logger_name: str
    message: str
    correlation_id: str
    simulation_step: int
    extra: Dict[str, Any]


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id.get(""),
            "simulation_step": simulation_step.get(0),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName"
            }:
                try:
                    json.dumps(value)  # Check if serializable
                    extra_fields[key] = value
                except (TypeError, ValueError):
                    extra_fields[key] = str(value)

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data)


class SimulationMetrics:
    """Collects and tracks simulation metrics for observability"""

    def __init__(self):
        self._metrics: Dict[str, Any] = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timers": {},
        }
        self._start_times: Dict[str, float] = {}

    def increment(self, name: str, value: int = 1, labels: Optional[Dict] = None):
        """Increment a counter metric"""
        key = self._make_key(name, labels)
        self._metrics["counters"][key] = self._metrics["counters"].get(key, 0) + value

    def gauge(self, name: str, value: float, labels: Optional[Dict] = None):
        """Set a gauge metric"""
        key = self._make_key(name, labels)
        self._metrics["gauges"][key] = value

    def histogram(self, name: str, value: float, labels: Optional[Dict] = None):
        """Add a value to a histogram"""
        key = self._make_key(name, labels)
        if key not in self._metrics["histograms"]:
            self._metrics["histograms"][key] = []
        self._metrics["histograms"][key].append(value)

    def start_timer(self, name: str):
        """Start a timer"""
        self._start_times[name] = time.perf_counter()

    def stop_timer(self, name: str, labels: Optional[Dict] = None) -> float:
        """Stop a timer and record the duration"""
        if name not in self._start_times:
            return 0.0
        duration = time.perf_counter() - self._start_times[name]
        self.histogram(f"{name}_duration_seconds", duration, labels)
        del self._start_times[name]
        return duration

    def _make_key(self, name: str, labels: Optional[Dict]) -> str:
        """Create a unique key for a metric with labels"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        summary = {
            "counters": dict(self._metrics["counters"]),
            "gauges": dict(self._metrics["gauges"]),
            "histograms": {},
        }

        # Calculate histogram summaries
        for key, values in self._metrics["histograms"].items():
            if values:
                sorted_values = sorted(values)
                n = len(sorted_values)
                summary["histograms"][key] = {
                    "count": n,
                    "min": sorted_values[0],
                    "max": sorted_values[-1],
                    "mean": sum(sorted_values) / n,
                    "p50": sorted_values[n // 2],
                    "p95": sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1],
                    "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
                }

        return summary

    def reset(self):
        """Reset all metrics"""
        self._metrics = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timers": {},
        }


# Global metrics instance
metrics = SimulationMetrics()


def setup_logging(
    level: str = "INFO",
    structured: bool = True,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for the simulation.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Use JSON structured logging
        log_file: Optional file path for log output

    Returns:
        Root logger instance
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Create formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def set_correlation_id(cid: Optional[str] = None):
    """Set the correlation ID for the current context"""
    correlation_id.set(cid or str(uuid.uuid4())[:8])


def set_simulation_step(step: int):
    """Set the current simulation step for logging context"""
    simulation_step.set(step)


class LoggedOperation:
    """Context manager for logging operation timing and success/failure"""

    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        log_level: int = logging.INFO,
        **extra_context
    ):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.log_level = log_level
        self.extra_context = extra_context
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.log(
            self.log_level,
            f"Starting {self.operation_name}",
            extra={"operation": self.operation_name, "status": "started", **self.extra_context}
        )
        metrics.start_timer(self.operation_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = metrics.stop_timer(self.operation_name)
        if exc_type is None:
            self.logger.log(
                self.log_level,
                f"Completed {self.operation_name} in {duration:.3f}s",
                extra={
                    "operation": self.operation_name,
                    "status": "success",
                    "duration_seconds": duration,
                    **self.extra_context
                }
            )
            metrics.increment(f"{self.operation_name}_success")
        else:
            self.logger.error(
                f"Failed {self.operation_name} after {duration:.3f}s: {exc_val}",
                extra={
                    "operation": self.operation_name,
                    "status": "failed",
                    "duration_seconds": duration,
                    "error_type": exc_type.__name__,
                    **self.extra_context
                },
                exc_info=True
            )
            metrics.increment(f"{self.operation_name}_failure")
        return False  # Don't suppress exceptions
