"""
AegisLang Metrics and Telemetry Module.

This module provides metrics collection and monitoring capabilities
using Prometheus-compatible metrics.

Usage:
    from aegislang.core.metrics import (
        setup_metrics,
        track_request,
        track_document_processing,
        track_clause_extraction,
    )

    # Setup metrics
    setup_metrics(app)

    # Track metrics
    with track_request("ingest"):
        process_document()

    track_document_processing(doc_id, duration, success=True)
    track_clause_extraction(doc_id, num_clauses)
"""

import time
import functools
from contextlib import contextmanager
from typing import Callable, Optional, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime
import threading


# =============================================================================
# Metrics Storage (In-Memory - Replace with Prometheus in production)
# =============================================================================

@dataclass
class Counter:
    """Thread-safe counter metric."""
    name: str
    description: str
    labels: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, value: int = 1, **label_values):
        """Increment counter."""
        key = tuple(sorted(label_values.items()))
        with self._lock:
            if key not in self.labels:
                self.labels[key] = 0
            self.labels[key] += value

    def get(self, **label_values) -> int:
        """Get counter value."""
        key = tuple(sorted(label_values.items()))
        with self._lock:
            return self.labels.get(key, 0)


@dataclass
class Histogram:
    """Thread-safe histogram metric."""
    name: str
    description: str
    buckets: tuple = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    observations: Dict[tuple, list] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, **label_values):
        """Record an observation."""
        key = tuple(sorted(label_values.items()))
        with self._lock:
            if key not in self.observations:
                self.observations[key] = []
            self.observations[key].append(value)

    def get_stats(self, **label_values) -> Dict[str, float]:
        """Get statistics for the histogram."""
        key = tuple(sorted(label_values.items()))
        with self._lock:
            values = self.observations.get(key, [])
            if not values:
                return {"count": 0, "sum": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

            sorted_values = sorted(values)
            count = len(values)

            return {
                "count": count,
                "sum": sum(values),
                "avg": sum(values) / count,
                "min": min(values),
                "max": max(values),
                "p50": sorted_values[int(count * 0.50)],
                "p95": sorted_values[int(count * 0.95)] if count >= 20 else sorted_values[-1],
                "p99": sorted_values[int(count * 0.99)] if count >= 100 else sorted_values[-1],
            }


@dataclass
class Gauge:
    """Thread-safe gauge metric."""
    name: str
    description: str
    values: Dict[tuple, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float, **label_values):
        """Set gauge value."""
        key = tuple(sorted(label_values.items()))
        with self._lock:
            self.values[key] = value

    def inc(self, value: float = 1.0, **label_values):
        """Increment gauge."""
        key = tuple(sorted(label_values.items()))
        with self._lock:
            if key not in self.values:
                self.values[key] = 0
            self.values[key] += value

    def dec(self, value: float = 1.0, **label_values):
        """Decrement gauge."""
        self.inc(-value, **label_values)

    def get(self, **label_values) -> float:
        """Get gauge value."""
        key = tuple(sorted(label_values.items()))
        with self._lock:
            return self.values.get(key, 0)


# =============================================================================
# Global Metrics Registry
# =============================================================================

class MetricsRegistry:
    """Global metrics registry."""

    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

        # Initialize default metrics
        self._init_default_metrics()

    def _init_default_metrics(self):
        """Initialize default application metrics."""
        # Request metrics
        self.register(Counter(
            name="aegislang_requests_total",
            description="Total number of API requests",
        ))

        self.register(Histogram(
            name="aegislang_request_duration_seconds",
            description="Request duration in seconds",
        ))

        self.register(Counter(
            name="aegislang_request_errors_total",
            description="Total number of request errors",
        ))

        # Document processing metrics
        self.register(Counter(
            name="aegislang_documents_processed_total",
            description="Total number of documents processed",
        ))

        self.register(Histogram(
            name="aegislang_document_processing_duration_seconds",
            description="Document processing duration in seconds",
        ))

        self.register(Gauge(
            name="aegislang_documents_in_progress",
            description="Number of documents currently being processed",
        ))

        # Clause extraction metrics
        self.register(Counter(
            name="aegislang_clauses_extracted_total",
            description="Total number of clauses extracted",
        ))

        self.register(Histogram(
            name="aegislang_clauses_per_document",
            description="Number of clauses extracted per document",
        ))

        # Compilation metrics
        self.register(Counter(
            name="aegislang_artifacts_generated_total",
            description="Total number of artifacts generated",
        ))

        self.register(Histogram(
            name="aegislang_compilation_duration_seconds",
            description="Artifact compilation duration in seconds",
        ))

        # Validation metrics
        self.register(Counter(
            name="aegislang_validations_total",
            description="Total number of validations performed",
        ))

        self.register(Counter(
            name="aegislang_validation_failures_total",
            description="Total number of validation failures",
        ))

        # LLM metrics
        self.register(Counter(
            name="aegislang_llm_requests_total",
            description="Total number of LLM API requests",
        ))

        self.register(Histogram(
            name="aegislang_llm_request_duration_seconds",
            description="LLM API request duration in seconds",
        ))

        self.register(Counter(
            name="aegislang_llm_tokens_used_total",
            description="Total number of LLM tokens used",
        ))

        # System metrics
        self.register(Gauge(
            name="aegislang_active_connections",
            description="Number of active connections",
        ))

    def register(self, metric: Any):
        """Register a metric."""
        with self._lock:
            self._metrics[metric.name] = metric

    def get(self, name: str) -> Optional[Any]:
        """Get a metric by name."""
        with self._lock:
            return self._metrics.get(name)

    def all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            return dict(self._metrics)


# Global registry instance
_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return _registry


# =============================================================================
# Metric Tracking Functions
# =============================================================================

def track_request_start(endpoint: str, method: str = "GET"):
    """Track the start of a request."""
    counter = _registry.get("aegislang_requests_total")
    if counter:
        counter.inc(endpoint=endpoint, method=method)

    gauge = _registry.get("aegislang_active_connections")
    if gauge:
        gauge.inc()


def track_request_end(endpoint: str, method: str, duration: float, status: int):
    """Track the end of a request."""
    histogram = _registry.get("aegislang_request_duration_seconds")
    if histogram:
        histogram.observe(duration, endpoint=endpoint, method=method)

    gauge = _registry.get("aegislang_active_connections")
    if gauge:
        gauge.dec()

    if status >= 400:
        counter = _registry.get("aegislang_request_errors_total")
        if counter:
            counter.inc(endpoint=endpoint, method=method, status=str(status))


@contextmanager
def track_request(endpoint: str, method: str = "GET"):
    """Context manager for tracking request duration."""
    start_time = time.time()
    track_request_start(endpoint, method)
    status = 200

    try:
        yield
    except Exception:
        status = 500
        raise
    finally:
        duration = time.time() - start_time
        track_request_end(endpoint, method, duration, status)


def track_document_processing(doc_id: str, duration: float, success: bool = True):
    """Track document processing metrics."""
    counter = _registry.get("aegislang_documents_processed_total")
    if counter:
        counter.inc(status="success" if success else "failure")

    histogram = _registry.get("aegislang_document_processing_duration_seconds")
    if histogram:
        histogram.observe(duration)


def track_document_start(doc_id: str):
    """Track start of document processing."""
    gauge = _registry.get("aegislang_documents_in_progress")
    if gauge:
        gauge.inc()


def track_document_end(doc_id: str):
    """Track end of document processing."""
    gauge = _registry.get("aegislang_documents_in_progress")
    if gauge:
        gauge.dec()


def track_clause_extraction(doc_id: str, num_clauses: int):
    """Track clause extraction metrics."""
    counter = _registry.get("aegislang_clauses_extracted_total")
    if counter:
        counter.inc(value=num_clauses)

    histogram = _registry.get("aegislang_clauses_per_document")
    if histogram:
        histogram.observe(num_clauses)


def track_artifact_generation(format: str, count: int = 1):
    """Track artifact generation metrics."""
    counter = _registry.get("aegislang_artifacts_generated_total")
    if counter:
        counter.inc(value=count, format=format)


def track_compilation(duration: float, format: str):
    """Track compilation duration."""
    histogram = _registry.get("aegislang_compilation_duration_seconds")
    if histogram:
        histogram.observe(duration, format=format)


def track_validation(success: bool):
    """Track validation metrics."""
    counter = _registry.get("aegislang_validations_total")
    if counter:
        counter.inc(status="success" if success else "failure")

    if not success:
        failures = _registry.get("aegislang_validation_failures_total")
        if failures:
            failures.inc()


def track_llm_request(duration: float, tokens: int = 0, model: str = "unknown"):
    """Track LLM API request metrics."""
    counter = _registry.get("aegislang_llm_requests_total")
    if counter:
        counter.inc(model=model)

    histogram = _registry.get("aegislang_llm_request_duration_seconds")
    if histogram:
        histogram.observe(duration, model=model)

    if tokens > 0:
        token_counter = _registry.get("aegislang_llm_tokens_used_total")
        if token_counter:
            token_counter.inc(value=tokens, model=model)


# =============================================================================
# Decorators
# =============================================================================

def timed(metric_name: str, **labels):
    """Decorator to track function execution time."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                histogram = _registry.get(metric_name)
                if histogram:
                    histogram.observe(duration, **labels)
        return wrapper
    return decorator


def counted(metric_name: str, **labels):
    """Decorator to count function calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            counter = _registry.get(metric_name)
            if counter:
                counter.inc(**labels)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Metrics Export
# =============================================================================

def export_metrics_json() -> Dict[str, Any]:
    """Export all metrics as JSON."""
    result = {}

    for name, metric in _registry.all_metrics().items():
        if isinstance(metric, Counter):
            result[name] = {
                "type": "counter",
                "description": metric.description,
                "values": {str(k): v for k, v in metric.labels.items()},
            }
        elif isinstance(metric, Histogram):
            result[name] = {
                "type": "histogram",
                "description": metric.description,
                "values": {
                    str(k): metric.get_stats(**dict(k))
                    for k in metric.observations.keys()
                },
            }
        elif isinstance(metric, Gauge):
            result[name] = {
                "type": "gauge",
                "description": metric.description,
                "values": {str(k): v for k, v in metric.values.items()},
            }

    return result


def export_metrics_prometheus() -> str:
    """Export metrics in Prometheus text format."""
    lines = []

    for name, metric in _registry.all_metrics().items():
        lines.append(f"# HELP {name} {metric.description}")

        if isinstance(metric, Counter):
            lines.append(f"# TYPE {name} counter")
            for labels, value in metric.labels.items():
                label_str = ",".join(f'{k}="{v}"' for k, v in labels) if labels else ""
                if label_str:
                    lines.append(f"{name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{name} {value}")

        elif isinstance(metric, Gauge):
            lines.append(f"# TYPE {name} gauge")
            for labels, value in metric.values.items():
                label_str = ",".join(f'{k}="{v}"' for k, v in labels) if labels else ""
                if label_str:
                    lines.append(f"{name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{name} {value}")

        elif isinstance(metric, Histogram):
            lines.append(f"# TYPE {name} histogram")
            for labels, observations in metric.observations.items():
                label_str = ",".join(f'{k}="{v}"' for k, v in labels) if labels else ""
                count = len(observations)
                total = sum(observations)

                # Bucket counts
                for bucket in metric.buckets:
                    bucket_count = sum(1 for o in observations if o <= bucket)
                    if label_str:
                        lines.append(f'{name}_bucket{{{label_str},le="{bucket}"}} {bucket_count}')
                    else:
                        lines.append(f'{name}_bucket{{le="{bucket}"}} {bucket_count}')

                # +Inf bucket
                if label_str:
                    lines.append(f'{name}_bucket{{{label_str},le="+Inf"}} {count}')
                    lines.append(f"{name}_sum{{{label_str}}} {total}")
                    lines.append(f"{name}_count{{{label_str}}} {count}")
                else:
                    lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                    lines.append(f"{name}_sum {total}")
                    lines.append(f"{name}_count {count}")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# FastAPI Integration
# =============================================================================

def setup_metrics(app):
    """
    Setup metrics middleware for FastAPI application.

    Usage:
        from fastapi import FastAPI
        from aegislang.core.metrics import setup_metrics

        app = FastAPI()
        setup_metrics(app)
    """
    from fastapi import Request, Response
    from fastapi.routing import APIRoute

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Middleware to track request metrics."""
        start_time = time.time()

        # Extract endpoint
        endpoint = request.url.path
        method = request.method

        track_request_start(endpoint, method)

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            status = 500
            raise
        finally:
            duration = time.time() - start_time
            track_request_end(endpoint, method, duration, status)

        return response

    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        return Response(
            content=export_metrics_prometheus(),
            media_type="text/plain",
        )

    @app.get("/api/v1/metrics")
    async def metrics_json_endpoint():
        """JSON metrics endpoint."""
        return export_metrics_json()


# =============================================================================
# Health Check with Metrics
# =============================================================================

def get_health_metrics() -> Dict[str, Any]:
    """Get health-related metrics for health check endpoint."""
    return {
        "requests_total": _registry.get("aegislang_requests_total").labels if _registry.get("aegislang_requests_total") else {},
        "documents_processed": _registry.get("aegislang_documents_processed_total").labels if _registry.get("aegislang_documents_processed_total") else {},
        "active_connections": _registry.get("aegislang_active_connections").values if _registry.get("aegislang_active_connections") else {},
        "documents_in_progress": _registry.get("aegislang_documents_in_progress").values if _registry.get("aegislang_documents_in_progress") else {},
    }
