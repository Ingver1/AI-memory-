"""Unified Memory System — Prometheus Metrics Utilities (v0.8-alpha)."""

from __future__ import annotations

import logging
import time
from typing import Callable, Coroutine

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

log = logging.getLogger(__name__)


# Helper function for creating counters
def prometheus_counter(name: str, description: str, labels: list[str] | None = None) -> Counter:
    """Create a Prometheus Counter with optional labels."""
    if labels:
        return Counter(name, description, labels)
    return Counter(name, description)


# ────────── Base collectors ──────────
MET_ERRORS_TOTAL = Counter("ums_errors_total", "Total errors", ["type", "component"])
LAT_DB_QUERY = Histogram("ums_db_query_latency_seconds", "DB query latency")
LAT_SEARCH = Histogram("ums_search_latency_seconds", "Vector search latency")
LAT_EMBEDDING = Histogram("ums_embedding_latency_seconds", "Embedding generation latency")
MET_POOL_EXHAUSTED = Counter("ums_pool_exhausted_total", "Connection pool exhausted events")

# ────────── System gauges ──────────
SYSTEM_CPU = Gauge("system_cpu_percent", "CPU usage %")
SYSTEM_MEM = Gauge("system_mem_percent", "Memory usage %")
PROCESS_UPTIME = Gauge("process_uptime_seconds", "Process uptime")

_START = time.monotonic()
PROCESS_UPTIME.set(0.0)


# ────────── Timing decorators ──────────
def _wrap_sync(metric: Histogram) -> Callable:
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with metric.time():
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def _wrap_async(metric: Histogram) -> Callable:
    def decorator(fn: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        async def wrapper(*args, **kwargs):
            with metric.time():
                return await fn(*args, **kwargs)

        return wrapper

    return decorator


measure_time = _wrap_sync
measure_time_async = _wrap_async


# ────────── Helpers ──────────
def update_system_metrics() -> None:
    """Push basic host metrics (requires *psutil*)."""
    try:
        import psutil

        SYSTEM_CPU.set(psutil.cpu_percent())
        SYSTEM_MEM.set(psutil.virtual_memory().percent)
        PROCESS_UPTIME.set(time.monotonic() - _START)
    except ImportError:
        log.debug("psutil not installed — skipping system gauges")


def get_prometheus_metrics() -> str:
    """Return metrics text for `/metrics` endpoint."""
    return generate_latest().decode()


def get_metrics_content_type() -> str:
    """Return content-type for metrics endpoint."""
    return CONTENT_TYPE_LATEST
