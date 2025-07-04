"""
Health-check and monitoring endpoints.
"""

import logging
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import Response

from memory_system.api.schemas import HealthResponse, StatsResponse
from memory_system.api.middleware import check_dependencies, session_tracker
from memory_system.core.store import EnhancedMemoryStore
from memory_system.utils.metrics import get_prometheus_metrics, get_metrics_content_type
from memory_system.config.settings import UnifiedSettings

log = logging.getLogger(__name__)
router = APIRouter(tags=["Health & Monitoring"])


# Dependency helpers (resolved via the DI container)
async def _store() -> EnhancedMemoryStore:
    from memory_system.api.app import get_memory_store_instance

    return await get_memory_store_instance()


def _settings() -> UnifiedSettings:
    from memory_system.api.app import get_settings_instance

    return get_settings_instance()


# ───────────────────────────  root  ───────────────────────────
@router.get("/", summary="Service info")
async def root() -> Dict[str, Any]:
    return {
        "service": "Unified Memory System",
        "version": "0.8-alpha",
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "api_version": "v1",
    }


# ───────────────────────  health / live / ready  ───────────────────────
@router.get("/health", response_model=HealthResponse, summary="Full health check")
async def health_check(
    memory_store: EnhancedMemoryStore = Depends(_store),
    settings: UnifiedSettings = Depends(_settings),
) -> HealthResponse:
    try:
        component = await memory_store.get_health()
        deps = await check_dependencies()

        overall_ok = component.healthy and all(deps.values())
        status = "healthy" if overall_ok else "degraded"

        stats = await memory_store.get_stats()

        return HealthResponse(
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=component.uptime,
            version="0.8-alpha",
            checks={**component.checks, **deps},
            memory_store_health={
                "total_memories": stats.get("total_memories", 0),
                "index_size": stats.get("index_size", 0),
                "cache_hit_rate": stats.get("cache_stats", {}).get("hit_rate", 0),
                "buffer_size": stats.get("buffer_size", 0),
            },
            api_enabled=settings.api.enable_api,
        )
    except Exception as e:
        log.error("Health check failed: %s", e, exc_info=True)
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=0,
            version="0.8-alpha",
            checks={"health_check_error": False},
            memory_store_health={"error": str(e)},
            api_enabled=settings.api.enable_api,
        )


@router.get("/health/live", summary="Liveness probe")
async def liveness_probe() -> Dict[str, str]:
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/ready", summary="Readiness probe")
async def readiness_probe(
    memory_store: EnhancedMemoryStore = Depends(_store),
) -> Dict[str, Any]:
    component = await memory_store.get_health()
    if component.healthy:
        return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}
    raise HTTPException(status_code=503, detail=f"Service not ready: {component.message}")


# ───────────────────────────  stats  ───────────────────────────
@router.get("/stats", response_model=StatsResponse, summary="System statistics")
async def get_stats(
    memory_store: EnhancedMemoryStore = Depends(_store),
    settings: UnifiedSettings = Depends(_settings),
) -> StatsResponse:
    stats = await memory_store.get_stats()

    # active session count (1-hour window)
    current_time = datetime.now(tz=timezone.utc).timestamp()
    active = sum(1 for ts in session_tracker.values() if ts > current_time - 3600)

    return StatsResponse(
        total_memories=stats.get("total_memories", 0),
        active_sessions=active,
        uptime_seconds=stats.get("uptime_seconds", 0),
        memory_store_stats=stats,
        api_stats={
            "cors_enabled": settings.api.enable_cors,
            "rate_limiting_enabled": settings.monitoring.enable_rate_limiting,
            "metrics_enabled": settings.monitoring.enable_metrics,
            "encryption_enabled": settings.security.encrypt_at_rest,
            "pii_filtering_enabled": settings.security.filter_pii,
            "backup_enabled": settings.reliability.backup_enabled,
            "model_name": settings.model.model_name,
            "api_version": "v1",
        },
    )


# ───────────────────────────  metrics  ───────────────────────────
@router.get("/metrics", summary="Prometheus metrics")
async def metrics(settings: UnifiedSettings = Depends(_settings)) -> Response:
    if not settings.monitoring.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")

    return Response(
        content=get_prometheus_metrics(),
        media_type=get_metrics_content_type(),
    )


# ───────────────────────────  version  ───────────────────────────
@router.get("/version", summary="Version info")
async def get_version() -> Dict[str, Any]:
    return {
        "version": "0.8-alpha",
        "api_version": "v1",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
    }
