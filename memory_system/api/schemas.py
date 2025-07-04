# schemas.py — Pydantic models for Unified Memory System API
#
# Version: 0.8‑alpha
"""Centralised data‑contracts used by the REST API layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_API_VERSION: str = "v1"
_SERVICE_VERSION: str = "0.8-alpha"


# ---------------------------------------------------------------------------
# Memory domain models
# ---------------------------------------------------------------------------
class MemoryBase(BaseModel):
    """Fields common to create / update operations."""

    text: str = Field(..., min_length=1, max_length=10_000)
    role: str = Field("user", max_length=32, description="Conversation role label")
    tags: List[str] = Field(default_factory=list, max_items=10)


class MemoryCreate(MemoryBase):
    """Payload for *create* operation."""

    user_id: Optional[str] = Field(
        None, description="Owner identifier (if omitted — resolved from auth context)"
    )


class MemoryUpdate(MemoryBase):
    """Partial update — all fields optional but at least one is required."""

    text: Optional[str] = None
    role: Optional[str] = None
    tags: Optional[List[str]] = None

    model_config = {
        "extra": "forbid",
        "validate_default": True,
    }


class MemoryRead(MemoryBase):
    """Full memory record as returned by the API."""

    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Query / search models
# ---------------------------------------------------------------------------
class MemoryQuery(BaseModel):
    """Vector or text query parameters."""

    query: str = Field(..., min_length=1, max_length=1_000)
    top_k: int = Field(10, ge=1, le=100)
    include_embeddings: bool = Field(
        False, description="Return raw vector embeddings in the response"
    )


class MemorySearchResult(MemoryRead):
    """Memory with an additional similarity score."""

    score: float = Field(..., ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Health & monitoring models (used by health routes)
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Global health report for the service."""

    status: str = Field(..., description="healthy | degraded | unhealthy")
    timestamp: str
    uptime_seconds: int = Field(..., ge=0)
    version: str = Field(_SERVICE_VERSION, description="Service version")
    checks: Dict[str, bool]
    memory_store_health: Dict[str, Any]
    api_enabled: bool


class StatsResponse(BaseModel):
    """Aggregated runtime metrics for dashboards / automation."""

    total_memories: int = Field(..., ge=0)
    active_sessions: int = Field(..., ge=0)
    uptime_seconds: int = Field(..., ge=0)
    memory_store_stats: Dict[str, Any]
    api_stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# Generic success / error wrappers (optional helpers)
# ---------------------------------------------------------------------------
class SuccessResponse(BaseModel):
    message: str = "success"
    api_version: str = _API_VERSION


class ErrorResponse(BaseModel):
    detail: str
    api_version: str = _API_VERSION
