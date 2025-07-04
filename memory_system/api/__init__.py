# memory_system/api/__init__.py
"""API module for Unified Memory System."""

from __future__ import annotations

__all__ = [
    "create_app",
    "HealthResponse", 
    "StatsResponse",
    "MemoryCreate",
    "MemoryRead", 
    "MemoryQuery",
]

def __getattr__(name: str):
    if name == "create_app":
        from memory_system.api.app import create_app
        return create_app
    elif name in ("HealthResponse", "StatsResponse", "MemoryCreate", "MemoryRead", "MemoryQuery"):
        from memory_system.api.schemas import (
            HealthResponse,
            StatsResponse, 
            MemoryCreate,
            MemoryRead,
            MemoryQuery,
        )
        return locals()[name]
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
