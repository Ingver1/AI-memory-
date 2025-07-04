# dependencies.py — FastAPI dependency helpers for Unified Memory System

# Version: 0.8‑alpha

"""
Reusable dependency callables exposed to FastAPI routes and middleware.

The goal is to keep route files free from import-heavy boilerplate while still
allowing unit tests to override singletons easily via FastAPI's dependency
override mechanism.

Exports:
• get_settings          – singleton of :class:`UnifiedSettings`
• get_memory_store      – singleton of :class:`SQLiteMemoryStore`
• require_api_enabled   – raises 503 if the API is disabled
"""
from __future__ import annotations

import functools
import logging

from fastapi import Depends, HTTPException, status

from memory_system.config import UnifiedSettings
from memory_system.store import SQLiteMemoryStore

__all__ = [
    "get_settings",
    "get_memory_store",
    "require_api_enabled",
]

log = logging.getLogger("ums.dependencies")


@functools.lru_cache()
def get_settings() -> UnifiedSettings:
    return UnifiedSettings()


@functools.lru_cache()
def get_memory_store() -> SQLiteMemoryStore:
    return SQLiteMemoryStore()


def require_api_enabled(settings: UnifiedSettings = Depends(get_settings)) -> None:
    if not settings.api_enabled:
        log.warning("API is disabled — blocking request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The API is currently disabled by configuration",
        )
