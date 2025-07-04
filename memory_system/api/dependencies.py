"""dependencies.py — FastAPI dependency helper functions for Unified Memory System."""

from __future__ import annotations

import functools
import logging

from fastapi import Depends, HTTPException, status

from memory_system.config.settings import UnifiedSettings
from memory_system.core.store import EnhancedMemoryStore

__all__ = ["get_settings", "get_memory_store", "require_api_enabled"]

log = logging.getLogger("ums.dependencies")

@functools.lru_cache()
def get_settings() -> UnifiedSettings:
    """Provide a cached UnifiedSettings instance (singleton)."""
    return UnifiedSettings()

@functools.lru_cache()
def get_memory_store() -> EnhancedMemoryStore:
    """Provide a cached EnhancedMemoryStore instance (singleton)."""
    return EnhancedMemoryStore(get_settings())  # Note: runs in sync for simplicity

def require_api_enabled(settings: UnifiedSettings = Depends(get_settings)) -> None:
    """FastAPI dependency that raises an HTTP 503 if the API is disabled in settings."""
    if not settings.api.enable_api:
        log.warning("API is disabled by configuration — blocking request.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The API is currently disabled by configuration.",
        )
