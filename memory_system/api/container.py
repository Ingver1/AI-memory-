"""container.py — lightweight dependency container for Unified Memory System."""

from __future__ import annotations

import asyncio
import functools
import logging
from pathlib import Path
from typing import Optional

from memory_system.config.settings import UnifiedSettings
from memory_system.core.store import EnhancedMemoryStore

__all__ = ["get_settings_instance", "get_memory_store_instance"]

log = logging.getLogger(__name__)

# Global cached instances
_settings_cache: Optional[UnifiedSettings] = None
_store_cache: Optional[EnhancedMemoryStore] = None

def get_settings_instance() -> UnifiedSettings:
    """Return a singleton UnifiedSettings instance (cached after first call)."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = UnifiedSettings()
        log.info("Settings initialized (profile=%s)", _settings_cache.profile)
    return _settings_cache

async def get_memory_store_instance() -> EnhancedMemoryStore:
    """Return a singleton EnhancedMemoryStore instance (initialized asynchronously)."""
    global _store_cache
    if _store_cache is None:
        settings = get_settings_instance()
        # Using asyncio.to_thread to avoid blocking on initialization (which may perform I/O)
        _store_cache = await asyncio.to_thread(EnhancedMemoryStore, settings)
        log.info("Memory store created in container")
    return _store_cache
