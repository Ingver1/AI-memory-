# container.py — lightweight dependency container for Unified Memory System
#
# Version: 0.8‑alpha
"""Central place that wires singletons shared across the whole application.

A *full‑blown* DI framework (like *punq*, *wired*, or *lagom*) would be
overkill for this project.  Instead we expose two lazily‑initialised globals:

* :pyfunc:`get_settings_instance` – returns a singleton of
  :class:`~memory_system.config.settings.UnifiedSettings`.
* :pyasyncfunc:`get_memory_store_instance` – returns a singleton of
  :class:`~memory_system.core.store.EnhancedMemoryStore` (created asynchronously
  because some back‑ends require I/O during bootstrap).

These helpers are imported by :pyfile:`app.py`, CLI utilities and tests, so
keep their signatures stable.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from pathlib import Path
from typing import Optional

from memory_system.config.settings import UnifiedSettings
from memory_system.core.store import EnhancedMemoryStore

__all__ = [
    "get_settings_instance",
    "get_memory_store_instance",
]

log = logging.getLogger(__name__)

###############################################################################
# Settings singleton                                                          #
###############################################################################

_settings_lock = asyncio.Lock()
_settings_instance: Optional[UnifiedSettings] = None


def _create_settings() -> UnifiedSettings:
    """Read environment variables and create *UnifiedSettings* once."""
    return UnifiedSettings()


@functools.lru_cache(maxsize=1)
def get_settings_instance() -> UnifiedSettings:  # noqa: D401 – simple helper
    """Return the global *UnifiedSettings* singleton (lazy‑loaded)."""
    global _settings_instance  # noqa: PLW0603 – we want to cache at module scope
    if _settings_instance is None:
        _settings_instance = _create_settings()
        log.info("Settings loaded: %s", _settings_instance.model_dump(mode="json"))
    return _settings_instance


###############################################################################
# Memory store singleton                                                      #
###############################################################################

_store_lock = asyncio.Lock()
_store_instance: Optional[EnhancedMemoryStore] = None


async def _create_memory_store(settings: UnifiedSettings) -> EnhancedMemoryStore:
    """Factory: instantiates the store and applies initial migrations."""
    db_path = Path(settings.storage.db_path).expanduser().resolve()
    store = EnhancedMemoryStore(db_path=db_path, vector_dim=settings.model.vector_dim)
    await store.migrate()
    log.info("Memory store initialised at %s", db_path)
    return store


async def get_memory_store_instance() -> EnhancedMemoryStore:  # noqa: D401
    """Return the global *EnhancedMemoryStore* singleton (lazy‑loaded)."""
    global _store_instance  # noqa: PLW0603
    if _store_instance is None:
        async with _store_lock:
            if _store_instance is None:  # double‑checked locking
                settings = get_settings_instance()
                _store_instance = await _create_memory_store(settings)
    return _store_instance
