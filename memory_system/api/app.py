"""app.py — FastAPI application factory for Unified Memory System

Version: 0.8‑alpha
"""

from __future__ import annotations

from memory_system.api.routes import health, admin, memory  # Add memory!

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local imports — keep them lazy to avoid circulars at import‑time
from memory_system.api.middleware import (
    MaintenanceModeMiddleware,
    RateLimitingMiddleware,
)
from memory_system.config.settings import UnifiedSettings
from memory_system.core.store import EnhancedMemoryStore

logger = logging.getLogger(__name__)

###############################################################################
# Global singletons                                                           #
###############################################################################

_settings: UnifiedSettings | None = None
_memory_store: EnhancedMemoryStore | None = None
_rate_limiter: RateLimitingMiddleware | None = None
_maintenance_middleware: MaintenanceModeMiddleware | None = None


# ---------------------------------------------------------------------------
# Dependency getters (FastAPI can resolve them via Depends())
# ---------------------------------------------------------------------------


def get_settings_instance() -> UnifiedSettings:
    """Returns the cached UnifiedSettings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = UnifiedSettings()  # loads from env vars / .env
        logger.info("✅ Settings loaded (env profile: %s)", _settings.profile)
    return _settings


async def get_memory_store_instance() -> EnhancedMemoryStore:
    """Lazy‑creates and returns the global EnhancedMemoryStore."""
    global _memory_store
    if _memory_store is None:
        settings = get_settings_instance()
        # Creating the store may involve network IO, do it in a thread.
        _memory_store = await asyncio.to_thread(EnhancedMemoryStore, settings)
        logger.info("✅ Memory store initialised")
    return _memory_store


def get_maintenance_middleware_instance() -> Optional[MaintenanceModeMiddleware]:
    """Returns the cached MaintenanceModeMiddleware instance if available."""
    return _maintenance_middleware


###############################################################################
# Lifespan — graceful startup/shutdown                                        #
###############################################################################


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manages application resources for the whole lifecycle."""
    # Startup ---------------------------------------------------------------
    settings = get_settings_instance()
    await get_memory_store_instance()

    logger.info(
        "🚀 Unified Memory System v%s started (profile=%s) …",
        settings.version,
        settings.profile,
    )
    yield  # ----------------------------------------------------------------
    # Shutdown --------------------------------------------------------------
    if _memory_store is not None:
        await _memory_store.close()
        logger.info("🛑 Memory store closed — goodbye!")


###############################################################################
# Application factory                                                         #
###############################################################################


def create_app() -> FastAPI:
    """Factory that builds and configures the FastAPI application."""
    settings = get_settings_instance()

    app = FastAPI(
        title="Unified Memory System",
        description="Enterprise‑grade memory system with vector search, FastAPI and monitoring",
        version="0.8-alpha",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # CORS (optional) -------------------------------------------------------
    if settings.api.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Custom middlewares ----------------------------------------------------
    maintenance = MaintenanceModeMiddleware(app)
    app.add_middleware(RateLimitingMiddleware)
    # Keep a reference so we can toggle at runtime if needed.
    global _maintenance_middleware
    _maintenance_middleware = maintenance

    # Routers ---------------------------------------------------------------

    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(admin.router, prefix=api_prefix)
    app.include_router(memory.router, prefix=api_prefix)
    
    return app



###############################################################################
# CLI entry‑point                                                             #
###############################################################################

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "memory_system.api.app:create_app",
        host="0.0.0.0",
        port=8000,
        factory=True,
        reload=True,
        log_level="info",
    )
