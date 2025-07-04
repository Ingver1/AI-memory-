"""app.py — FastAPI application factory for Unified Memory System (v0.8-alpha)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routes and middleware lazily to avoid circular imports
from memory_system.api.routes import health, admin, memory
from memory_system.api.middleware import MaintenanceModeMiddleware, RateLimitingMiddleware
from memory_system.config.settings import UnifiedSettings
from memory_system.core.store import EnhancedMemoryStore

logger = logging.getLogger(__name__)

# Global singleton instances (populated on app startup)
_settings: Optional[UnifiedSettings] = None
_memory_store: Optional[EnhancedMemoryStore] = None
_rate_limiter: Optional[RateLimitingMiddleware] = None
_maintenance_middleware: Optional[MaintenanceModeMiddleware] = None

# Dependency getters for FastAPI

def get_settings_instance() -> UnifiedSettings:
    """Get or create the global UnifiedSettings singleton."""
    global _settings
    if _settings is None:
        _settings = UnifiedSettings()  # Loads settings (from env, .env, etc.)
        logger.info("✅ Settings loaded (environment profile: %s)", _settings.profile)
    return _settings

async def get_memory_store_instance() -> EnhancedMemoryStore:
    """Get or initialize the global EnhancedMemoryStore singleton."""
    global _memory_store
    if _memory_store is None:
        settings = get_settings_instance()
        # Create the EnhancedMemoryStore in a thread to avoid blocking the event loop
        _memory_store = await asyncio.to_thread(EnhancedMemoryStore, settings)
        logger.info("✅ Memory store initialized")
    return _memory_store

def get_maintenance_middleware_instance() -> Optional[MaintenanceModeMiddleware]:
    """Return the MaintenanceModeMiddleware instance, if initialized."""
    return _maintenance_middleware

# Application lifespan context for startup/shutdown

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Context manager for FastAPI app lifespan (setup and cleanup)."""
    # On startup: initialize settings and memory store
    settings = get_settings_instance()
    await get_memory_store_instance()
    logger.info("🚀 Unified Memory System v%s started (profile=%s)...", settings.version, settings.profile)
    yield  # yield control back to FastAPI (run the app)
    # On shutdown: clean up resources
    if _memory_store is not None:
        await _memory_store.close()
        logger.info("🛑 Memory store closed – shutting down.")

def create_app() -> FastAPI:
    """Application factory that configures and returns a FastAPI app for UMS."""
    settings = get_settings_instance()
    app = FastAPI(
        title="Unified Memory System",
        description="Enterprise-grade memory system with vector search, FastAPI, and monitoring",
        version=settings.version or "0.8-alpha",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    # Conditional CORS configuration
    if settings.api.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # Install global middlewares
    maintenance = MaintenanceModeMiddleware(app)
    app.add_middleware(RateLimitingMiddleware)
    # Keep reference to maintenance middleware for runtime toggling
    global _maintenance_middleware
    _maintenance_middleware = maintenance

    # Include API route routers with version prefix
    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(admin.router, prefix=api_prefix)
    app.include_router(memory.router, prefix=api_prefix)

    return app

# CLI entry point for running directly
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
