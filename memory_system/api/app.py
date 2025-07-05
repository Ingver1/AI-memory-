"""memory_system.api.app
=======================
FastAPI application setup with:
* Lifespan‑managed `SQLiteMemoryStore` (no hidden globals).
* OpenTelemetry middleware for distributed tracing.
* Basic `/health/live` and `/health/ready` endpoints for liveness & readiness probes.
"""
from __future__ import annotations

import logging
import os
from typing import AsyncIterator, Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from memory_system.config.settings import Settings, get_settings, configure_logging
from memory_system.core.store import create_memory_store, SQLiteMemoryStore, get_memory_store
from memory_system.unified_memory import add, search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
router = APIRouter(tags=["Memory"], prefix="/memory")

@router.post("/", summary="Add memory", response_description="Memory UUID")
async def add_memory(request: Request, body: Dict[str, Any]) -> Dict[str, str]:
    """Add a new piece of memory."""
    store: SQLiteMemoryStore = get_memory_store(request)
    uid = await add(body["text"], metadata=body.get("metadata", {}), store=store)
    return {"id": uid}

@router.get("/search", summary="Search memory", response_description="Search results")
async def search_memory(request: Request, q: str, limit: int = 5):
    """Semantic search across stored memories."""
    store: SQLiteMemoryStore = get_memory_store(request)
    return await search(q, limit=limit, store=store)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(settings: Settings | None = None) -> FastAPI:  # pragma: no cover
    settings = settings or get_settings()

    configure_logging()

    app = FastAPI(title="AI‑memory‑ API", version="0.8.0")

    # CORS (can be tightened in prod)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OpenTelemetry
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        FastAPIInstrumentor.instrument_app(app)

    # Health probes ---------------------------------------------------------
    @app.get("/health/live", include_in_schema=False)
    async def live() -> Dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", include_in_schema=False)
    async def ready(request: Request) -> Dict[str, str]:
        try:
            store: SQLiteMemoryStore = get_memory_store(request)
            await store.ping()
            return {"status": "ready"}
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Readiness check failed: %s", exc)
            return {"status": "unready"}

    # Lifespan --------------------------------------------------------------
    @app.on_event("startup")
    async def _startup() -> None:  # noqa: WPS430  (nested function OK here)
        app.state.store = await create_memory_store(settings)
        logger.info("SQLiteMemoryStore initialised")

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # noqa: WPS430
        store: SQLiteMemoryStore = app.state.store  # type: ignore[attr-defined]
        await store.close()
        logger.info("SQLiteMemoryStore closed")

    # Dependency bridge -----------------------------------------------------
    app.dependency_overrides[get_memory_store] = lambda req: req.app.state.store  # type: ignore[arg-type]

    # Routers ---------------------------------------------------------------
    app.include_router(router)

    return app

# ---------------------------------------------------------------------------
# Entry‑point for `uvicorn memory_system.api.app:app`
# ---------------------------------------------------------------------------
settings = get_settings()
app = create_app(settings)
