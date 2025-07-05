"""memory_system.api.app
=======================

FastAPI application instance for **AI‑memory‑**.

Key points
----------
* **OpenTelemetry** middleware for distributed tracing (HTTP spans).
* Lifespan‑managed singleton of :class:`~memory_system.core.store.SQLiteMemoryStore`
  (no hidden globals → better test isolation).
* Modular routers with Swagger tags and example payloads.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Generator, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from memory_system.config.settings import Settings, configure_logging, get_settings
from memory_system.core.store import Memory, SQLiteMemoryStore, get_memory_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan management                                                           
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources and tear them down at shutdown."""

    settings: Settings = get_settings()
    configure_logging(settings)

    # 1. Setup OpenTelemetry (stdout exporter for PoC; replace in prod)
    resource = Resource(attributes={SERVICE_NAME: settings.service_name})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    # 2. Create memory store and attach to app state
    memory_store = SQLiteMemoryStore(
        dsn=settings.sqlite_dsn,
        pool_size=settings.sqlite_pool_size,
    )
    await memory_store.init()
    app.state.memory_store = memory_store
    logger.info("SQLiteMemoryStore initialised (pool=%s)…", settings.sqlite_pool_size)

    try:
        yield
    finally:
        await memory_store.close()
        logger.info("SQLiteMemoryStore closed — bye!")


# ---------------------------------------------------------------------------
# FastAPI instance                                                              
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI‑memory− API",
    version="0.9.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Memory", "description": "Create/search textual memories"},
        {"name": "Health", "description": "Liveness and readiness probes"},
    ],
)

# Instrument AFTER creation (so lifespan is wrapped as well)
FastAPIInstrumentor.instrument_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers                                                                      
# ---------------------------------------------------------------------------

@app.post(
    "/memory",
    tags=["Memory"],
    status_code=status.HTTP_201_CREATED,
    summary="Add a memory",
    response_model=Memory,
    responses={
        201: {"description": "Memory stored", "model": Memory},
        400: {"description": "Invalid payload"},
    },
    openapi_extra={
        "examples": {
            "basic": {
                "summary": "Minimal payload",
                "value": {
                    "content": "I met John at the café today.",
                    "metadata": {"importance": 0.8, "tags": ["social", "journal"]},
                },
            }
        }
    },
)
async def add_memory(
    payload: Memory,
    store: SQLiteMemoryStore = Depends(get_memory_store),
) -> Memory:  # pragma: no cover — thin router layer
    return await store.add_memory(payload)


@app.get(
    "/memory/{memory_id}",
    tags=["Memory"],
    summary="Retrieve a memory by ID",
    response_model=Memory,
)
async def get_memory(
    memory_id: str = Path(..., description="UUID of the stored memory"),
    store: SQLiteMemoryStore = Depends(get_memory_store),
) -> Memory:  # pragma: no cover
    memory = await store.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@app.get(
    "/health/liveness",
    tags=["Health"],
    summary="Basic liveness probe",
)
async def liveness() -> dict[str, str]:  # pragma: no cover
    return {"status": "ok"}


@app.get(
    "/health/readiness",
    tags=["Health"],
    summary="Database readiness probe",
)
async def readiness(store: SQLiteMemoryStore = Depends(get_memory_store)) -> dict[str, str]:  # pragma: no cover
    try:
        await store.ping()
        return {"status": "ready"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail="DB not ready") from exc
