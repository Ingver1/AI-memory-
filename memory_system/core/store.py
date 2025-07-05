"""memory_system.core.store
=================================
Asynchronous SQLite‑backed memory store with JSON1 metadata support and
connection pooling via **aiosqlite**.  Designed to be injected through a
FastAPI lifespan context — no hidden singletons.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import aiosqlite

logger = logging.getLogger(__name__)

###############################################################################
# Data model
###############################################################################

@dataclass(slots=True, frozen=True)
class Memory:
    """A single memory entry."""

    id: str
    text: str
    created_at: dt.datetime
    importance: float = 0.0  # 0..1
    metadata: Dict[str, Any] | None = None

    @staticmethod
    def new(text: str, *, importance: float = 0.0, metadata: Optional[Dict[str, Any]] = None) -> "Memory":
        return Memory(
            id=str(uuid.uuid4()),
            text=text,
            created_at=dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc),
            importance=importance,
            metadata=metadata or {},
        )

###############################################################################
# Store implementation
###############################################################################

class SQLiteMemoryStore:
    """Async store that leverages SQLite JSON1 for flexible metadata queries."""

    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS memories (
        id          TEXT PRIMARY KEY,
        text        TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        importance  REAL DEFAULT 0,
        metadata    JSON
    );
    """

    def __init__(self, dsn: str = "file:memories.db?mode=rwc", *, pool_size: int = 5) -> None:
        self._dsn = dsn
        self._pool_size = pool_size
        self._pool: asyncio.LifoQueue[aiosqlite.Connection] = asyncio.LifoQueue(maxsize=pool_size)
        self._initialised: bool = False
        self._lock = asyncio.Lock()  # protects initialisation & pool resize

    # ---------------------------------------------------------------------
    # Low‑level connection helpers
    # ---------------------------------------------------------------------
    async def _acquire(self) -> aiosqlite.Connection:
        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            conn = await aiosqlite.connect(self._dsn, uri=True, timeout=30)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = aiosqlite.Row
            return conn

    async def _release(self, conn: aiosqlite.Connection) -> None:
        try:
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            await conn.close()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    async def initialise(self) -> None:
        """Create table / indices once per process."""
        if self._initialised:
            return
        async with self._lock:
            if self._initialised:
                return
            conn = await self._acquire()
            try:
                await conn.execute(self._CREATE_SQL)
                await conn.commit()
                self._initialised = True
            finally:
                await self._release(conn)
            logger.info("SQLiteMemoryStore initialised (dsn=%s)", self._dsn)

    async def aclose(self) -> None:
        """Close all pooled connections."""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------
    async def add(self, mem: Memory) -> None:
        await self.initialise()
        conn = await self._acquire()
        try:
            await conn.execute(
                "INSERT INTO memories (id, text, created_at, importance, metadata) VALUES (?, ?, ?, ?, json(?))",
                (
                    mem.id,
                    mem.text,
                    mem.created_at.isoformat(),
                    mem.importance,
                    json.dumps(mem.metadata) if mem.metadata else "null",
                ),
            )
            await conn.commit()
        finally:
            await self._release(conn)

    async def get(self, memory_id: str) -> Optional[Memory]:
        await self.initialise()
        conn = await self._acquire()
        try:
            row = await conn.execute_fetchone(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )
            return self._row_to_memory(row) if row else None
        finally:
            await self._release(conn)

    async def delete(self, memory_id: str) -> None:
        await self.initialise()
        conn = await self._acquire()
        try:
            await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            await conn.commit()
        finally:
            await self._release(conn)

    async def search(
        self,
        text_query: Optional[str] = None,
        *,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[Memory]:
        """Simple LIKE + JSON1 metadata search (no vectors here)."""
        await self.initialise()
        conn = await self._acquire()
        try:
            where: List[str] = []
            params: List[Any] = []
            if text_query:
                where.append("text LIKE ?")
                params.append(f"%{text_query}%")
            if metadata_filters:
                for key, val in metadata_filters.items():
                    where.append("json_extract(metadata, ?) = ?")
                    params.extend([f"$.{key}", val])
            sql = "SELECT * FROM memories"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = await conn.execute_fetchall(sql, params)
            return [self._row_to_memory(r) for r in rows]
        finally:
            await self._release(conn)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_memory(row: aiosqlite.Row) -> Memory:
        return Memory(
            id=row["id"],
            text=row["text"],
            created_at=dt.datetime.fromisoformat(row["created_at"]).replace(tzinfo=dt.timezone.utc),
            importance=row["importance"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

###############################################################################
# FastAPI integration helpers (optional import‑time dep)
###############################################################################

try:
    from fastapi import FastAPI, Request
except ModuleNotFoundError:  # pragma: no cover – FastAPI not installed in some envs
    FastAPI = Request = None  # type: ignore


async def lifespan_context(app: "FastAPI") -> AsyncIterator[None]:  # pragma: no cover
    """FastAPI lifespan function that attaches a SQLiteMemoryStore to ``app.state``."""

    store = SQLiteMemoryStore()
    await store.initialise()
    app.state.memory_store = store
    try:
        yield
    finally:
        await store.aclose()


def get_memory_store(request: "Request") -> SQLiteMemoryStore:  # pragma: no cover
    return request.app.state.memory_store
    
from memory_system.core.enhanced_store import EnhancedMemoryStore, HealthComponent  # Ensure EnhancedMemoryStore & HealthComponent are accessible via core.store
