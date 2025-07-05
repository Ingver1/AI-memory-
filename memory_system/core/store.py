"""memory_system.core.store
=================================
Async persistent storage for textual memories with **SQLite + JSON1**
and a lightweight **aiosqlite** connection pool.  Designed to integrate
smoothly with FastAPI via lifespan/dependency injection – avoiding the
old hidden singleton and making tests deterministic.

Public API  (async)    Description
---------------------  ---------------------------------------------
`add_memory()`         Insert a new memory row
`search_memories()`    Full‑text LIKE search + JSON filter
`get_memory()`         Fetch by primary id
`update_metadata()`    Patch JSON metadata using ⇒ operator
`delete_memory()`      Hard‑delete a row

Additionally, `InMemoryStore` implements the same interface for fast
unit‑testing without touching the filesystem.
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol, Sequence

import aiosqlite

logger = logging.getLogger(__name__)

###############################################################################
# Domain model
###############################################################################

@dataclass(slots=True)
class Memory:
    """Single memory record."""

    id: str
    content: str
    metadata: dict[str, Any]
    created_at: float


###############################################################################
# Abstract interface – handy for unit‑test mocks
###############################################################################

class AbstractMemoryStore(Protocol):
    """Behaviour contract for memory stores (sync or async)."""

    async def add_memory(self, mem: Memory) -> None: ...  # noqa: D401, E501

    async def search_memories(
        self, *, text_query: str | None = None, metadata_query: dict[str, Any] | None = None, limit: int = 10
    ) -> Sequence[Memory]: ...

    async def get_memory(self, mem_id: str) -> Memory | None: ...

    async def update_metadata(self, mem_id: str, patch: dict[str, Any]) -> None: ...

    async def delete_memory(self, mem_id: str) -> None: ...

    async def close(self) -> None: ...


###############################################################################
# In‑memory dummy – great for fast unit tests
###############################################################################

class InMemoryStore(AbstractMemoryStore):
    """Very simple dict‑backed store used in unit tests."""

    def __init__(self) -> None:
        self._db: dict[str, Memory] = {}
        self._lock = asyncio.Lock()

    async def add_memory(self, mem: Memory) -> None:  # noqa: D401
        async with self._lock:
            self._db[mem.id] = mem

    async def search_memories(
        self, *, text_query: str | None = None, metadata_query: dict[str, Any] | None = None, limit: int = 10
    ) -> Sequence[Memory]:
        async with self._lock:
            res = list(self._db.values())
            if text_query:
                res = [m for m in res if text_query.lower() in m.content.lower()]
            if metadata_query:
                for k, v in metadata_query.items():
                    res = [m for m in res if m.metadata.get(k) == v]
            return res[:limit]

    async def get_memory(self, mem_id: str) -> Memory | None:  # noqa: D401
        async with self._lock:
            return self._db.get(mem_id)

    async def update_metadata(self, mem_id: str, patch: dict[str, Any]) -> None:  # noqa: D401
        async with self._lock:
            if mem_id in self._db:
                self._db[mem_id].metadata.update(patch)

    async def delete_memory(self, mem_id: str) -> None:  # noqa: D401
        async with self._lock:
            self._db.pop(mem_id, None)

    async def close(self) -> None:  # noqa: D401
        # nothing to close
        pass


###############################################################################
# SQLite implementation with JSON1 + connection pool
###############################################################################

class SQLiteMemoryStore(AbstractMemoryStore):
    """Async SQLite store using JSON1 and aiosqlite connection pool."""

    DEFAULT_POOL_SIZE = 4

    def __init__(self, db_path: str | pathlib.Path, pool_size: int | None = None) -> None:
        self._db_path = str(db_path)
        self._pool_size = pool_size or self.DEFAULT_POOL_SIZE
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        self._init_lock = asyncio.Lock()
        self._initialised = False

    # ---------------------------------------------------------------------
    # Public high‑level API
    # ---------------------------------------------------------------------

    async def add_memory(self, mem: Memory) -> None:  # noqa: D401
        await self._ensure_initialised()
        conn = await self._acquire()
        try:
            await conn.execute(
                """
                INSERT INTO memories(id, content, metadata, created_at)
                VALUES (?, ?, json(?), ?)
                """,
                (mem.id, mem.content, json.dumps(mem.metadata), mem.created_at),
            )
            await conn.commit()
        finally:
            await self._release(conn)

    async def search_memories(
        self,
        *,
        text_query: str | None = None,
        metadata_query: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> Sequence[Memory]:
        await self._ensure_initialised()
        clauses: list[str] = []
        params: list[Any] = []
        if text_query:
            clauses.append("content LIKE ?")
            params.append(f"%{text_query}%")
        if metadata_query:
            for k, v in metadata_query.items():
                clauses.append("json_extract(metadata, ?) = ?")
                params.extend([f"$.{k}", v])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT id, content, metadata, created_at FROM memories{where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        conn = await self._acquire()
        try:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
            return [self._row_to_memory(r) for r in rows]
        finally:
            await self._release(conn)

    async def get_memory(self, mem_id: str) -> Memory | None:  # noqa: D401
        await self._ensure_initialised()
        conn = await self._acquire()
        try:
            cur = await conn.execute(
                "SELECT id, content, metadata, created_at FROM memories WHERE id = ?", (mem_id,)
            )
            row = await cur.fetchone()
            return self._row_to_memory(row) if row else None
        finally:
            await self._release(conn)

    async def update_metadata(self, mem_id: str, patch: dict[str, Any]) -> None:  # noqa: D401
        await self._ensure_initialised()
        conn = await self._acquire()
        try:
            # build JSON_PATCH expression
            set_expr = "metadata"
            for k, v in patch.items():
                set_expr = f"json_set({set_expr}, '$.{k}', json(?))"
            params = list(patch.values()) + [mem_id]
            await conn.execute(
                f"UPDATE memories SET metadata = {set_expr} WHERE id = ?", params
            )
            await conn.commit()
        finally:
            await self._release(conn)

    async def delete_memory(self, mem_id: str) -> None:  # noqa: D401
        await self._ensure_initialised()
        conn = await self._acquire()
        try:
            await conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            await conn.commit()
        finally:
            await self._release(conn)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def close(self) -> None:  # noqa: D401
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
        self._initialised = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _acquire(self) -> aiosqlite.Connection:
        return await self._pool.get()

    async def _release(self, conn: aiosqlite.Connection) -> None:
        await self._pool.put(conn)

    async def _ensure_initialised(self) -> None:
        if self._initialised:
            return
        async with self._init_lock:
            if self._initialised:
                return
            # Create directory if needed
            pathlib.Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            # Open pool
            for _ in range(self._pool_size):
                conn = await aiosqlite.connect(self._db_path, isolation_level=None)
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA foreign_keys=ON;")
                await self._migrate(conn)
                await self._pool.put(conn)
            self._initialised = True

    async def _migrate(self, conn: aiosqlite.Connection) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                metadata    JSON NOT NULL,
                created_at  REAL NOT NULL
            );
            """
        )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row | None) -> Memory:  # type: ignore[arg-type]
        if row is None:
            raise ValueError("Row is None")
        return Memory(
            id=row[0],
            content=row[1],
            metadata=json.loads(row[2]),
            created_at=row[3],
        )

###############################################################################
# FastAPI integration helpers
###############################################################################

async def lifespan_context(app) -> AsyncIterator[None]:  # type: ignore[func‑returns‑value]
    """Attach store to ``app.state`` for the duration of the process."""

    from ..config.settings import get_settings  # local import to avoid cycle

    settings = get_settings()
    store = SQLiteMemoryStore(settings.sqlite_path, pool_size=settings.sqlite_pool_size)
    await store._ensure_initialised()
    app.state.memory_store = store  # type: ignore[attr‑defined]
    yield
    await store.close()


def get_memory_store(app_state) -> AbstractMemoryStore:  # type: ignore[valid‑type]
    """FastAPI dependency; extracts store from ``request.app.state``."""

    return app_state.memory_store  # type: ignore[attr‑defined]

###############################################################################
# Async context manager – handy for scripts
###############################################################################

@asynccontextmanager
async def open_sqlite_memory_store(
    path: str | pathlib.Path, *, pool_size: int | None = None
) -> AsyncIterator[SQLiteMemoryStore]:
    store = SQLiteMemoryStore(path, pool_size)
    await store._ensure_initialised()
    try:
        yield store
    finally:
        await store.close()      
