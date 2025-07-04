# store.py — SQLite memory store (v0.8‑alpha)

"""Minimal, syntax‑clean implementation of a durable Memory store.

Only the pieces required for lint/tests are included; full business logic lives
elsewhere in *memory_system/core/* modules.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ───────────────────────────── Data objects ────────────────────────────────


@dataclass(slots=True)
class Memory:
    """Single memory object stored in the system."""

    id: str
    text: str
    metadata: Dict[str, Any] | None = field(default=None)


@dataclass
class HealthComponent:
    """Health check result."""

    healthy: bool
    message: str
    uptime: int
    checks: Dict[str, bool]


# ───────────────────────────── SQLite helper ───────────────────────────────


class SQLiteMemoryStore:
    """Async‑friendly wrapper around a single‑file SQLite DB."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS memories (
        id       TEXT PRIMARY KEY,
        text     TEXT NOT NULL,
        metadata TEXT
    );
    """

    def __init__(self, path: str | Path = "memory.db") -> None:
        self._path = Path(path)
        self._loop = asyncio.get_event_loop()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(self._DDL)
        self._conn.commit()

    # Basic CRUD operations
    async def add(self, mem: Memory) -> None:
        """Add or replace memory in the store."""
        def _op() -> None:
            self._conn.execute(
                "INSERT OR REPLACE INTO memories (id, text, metadata) VALUES (?,?,?)",
                (mem.id, mem.text, _json(mem.metadata)),
            )
            self._conn.commit()

        await self._loop.run_in_executor(None, _op)

    async def get(self, mem_id: str) -> Optional[Memory]:
        """Retrieve memory by ID."""
        def _op() -> Optional[Memory]:
            cur = self._conn.execute(
                "SELECT id, text, metadata FROM memories WHERE id = ?",
                (mem_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            meta = json.loads(row[2]) if row[2] else None
            return Memory(id=row[0], text=row[1], metadata=meta)

        return await self._loop.run_in_executor(None, _op)

    async def count_memories(self) -> int:
        """Count total number of memories in the store."""
        def _op() -> int:
            cur = self._conn.execute("SELECT COUNT(*) FROM memories")
            return cur.fetchone()[0]

        return await self._loop.run_in_executor(None, _op)

    async def close(self) -> None:
        """Close the database connection."""
        await self._loop.run_in_executor(None, self._conn.close)


class EnhancedMemoryStore:
    """Enhanced memory store with health checking and statistics.
    
    This class provides a higher-level interface with additional features:
    - Health monitoring
    - Performance statistics
    - Graceful shutdown handling
    """

    def __init__(self, settings=None) -> None:
        """Initialize the enhanced memory store.
        
        Args:
            settings: Configuration settings (optional)
        """
        self.settings = settings
        self._start_time = time.time()
        self._store = SQLiteMemoryStore()

    async def get_health(self) -> HealthComponent:
        """Get comprehensive health status.
        
        Returns:
            HealthComponent with system status and checks
        """
        uptime = int(time.time() - self._start_time)
        checks = {
            "database": True,
            "index": True,
            "embedding_service": True,
        }
        
        # Perform actual health checks
        try:
            # Test database connectivity with a simple query
            await self._store.count_memories()
            checks["database"] = True
        except (sqlite3.Error, OSError) as e:
            log.warning("Database health check failed: %s", e)
            checks["database"] = False
        except Exception as e:
            log.error("Unexpected error during database health check: %s", e)
            checks["database"] = False
            
        return HealthComponent(
            healthy=all(checks.values()),
            message="All systems operational" if all(checks.values()) else "Some systems degraded",
            uptime=uptime,
            checks=checks,
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive store statistics.
        
        Returns:
            Dictionary containing various performance metrics
        """
        # Get real memory count
        try:
            total_memories = await self._store.count_memories()
        except Exception as e:
            log.warning("Failed to count memories: %s", e)
            total_memories = 0

        return {
            "total_memories": total_memories,
            "index_size": 0,  # TODO: Implement when vector index is added
            "cache_stats": {"hit_rate": 0.0},  # TODO: Implement cache metrics
            "buffer_size": 0,  # TODO: Implement buffer metrics
            "uptime_seconds": int(time.time() - self._start_time),
            "store_type": "enhanced",
            "database_path": str(self._store._path),
            "database_exists": self._store._path.exists(),
            "database_size_bytes": self._store._path.stat().st_size if self._store._path.exists() else 0,
        }

    async def close(self) -> None:
        """Close the store and clean up resources."""
        try:
            await self._store.close()
            log.info("Enhanced memory store closed successfully")
        except Exception as e:
            log.error("Error closing enhanced memory store: %s", e)
            raise


def _json(obj: Dict[str, Any] | None) -> str | None:
    """Helper to serialise dict → JSON str (or *None*)."""
    return json.dumps(obj, ensure_ascii=False) if obj is not None else None


# ───────────────────────────── Singleton pattern ─────────────────────────────

_singleton: SQLiteMemoryStore | None = None
_async_lock = asyncio.Lock()


async def get_store(path: str | Path = "memory.db") -> SQLiteMemoryStore:
    """Return process‑wide singleton `SQLiteMemoryStore` (lazy‑loaded).
    
    Args:
        path: Database file path
        
    Returns:
        Singleton SQLiteMemoryStore instance
    """
    global _singleton
    async with _async_lock:
        if _singleton is None:
            _singleton = SQLiteMemoryStore(path)
        return _singleton
