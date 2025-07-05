""" vector_store.py — Async durable vector store for Unified Memory System

This rewrite introduces:

AbstractVectorStore    – interface for mocks/fakes during unit‑testing

asyncio.Lock           – coroutine‑safe lock instead of threading.RLock

Background maintenance     – periodic blob compaction & replication

Fully async public API     – suitable for FastAPI / anyio apps


The binary blob (*.bin) keeps raw float32 vectors, while an SQLite side‑table (*.db) stores offset/length/hash metadata.  All heavy blocking I/O is off‑loaded to the default thread‑pool so the event‑loop stays responsive. """

from future import annotations

import asyncio import hashlib import os import shutil import sqlite3 from abc import ABC, abstractmethod from pathlib import Path from typing import Dict, Optional

import numpy as np from numpy.typing import NDArray

---------------------------------------------------------------------------

Constants & helpers

---------------------------------------------------------------------------

_PAGE_SIZE: int = os.sysconf("SC_PAGESIZE") if hasattr(os, "sysconf") else 4096 _FLOAT_SIZE: int = 4  # bytes per float32

class StorageError(Exception): """Raised on any persistence failure (I/O, corruption)."""

class ValidationError(Exception): """Raised when user input is invalid (wrong dtype/dim/etc.)."""

class _Meta: """In‑memory metadata for a single vector."""

__slots__ = ("offset", "length", "sha256")

def __init__(self, offset: int, length: int, sha256: str) -> None:  # noqa: D401
    self.offset = offset  # byte offset in blob file
    self.length = length  # number of float32 elements
    self.sha256 = sha256  # integrity hash

# Convenience -------------------------------------------------------

@property
def nbytes(self) -> int:
    """Total byte‑length of the vector payload in the blob file."""

    return self.length * _FLOAT_SIZE

---------------------------------------------------------------------------

Public interface (for DI & mocking)

---------------------------------------------------------------------------

class AbstractVectorStore(ABC): """Minimal async contract for vector persistence.

Sub‑classes (real or fake) must implement *async* CRUD operations so they
can be injected into FastAPI routes or tested in isolation.
"""

# ────────────────────── CRUD ──────────────────────

@abstractmethod
async def add_vector(self, vector_id: str, vec: NDArray[np.float32]) -> None:  # noqa: D401,E501
    """Persist a new vector under *vector_id* (must be unique)."""

@abstractmethod
async def get_vector(self, vector_id: str) -> NDArray[np.float32]:
    """Load the vector payload into RAM.  Raises *StorageError* if missing."""

@abstractmethod
async def remove_vector(self, vector_id: str) -> None:
    """Delete vector metadata (space reclaimed by the compactor)."""

@abstractmethod
async def list_ids(self) -> list[str]:
    """Return *all* vector ids currently stored."""

# ─────────────────── lifecycle ───────────────────

@abstractmethod
async def flush(self) -> None:
    """Ensure on‑disk durability (fsync + commit)."""

@abstractmethod
async def close(self) -> None:
    """Flush and shutdown background tasks (idempotent)."""

---------------------------------------------------------------------------

Concrete implementation

---------------------------------------------------------------------------

class VectorStore(AbstractVectorStore):  # noqa: D101 – top‑level docstring above """Durable on‑disk vector store (async).

Parameters
----------
path
    Base path *without* extension (``.bin`` / ``.db`` will be appended).
dim
    Expected dimensionality. ``0`` → set on first insert.
maintenance_interval
    Period for compaction & backup jobs (seconds, min=60).
backup_dir
    Optional directory where ``*.bin`` and ``*.db`` are replicated.
"""

def __init__(
    self,
    path: str | Path,
    *,
    dim: int = 0,
    maintenance_interval: int = 600,
    backup_dir: str | Path | None = None,
) -> None:
    # ───── runtime config ─────
    self._base_path = Path(path)
    self._bin_path = self._base_path.with_suffix(".bin")
    self._db_path = self._base_path.with_suffix(".db")
    self._dim = dim

    # ───── process state ─────
    self._file: Optional[os.FileIO] = None
    self._conn: Optional[sqlite3.Connection] = None
    self._lock = asyncio.Lock()
    self._cache: Dict[str, _Meta] = {}

    self._maintenance_interval = max(maintenance_interval, 60)
    self._backup_dir = Path(backup_dir) if backup_dir else None
    self._compaction_task: Optional[asyncio.Task] = None
    self._replication_task: Optional[asyncio.Task] = None
    self._closed = False

    # ───── bootstrap ─────
    self._open_files()
    self._init_db()
    self._load_cache()
    self._start_background_tasks()

# ------------------------------------------------------------------
# Async public API
# ------------------------------------------------------------------

async def add_vector(self, vector_id: str, vec: NDArray[np.float32]) -> None:  # noqa: D401,E501
    """Persist a *new* vector.

    Heavy I/O (write) is dispatched to a thread‑pool to keep the event‑loop
    snappy even on slow disks.
    """

    if not isinstance(vec, np.ndarray) or vec.dtype != np.float32:
        raise ValidationError("vector must be numpy.ndarray[float32]")
    if vec.ndim != 1:
        raise ValidationError("vector must be 1‑D")

    async with self._lock:
        if vector_id in self._cache:
            raise ValidationError("duplicate vector id")
        if self._dim and vec.size != self._dim:
            raise ValidationError(f"expected dim {self._dim}, got {vec.size}")
        if self._dim == 0:
            self._dim = vec.size

        sha256 = hashlib.sha256(vec.tobytes()).hexdigest()
        offset = await asyncio.get_running_loop().run_in_executor(  # heavy I/O
            None, self._append_blob, vec
        )

        meta = _Meta(offset=offset, length=vec.size, sha256=sha256)
        assert self._conn is not None
        try:
            self._conn.execute(
                "INSERT INTO vectors (id, offset, length, sha256) VALUES (?,?,?,?)",
                (vector_id, meta.offset, meta.length, meta.sha256),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:  # pragma: no cover
            raise ValidationError("duplicate vector id") from exc
        self._cache[vector_id] = meta

# ------------------------------------------------------------------

async def get_vector(self, vector_id: str) -> NDArray[np.float32]:
    meta = self._cache.get(vector_id)
    if meta is None:
        raise StorageError("vector not found")

    async with self._lock:
        data = await asyncio.get_running_loop().run_in_executor(  # heavy I/O
            None, self._read_blob, meta
        )
    return np.frombuffer(data, dtype=np.float32).copy()  # detach from mmap

# ------------------------------------------------------------------

async def remove_vector(self, vector_id: str) -> None:
    meta = self._cache.pop(vector_id, None)
    if meta is None:
        raise StorageError("vector not found")

    async with self._lock:
        assert self._conn is not None
        self._conn.execute("DELETE FROM vectors WHERE id=?", (vector_id,))
        self._conn.commit()
        # blob space reclaimed later by compactor

# ------------------------------------------------------------------

async def list_ids(self) -> list[str]:
    return list(self._cache.keys())

# ------------------------------------------------------------------

async def flush(self) -> None:
    async with self._lock:
        await asyncio.get_running_loop().run_in_executor(None, self._flush_sync)

# ------------------------------------------------------------------

async def close(self) -> None:
    if self._closed:
        return
    self._closed = True

    # cancel & await maintenance tasks
    for task in (self._compaction_task, self._replication_task):
        if task:
            task.cancel()
    await asyncio.gather(
        *(t for t in (self._compaction_task, self._replication_task) if t),
        return_exceptions=True,
    )

    async with self._lock:
        await asyncio.get_running_loop().run_in_executor(None, self._close_sync)

# ------------------------------------------------------------------
# Internal – synchronous helpers (executed in thread‑pool)
# ------------------------------------------------------------------

def _open_files(self) -> None:
    self._bin_path.parent.mkdir(parents=True, exist_ok=True)
    self._file = open(self._bin_path, "a+b", buffering=0)
    self._conn = sqlite3.connect(self._db_path)
    self._conn.execute("PRAGMA journal_mode=WAL")
    self._conn.execute("PRAGMA synchronous=NORMAL")

def _init_db(self) -> None:
    assert self._conn is not None
    self._conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vectors (
            id TEXT PRIMARY KEY,
            offset INTEGER NOT NULL,
            length INTEGER NOT NULL,
            sha256 TEXT NOT NULL
        )
        """
    )
    self._conn.commit()

def _load_cache(self) -> None:
    assert self._conn is not None
    cur = self._conn.execute("SELECT id, offset, length, sha256 FROM vectors")
    self._cache = {
        row[0]: _Meta(offset=row[1], length=row[2], sha256=row[3]) for row in cur
    }

# ---- low‑level I/O -------------------------------------------------

def _append_blob(self, vec: NDArray[np.float32]) -> int:
    """Append raw bytes to blob file and return starting offset."""

    assert self._file is not None
    data = vec.tobytes()
    offset = self._file.seek(0, os.SEEK_END)
    self._file.write(data)
    pad = (-len(data)) % _PAGE_SIZE  # keep page‑aligned for mmap perf
    if pad:
        self._file.write(b"\0" * pad)
    self._file.flush()
    os.fsync(self._file.fileno())
    return offset

def _read_blob(self, meta: _Meta) -> bytes:
    assert self._file is not None
    self._file.seek(meta.offset)
    data = self._file.read(meta.nbytes)
    if hashlib.sha256(data).hexdigest() != meta.sha256:
        raise StorageError("checksum mismatch")
    return data

def _flush_sync(self) -> None:
    if self._file:
        self._file.flush()
        os.fsync(self._file.fileno())
    if self._conn:
        self._conn.commit()

def _close_sync(self) -> None:
    self._flush_sync()
    if self._conn:
        self._conn.close()
        self._conn = None
    if self._file:
        self._file.close()
        self._file = None

# ------------------------------------------------------------------
# Background maintenance workers
# ------------------------------------------------------------------

def _start_background_tasks(self) -> None:
    """Spawn compaction & replication coroutines."""

    loop = asyncio.get_running_loop()
    self._compaction_task = loop.create_task(self._periodic_compaction())
    if self._backup_dir:
        self._replication_task = loop.create_task(self._periodic_backup())

# ────────────────────────────────────────────────────────────────────

async def _periodic_compaction(self) -> None:
    try:
        while True:
            await asyncio.sleep(self._maintenance_interval)
            await self._compact_blob()
    except asyncio.CancelledError:
        return

async def _periodic_backup(self) -> None:
    try:
        while True:
            await asyncio.sleep(self._maintenance_interval * 3)
            await self._rep
