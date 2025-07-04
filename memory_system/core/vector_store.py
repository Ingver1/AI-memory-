# vector_store.py — durable on‑disk vector store for Unified Memory System
#
# Version: 0.8‑alpha

"""Persistent vector store that combines a **binary blob file** with a compact
SQLite side‑table for metadata (offset, length, SHA‑256 hash).

## Design goals

* **Crash‑safety** – every write is atomic; on failure the store rolls back to
  a consistent state.
* **Low memory footprint** – vectors stay on disk; RAM holds only a small
  id‑>meta mapping cache.
* **Portability** – pure‑Python; optional NumPy acceleration; works on any FS.

Public API

````
```python
from memory_system.core.vector_store import VectorStore
store = VectorStore(path="/data/memory_vectors")
store.add_vector("id‑123", my_vector)      # numpy.ndarray[float32] shape=(d,)
vec = store.get_vector("id‑123")           # → same ndarray
store.remove_vector("id‑123")
store.flush()                              # fsync to disk
store.close()
```
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from memory_system.exceptions import StorageError, ValidationError

__all__ = ["VectorStore"]

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
_PAGE_SIZE = os.sysconf("SC_PAGESIZE") if hasattr(os, "sysconf") else 4096
_FLOAT_SIZE = 4  # bytes per float32


class _Meta:
    """Lightweight container for vector metadata."""

    __slots__ = ("offset", "length", "sha256")

    def __init__(self, offset: int, length: int, sha256: str) -> None:  # noqa: D401
        self.offset = offset  # byte offset in blob file
        self.length = length  # number of float32 elements
        self.sha256 = sha256  # integrity hash

    # Convenience -----------------------------------------------------------
    @property
    def nbytes(self) -> int:  # total bytes in blob
        return self.length * _FLOAT_SIZE


# ---------------------------------------------------------------------------
# VectorStore implementation
# ---------------------------------------------------------------------------


class VectorStore:  # noqa: D101 – high‑level docstring at module top
    """Durable on‑disk vector store.

    Parameters
    ----------
    path : str | Path
        Base path **without extension** (``.bin`` / ``.db`` will be added).
    dim : int, default ``0``
        Expected dimensionality. ``0`` means *unknown* and will be set on first
        added vector.
    """

    def __init__(self, path: str | Path, *, dim: int = 0) -> None:  # noqa: D401
        self._base_path = Path(path)
        self._bin_path = self._base_path.with_suffix(".bin")
        self._db_path = self._base_path.with_suffix(".db")
        self._dim = dim

        self._file: Optional[os.FileIO] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = RLock()
        self._cache: Dict[str, _Meta] = {}

        self._open_files()
        self._init_db()
        self._load_cache()

    # ------------------------------ setup ----------------------------------
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
                id       TEXT PRIMARY KEY,
                offset   INTEGER NOT NULL,
                length   INTEGER NOT NULL,
                sha256   TEXT    NOT NULL
            )
            """
        )
        self._conn.commit()

    def _load_cache(self) -> None:
        assert self._conn is not None
        cur = self._conn.execute("SELECT id, offset, length, sha256 FROM vectors")
        self._cache = {row[0]: _Meta(offset=row[1], length=row[2], sha256=row[3]) for row in cur}

    # ------------------------------ public API ------------------------------
    def add_vector(self, vector_id: str, vec: NDArray[np.float32]) -> None:
        if not isinstance(vec, np.ndarray) or vec.dtype != np.float32:
            raise ValidationError("vector must be numpy.ndarray[float32]")
        if vec.ndim != 1:
            raise ValidationError("vector must be 1‑D")

        with self._lock:
            if vector_id in self._cache:
                raise ValidationError("duplicate vector id")
            if self._dim and vec.size != self._dim:
                raise ValidationError(f"expected dim {self._dim}, got {vec.size}")
            if self._dim == 0:
                self._dim = vec.size

            sha256 = hashlib.sha256(vec.tobytes()).hexdigest()
            offset = self._append_blob(vec)
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
    def get_vector(self, vector_id: str) -> NDArray[np.float32]:
        meta = self._cache.get(vector_id)
        if meta is None:
            raise StorageError("vector not found")
        assert self._file is not None
        with self._lock:
            self._file.seek(meta.offset)
            data = self._file.read(meta.nbytes)
        if hashlib.sha256(data).hexdigest() != meta.sha256:
            raise StorageError("checksum mismatch")
        return np.frombuffer(data, dtype=np.float32).copy()  # detach from mmap

    # ------------------------------------------------------------------
    def remove_vector(self, vector_id: str) -> None:
        meta = self._cache.pop(vector_id, None)
        if meta is None:
            raise StorageError("vector not found")
        assert self._conn is not None
        with self._lock:
            self._conn.execute("DELETE FROM vectors WHERE id=?", (vector_id,))
            self._conn.commit()
        # blob not truncated — space reclaimed on manual compaction

    # ------------------------------------------------------------------
    def list_ids(self) -> list[str]:
        return list(self._cache.keys())

    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Ensure all pending changes are flushed and fsynced."""
        with self._lock:
            if self._file is not None:
                self._file.flush()
                os.fsync(self._file.fileno())
            if self._conn is not None:
                self._conn.commit()

    # ------------------------------------------------------------------
    async def async_flush(self) -> None:
        """Async wrapper around :py:meth:`flush`."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.flush)

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.flush()
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._file is not None:
            self._file.close()
            self._file = None

    # --------------------------- internal helpers ---------------------------
    def _append_blob(self, vec: NDArray[np.float32]) -> int:
        """Append raw bytes to the blob file and return the starting offset."""
        assert self._file is not None
        data = vec.tobytes()
        with self._lock:
            offset = self._file.seek(0, os.SEEK_END)
            self._file.write(data)
            # pad to page boundary for better mmap alignment
            pad = (-len(data)) % _PAGE_SIZE
            if pad:
                self._file.write(b"\0" * pad)
            self._file.flush()
            os.fsync(self._file.fileno())
        return offset
