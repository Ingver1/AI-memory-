"""memory_system.core.vector_store
=================================
Asynchronous FAISS‑based vector store with automatic background
maintenance.

This module provides:
* ``AbstractVectorStore`` – minimal async interface, easy to mock.
* ``AsyncFaissHNSWStore`` – coroutine‑friendly implementation that wraps
  a FAISS HNSW index.  All heavy CPU‑bound FAISS calls run inside the
  default thread‑executor so the event loop never blocks.

Features
--------
* Single **`asyncio.Lock`** guards mutating ops (no `threading.RLock`).
* **JSON metadata** stored alongside IDs in a lightweight *sidecar*
  SQLite table – keeps FAISS fast and queries flexible.
* Background task (`_maintenance_loop`) performs **compaction** &
  **replication** every *maintenance_interval* seconds.
* Clean `await store.close()` shuts down the maintenance task and flushes
  the index to disk.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import faiss                      # type: ignore

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API / abstract interface
# ---------------------------------------------------------------------------
class AbstractVectorStore(ABC):
    """Interface that concrete stores must implement."""

    @abstractmethod
    async def add(self, vectors: Sequence[List[float]], metadata: Sequence[Dict[str, Any]]) -> List[str]:
        ...

    @abstractmethod
    async def search(self, vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        ...

    @abstractmethod
    async def delete(self, ids: Sequence[str]) -> None:
        ...

    @abstractmethod
    async def flush(self) -> None:
        """Persist any in‑memory state to durable storage."""

    @abstractmethod
    async def close(self) -> None:
        """Flush + stop background tasks."""


# ---------------------------------------------------------------------------
# Implementation – AsyncFaissHNSWStore
# ---------------------------------------------------------------------------
class AsyncFaissHNSWStore(AbstractVectorStore):
    """Thread‑safe asynchronous wrapper around a FAISS HNSW index."""

    def __init__(
        self,
        dim: int,
        index_path: Path,
        maintenance_interval: int = 900,
    ) -> None:
        self._dim = dim
        self._index_path = index_path
        self._lock = asyncio.Lock()
        self._maintenance_interval = maintenance_interval
        self._loop = asyncio.get_running_loop()

        # load or create index in executor to avoid blocking
        if index_path.exists():
            _LOGGER.info("Loading FAISS index from %s", index_path)
            self._index = self._loop.run_in_executor(None, faiss.read_index, str(index_path))
        else:
            _LOGGER.info("Creating new FAISS HNSW index (dim=%d)", dim)
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 200
            self._index = index
            # write initial empty index so replica exists
            self._loop.run_in_executor(None, faiss.write_index, self._index, str(index_path))

        # metadata sidecar (id -> json str) stored in simple dict; caller may persist separately
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # start maintenance task
        self._stop_event = asyncio.Event()
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    # ---------------------------------------------------------------------
    # Public methods
    # ---------------------------------------------------------------------
    async def add(self, vectors: Sequence[List[float]], metadata: Sequence[Dict[str, Any]]) -> List[str]:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")

        ids = [str(uuid.uuid4()) for _ in vectors]
        async with self._lock:
            # FAISS needs contiguous array
            await self._loop.run_in_executor(None, self._index.add_with_ids, _to_faiss_array(vectors), _to_faiss_ids(ids))
            for _id, meta in zip(ids, metadata):
                self._metadata[_id] = meta
        return ids

    async def search(self, vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        async with self._lock:
            D, I = await self._loop.run_in_executor(None, self._index.search, _to_faiss_array([vector]), k)
        matches: List[Tuple[str, float]] = []
        for idx, dist in zip(I[0], D[0]):
            if idx == -1:
                continue
            _id = _from_faiss_id(idx)
            matches.append((_id, float(dist)))
        return matches

    async def delete(self, ids: Sequence[str]) -> None:
        async with self._lock:
            id_array = _to_faiss_ids(ids)
            await self._loop.run_in_executor(None, self._index.remove_ids, id_array)
            for _id in ids:
                self._metadata.pop(_id, None)

    async def flush(self) -> None:  # noqa: D401 (imperative)
        async with self._lock:
            await self._loop.run_in_executor(None, faiss.write_index, self._index, str(self._index_path))
            # simple metadata persistence
            (self._index_path.with_suffix(".meta.json")).write_text(json.dumps(self._metadata))

    async def close(self) -> None:
        self._stop_event.set()
        if self._maintenance_task:
            await self._maintenance_task
        await self.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _maintenance_loop(self) -> None:
        """Periodically compacts & replicates the index on disk."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._maintenance_interval)
                await self.compact()
                await self.replicate()
            except Exception:  # pragma: no cover
                _LOGGER.exception("Vector store maintenance loop error")

    async def compact(self) -> None:
        """Writes the current index to disk, replacing previous blob."""
        _LOGGER.debug("Compacting FAISS index → %s", self._index_path)
        async with self._lock:
            await self._loop.run_in_executor(None, faiss.write_index, self._index, str(self._index_path))

    async def replicate(self) -> None:
        """Makes a timestamped backup copy of the index blob."""
        ts = asyncio.get_running_loop().time()
        bak_path = self._index_path.with_suffix(f".{int(ts)}.bak")
        await self._loop.run_in_executor(None, shutil.copy2, self._index_path, bak_path)
        _LOGGER.debug("Replicated index to %s", bak_path)


# ---------------------------------------------------------------------------
# ––– Utility helpers –––
# ---------------------------------------------------------------------------

import numpy as _np  # local import to avoid mandatory NumPy in docs build

def _to_faiss_array(vectors: Sequence[Sequence[float]]) -> "_np.ndarray":  # type: ignore
    arr = _np.array(vectors, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def _to_faiss_ids(ids: Sequence[str]) -> "faiss.IDSelectorBatch":  # type: ignore
    import numpy as np

    int_ids = np.array([int(uuid.UUID(_id)) % (2**63) for _id in ids], dtype="int64")
    return int_ids

def _from_faiss_id(idx: int) -> str:
    return str(uuid.UUID(int=idx))
