"""memory_system.unified_memory
================================
High‑level, **framework‑agnostic** helper functions that wrap the lower
level storage / vector components.  Nothing in here depends on FastAPI
or any other web framework – the goal is to let notebooks, background
jobs, or other services reuse the same persistence logic without pulling
in heavy HTTP deps.

Notes
-----
* All functions are **async**.  They accept an optional ``store`` kwarg –
  any object that implements ``add_memory``, ``search_memory``,
  ``delete_memory`` and ``update_metadata``.  If not supplied the helper
  tries to obtain the application‑scoped store via
  :pyfunc:`memory_system.core.store.get_memory_store`.
* Docstrings follow **PEP 257** and type hints are 100 % complete so that
  MyPy / Ruff‑strict pass cleanly.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import uuid
from typing import Any
from collections.abc import MutableMapping, Sequence

from .core.store import (
    Memory,
    MemoryStoreProtocol,  # structural – both SQLiteMemoryStore & InMemoryStore implement.
    get_memory_store,
)

logger = logging.getLogger("memory_system.unified_memory")

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


aSYNC_TIMEOUT = 5  # seconds – safety net for accidental long‑running operations.


async def _resolve_store(
    store: MemoryStoreProtocol | None = None,
) -> MemoryStoreProtocol:
    """Return a concrete store instance.

    The rules are:
    1. If *store* is given → use it.
    2. Else try to obtain the application‑scoped store via
       :func:`memory_system.core.store.get_memory_store`.
    """

    if store is not None:
        return store

    resolved = get_memory_store()
    if resolved is None:
        raise RuntimeError("Memory store has not been initialised.")
    return resolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def add(
    text: str,
    metadata: MutableMapping[str, Any] | None = None,
    *,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Persist a *text* record with optional *metadata* and return a **Memory**.

    Parameters
    ----------
    text:
        Raw textual content of the memory.
    metadata:
        Arbitrary key/‑value mapping (JSON‑serialisable).  Reserved keys
        such as ``created_at`` or ``memory_id`` will be overwritten.
    store:
        Optional explicit store object implementing the protocol.  If
        *None* the process‑wide default is used.
    """

    memory = Memory(
        memory_id=str(uuid.uuid4()),
        text=text,
        metadata=metadata or {},
        created_at=_dt.datetime.utcnow(),
    )

    st = await _resolve_store(store)
    await asyncio.wait_for(st.add_memory(memory), timeout=aSYNC_TIMEOUT)
    logger.debug("Memory %s added (%d chars).", memory.memory_id, len(text))
    return memory


async def search(
    query: str,
    k: int = 5,
    *,
    metadata_filter: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Sequence[Memory]:
    """Semantic search across stored memories.

    Parameters
    ----------
    query:
        Search phrase.
    k:
        Maximum number of results.
    metadata_filter:
        Optional mapping – only memories whose metadata contains all
        specified keys/values will be considered.
    store:
        Explicit store object or *None* for the default.
    """
    st = await _resolve_store(store)
    results = await asyncio.wait_for(
        st.search_memory(query=query, k=k, metadata_filter=metadata_filter),
        timeout=aSYNC_TIMEOUT,
    )
    logger.debug("Search for '%s' returned %d result(s).", query, len(results))
    return results


async def delete(
    memory_id: str,
    *,
    store: MemoryStoreProtocol | None = None,
) -> None:
    """Delete a memory by ``memory_id`` if it exists."""
    st = await _resolve_store(store)
    await asyncio.wait_for(st.delete_memory(memory_id), timeout=aSYNC_TIMEOUT)
    logger.debug("Memory %s deleted.", memory_id)


async def update(
    memory_id: str,
    *,
    text: str | None = None,
    metadata: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Update text and/or metadata of an existing memory and return the new object."""
    st = await _resolve_store(store)
    updated = await asyncio.wait_for(
        st.update_memory(memory_id, text=text, metadata=metadata), timeout=aSYNC_TIMEOUT
    )
    logger.debug("Memory %s updated.", memory_id)
    return updated


async def list_recent(
    n: int = 20,
    *,
    store: MemoryStoreProtocol | None = None,
) -> Sequence[Memory]:
    """Return *n* most recently added memories in descending chronological order."""
    st = await _resolve_store(store)
    recent = await asyncio.wait_for(st.list_recent(n=n), timeout=aSYNC_TIMEOUT)
    logger.debug("Fetched %d recent memories.", len(recent))
    return recent
