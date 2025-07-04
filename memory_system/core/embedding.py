# embedding.py — text‑embedding backend for Unified Memory System
#
# Version: 0.8‑alpha
"""High‑level embedding service with:
    • **Lazy model loading** & fallback to a lightweight default.
    • **Smart cache**       – TTL + LRU keeping memory footprint predictable.
    • **Batch processor**    – background thread that multiplexes single‑text
      requests into vectorised *Sentence‑Transformers* calls.
    • **Graceful shutdown**  – makes sure no futures dangle after exit.

This file replaces early revisions that suffered from race conditions,
redundant locks and raw byte decoding.  It now aligns with the rest of the
v0.8‑alpha codebase (async‑friendly, English‑only comments).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from memory_system.config.settings import UnifiedSettings
from memory_system.utils.cache import SmartCache
from memory_system.utils.metrics import LAT_EMBEDDING, MET_ERRORS_TOTAL

__all__ = [
    "EmbeddingError",
    "EmbeddingJob",
    "EnhancedEmbeddingService",
]

log = logging.getLogger(__name__)

###############################################################################
# Exceptions & DTOs                                                           #
###############################################################################


class EmbeddingError(RuntimeError):
    """Raised when the embedding pipeline fails irrecoverably."""


@dataclass(slots=True, frozen=True)
class EmbeddingJob:
    """Internal container binding text to its awaiting *Future*."""

    text: str
    future: asyncio.Future[np.ndarray]


###############################################################################
# Service implementation                                                      #
###############################################################################


class EnhancedEmbeddingService:
    """Thread‑safe, cache‑aware embedding service.

    The public surface is *small* – only :py:meth:`encode` plus a few utility
    helpers.  All heavy lifting happens in a single background thread so that
    the asyncio event‑loop stays responsive.
    """

    # ---------------------------------------------------------------------
    def __init__(self, model_name: str, settings: Optional[UnifiedSettings] = None) -> None:
        self.model_name = model_name
        self.settings = settings or UnifiedSettings.for_development()
        self._model_lock = threading.RLock()
        self._model: Optional[SentenceTransformer] = None

        # Cache setup ---------------------------------------------------
        self.cache = SmartCache(
            max_size=self.settings.performance.cache_size,
            ttl=self.settings.performance.cache_ttl_seconds,
        )

        # Batching state -------------------------------------------------
        self._queue: List[EmbeddingJob] = []
        self._queue_lock = threading.RLock()
        self._queue_condition = threading.Condition(self._queue_lock)
        self._shutdown = threading.Event()

        # Runtime flags --------------------------------------------------
        self._batch_thread: Optional[threading.Thread] = None

        self._load_model()  # eager by default – keeps first request fast
        self._start_processor()  # background batching
        log.info("Embedding service ready (model=%s)", self.model_name)

    # ------------------------------------------------------------------
    # Model management                                                  #
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Lazy‑load *Sentence‑Transformers* model with fallback."""
        with self._model_lock:
            if self._model is not None:
                return

            try:
                log.info("Loading embedding model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                log.info(
                    "Model loaded: %s (dim=%d)",
                    self.model_name,
                    self._model.get_sentence_embedding_dimension(),
                )
            except Exception as exc:  # noqa: BLE001 – external lib may raise anything
                log.warning("Loading failed: %s", exc)
                fallback = "all-MiniLM-L6-v2"
                if self.model_name != fallback:
                    try:
                        log.info("Attempting fallback model: %s", fallback)
                        self._model = SentenceTransformer(fallback)
                        self.model_name = fallback
                        log.info("Fallback model loaded: %s", fallback)
                    except Exception as fexc:
                        log.error("Fallback model failed: %s", fexc)
                        raise EmbeddingError("Could not load any embedding model") from fexc
                else:
                    raise EmbeddingError("Could not load embedding model") from exc

    # ------------------------------------------------------------------
    # Batch background thread                                           #
    # ------------------------------------------------------------------
    def _start_processor(self) -> None:
        """Start the background batching thread (idempotent)."""
        if self._batch_thread and self._batch_thread.is_alive():
            return

        self._batch_thread = threading.Thread(
            target=self._batch_loop, name="emb-batcher", daemon=True
        )
        self._batch_thread.start()

    def _batch_loop(self) -> None:  # runs in *dedicated* thread
        batch_size = self.settings.model.batch_add_size
        log.debug("Batch processor entered loop (size=%d)", batch_size)

        while not self._shutdown.is_set():
            with self._queue_condition:
                if not self._queue:
                    self._queue_condition.wait(timeout=0.05)
                    continue
                # Slice batch
                batch, self._queue = self._queue[:batch_size], self._queue[batch_size:]

            # Process outside of lock --------------------------------
            try:
                texts = [job.text for job in batch]
                vectors = self._encode_direct(texts)
                for job, vec in zip(batch, vectors):
                    if not job.future.done():
                        job.future.set_result(vec)
            except Exception as exc:  # noqa: BLE001
                MET_ERRORS_TOTAL.labels(type="embedding", component="batch_loop").inc()
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(EmbeddingError(str(exc)))

        log.debug("Batch processor loop exited")

    # ------------------------------------------------------------------
    # Public API                                                       #
    # ------------------------------------------------------------------
    async def encode(
        self, text: Union[str, Sequence[str]]
    ) -> np.ndarray:  # noqa: D401 – simple verb is fine
        """Return a vector embedding for *text* (str or list)."""
        if isinstance(text, str):
            return await self._encode_single(text)
        return await self._encode_multi(list(text))

    # Single -----------------------------------------------------------
    async def _encode_single(self, text: str) -> np.ndarray:
        # Cache first
        key = self._cache_key(text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached.reshape(1, -1)

        # Batch enqueue -------------------------------------------
        loop = asyncio.get_event_loop()
        future: asyncio.Future[np.ndarray] = loop.create_future()
        job = EmbeddingJob(text=text, future=future)

        with self._queue_condition:
            self._queue.append(job)
            self._queue_condition.notify()

        try:
            vec = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            future.cancel()
            raise EmbeddingError("Embedding timed out")

        self.cache.put(key, vec)
        return vec.reshape(1, -1)

    # Multi ------------------------------------------------------------
    async def _encode_multi(self, texts: List[str]) -> np.ndarray:
        # Try all‑hit cache shortcut --------------------------------
        keys = [self._cache_key(t) for t in texts]
        hits = [self.cache.get(k) for k in keys]
        if all(v is not None for v in hits):
            return np.stack(hits)  # type: ignore[arg-type]

        # Fallback to direct encoding ------------------------------
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, self._encode_direct, texts)

        for k, v in zip(keys, vectors):
            self.cache.put(k, v)
        return vectors

    # ------------------------------------------------------------------
    # Internal helpers                                                 #
    # ------------------------------------------------------------------
    def _encode_direct(self, texts: List[str]) -> np.ndarray:
        start = time.perf_counter()
        vecs = self._safe_model().encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        LAT_EMBEDDING.observe(time.perf_counter() - start)
        return vecs

    def _safe_model(self) -> SentenceTransformer:
        """Return the loaded model (lazy‑loading under lock)."""
        if self._model is None:
            self._load_model()
        # _model is definitely set now
        return self._model  # type: ignore[return-value]

    @staticmethod
    def _cache_key(text: str | Sequence[str]) -> str:  # noqa: D401 – no imperative mood needed
        joined = text if isinstance(text, str) else "|".join(text)
        return hashlib.md5(joined.encode(), usedforsecurity=False).hexdigest()

    # ------------------------------------------------------------------
    # Diagnostics & shutdown                                           #
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._queue_lock:
            queue_len = len(self._queue)
        return {
            "model": self.model_name,
            "dimension": self._safe_model().get_sentence_embedding_dimension(),
            "cache": self.cache.get_stats(),
            "queue_size": queue_len,
            "shutdown": self._shutdown.is_set(),
        }

    def shutdown(self) -> None:
        """Signal graceful shutdown and flush remaining jobs."""
        if self._shutdown.is_set():
            return

        self._shutdown.set()
        with self._queue_condition:
            self._queue_condition.notify_all()

        if self._batch_thread and self._batch_thread.is_alive():
            self._batch_thread.join(timeout=5.0)

        # Cancel leftovers --------------------------------------
        with self._queue_condition:
            for job in self._queue:
                if not job.future.done():
                    job.future.cancel()
            self._queue.clear()

        self.cache.clear()
        log.info("Embedding service shut down")

    # Context‑manager sugar --------------------------------------------
    def __enter__(self):  # noqa: D401 – single‑word imperative is fine
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401 – docstring not needed
        self.shutdown()
