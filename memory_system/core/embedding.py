# embedding.py — text-embedding backend for Unified Memory System
#
# Version: 0.8-alpha
"""High-level embedding service with:
- **Lazy model loading** & fallback to a lightweight default.
- **Smart cache** – TTL + LRU keeping memory footprint predictable.
- **Batch processor** – background thread that multiplexes single-text requests into vectorized *Sentence-Transformers* calls.
- **Graceful shutdown** – ensures no futures dangle after exit.

This implementation replaces earlier versions that had race conditions and redundant locks.
It is now async-friendly and uses English-only comments.
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

__all__ = ["EmbeddingError", "EmbeddingJob", "EnhancedEmbeddingService"]

log = logging.getLogger(__name__)

###############################################################################
# Exceptions & Data Containers
###############################################################################

class EmbeddingError(RuntimeError):
    """Raised when the embedding pipeline fails irrecoverably."""

@dataclass(slots=True, frozen=True)
class EmbeddingJob:
    """Internal container binding a text to its awaiting Future result."""
    text: str
    future: asyncio.Future[np.ndarray]

###############################################################################
# Embedding Service Implementation
###############################################################################

class EmbeddingService:
    """Thread-safe, cache-aware embedding service.
    
    Public API consists of the `encode` method (async) and a few internal helpers.
    All heavy lifting happens in a single background thread to keep the asyncio event loop responsive.
    """

    def __init__(self, model_name: str, settings: Optional[UnifiedSettings] = None) -> None:
        self.model_name = model_name
        self.settings = settings or UnifiedSettings.for_development()
        self._model_lock = threading.RLock()
        self._model: Optional[SentenceTransformer] = None

        # Initialize cache for embeddings (size and TTL from settings)
        self.cache = SmartCache(
            max_size=self.settings.performance.cache_size,
            ttl=self.settings.performance.cache_ttl_seconds,
        )

        # Batch processing state
        self._queue: List[EmbeddingJob] = []
        self._queue_lock = threading.RLock()
        self._queue_condition = threading.Condition(self._queue_lock)
        self._shutdown = threading.Event()

        # Background batch processing thread handle
        self._batch_thread: Optional[threading.Thread] = None

        # Eager initialization: load model and start batch thread
        self._load_model()      # load model (or fallback)
        self._start_processor()  # start background batching thread
        log.info("Embedding service ready (model=%s)", self.model_name)

    # Model management

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model with a fallback on failure."""
        with self._model_lock:
            if self._model is not None:
                return
            try:
                log.info("Loading embedding model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                log.info("Model loaded: %s (dim=%d)", self.model_name, self._model.get_sentence_embedding_dimension())
            except Exception as exc:
                log.warning("Primary model loading failed: %s", exc)
                fallback = "all-MiniLM-L6-v2"
                if self.model_name != fallback:
                    try:
                        log.info("Attempting fallback model: %s", fallback)
                        self._model = SentenceTransformer(fallback)
                        self.model_name = fallback
                        log.info("Fallback model loaded: %s", fallback)
                    except Exception as fexc:
                        log.error("Fallback model failed to load: %s", fexc)
                        raise EmbeddingError("Could not load any embedding model") from fexc
                else:
                    raise EmbeddingError("Could not load embedding model") from exc

    def _start_processor(self) -> None:
        """Start the background batching thread (idempotent)."""
        if self._batch_thread and self._batch_thread.is_alive():
            return
        self._batch_thread = threading.Thread(target=self._batch_loop, name="emb-batcher", daemon=True)
        self._batch_thread.start()

    def _batch_loop(self) -> None:
        """Background thread loop that processes queued embedding jobs in batches."""
        batch_size = self.settings.model.batch_add_size
        log.debug("Batch processor loop started (batch_size=%d)", batch_size)
        while not self._shutdown.is_set():
            with self._queue_condition:
                if not self._queue:
                    # Wait for a short time for new jobs
                    self._queue_condition.wait(timeout=0.05)
                    continue
                # Slice a batch from the queue
                batch = self._queue[:batch_size]
                self._queue = self._queue[batch_size:]
            # Process batch outside the lock
            try:
                texts = [job.text for job in batch]
                vectors = self._encode_direct(texts)  # synchronous call
                for job, vec in zip(batch, vectors):
                    if not job.future.done():
                        job.future.set_result(vec)
            except Exception as exc:
                MET_ERRORS_TOTAL.labels(type="embedding", component="batch_loop").inc()
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(EmbeddingError(str(exc)))
        log.debug("Batch processor loop exited")

    # Public API

    async def encode(self, text: Union[str, Sequence[str]]) -> np.ndarray:
        """Return a vector embedding for the given text (string or sequence of strings)."""
        if isinstance(text, str):
            # Single text -> returns shape (1, dim) array
            return await self._encode_single(text)
        else:
            # Sequence of texts -> returns shape (n, dim) array
            return await self._encode_multi(list(text))

    # Internal async helpers

    async def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single string into a 1 x dim embedding vector (as numpy array)."""
        # Attempt cache lookup first
        key = self._cache_key(text)
        cached = self.cache.get(key)
        if cached is not None:
            # Return cached embedding, ensure shape (1, dim)
            return cached.reshape(1, -1)
        # Not in cache: enqueue for batch processing
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
        # Cache the new embedding result for future reuse
        self.cache.put(key, vec)
        return vec.reshape(1, -1)

    async def _encode_multi(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into an array of embeddings."""
        # For multiple texts, we can process them directly (and possibly cache each)
        vectors = []
        for text in texts:
            vec = await self._encode_single(text)
            vectors.append(vec)  # each vec is 1 x dim
        # Concatenate results into one array
        return np.vstack(vectors)

    def _encode_direct(self, texts: List[str]) -> np.ndarray:
        """Directly encode a batch of texts (runs in background thread)."""
        if self._model is None:
            raise EmbeddingError("Embedding model is not loaded")
        return self._model.encode(texts)  # returns an array of embeddings

    def _cache_key(self, text: str) -> str:
        """Compute a cache key for a given text input."""
        # Use a hash for caching (e.g., SHA1 for speed, here MD5 via hashlib for simplicity)
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    async def close(self) -> None:
        """Gracefully shut down the embedding service, stopping the batch thread."""
        self._shutdown.set()
        # Wait for batch thread to finish if running
        if self._batch_thread:
            self._batch_thread.join(timeout=1.0)
        log.info("Embedding service closed.")
