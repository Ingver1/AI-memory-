# memory_system/utils/cache.py
"""
Simple cache implementation for Unified Memory System with TTL and LRU support.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
from collections import OrderedDict
import time

class SmartCache:
    """In-memory cache with optional TTL (time-to-live) and LRU eviction."""

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        Initialize the cache.

        :param max_size: maximum number of items to store before evicting least recently used.
        :param ttl: time-to-live for items in seconds. Items older than this will be expired.
        """
        self.max_size = max_size
        self.ttl = ttl
        # Using OrderedDict to maintain insertion order for LRU policy
        self._data: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def _is_expired(self, entry_time: float) -> bool:
        """Check if a cached entry timestamp is expired relative to TTL."""
        return (time.time() - entry_time) > self.ttl

    def get(self, key: str) -> Any:
        """
        Retrieve value from cache by key.

        Returns the cached value if present and not expired, otherwise None.
        Updates hit/miss statistics and moves the key to end (most recently used) if found.
        """
        if key in self._data:
            value, ts = self._data[key]
            if self._is_expired(ts):
                # Remove expired item
                self._data.pop(key, None)
                self._misses += 1
                return None
            # Item is valid
            self._hits += 1
            # Refresh order (move to end as recently used)
            self._data.move_to_end(key)
            return value
        else:
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """
        Store a value in the cache under the given key.

        If adding the item exceeds max_size, evict the least recently used item.
        """
        current_time = time.time()
        # If key exists, remove it first to update order
        if key in self._data:
            self._data.pop(key)
        self._data[key] = (value, current_time)
        # Evict least recently used if over capacity
        if len(self._data) > self.max_size:
            # popitem(last=False) pops the first (oldest) item
            self._data.popitem(last=False)

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._data.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, float]:
        """
        Get cache statistics.

        :return: Dictionary with current size, max size, hit rate.
        """
        size = len(self._data)
        # Calculate hit rate as fraction of hits in total access attempts
        total_accesses = self._hits + self._misses
        hit_rate = (self._hits / total_accesses) if total_accesses > 0 else 0.0
        return {"size": size, "max_size": self.max_size, "hit_rate": hit_rate}
