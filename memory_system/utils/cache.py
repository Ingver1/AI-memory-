"""Simple cache implementation for Unified Memory System."""

from __future__ import annotations
from typing import Any, Dict

class SmartCache:
    """In-memory cache with optional max size and time-to-live (TTL) support."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300) -> None:
        """
        Initialize the cache.
        :param max_size: Maximum number of items to store.
        :param ttl: Time-to-live for each cache entry in seconds (not enforced in this simple implementation).
        """
        self.max_size = max_size
        self.ttl = ttl
        self._data: Dict[str, Any] = {}
        # NOTE: TTL functionality is not fully implemented; entries are not auto-expiring in this simple cache.

    def get(self, key: str) -> Any:
        """Retrieve a value from the cache by key, or None if not present."""
        return self._data.get(key)

    def put(self, key: str, value: Any) -> None:
        """Store a value in the cache under the given key. Evict oldest if over max_size."""
        if len(self._data) >= self.max_size:
            # Evict an arbitrary item (FIFO eviction strategy for simplicity)
            oldest_key = next(iter(self._data))
            self._data.pop(oldest_key, None)
        self._data[key] = value

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._data.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the cache."""
        # A hit rate statistic could be maintained with additional tracking; here we return 0.0 as a placeholder.
        return {
            "size": len(self._data),
            "max_size": self.max_size,
            "hit_rate": 0.0
        }
