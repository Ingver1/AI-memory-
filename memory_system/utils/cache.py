# utils/cache.py — Simple cache utilities
#
# Version: 0.8‑alpha
"""Simple cache implementation for Unified Memory System."""

from __future__ import annotations

from typing import Any, Dict


class SmartCache:
    """Simple in-memory cache."""

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._data: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        """Get value from cache."""
        return self._data.get(key)

    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        self._data[key] = value

    def clear(self) -> None:
        """Clear cache."""
        self._data.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {"size": len(self._data), "max_size": self.max_size, "hit_rate": 0.0}
