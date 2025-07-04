"""middleware.py — global FastAPI middlewares for Unified Memory System

Version: 0.8-alpha

This module ships three core middlewares and a lightweight dependency checker:
• SessionTracker — in-memory helper that records the last activity timestamp per user.
• RateLimitingMiddleware — token-bucket rate limiting per user / IP.
• MaintenanceModeMiddleware — graceful shutdown gate that denies traffic while the
  service is in maintenance.
• check_dependencies() — async helper used by health routes to verify that optional
  third-party libraries are importable at runtime.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import logging
import os
import time
from collections import deque
from typing import Dict, MutableMapping, Set

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

__all__ = [
    "SessionTracker",
    "RateLimitingMiddleware",
    "MaintenanceModeMiddleware",
    "check_dependencies",
]

log = logging.getLogger(__name__)

###############################################################################
# Session tracking helper                                                     #
###############################################################################


class SessionTracker:
    """Thread-safe tracker that records last activity timestamps.

    A single process-wide instance is usually enough. In production you can
    replace it with Redis or another distributed key/value store.
    """

    _last_seen: MutableMapping[str, float] = {}
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def mark(cls, user_id: str) -> None:
        """Register current UTC timestamp for *user_id*."""
        async with cls._lock:
            cls._last_seen[user_id] = time.time()

    @classmethod
    async def active_count(cls, window_seconds: int = 3600) -> int:
        """Return number of users seen in the last *window_seconds*."""
        threshold = time.time() - window_seconds
        async with cls._lock:
            return sum(1 for ts in cls._last_seen.values() if ts >= threshold)

    def values(self):
        """Return all tracked timestamps."""
        return self._last_seen.values()


# Global session tracker instance
session_tracker = SessionTracker()


###############################################################################
# Rate limiting                                                               #
###############################################################################


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple token-bucket rate limiting per user / IP.

    Parameters
    ----------
    max_requests
        Allowed requests per sliding window.
    window_seconds
        Duration of the sliding window in seconds.
    bypass_endpoints
        Set of URL paths that skip rate limiting (health, docs, etc.).
    """

    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
        bypass_endpoints: Set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.bypass = bypass_endpoints or {
            "/api/v1/health",
            "/api/v1/version",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/favicon.ico",
        }
        # user_id -> deque[timestamps]
        self._hits: Dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------------------
    @staticmethod
    def _get_user_id(request: Request) -> str:
        """Derive a stable identifier from Authorization header or client IP."""
        auth = request.headers.get("authorization") or getattr(request.client, "host", "unknown")
        return hashlib.sha256(auth.encode()).hexdigest()

    # ---------------------------------------------------------------------
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:  # type: ignore[override]
        if request.url.path in self.bypass:
            return await call_next(request)

        user_id = self._get_user_id(request)
        now = time.time()

        async with self._lock:
            bucket = self._hits.setdefault(user_id, deque())
            # Drop timestamps older than the window
            while bucket and bucket[0] <= now - self.window:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = int(bucket[0] + self.window - now) + 1
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            bucket.append(now)

        await session_tracker.mark(user_id)
        return await call_next(request)


###############################################################################
# Maintenance mode                                                            #
###############################################################################


class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    """Blocks all requests while the service is under maintenance.

    Toggle via the *UMS_MAINTENANCE* environment variable or by calling
    `enable()` / `disable()` on the middleware instance.
    """

    def __init__(self, app, allowed_paths: Set[str] | None = None) -> None:
        super().__init__(app)
        self.allowed_paths: Set[str] = allowed_paths or {
            "/api/v1/admin/maintenance-mode",
            "/health",
            "/healthz",
            "/readyz",
        }
        self._enabled: bool = os.getenv("UMS_MAINTENANCE", "0") == "1"

    # Public helpers -------------------------------------------------------
    def enable(self) -> None:
        """Enable maintenance mode at runtime."""
        self._enabled = True

    def disable(self) -> None:
        """Disable maintenance mode at runtime."""
        self._enabled = False

    # ---------------------------------------------------------------------
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:  # type: ignore[override]
        if self._enabled and request.url.path not in self.allowed_paths:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Service temporarily unavailable due to maintenance",
                    "retry_after": 60,
                },
                headers={"Retry-After": "60"},
            )
        return await call_next(request)


###############################################################################
# Dependency checker                                                          #
###############################################################################

_REQUIRED_MODULES = {
    "faiss": "FAISS (vector search backend)",
    "sentence_transformers": "SentenceTransformers (embedding models)",
}


async def check_dependencies() -> Dict[str, bool]:
    """Return a mapping of optional module names to their availability."""
    results: Dict[str, bool] = {}
    for module_name in _REQUIRED_MODULES:
        try:
            await asyncio.to_thread(importlib.import_module, module_name)
            results[module_name] = True
        except Exception:  # noqa: BLE001 — catch anything (importlib errors, etc.)
            results[module_name] = False
    return results
