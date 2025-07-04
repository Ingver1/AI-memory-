# memory_system/utils/__init__.py
"""Utilities module for Unified Memory System."""

from __future__ import annotations

__all__ = [
    "SmartCache",
    "EnhancedPIIFilter", 
    "SecureTokenManager",
    "EncryptionManager",
    "PasswordManager",
    "RateLimiter",
    "MemorySystemError",
    "ValidationError", 
    "StorageError",
    "EmbeddingError",
    "get_prometheus_metrics",
    "prometheus_counter",
]

def __getattr__(name: str):
    if name == "SmartCache":
        from memory_system.utils.cache import SmartCache
        return SmartCache
    elif name in (
        "EnhancedPIIFilter",
        "SecureTokenManager",
        "EncryptionManager", 
        "PasswordManager",
        "RateLimiter",
    ):
        from memory_system.utils.security import (
            EnhancedPIIFilter,
            SecureTokenManager,
            EncryptionManager,
            PasswordManager, 
            RateLimiter,
        )
        return locals()[name]
    elif name in (
        "MemorySystemError",
        "ValidationError",
        "StorageError", 
        "EmbeddingError",
    ):
        from memory_system.utils.exceptions import (
            MemorySystemError,
            ValidationError,
            StorageError,
            EmbeddingError,
        )
        return locals()[name]
    elif name in ("get_prometheus_metrics", "prometheus_counter"):
        from memory_system.utils.metrics import get_prometheus_metrics, prometheus_counter
        return locals()[name]
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
