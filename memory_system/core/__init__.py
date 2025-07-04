# memory_system/core/__init__.py
"""Core module for Unified Memory System."""

from __future__ import annotations

__all__ = [
    "EnhancedMemoryStore",
    "EnhancedEmbeddingService", 
    "FaissHNSWIndex",
    "VectorStore",
    "Memory",
    "HealthComponent",
]

def __getattr__(name: str):
    if name == "EnhancedMemoryStore":
        from memory_system.core.store import EnhancedMemoryStore
        return EnhancedMemoryStore
    elif name == "EnhancedEmbeddingService":
        from memory_system.core.embedding import EnhancedEmbeddingService
        return EnhancedEmbeddingService
    elif name == "FaissHNSWIndex":
        from memory_system.core.index import FaissHNSWIndex
        return FaissHNSWIndex
    elif name == "VectorStore":
        from memory_system.core.vector_store import VectorStore
        return VectorStore
    elif name in ("Memory", "HealthComponent"):
        from memory_system.core.store import Memory, HealthComponent
        return locals()[name]
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
