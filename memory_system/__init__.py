"""Unified Memory System v0.8-alpha

A production-grade long-term memory backend for AI agents, assistants, and 
generative-retrieval workloads.

This package provides:
- FastAPI-based REST API for memory management
- Vector search capabilities with FAISS
- Semantic embeddings with sentence-transformers
- Security features (PII filtering, encryption)
- Monitoring and health checks
- Scalable SQLite backend with connection pooling
"""

from __future__ import annotations

import logging
from typing import Any, Dict

__version__ = "0.8.0a0"
__author__ = "Enhanced Memory Team"
__email__ = "kayel20221967@gmail.com"
__description__ = "Enterprise-grade memory system with vector search, FastAPI and monitoring"

# Configure default logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API exports
__all__ = [
    "__version__",
    "UnifiedSettings",
    "EnhancedMemoryStore", 
    "EmbeddingService",
    "create_app",
    "get_settings",
]

# Lazy imports to avoid heavy dependencies during package import
def __getattr__(name: str) -> Any:
    if name == "UnifiedSettings":
        from memory_system.config.settings import UnifiedSettings
        return UnifiedSettings
    elif name == "EnhancedMemoryStore":
        from memory_system.core.store import EnhancedMemoryStore
        return EnhancedMemoryStore
    elif name == "EmbeddingService":  # Fix this
        from memory_system.core.embedding import EnhancedEmbeddingService
        return EnhancedEmbeddingService  # Return the correct class
    elif name == "create_app":
        from memory_system.api.app import create_app
        return create_app
    elif name == "get_settings":
        from memory_system.config.settings import get_settings
        return get_settings
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_version_info() -> Dict[str, Any]:
    """Get detailed version information."""
    return {
        "version": __version__,
        "author": __author__, 
        "email": __email__,
        "description": __description__,
    }
