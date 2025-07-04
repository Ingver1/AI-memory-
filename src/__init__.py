AI Memory System - A comprehensive memory management system for AI applications.

This package provides persistent memory capabilities for AI agents, including:
- Short-term and long-term memory storage
- Semantic search and retrieval
- Memory consolidation and decay
- Vector database integration
- REST API for memory operations

Author: AI Memory System
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Memory System"

from .memory.memory_manager import MemoryManager
from .memory.storage import MemoryStorage
from .memory.embeddings import EmbeddingManager
from .memory.retrieval import MemoryRetrieval
from .models.memory_models import MemoryItem, MemoryType
from .config.settings import Settings

__all__ = [
    "MemoryManager",
    "MemoryStorage", 
    "EmbeddingManager",
    "MemoryRetrieval",
    "MemoryItem",
    "MemoryType",
    "Settings"
]
