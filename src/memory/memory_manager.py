Memory Manager - Core memory management functionality.

This module provides the central memory management system that coordinates
storage, retrieval, and consolidation of memories across different backends.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import asdict

from ..models.memory_models import MemoryItem, MemoryType
from ..config.settings import Settings
from .storage import MemoryStorage
from .embeddings import EmbeddingManager
from .retrieval import MemoryRetrieval

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Central memory management system that coordinates all memory operations.
    
    This class provides a high-level interface for memory storage, retrieval,
    and consolidation operations across different storage backends.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the memory manager.
        
        Args:
            settings: Configuration settings for the memory system
        """
        self.settings = settings
        self.storage = None
        self.embedding_manager = None
        self.retrieval = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """
        Initialize all memory system components.
        
        This method sets up storage, embedding, and retrieval components
        required for the memory system to function properly.
        """
        try:
            logger.info("Initializing memory manager components...")
            
            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager(self.settings)
            await self.embedding_manager.initialize()
            
            # Initialize storage
            self.storage = MemoryStorage(self.settings, self.embedding_manager)
            await self.storage.initialize()
            
            # Initialize retrieval system
            self.retrieval = MemoryRetrieval(self.settings, self.storage)
            await self.retrieval.initialize()
            
            self._initialized = True
            logger.info("Memory manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            raise
            
    async def shutdown(self) -> None:
        """
        Gracefully shutdown all memory system components.
        """
        if self.storage:
            await self.storage.shutdown()
            
        if self.embedding_manager:
            await self.embedding_manager.shutdown()
            
        if self.retrieval:
            await self.retrieval.shutdown()
            
        self._initialized = False
        logger.info("Memory manager shutdown complete")
        
    async def store_memory(
        self,
        content: str,
        user_id: str,
        memory_type: str = "long_term",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a new memory in the system.
        
        Args:
            content: The memory content to store
            user_id: Unique identifier for the user
            memory_type: Type of memory (short_term, long_term)
            metadata: Additional metadata for the memory
            
        Returns:
            Memory ID of the stored memory
            
        Raises:
            RuntimeError: If manager is not initialized
            ValueError: If invalid parameters are provided
        """
        if not self._initialized:
            raise RuntimeError("Memory manager not initialized")
            
        # Validate inputs
        if not content.strip():
            raise ValueError("Memory content cannot be empty")
            
        if not user_id.strip():
            raise ValueError("User ID cannot be empty")
            
        # Validate memory type
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            raise ValueError(f"Invalid memory type: {memory_type}")
            
        try:
            # Create memory item
            memory_item = MemoryItem(
                content=content.strip(),
                user_id=user_id.strip(),
                memory_type=mem_type,
                metadata=metadata or {},
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=0,
                importance_score=0.0
            )
            
            # Generate embedding for the memory
            embedding = await self.embedding_manager.generate_embedding(content)
            memory_item.embedding = embedding
            
            # Calculate initial importance score
            memory_item.importance_score = await self._calculate_importance_score(memory_item)
            
            # Store the memory
            memory_id = await self.storage.store_memory(memory_item)
            
            logger.info(f"Memory stored successfully: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
            
    async def retrieve_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        memory_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve memories based on a query.
        
        Args:
            query: Search query for memory retrieval
            user_id: Unique identifier for the user
            limit: Maximum number of memories to retrieve
            memory_type: Filter by memory type (optional)
            
        Returns:
            List of memory dictionaries
            
        Raises:
            RuntimeError: If manager is not initialized
            ValueError: If invalid parameters are provided
        """
        if not self._initialized:
            raise RuntimeError("Memory manager not initialized")
            
        # Validate inputs
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        if not user_id.strip():
            raise ValueError("User ID cannot be empty")
            
        if limit <= 0:
            raise ValueError("Limit must be positive")
            
        try:
            # Retrieve memories using the retrieval system
            memories = await self.retrieval.retrieve_memories(
                query=query.strip(),
                user_id=user_id.strip(),
                limit=limit,
                memory_type=memory_type
            )
            
            # Update access counts and last accessed times
            for memory in memories:
                await self._update_memory_access(memory['id'])
                
            logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise
            
    async def consolidate_memories(self, user_id: str) -> None:
        """
        Consolidate memories for a specific user.
        
        This method performs memory consolidation operations including:
        - Importance scoring updates
        - Redundancy removal
        - Memory decay application
        - Memory optimization
        
        Args:
            user_id: Unique identifier for the user
            
        Raises:
            RuntimeError: If manager is not initialized
            ValueError: If invalid user ID is provided
        """
        if not self._initialized:
            raise RuntimeError("Memory manager not initialized")
            
        if not user_id.strip():
            raise ValueError("User ID cannot be empty")
            
        try:
            logger.info(f"Starting memory consolidation for user {user_id}")
            
            # Get all memories for the user
            all_memories = await self.storage.get_user_memories(user_id)
            
            if not all_memories:
                logger.info(f"No memories found for user {user_id}")
                return
                
            # Apply memory decay
            await self._apply_memory_decay(all_memories)
            
            # Update importance scores
            await self._update_importance_scores(all_memories)
            
            # Remove redundant memories
            await self._remove_redundant_memories(all_memories)
            
            # Consolidate short-term memories to long-term
            await self._consolidate_short_term_memories(user_id)
            
            logger.info(f"Memory consolidation completed for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            raise
            
    async def _calculate_importance_score(self, memory_item: MemoryItem) -> float:
        """
        Calculate the importance score for a memory item.
        
        Args:
            memory_item: The memory item to score
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        # Base importance score calculation
        score = 0.5
        
        # Increase score based on content length (longer content might be more important)
        content_length_factor = min(len(memory_item.content) / 1000, 0.3)
        score += content_length_factor
        
        # Increase score if memory has metadata
        if memory_item.metadata:
            score += 0.1
            
        # Increase score for certain memory types
        if memory_item.memory_type == MemoryType.LONG_TERM:
            score += 0.1
            
        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, score))
        
    async def _update_memory_access(self, memory_id: str) -> None:
        """
        Update memory access statistics.
        
        Args:
            memory_id: ID of the memory to update
        """
        try:
            await self.storage.update_memory_access(memory_id)
        except Exception as e:
            logger.error(f"Failed to update memory access: {e}")
            
    async def _apply_memory_decay(self, memories: List[MemoryItem]) -> None:
        """
        Apply memory decay to reduce importance of old, unused memories.
        
        Args:
            memories: List of memory items to apply decay to
        """
        current_time = datetime.utcnow()
        
        for memory in memories:
            # Calculate time since last access
            time_since_access = current_time - memory.last_accessed
            days_since_access = time_since_access.days
            
            # Apply decay based on time and access frequency
            if days_since_access > 30:  # 30 days threshold
                decay_factor = 0.95 ** (days_since_access / 30)
                memory.importance_score *= decay_factor
                
                # Update in storage
                await self.storage.update_memory_importance(memory.id, memory.importance_score)
                
    async def _update_importance_scores(self, memories: List[MemoryItem]) -> None:
        """
        Update importance scores based on access patterns.
        
        Args:
            memories: List of memory items to update
        """
        for memory in memories:
            # Increase importance based on access frequency
            access_boost = min(memory.access_count / 100, 0.2)
            memory.importance_score = min(1.0, memory.importance_score + access_boost)
            
            # Update in storage
            await self.storage.update_memory_importance(memory.id, memory.importance_score)
            
    async def _remove_redundant_memories(self, memories: List[MemoryItem]) -> None:
        """
        Remove redundant or duplicate memories.
        
        Args:
            memories: List of memory items to check for redundancy
        """
        # Simple redundancy removal based on content similarity
        for i, memory1 in enumerate(memories):
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if memory1.embedding is not None and memory2.embedding is not None:
                    similarity = await self.embedding_manager.calculate_similarity(
                        memory1.embedding, memory2.embedding
                    )
                    
                    # If very similar and one has lower importance, remove it
                    if similarity > 0.95:
                        if memory1.importance_score < memory2.importance_score:
                            await self.storage.delete_memory(memory1.id)
                            logger.info(f"Removed redundant memory: {memory1.id}")
                        else:
                            await self.storage.delete_memory(memory2.id)
                            logger.info(f"Removed redundant memory: {memory2.id}")
                            
    async def _consolidate_short_term_memories(self, user_id: str) -> None:
        """
        Consolidate important short-term memories to long-term storage.
        
        Args:
            user_id: User ID for memory consolidation
        """
        # Get short-term memories
        short_term_memories = await self.storage.get_user_memories(
            user_id, memory_type=MemoryType.SHORT_TERM
        )
        
        # Consolidate high-importance short-term memories
        consolidation_threshold = self.settings.memory.memory_types.long_term.importance_threshold
        
        for memory in short_term_memories:
            if memory.importance_score >= consolidation_threshold:
                # Convert to long-term memory
                memory.memory_type = MemoryType.LONG_TERM
                await self.storage.update_memory_type(memory.id, MemoryType.LONG_TERM)
                logger.info(f"Consolidated short-term memory to long-term: {memory.id}")
                
    def get_stats(self) -> Dict:
        """
        Get memory system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        if not self._initialized:
            return {"error": "Memory manager not initialized"}
            
        try:
            return {
                "initialized": self._initialized,
                "storage_backend": self.settings.memory.storage_type,
                "embedding_model": self.settings.memory.embedding.model,
                "vector_db_provider": self.settings.memory.vector_db.provider,
                "total_memories": self.storage.get_total_memory_count() if self.storage else 0,
                "status": "healthy"
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
