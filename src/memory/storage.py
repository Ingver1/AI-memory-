Memory Storage - Storage backend implementations for the memory system.

This module provides different storage backends for memory persistence,
including vector databases, file systems, and hybrid approaches.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from ..models.memory_models import MemoryItem, MemoryType
from ..config.settings import Settings
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class MemoryStorage:
    """
    Memory storage system supporting multiple backends.
    
    This class provides a unified interface for memory storage operations
    across different backends including vector databases and file systems.
    """
    
    def __init__(self, settings: Settings, embedding_manager: EmbeddingManager):
        """
        Initialize the memory storage system.
        
        Args:
            settings: Configuration settings for the storage system
            embedding_manager: Embedding manager for vector operations
        """
        self.settings = settings
        self.embedding_manager = embedding_manager
        self.chroma_client = None
        self.chroma_collection = None
        self.file_storage_path = Path("data/memories")
        self._initialized = False
        
    async def initialize(self) -> None:
        """
        Initialize the storage backend.
        
        This method sets up the chosen storage backend and prepares
        it for memory operations.
        """
        try:
            logger.info("Initializing memory storage...")
            
            # Create file storage directory if it doesn't exist
            self.file_storage_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB
            if self.settings.memory.vector_db.provider == "chroma":
                await self._initialize_chroma()
            else:
                logger.warning(f"Unsupported vector DB provider: {self.settings.memory.vector_db.provider}")
                
            self._initialized = True
            logger.info("Memory storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory storage: {e}")
            raise
            
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the storage backend.
        """
        if self.chroma_client:
            # ChromaDB client doesn't need explicit shutdown
            pass
            
        self._initialized = False
        logger.info("Memory storage shutdown complete")
        
    async def _initialize_chroma(self) -> None:
        """
        Initialize ChromaDB client and collection.
        """
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.Client(
                Settings=ChromaSettings(
                    persist_directory="data/chroma_db",
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection
            collection_name = self.settings.memory.vector_db.index_name
            
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    name=collection_name
                )
                logger.info(f"Using existing ChromaDB collection: {collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.chroma_collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new ChromaDB collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
            
    async def store_memory(self, memory_item: MemoryItem) -> str:
        """
        Store a memory item in the storage backend.
        
        Args:
            memory_item: The memory item to store
            
        Returns:
            Memory ID of the stored memory
            
        Raises:
            RuntimeError: If storage is not initialized
            ValueError: If memory item is invalid
        """
        if not self._initialized:
            raise RuntimeError("Memory storage not initialized")
            
        if not memory_item.content.strip():
            raise ValueError("Memory content cannot be empty")
            
        try:
            # Generate unique ID if not provided
            if not memory_item.id:
                memory_item.id = str(uuid.uuid4())
                
            # Store in vector database
            await self._store_in_vector_db(memory_item)
            
            # Store in file system as backup
            await self._store_in_file_system(memory_item)
            
            logger.info(f"Memory stored successfully: {memory_item.id}")
            return memory_item.id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
            
    async def _store_in_vector_db(self, memory_item: MemoryItem) -> None:
        """
        Store memory item in the vector database.
        
        Args:
            memory_item: The memory item to store
        """
        if not self.chroma_collection:
            raise RuntimeError("ChromaDB collection not initialized")
            
        # Prepare metadata
        metadata = {
            "user_id": memory_item.user_id,
            "memory_type": memory_item.memory_type.value,
            "created_at": memory_item.created_at.isoformat(),
            "last_accessed": memory_item.last_accessed.isoformat(),
            "access_count": memory_item.access_count,
            "importance_score": memory_item.importance_score,
            **memory_item.metadata
        }
        
        # Store in ChromaDB
        self.chroma_collection.add(
            ids=[memory_item.id],
            embeddings=[memory_item.embedding.tolist()],
            metadatas=[metadata],
            documents=[memory_item.content]
        )
        
    async def _store_in_file_system(self, memory_item: MemoryItem) -> None:
        """
        Store memory item in the file system as backup.
        
        Args:
            memory_item: The memory item to store
        """
        user_dir = self.file_storage_path / memory_item.user_id
        user_dir.mkdir(exist_ok=True)
        
        memory_file = user_dir / f"{memory_item.id}.json"
        
        # Convert memory item to dictionary
        memory_dict = {
            "id": memory_item.id,
            "content": memory_item.content,
            "user_id": memory_item.user_id,
            "memory_type": memory_item.memory_type.value,
            "metadata": memory_item.metadata,
            "created_at": memory_item.created_at.isoformat(),
            "last_accessed": memory_item.last_accessed.isoformat(),
            "access_count": memory_item.access_count,
            "importance_score": memory_item.importance_score,
            "embedding": memory_item.embedding.tolist() if memory_item.embedding is not None else None
        }
        
        # Save to file
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory_dict, f, indent=2, ensure_ascii=False)
            
    async def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        user_id: str,
        limit: int = 5,
        memory_type: Optional[MemoryType] = None
    ) -> List[Dict]:
        """
        Retrieve memories based on query embedding.
        
        Args:
            query_embedding: Embedding vector for the query
            user_id: User ID to filter memories
            limit: Maximum number of memories to retrieve
            memory_type: Filter by memory type (optional)
            
        Returns:
            List of memory dictionaries
            
        Raises:
            RuntimeError: If storage is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Memory storage not initialized")
            
        if not self.chroma_collection:
            raise RuntimeError("ChromaDB collection not initialized")
            
        try:
            # Build where clause for filtering
            where_clause = {"user_id": user_id}
            if memory_type:
                where_clause["memory_type"] = memory_type.value
                
            # Query ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                where=where_clause,
                n_results=limit
            )
            
            # Format results
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    memory_dict = {
                        "id": results['ids'][0][i],
                        "content": doc,
                        "user_id": metadata['user_id'],
                        "memory_type": metadata['memory_type'],
                        "created_at": metadata['created_at'],
                        "last_accessed": metadata['last_accessed'],
                        "access_count": metadata['access_count'],
                        "importance_score": metadata['importance_score'],
                        "similarity_score": 1.0 - results['distances'][0][i] if results['distances'] else 0.0
                    }
                    memories.append(memory_dict)
                    
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise
            
    async def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None
    ) -> List[MemoryItem]:
        """
        Get all memories for a specific user.
        
        Args:
            user_id: User ID to get memories for
            memory_type: Filter by memory type (optional)
            
        Returns:
            List of memory items
            
        Raises:
            RuntimeError: If storage is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Memory storage not initialized")
            
        try:
            # Load from file system
            user_dir = self.file_storage_path / user_id
            if not user_dir.exists():
                return []
                
            memories = []
            for memory_file in user_dir.glob("*.json"):
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        memory_dict = json.load(f)
                        
                    # Convert to MemoryItem
                    memory_item = MemoryItem(
                        id=memory_dict['id'],
                        content=memory_dict['content'],
                        user_id=memory_dict['user_id'],
                        memory_type=MemoryType(memory_dict['memory_type']),
                        metadata=memory_dict['metadata'],
                        created_at=datetime.fromisoformat(memory_dict['created_at']),
                        last_accessed=datetime.fromisoformat(memory_dict['last_accessed']),
                        access_count=memory_dict['access_count'],
                        importance_score=memory_dict['importance_score'],
                        embedding=np.array(memory_dict['embedding']) if memory_dict['embedding'] else None
                    )
                    
                    # Filter by memory type if specified
                    if memory_type is None or memory_item.memory_type == memory_type:
                        memories.append(memory_item)
                        
                except Exception as e:
                    logger.error(f"Failed to load memory from {memory_file}: {e}")
                    continue
                    
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get user memories: {e}")
            raise
            
    async def update_memory_access(self, memory_id: str) -> None:
        """
        Update memory access statistics.
        
        Args:
            memory_id: ID of the memory to update
            
        Raises:
            RuntimeError: If storage is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Memory storage not initialized")
            
        try:
            # Update in ChromaDB
            if self.chroma_collection:
                # Get current memory
                result = self.chroma_collection.get(ids=[memory_id])
                if result['metadatas'] and result['metadatas'][0]:
                    metadata = result['metadatas'][0]
                    metadata['access_count'] += 1
                    metadata['last_accessed'] = datetime.utcnow().isoformat()
                    
                    # Update in ChromaDB
                    self.chroma_collection.update(
                        ids=[memory_id],
                        metadatas=[metadata]
                    )
                    
            # Update in file system
            await self._update_memory_file_access(memory_id)
            
        except Exception as e:
            logger.error(f"Failed to update memory access: {e}")
            raise
            
    async def _update_memory_file_access(self, memory_id: str) -> None:
        """
        Update memory access statistics in file system.
        
        Args:
            memory_id: ID of the memory to update
        """
        # Find the memory file
        for user_dir in self.file_storage_path.iterdir():
            if user_dir.is_dir():
                memory_file = user_dir / f"{memory_id}.json"
                if memory_file.exists():
                    try:
                        with open(memory_file, 'r', encoding='utf-8') as f:
                            memory_dict = json.load(f)
                            
                        memory_dict['access_count'] += 1
                        memory_dict['last_accessed'] = datetime.utcnow().isoformat()
                        
                        with open(memory_file, 'w', encoding='utf-8') as f:
                            json.dump(memory_dict, f, indent=2, ensure_ascii=False)
                            
                        break
                    except Exception as e:
                        logger.error(f"Failed to update memory file {memory_file}: {e}")
                        
    async def update_memory_importance(self, memory_id: str, importance_score: float) -> None:
        """
        Update memory importance score.
        
        Args:
            memory_id: ID of the memory to update
            importance_score: New importance score
            
        Raises:
            RuntimeError: If storage is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Memory storage not initialized")
            
        try:
            # Update in ChromaDB
            if self.chroma_collection:
                result = self.chroma_collection.get(ids=[memory_id])
                if result['metadatas'] and result['metadatas'][0]:
                    metadata = result['metadatas'][0]
                    metadata['importance_score'] = importance_score
                    
                    self.chroma_collection.update(
                        ids=[memory_id],
                        metadatas=[metadata]
                    )
                    
            # Update in file system
            await self._update_memory_file_importance(memory_id, importance_score)
            
        except Exception as e:
            logger.error(f"Failed to update memory importance: {e}")
            raise
            
    async def _update_memory_file_importance(self, memory_id: str, importance_score: float) -> None:
        """
        Update memory importance score in file system.
        
        Args:
            memory_id: ID of the memory to update
            importance_score: New importance score
        """
        for user_dir in self.file_storage_path.iterdir():
            if user_dir.is_dir():
                memory_file = user_dir / f"{memory_id}.json"
                if memory_file.exists():
                    try:
                        with open(memory_file, 'r', encoding='utf-8') as f:
                            memory_dict = json.load(f)
                            
                        memory_dict['importance_score'] = importance_score
                        
                        with open(memory_file, 'w', encoding='utf-8') as f:
                            json.dump(memory_dict, f, indent=2, ensure_ascii=False)
                            
                        break
                    except Exception as e:
                        logger.error(f"Failed to update memory file {memory_file}: {e}")
                        
    async def update_memory_type(self, memory_id: str, memory_type: MemoryType) -> None:
        """
        Update memory type.
        
        Args:
            memory_id: ID of the memory to update
            memory_type: New memory type
            
        Raises:
            RuntimeError: If storage is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Memory storage not initialized")
            
        try:
            # Update in ChromaDB
            if self.chroma_collection:
                result = self.chroma_collection.get(ids=[memory_id])
                if result['metadatas'] and result['metadatas'][0]:
                    metadata = result['metadatas'][0]
                    metadata['memory_type'] = memory_type.value
                    
                    self.chroma_collection.update(
                        ids=[memory_id],
                        metadatas=[metadata]
                    )
                    
            # Update in file system
            await self._update_memory_file_type(memory_id, memory_type)
            
        except Exception as e:
            logger.error(f"Failed to update memory type: {e}")
            raise
            
    async def _update_memory_file_type(self, memory_id: str, memory_type: MemoryType) -> None:
        """
        Update memory type in file system.
        
        Args:
            memory_id: ID of the memory to update
            memory_type: New memory type
        """
        for user_dir in self.file_storage_path.iterdir():
            if user_dir.is_dir():
                memory_file = user_dir / f"{memory_id}.json"
                if memory_file.exists():
                    try:
                        with open(memory_file, 'r', encoding='utf-8') as f:
                            memory_dict = json.load(f)
                            
                        memory_dict['memory_type'] = memory_type.value
                        
                        with open(memory_file, 'w', encoding='utf-8') as f:
                            json.dump(memory_dict, f, indent=2, ensure_ascii=False)
                            
                        break
                    except Exception as e:
                        logger.error(f"Failed to update memory file {memory_file}: {e}")
                        
    async def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory from storage.
        
        Args:
            memory_id: ID of the memory to delete
            
        Raises:
            RuntimeError: If storage is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Memory storage not initialized")
            
        try:
            # Delete from ChromaDB
            if self.chroma_collection:
                self.chroma_collection.delete(ids=[memory_id])
                
            # Delete from file system
            await self._delete_memory_file(memory_id)
            
            logger.info(f"Memory deleted successfully: {memory_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            raise
            
    async def _delete_memory_file(self, memory_id: str) -> None:
        """
        Delete memory file from file system.
        
        Args:
            memory_id: ID of the memory to delete
        """
        for user_dir in self.file_storage_path.iterdir():
            if user_dir.is_dir():
                memory_file = user_dir / f"{memory_id}.json"
                if memory_file.exists():
                    try:
                        memory_file.unlink()
                        break
                    except Exception as e:
                        logger.error(f"Failed to delete memory file {memory_file}: {e}")
                        
    def get_total_memory_count(self) -> int:
        """
        Get total number of memories stored.
        
        Returns:
            Total number of memories
        """
        if not self._initialized:
            return 0
            
        try:
            if self.chroma_collection:
                return self.chroma_collection.count()
            else:
                # Count from file system
                count = 0
                for user_dir in self.file_storage_path.iterdir():
                    if user_dir.is_dir():
                        count += len(list(user_dir.glob("*.json")))
                return count
        except Exception as e:
            logger.error(f"Failed to get total memory count: {e}")
            return 0
