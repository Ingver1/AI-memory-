"""Memory management API routes for Unified Memory System."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from memory_system.api.schemas import (
    MemoryCreate,
    MemoryRead,
    MemoryUpdate,
    MemoryQuery,
    MemorySearchResult,
    SuccessResponse,
)
from memory_system.config.settings import UnifiedSettings
from memory_system.core.store import EnhancedMemoryStore
from memory_system.core.embedding import EnhancedEmbeddingService
from memory_system.utils.exceptions import (
    MemorySystemError,
    ValidationError,
    StorageError,
    EmbeddingError,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["Memory Management"])


# ────────────────────────────────────────────────────────────────────────
# Dependencies
# ────────────────────────────────────────────────────────────────────────
async def get_memory_store() -> EnhancedMemoryStore:
    """Get memory store instance."""
    from memory_system.api.app import get_memory_store_instance
    return await get_memory_store_instance()


def get_settings() -> UnifiedSettings:
    """Get settings instance."""
    from memory_system.api.app import get_settings_instance
    return get_settings_instance()


async def get_embedding_service(settings: UnifiedSettings = Depends(get_settings)) -> EnhancedEmbeddingService:
    """Get embedding service instance."""
    # This would be cached in a real implementation
    return EnhancedEmbeddingService(settings.model.model_name, settings)


# ────────────────────────────────────────────────────────────────────────
# Memory CRUD Operations
# ────────────────────────────────────────────────────────────────────────
@router.post("/", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
async def create_memory(
    memory_data: MemoryCreate,
    store: EnhancedMemoryStore = Depends(get_memory_store),
    embedding_service: EnhancedEmbeddingService = Depends(get_embedding_service),
) -> MemoryRead:
    """Create a new memory with automatic embedding generation."""
    try:
        # Generate embedding for the text
        embedding = await embedding_service.encode(memory_data.text)
        
        # Create memory record
        # In a real implementation, this would use the actual store
        memory_id = f"mem_{hash(memory_data.text)}"
        
        memory = MemoryRead(
            id=memory_id,
            user_id=memory_data.user_id or "anonymous",
            text=memory_data.text,
            role=memory_data.role,
            tags=memory_data.tags,
            created_at=datetime.now(datetime.UTC) - timedelta(hours=1),
            updated_at=datetime.now(datetime.UTC) - timedelta(hours=1),
        )
        
        log.info(f"Created memory {memory_id} for user {memory.user_id}")
        return memory
        
    except EmbeddingError as e:                                      # B904
         log_error("Failed to generate query embedding: %s", e)
         raise HTTPException(
             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
             detail="Failed to generate query embedding",
         ) from e

    except Exception as e:                                          # B904
         log_error("Failed to search memories: %s", e)
         raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail="Failed to search memories",
         ) from e


@router.get("/{memory_id}", response_model=MemoryRead)
async def get_memory(
    memory_id: str,
    user_id: Optional[str] = Query(None, description="User ID filter"),
    store: EnhancedMemoryStore = Depends(get_memory_store),
) -> MemoryRead:
    """Retrieve a specific memory by ID."""
    try:
        # In a real implementation, this would query the store
        # For now, return a mock memory
        from datetime import datetime, timezone
        
        memory = MemoryRead(
            id=memory_id,
            user_id=user_id or "anonymous",
            text=f"Sample memory content for {memory_id}",
            role="assistant",
            tags=["sample"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        return memory
        
    except StorageError as e:
        log.error(f"Failed to retrieve memory {memory_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found"
        )
    except Exception as e:
        log.error(f"Failed to get memory {memory_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve memory"
        )


@router.put("/{memory_id}", response_model=MemoryRead)
async def update_memory(
    memory_id: str,
    memory_update: MemoryUpdate,
    user_id: Optional[str] = Query(None, description="User ID filter"),
    store: EnhancedMemoryStore = Depends(get_memory_store),
    embedding_service: EnhancedEmbeddingService = Depends(get_embedding_service),
) -> MemoryRead:
    """Update an existing memory."""
    try:
        # Validate that at least one field is being updated
        if not any([memory_update.text, memory_update.role, memory_update.tags]):
            raise ValidationError("At least one field must be updated")
        
        # If text is being updated, regenerate embedding
        if memory_update.text:
            embedding = await embedding_service.encode(memory_update.text)
        
        # In a real implementation, this would update the store
        from datetime import datetime, timezone
        
        memory = MemoryRead(
            id=memory_id,
            user_id=user_id or "anonymous",
            text=memory_update.text or f"Updated memory content for {memory_id}",
            role=memory_update.role or "assistant",
            tags=memory_update.tags or ["updated"],
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            updated_at=datetime.now(timezone.utc),
        )
        
        log.info(f"Updated memory {memory_id} for user {memory.user_id}")
        return memory
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except EmbeddingError as e:
        log.error(f"Failed to generate embedding for update: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to generate embedding: {str(e)}"
        )
    except StorageError as e:
        log.error(f"Failed to update memory {memory_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found"
        )
    except Exception as e:
        log.error(f"Failed to update memory {memory_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update memory"
        )


@router.delete("/{memory_id}", response_model=SuccessResponse)
async def delete_memory(
    memory_id: str,
    user_id: Optional[str] = Query(None, description="User ID filter"),
    store: EnhancedMemoryStore = Depends(get_memory_store),
) -> SuccessResponse:
    """Delete a specific memory."""
    try:
        # In a real implementation, this would delete from the store
        log.info(f"Deleted memory {memory_id} for user {user_id}")
        return SuccessResponse(message=f"Memory {memory_id} deleted successfully")
        
    except StorageError as e:
        log.error(f"Failed to delete memory {memory_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found"
        )
    except Exception as e:
        log.error(f"Failed to delete memory {memory_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete memory"
        )


# ────────────────────────────────────────────────────────────────────────
# Memory Search Operations
# ────────────────────────────────────────────────────────────────────────
@router.post("/search", response_model=List[MemorySearchResult])
async def search_memories(
    query: MemoryQuery,
    user_id: Optional[str] = Query(None, description="User ID filter"),
    store: EnhancedMemoryStore = Depends(get_memory_store),
    embedding_service: EnhancedEmbeddingService = Depends(get_embedding_service),
) -> List[MemorySearchResult]:
    """Search memories using semantic similarity."""
    try:
        # Generate embedding for the query
        query_embedding = await embedding_service.encode(query.query)
        
        # In a real implementation, this would search the vector index
        # For now, return mock results
        from datetime import datetime, timezone
        
        results = []
        for i in range(min(query.top_k, 3)):  # Mock 3 results
            score = 0.9 - (i * 0.1)  # Decreasing similarity scores
            
            result = MemorySearchResult(
                id=f"mem_{i}",
                user_id=user_id or "anonymous",
                text=f"Mock search result {i+1} for query: {query.query}",
                role="assistant",
                tags=["search", "result"],
                created_at=datetime.now(timezone.utc) - timedelta(hours=i),
                updated_at=datetime.now(timezone.utc) - timedelta(hours=i),
                score=score,
                embedding=query_embedding.flatten().tolist() if query.include_embeddings else None,
            )
            results.append(result)
        
        log.info(f"Search query '{query.query}' returned {len(results)} results")
        return results
        
    except EmbeddingError as e:
        log.error(f"Failed to generate query embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to generate query embedding: {str(e)}"
        )
    except Exception as e:
        log.error(f"Failed to search memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search memories"
        )


@router.get("/", response_model=list[MemoryRead])
async def list_memories(
    user_id: str | None = Query(None, description="User ID filter"),
) -> list[MemoryRead]:
    """Return all memories, optionally filtered by user_id."""
    store: EnhancedMemoryStore = request.app.state.store  # type: ignore[attr-defined]
    raw_rows = await store.list_memories(user_id=user_id)
    return [MemoryRead.model_validate(r) for r in raw_rows]
    
    try:
        # In a real implementation, this would query the store with filters
        from datetime import datetime, timezone
        
        memories = []
        for i in range(min(limit, 10)):  # Mock up to 10 memories
            memory = MemoryRead(
                id=f"mem_{offset + i}",
                user_id=user_id or "anonymous",
                text=f"Memory {offset + i} content",
                role=role or "assistant",
                tags=tags or ["sample"],
                created_at=datetime.now(timezone.utc) - timedelta(hours=i),
                updated_at=datetime.now(timezone.utc) - timedelta(hours=i),
            )
            memories.append(memory)
        
        log.info(f"Listed {len(memories)} memories with filters: user_id={user_id}, role={role}, tags={tags}")
        return memories
        
    except Exception as e:
        log.error(f"Failed to list memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list memories"
        )


# ────────────────────────────────────────────────────────────────────────
# Batch Operations
# ────────────────────────────────────────────────────────────────────────
@router.post("/batch", response_model=List[MemoryRead])
async def create_memories_batch(
    memories: List[MemoryCreate],
    store: EnhancedMemoryStore = Depends(get_memory_store),
    embedding_service: EnhancedEmbeddingService = Depends(get_embedding_service),
) -> List[MemoryRead]:
    """Create multiple memories in a single batch operation."""
    try:
        if len(memories) > 100:
            raise ValidationError("Batch size cannot exceed 100 memories")
        
        # Generate embeddings for all texts
        texts = [memory.text for memory in memories]
        embeddings = await embedding_service.encode(texts)
        
        # In a real implementation, this would batch insert into the store
        from datetime import datetime, timezone
        
        created_memories = []
        for i, memory_data in enumerate(memories):
            memory_id = f"batch_mem_{i}_{hash(memory_data.text)}"
            
            memory = MemoryRead(
                id=memory_id,
                user_id=memory_data.user_id or "anonymous",
                text=memory_data.text,
                role=memory_data.role,
                tags=memory_data.tags,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            created_memories.append(memory)
        
        log.info(f"Created batch of {len(created_memories)} memories")
        return created_memories
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except EmbeddingError as e:
        log.error(f"Failed to generate embeddings for batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to generate embeddings: {str(e)}"
        )
    except Exception as e:
        log.error(f"Failed to create memory batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create memory batch"
        )


@router.delete("/batch", response_model=SuccessResponse)
async def delete_memories_batch(
    memory_ids: List[str],
    user_id: Optional[str] = Query(None, description="User ID filter"),
    store: EnhancedMemoryStore = Depends(get_memory_store),
) -> SuccessResponse:
    """Delete multiple memories in a single batch operation."""
    try:
        if len(memory_ids) > 100:
            raise ValidationError("Batch size cannot exceed 100 memory IDs")
        
        # In a real implementation, this would batch delete from the store
        deleted_count = len(memory_ids)
        
        log.info(f"Deleted batch of {deleted_count} memories for user {user_id}")
        return SuccessResponse(message=f"Successfully deleted {deleted_count} memories")
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        log.error(f"Failed to delete memory batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete memory batch"
        )


# ────────────────────────────────────────────────────────────────────────
# Memory Statistics
# ────────────────────────────────────────────────────────────────────────
@router.get("/stats", response_model=dict)
async def get_memory_stats(
    user_id: Optional[str] = Query(None, description="User ID filter"),
    store: EnhancedMemoryStore = Depends(get_memory_store),
) -> dict:
    """Get memory statistics for a user or globally."""
    try:
        # In a real implementation, this would query the store for stats
        stats = {
            "total_memories": 42,
            "user_memories": 15 if user_id else 42,
            "total_size_bytes": 1024 * 1024,  # 1MB
            "average_text_length": 150,
            "most_common_tags": ["ai", "ml", "nlp"],
            "memory_distribution": {
                "user": 25,
                "assistant": 17,
            },
            "recent_activity": {
                "created_today": 5,
                "updated_today": 3,
                "deleted_today": 1,
            }
        }
        
        if user_id:
            stats["user_id"] = user_id
        
        return stats
        
    except Exception as e:
        log.error(f"Failed to get memory stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get memory statistics"
        )


# ────────────────────────────────────────────────────────────────────────
# Import missing imports
# ────────────────────────────────────────────────────────────────────────
from datetime import datetime, timezone, timedelta
