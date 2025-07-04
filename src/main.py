Main entry point for the AI Memory System.

This module provides the main application interface and coordinates
all memory system components.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from pathlib import Path

from .config.settings import Settings
from .memory.memory_manager import MemoryManager
from .api.routes import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIMemorySystem:
    """
    Main AI Memory System class that orchestrates all memory operations.
    
    This class provides a high-level interface for memory management,
    including storage, retrieval, and consolidation operations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AI Memory System.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.settings = Settings(config_path)
        self.memory_manager = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """
        Initialize the memory system components.
        
        This method sets up the memory manager and all required components
        for the AI memory system to function properly.
        """
        try:
            logger.info("Initializing AI Memory System...")
            
            # Initialize memory manager
            self.memory_manager = MemoryManager(self.settings)
            await self.memory_manager.initialize()
            
            self._initialized = True
            logger.info("AI Memory System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Memory System: {e}")
            raise
            
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the memory system.
        
        This method ensures all components are properly closed and
        any pending operations are completed.
        """
        if self.memory_manager:
            await self.memory_manager.shutdown()
            
        self._initialized = False
        logger.info("AI Memory System shutdown complete")
        
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
            RuntimeError: If system is not initialized
            ValueError: If invalid parameters are provided
        """
        if not self._initialized:
            raise RuntimeError("Memory system not initialized. Call initialize() first.")
            
        if not content.strip():
            raise ValueError("Memory content cannot be empty")
            
        if not user_id.strip():
            raise ValueError("User ID cannot be empty")
            
        try:
            memory_id = await self.memory_manager.store_memory(
                content=content,
                user_id=user_id,
                memory_type=memory_type,
                metadata=metadata or {}
            )
            
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
            RuntimeError: If system is not initialized
            ValueError: If invalid parameters are provided
        """
        if not self._initialized:
            raise RuntimeError("Memory system not initialized. Call initialize() first.")
            
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        if not user_id.strip():
            raise ValueError("User ID cannot be empty")
            
        if limit <= 0:
            raise ValueError("Limit must be positive")
            
        try:
            memories = await self.memory_manager.retrieve_memories(
                query=query,
                user_id=user_id,
                limit=limit,
                memory_type=memory_type
            )
            
            logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise
            
    async def consolidate_memories(self, user_id: str) -> None:
        """
        Consolidate memories for a specific user.
        
        This method performs memory consolidation operations including
        importance scoring, redundancy removal, and memory optimization.
        
        Args:
            user_id: Unique identifier for the user
            
        Raises:
            RuntimeError: If system is not initialized
            ValueError: If invalid user ID is provided
        """
        if not self._initialized:
            raise RuntimeError("Memory system not initialized. Call initialize() first.")
            
        if not user_id.strip():
            raise ValueError("User ID cannot be empty")
            
        try:
            await self.memory_manager.consolidate_memories(user_id)
            logger.info(f"Memory consolidation completed for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            raise
            
    def get_stats(self) -> Dict:
        """
        Get memory system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        if not self._initialized:
            return {"error": "System not initialized"}
            
        try:
            return self.memory_manager.get_stats()
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


async def main() -> None:
    """
    Main function to run the AI Memory System.
    
    This function demonstrates basic usage of the memory system
    and can be used for testing purposes.
    """
    # Initialize the memory system
    system = AIMemorySystem()
    
    try:
        await system.initialize()
        
        # Example usage
        print("AI Memory System initialized successfully!")
        
        # Store a test memory
        memory_id = await system.store_memory(
            content="User prefers dark mode interface",
            user_id="test_user",
            memory_type="long_term"
        )
        print(f"Stored memory: {memory_id}")
        
        # Retrieve memories
        memories = await system.retrieve_memories(
            query="interface preferences",
            user_id="test_user",
            limit=5
        )
        print(f"Retrieved {len(memories)} memories")
        
        # Display statistics
        stats = system.get_stats()
        print(f"System stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
    finally:
        await system.shutdown()


def run_server() -> None:
    """
    Run the AI Memory System as a web server.
    
    This function starts the FastAPI server with the memory system
    endpoints available for HTTP requests.
    """
    import uvicorn
    
    # Create FastAPI app
    app = create_app()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
