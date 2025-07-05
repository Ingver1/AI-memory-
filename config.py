"""Configuration management using Pydantic BaseSettings for the AI Memory system."""
import os
from typing import Optional, List
from pydantic import BaseSettings, validator

class MemoryConfig(BaseSettings):
    """Configuration settings for the AI Memory system."""
    # Database settings
    database_url: str = "sqlite:///memory.db"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Memory settings
    max_memories: int = 100000
    memory_cleanup_threshold: float = 0.8

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Search settings
    search_limit_max: int = 100
    search_limit_default: int = 10
    similarity_threshold: float = 0.7

    # API settings
    api_rate_limit: str = "100/hour"
    api_timeout: int = 30  # seconds
    max_content_length: int = 10000

    # Security settings
    secret_key: str  # no default, must be provided in environment or .env
    allowed_origins: List[str] = ["http://localhost:3000"]

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @validator('secret_key')
    def secret_key_must_be_set(cls, v):
        if not v or not v.strip():
            raise ValueError("SECRET_KEY must be set in environment or .env file")
        return v

    @validator('memory_cleanup_threshold')
    def threshold_must_be_valid(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Memory cleanup threshold must be between 0 and 1")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiate a global config object to be used across the application
config = MemoryConfig()
