"""settings.py — central configuration using Pydantic for Unified Memory System."""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseSettings, Field

class APISettings(BaseSettings):
    enable_api: bool = Field(True, description="Master switch to enable/disable API")
    enable_cors: bool = Field(False, description="Enable Cross-Origin Resource Sharing")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"], description="Allowed CORS origins")
    host: str = Field("0.0.0.0", description="Host to bind the server to")
    port: int = Field(8000, description="Port for the API server")

class DatabaseSettings(BaseSettings):
    db_path: str = Field("data/ums.db", description="Path to the SQLite database file")

class PerformanceSettings(BaseSettings):
    cache_size: int = Field(1000, description="Max cache entries for embeddings")
    cache_ttl_seconds: int = Field(300, description="Time-to-live for cache entries")
    max_workers: int = Field(1, description="Max worker threads for background tasks")

class MonitoringSettings(BaseSettings):
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    enable_rate_limiting: bool = Field(True, description="Enable global rate limiting")

class SecuritySettings(BaseSettings):
    encrypt_at_rest: bool = Field(False, description="Enable encryption for stored data")
    filter_pii: bool = Field(True, description="Enable automatic PII filtering in logs")
    secret_key: str = Field("changeme", description="Secret key for encryption/JWT", env="UMS_SECRET_KEY")

class ReliabilitySettings(BaseSettings):
    backup_enabled: bool = Field(False, description="Enable automatic backups")
    backup_interval_minutes: int = Field(60 * 24, description="Backup interval in minutes")  # default daily

class ModelSettings(BaseSettings):
    model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Default embedding model name")
    batch_add_size: int = Field(8, description="Batch size for adding memories (embedding batch size)")

class UnifiedSettings(BaseSettings):
    profile: str = Field("development", description="Environment profile name")
    version: str = Field("0.8-alpha", description="Version string for the service")

    # Nested config sections
    api: APISettings = Field(default_factory=APISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    reliability: ReliabilitySettings = Field(default_factory=ReliabilitySettings)
    model: ModelSettings = Field(default_factory=ModelSettings)

    class Config:
        env_prefix = 'UMS_'  # Environment variables prefix (e.g., UMS_API_PORT)
        env_file = '.env'    # Load default values from .env file if present

    @classmethod
    def for_development(cls) -> UnifiedSettings:
        """Alternate constructor for development defaults."""
        return cls(profile="development")

    @classmethod
    def for_testing(cls) -> UnifiedSettings:
        """Alternate constructor for testing environment."""
        return cls(profile="testing", monitoring={"enable_metrics": False}, security={"encrypt_at_rest": False})
