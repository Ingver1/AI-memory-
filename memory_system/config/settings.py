# settings.py — central configuration for Unified Memory System
#
# Version: 0.8‑alpha
"""Typed, environment‑aware configuration powered by **Pydantic v2**.

The class hierarchy mirrors functional domains:
    • *DatabaseConfig*      – SQLite/FAISS paths, connection pool size.
    • *ModelConfig*         – embedding model and HNSW parameters.
    • *SecurityConfig*      – encryption, PII filter, API token, rate limits.
    • *PerformanceConfig*   – worker pool, cache size/TTL.
    • *ReliabilityConfig*   – retries, backups.
    • *APIConfig*           – FastAPI host/port, CORS, enable flag.
    • *MonitoringConfig*    – Prometheus, rate limiting, log level.

:class:`UnifiedSettings` wraps them into a single **BaseSettings** subclass that
reads from ``.env`` (using nested *ENV_VAR__SUBKEY* convention) and provides
helper factories for development / testing / production.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Final

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DatabaseConfig",
    "ModelConfig",
    "SecurityConfig",
    "PerformanceConfig",
    "ReliabilityConfig",
    "APIConfig",
    "MonitoringConfig",
    "UnifiedSettings",
    "get_settings",
]

log = logging.getLogger(__name__)

###############################################################################
# Defaults & constants                                                        #
###############################################################################
DEFAULT_MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE: Final[int] = 128
DEFAULT_CACHE_SIZE: Final[int] = 1_000
DEFAULT_POOL_SIZE: Final[int] = 10
DEFAULT_PROM_PORT: Final[int] = 9100

###############################################################################
# Section models                                                              #
###############################################################################


class DatabaseConfig(BaseModel):
    """Database paths and pool size."""

    db_path: Path = Path("data/memory.db")
    vec_path: Path = Path("data/memory.vectors")
    cache_path: Path = Path("data/memory.cache")
    connection_pool_size: int = DEFAULT_POOL_SIZE

    model_config = {"frozen": True}


class ModelConfig(BaseModel):
    """Embedding model and HNSW tuning."""

    model_name: str = DEFAULT_MODEL_NAME
    batch_add_size: int = DEFAULT_BATCH_SIZE
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    vector_dim: int = 384

    model_config = {"frozen": True}


class SecurityConfig(BaseModel):
    """Encryption, PII filtering, API token and rate limits."""

    encrypt_at_rest: bool = False
    encryption_key: str = ""
    filter_pii: bool = True
    max_text_length: int = 10_000
    rate_limit_per_minute: int = 1_000
    api_token: str = "your-secret-token-change-me"

    # ------------------------------------------------------------------
    @field_validator("api_token")
    @classmethod
    def _validate_token(cls, v: str) -> str:  # noqa: D401
        if len(v) < 8:
            raise ValueError("API token must be at least 8 characters long")
        return v

    # ------------------------------------------------------------------
    @field_validator("encryption_key")
    @classmethod
    def _validate_key(cls, v: str) -> str:  # noqa: D401
        if v:
            try:
                Fernet(v.encode())
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid encryption key: {exc}") from exc
        return v


class PerformanceConfig(BaseModel):
    """Worker pool and cache parameters."""

    max_workers: int = 4
    cache_size: int = DEFAULT_CACHE_SIZE
    cache_ttl_seconds: int = 300
    rebuild_interval_seconds: int = 3_600

    # ------------------------------------------------------------------
    @field_validator("max_workers")
    @classmethod
    def _validate_workers(cls, v: int) -> int:  # noqa: D401
        if not 1 <= v <= 32:
            raise ValueError("max_workers must be between 1 and 32")
        return v

    model_config = {"frozen": True}


class ReliabilityConfig(BaseModel):
    """Retry policy and backups."""

    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    backup_enabled: bool = True
    backup_interval_hours: int = 24

    model_config = {"frozen": True}


class APIConfig(BaseModel):
    """FastAPI host/port and feature flags."""

    host: str = "0.0.0.0"
    port: int = 8000
    enable_cors: bool = True
    enable_api: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ------------------------------------------------------------------
    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:  # noqa: D401
        if not 1024 <= v <= 65_535:
            raise ValueError("port must be between 1024 and 65535")
        return v

    model_config = {"frozen": True}


class MonitoringConfig(BaseModel):
    """Prometheus exporter, rate limiting, logging."""

    enable_metrics: bool = True
    enable_rate_limiting: bool = True
    prom_port: int = DEFAULT_PROM_PORT
    health_check_interval: int = 30  # seconds
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    @field_validator("prom_port")
    @classmethod
    def _validate_pp(cls, v: int) -> int:  # noqa: D401
        if not 1024 <= v <= 65_535:
            raise ValueError("prom_port must be between 1024 and 65535")
        return v

    model_config = {"frozen": True}


###############################################################################
# Unified settings                                                            #
###############################################################################


class UnifiedSettings(BaseSettings):
    """Full configuration object loaded from environment or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    # Sub‑sections -------------------------------------------------------
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Basic properties
    version: str = "0.8-alpha"
    profile: str = "development"

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _post_init(self) -> "UnifiedSettings":  # noqa: D401
        """Create directories and inject encryption key if required."""
        # Ensure all parent directories exist.
        for path in (
            self.database.db_path,
            self.database.vec_path,
            self.database.cache_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        # Auto‑generate encryption key if needed.
        if self.security.encrypt_at_rest and not self.security.encryption_key:
            new_key = Fernet.generate_key().decode()
            self.security = self.security.model_copy(update={"encryption_key": new_key})
            log.warning(
                "🔐 Generated new encryption key. Store it in .env as "
                "SECURITY__ENCRYPTION_KEY=<key> to persist across restarts."
            )
        return self

    # Convenience helpers ------------------------------------------------
    @classmethod
    def for_testing(cls) -> "UnifiedSettings":
        """Settings tuned for pytest / CI."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="ums-test-"))
        return cls(
            database=DatabaseConfig(
                db_path=tmp_dir / "test.db",
                vec_path=tmp_dir / "test.vec",
                cache_path=tmp_dir / "test.cache",
                connection_pool_size=2,
            ),
            performance=PerformanceConfig(
                max_workers=2,
                cache_size=100,
                cache_ttl_seconds=10,
            ),
            api=APIConfig(port=0),  # random free port
            monitoring=MonitoringConfig(enable_metrics=False, health_check_interval=5),
            security=SecurityConfig(api_token="test-token-12345678", rate_limit_per_minute=10_000),
        )

    @classmethod
    def for_production(cls) -> "UnifiedSettings":
        """Opinionated production preset."""
        return cls(
            database=DatabaseConfig(connection_pool_size=20),
            performance=PerformanceConfig(max_workers=8, cache_size=5_000),
            security=SecurityConfig(
                encrypt_at_rest=True,
                filter_pii=True,
                rate_limit_per_minute=1_000,
            ),
            monitoring=MonitoringConfig(enable_metrics=True, log_level="INFO"),
        )

    @classmethod
    def for_development(cls) -> "UnifiedSettings":
        """Developer‑friendly preset."""
        return cls(
            database=DatabaseConfig(connection_pool_size=5),
            performance=PerformanceConfig(max_workers=2, cache_size=500),
            monitoring=MonitoringConfig(
                enable_metrics=True, log_level="DEBUG", health_check_interval=10
            ),
            api=APIConfig(enable_cors=True),
        )

    # ------------------------------------------------------------------
    def get_database_url(self) -> str:
        """Return an SQLAlchemy‑compatible DSN."""
        return f"sqlite:///{self.database.db_path}"

    # Production sanity‑check -------------------------------------------
    def validate_production_ready(self) -> list[str]:
        """Return a list of issues; empty list means OK."""
        issues: list[str] = []
        if self.security.api_token == "your-secret-token-change-me":
            issues.append("API token is not set")
        if self.security.encrypt_at_rest and not self.security.encryption_key:
            issues.append("Encryption enabled but key is missing")
        if self.performance.max_workers < 4:
            issues.append("Too few workers for production load")
        if not self.monitoring.enable_metrics:
            issues.append("Prometheus metrics should be enabled in production")
        if self.monitoring.log_level.upper() == "DEBUG":
            issues.append("Debug logging must be disabled in production")
        return issues

    # Serialise a redacted summary --------------------------------------
    def get_config_summary(self) -> dict[str, Any]:
        """Return a sanitized settings snapshot suitable for logs."""
        summary = {
            "database": {
                "pool_size": self.database.connection_pool_size,
                "db_exists": self.database.db_path.exists(),
            },
            "model": {"name": self.model.model_name},
            "security": {
                "encrypt_at_rest": self.security.encrypt_at_rest,
                "filter_pii": self.security.filter_pii,
                "has_key": bool(self.security.encryption_key),
            },
            "performance": {
                "max_workers": self.performance.max_workers,
                "cache_size": self.performance.cache_size,
            },
            "api": {
                "port": self.api.port,
                "cors": self.api.enable_cors,
            },
            "monitoring": {
                "metrics": self.monitoring.enable_metrics,
                "rate_limiting": self.monitoring.enable_rate_limiting,
                "level": self.monitoring.log_level,
            },
        }
        return summary

    # Persist / restore --------------------------------------------------
    def save_to_file(self, path: Path) -> None:
        """Persist a redacted config summary as JSON."""
        path.write_text(json.dumps(self.get_config_summary(), indent=2))
        log.info("Configuration saved → %s", path)

    @classmethod
    def load_from_file(cls, path: Path) -> "UnifiedSettings":
        """Load settings from a previously saved JSON summary."""
        return cls(**json.loads(path.read_text()))


###############################################################################
# Factory helper                                                              #
###############################################################################

env_default = os.getenv("ENVIRONMENT", "development").lower()

_ENV_PRESETS = {
    "production": UnifiedSettings.for_production,
    "development": UnifiedSettings.for_development,
    "testing": UnifiedSettings.for_testing,
}


def get_settings(env: str | None = None) -> UnifiedSettings:
    """Return a configured :class:`UnifiedSettings` instance."""
    preset = _ENV_PRESETS.get((env or env_default).lower())
    if preset is None:
        log.warning("Unknown environment '%s'; falling back to default", env)
        return UnifiedSettings()
    return preset()
