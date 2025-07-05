"""memory_system.config.settings
================================
Runtime configuration for **AI‑memory‑**.

This module provides a single `Settings` object powered by
`pydantic‑settings` (v2) that transparently merges configuration from
**environment variables**, **.env**, **TOML** and **YAML** files. The
loading order (highest → lowest priority):

1. Environment variables (12‑factor first)
2. Values passed via `Settings(...` kwargs)
3. External YAML (``settings.yaml`` or path in ``AI_MEMORY_SETTINGS``)
4. External TOML (``settings.toml`` or path in ``AI_MEMORY_SETTINGS``)
5. ``.env`` file in project root (if present)
6. File‑secrets directory (Kubernetes‑style)

The module also exposes helpers:

* `get_settings()` – cached accessor for DI/tests.
* `configure_logging()` – sets up logging from ``logging.yaml`` and
  honours `LOG_LEVEL_PER_MODULE` for fine‑grained control.

Usage
-----
```python
from memory_system.config.settings import get_settings, configure_logging

settings = get_settings()
configure_logging(settings)
```
"""
from __future__ import annotations

import json
import logging
import logging.config
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Tuple

from pydantic import Field, PositiveInt, SecretStr, constr
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSourceCallable,
)

# ---------------------------------------------------------------------------
# Optional dependencies: TOML / YAML
# ---------------------------------------------------------------------------
try:
    import tomllib  # Python ≥3.11
except ModuleNotFoundError:  # pragma: no cover – fallback for 3.10
    import tomli as tomllib  # type: ignore

try:
    import yaml  # PyYAML (optional)
except ModuleNotFoundError:  # pragma: no cover – runtime warning only
    yaml = None  # type: ignore
    logging.getLogger(__name__).warning(
        "PyYAML not installed – YAML config support disabled.")


# ---------------------------------------------------------------------------
# Helpers to load external config files
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fp:  # tomllib requires binary mode
        return tomllib.load(fp)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not yaml or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

_log_level_type = constr(regex=r"^(CRITICAL|ERROR|WARNING|INFO|DEBUG|NOTSET)$", strip_whitespace=True)

class Settings(BaseSettings):
    """Application runtime settings (validated & type‑safe)."""

    # ---------------------------------------------------------------------
    # Core service
    # ---------------------------------------------------------------------
    debug: bool = Field(False, description="Run service in debug mode")
    host: str = Field("0.0.0.0", description="Bind address")
    port: PositiveInt = Field(8000, description="Bind port")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: _log_level_type = Field(
        "INFO", description="Root log level")
    log_level_per_module: Dict[str, _log_level_type] | None = Field(
        default=None,
        description="Per‑module log levels, e.g. '{\n    \"uvicorn.error\": \"WARNING\"\n}'.",
    )

    # ------------------------------------------------------------------
    # Storage paths
    # ------------------------------------------------------------------
    sqlite_path: str = Field("memory.db", description="SQLite database file")
    vector_store_path: str = Field(
        "memory_vectors.faiss", description="FAISS index path")

    # ------------------------------------------------------------------
    # Security / auth
    # ------------------------------------------------------------------
    jwt_secret: SecretStr = Field(..., description="JWT signing secret")
    encryption_key: SecretStr = Field(..., description="Fernet encryption key")

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------
    compaction_interval_sec: PositiveInt = Field(
        3600, description="Interval for blob compaction job")
    replication_interval_sec: PositiveInt = Field(
        3600, description="Interval for replica sync job")

    # ------------------------------------------------------------------
    # Pydantic settings config
    # ------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_prefix="AI_",  # All env vars start with "AI_"
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # custom sources (TOML / YAML)
    # ------------------------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        init_settings: PydanticBaseSettingsSourceCallable,
        env_settings: PydanticBaseSettingsSourceCallable,
        dotenv_settings: PydanticBaseSettingsSourceCallable,
        file_secret_settings: PydanticBaseSettingsSourceCallable,
    ) -> Tuple[PydanticBaseSettingsSourceCallable, ...]:
        """Define custom loading order with TOML & YAML support."""

        def yaml_settings(_: BaseSettings) -> Dict[str, Any]:
            env_val = os.getenv("AI_MEMORY_SETTINGS")
            path = Path(env_val) if env_val else Path("settings.yaml")
            return _load_yaml(path)

        def toml_settings(_: BaseSettings) -> Dict[str, Any]:
            env_val = os.getenv("AI_MEMORY_SETTINGS")
            path = Path(env_val) if env_val else Path("settings.toml")
            return _load_toml(path)

        # Precedence: ENV → init → YAML → TOML → .env → secrets
        return (
            env_settings,
            init_settings,
            yaml_settings,
            toml_settings,
            dotenv_settings,
            file_secret_settings,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:  # pragma: no cover
    """Return a cached `Settings` instance (singleton‑like)."""
    return Settings()


def configure_logging(settings: Settings | None = None) -> None:  # pragma: no cover
    """Configure logging from ``logging.yaml`` and apply overrides.

    If the YAML file is missing, falls back to ``basicConfig``. Call this
    once at app startup (e.g. in ``main.py``).
    """
    settings = settings or get_settings()
    cfg_path = Path(__file__).resolve().parent.parent / "logging.yaml"
    if cfg_path.exists() and yaml:
        try:
            with cfg_path.open("r", encoding="utf-8") as fp:
                config_dict = yaml.safe_load(fp)
            logging.config.dictConfig(config_dict)
        except Exception:  # pragma: no cover
            logging.basicConfig(level=settings.log_level)
            logging.getLogger(__name__).exception(
                "Failed to load logging.yaml – falling back to basicConfig")
    else:
        logging.basicConfig(level=settings.log_level)

    # fine‑grained overrides
    if settings.log_level_per_module:
        for mod, lvl in settings.log_level_per_module.items():
            logging.getLogger(mod).setLevel(lvl)


# ---------------------------------------------------------------------------
# CLI helper (optional)
# ---------------------------------------------------------------------------

def _cli_preview() -> None:  # pragma: no cover – quick debug helper
    """Print current settings as JSON (for debugging)."""

    import argparse

    parser = argparse.ArgumentParser(description="Print merged settings")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    s = get_settings()
    if args.json:
        print(json.dumps(s.model_dump(), indent=2, default=str))
    else:
        for k, v in s.model_dump().items():
            print(f"{k:30} : {v}")


if __name__ == "__main__":  # pragma: no cover
    _cli_preview()
