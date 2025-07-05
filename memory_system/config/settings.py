"""memory_system.config.settings
================================
Centralised runtime configuration for *AI‑memory‑*.

Loads settings from (highest priority → lowest):
1. **Environment variables** (`AI_` prefix)
2. Explicit kwargs when instantiating `Settings(...)`
3. External YAML file                → `$AI_MEMORY_SETTINGS_YAML`
4. External TOML file                → `$AI_MEMORY_SETTINGS_TOML`
5. `.env` file in project root

Both YAML and TOML are optional.  Missing files are skipped silently.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover – fallback for CI images < 3.11
    import tomli as tomllib  # type: ignore

try:
    import yaml
except ImportError:  # optional dependency
    yaml = None  # type: ignore

from pydantic import Field, PositiveInt, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOG = logging.getLogger(__name__)

####################################################################################
# Helper loaders — must live *above* Settings class so customise_sources can refer #
####################################################################################

def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    _LOG.debug("Loading settings from TOML: %s", path)
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    _LOG.debug("Loading settings from YAML: %s", path)
    with path.open("r", encoding="utf‑8") as f:
        return yaml.safe_load(f) or {}


############################
# Main Settings definition #
############################

class Settings(BaseSettings):
    """Application settings validated by *pydantic‑settings*."""

    # ---------------------------------------------------------------------
    # Database / storage
    # ---------------------------------------------------------------------
    db_path: Path = Field(Path("./data/memory.sqlite"), description="Path to SQLite db file")
    faiss_index_path: Path = Field(Path("./data/index.faiss"), description="Vector index file")
    pool_size: PositiveInt = Field(8, description="aiosqlite connection pool size")

    # ---------------------------------------------------------------------
    # Security
    # ---------------------------------------------------------------------
    jwt_secret: SecretStr = Field(..., env="AI_JWT_SECRET", description="JWT signing key")

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------
    log_level: str = Field("INFO", description="Root log level")
    log_json: bool = Field(False, description="Emit logs as JSON")
    log_level_per_module: Dict[str, str] = Field(default_factory=dict)

    # ---------------------------------------------------------------------
    # OpenTelemetry
    # ---------------------------------------------------------------------
    otlp_endpoint: str | None = Field(None, description="OTLP exporter endpoint")

    # Pydantic‑settings config
    model_config = SettingsConfigDict(env_prefix="AI_", env_file=".env", extra="ignore")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def _validate_level(cls, v: str) -> str:  # noqa: D401 – imperative name
        v_upper = v.upper()
        valid = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if v_upper not in valid:
            raise ValueError(f"Invalid log level: {v}")
        return v_upper

    # ------------------------------------------------------------------
    # customise_sources — merge ENV → kwargs → YAML → TOML → .env defaults
    # ------------------------------------------------------------------

    @classmethod
    def settings_customise_sources(cls, init_settings, env_settings, dotenv_settings, file_secret_settings):  # type: ignore[override]
        def yaml_settings():
            path = Path(os.getenv("AI_MEMORY_SETTINGS_YAML", "settings.yaml"))
            return _load_yaml(path)

        def toml_settings():
            path = Path(os.getenv("AI_MEMORY_SETTINGS_TOML", "settings.toml"))
            return _load_toml(path)

        # Order: ENV → kwargs → YAML → TOML → .env → secrets
        return (
            env_settings,
            init_settings,
            yaml_settings,
            toml_settings,
            dotenv_settings,
            file_secret_settings,
        )


############################################
# Public helper to configure Python logging #
############################################

def configure_logging(config: "Settings" | None = None) -> None:  # noqa: D401
    """Load ``logging.yaml`` and apply overrides from *Settings*."""

    import logging.config
    import yaml  # local import: only needed here

    cfg_path = Path(__file__).with_name("..") / ".." / ".." / "logging.yaml"
    cfg_path = cfg_path.resolve()

    with cfg_path.open("r", encoding="utf‑8") as f:
        logging_cfg = yaml.safe_load(f)

    # Apply env overrides
    settings = config or Settings()  # pragma: no cover – default call
    logging_cfg["root"]["level"] = settings.log_level

    if settings.log_json:
        logging_cfg["root"]["handlers"] = ["json_console"]

    # Fine‑grained module levels
    for mod, lvl in settings.log_level_per_module.items():
        logging_cfg.setdefault("loggers", {}).setdefault(mod, {"level": lvl, "handlers": []})

    logging.config.dictConfig(logging_cfg)
    logging.getLogger(__name__).debug("Logging configured (json=%s)", settings.log_json)


################
# Cached getter #
################

_settings_instance: Settings | None = None

def get_settings() -> Settings:
    """Return a cached Settings instance, constructing it lazily."""
    global _settings_instance  # noqa: PLW0603
    if _settings_instance is None:
        _settings_instance = Settings()  # type: ignore[call-arg]
    return _settings_instance
