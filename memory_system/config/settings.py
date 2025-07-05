"""memory_system.config.settings
================================
Runtime configuration & logging setup for **AI‑memory‑**.

Highlights
----------
* **Multi‑format configuration** – values are loaded in the following order (lowest → highest priority):
  1. ``default.toml`` inside the package (ships sane defaults)
  2. External ``settings.toml`` (cwd or path via ENV ``AI_MEMORY_SETTINGS``)
  3. External ``settings.yaml`` / ``.yml`` (same lookup)
  4. ``.env`` file (dotenv)
  5. Environment variables

* **Strong validation** – ranges via ``PositiveInt``, secrets via ``SecretStr``.
* **Per‑module log levels** – configurable with ``LOG_LEVEL_PER_MODULE`` env (``module=LEVEL[,module=LEVEL...]``).
* **One‑shot singleton accessor** – ``get_settings()`` returns a cached instance in the fastest possible way.

This file uses **Pydantic‑Settings v2** (`pydantic-settings` package).
"""

from __future__ import annotations

import json
import os
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Mapping, MutableMapping

import tomllib  # Python ≥3.11 (fallback provided below)

import yaml  # PyYAML
from pydantic import BaseModel, Field, PositiveInt, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS_FILE = "default.toml"
EXTERNAL_SETTINGS_FILE_NAMES = ("settings.toml", "settings.yaml", "settings.yml")
ENV_SETTINGS_PATH_KEY = "AI_MEMORY_SETTINGS"  # path override

LogLevel = str  # for readability; actual validation done later


def _toml_loader(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text())


def _yaml_loader(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Runtime settings for the whole service."""

    # ---------------------------------------------------------------------
    # Core service options
    # ---------------------------------------------------------------------

    app_name: str = Field("AI‑memory‑", description="Service human‑readable name")
    host: str = Field("0.0.0.0", description="Host interface to bind")
    port: PositiveInt = Field(8000, description="TCP port to expose")
    workers: PositiveInt = Field(1, description="Uvicorn workers count")

    # ---------------------------------------------------------------------
    # Security & crypto
    # ---------------------------------------------------------------------

    fernet_key: SecretStr = Field(..., description="Base64‑encoded Fernet key (32 bytes)")
    allowed_hosts: List[str] = Field(default_factory=list, description="CORS / Host header whitelist")

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------

    log_level: LogLevel = Field("INFO", description="Root log level (DEBUG, INFO…)")
    log_level_per_module: str | None = Field(
        None,
        alias="LOG_LEVEL_PER_MODULE",
        description="Comma‑separated list: mod.path=LEVEL,another.mod=LEVEL",
    )

    # ---------------------------------------------------------------------
    # Background maintenance
    # ---------------------------------------------------------------------

    compaction_interval_seconds: PositiveInt = Field(3600, description="Blob compaction interval")
    replication_interval_seconds: PositiveInt = Field(3600, description="Backup replication interval")
    backup_path: Path = Field(Path("./backups"), description="Local path for replica snapshots")

    # ------------------------------------------------------------------
    # Pydantic Settings config
    # ------------------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AI_MEMORY_",  # e.g. AI_MEMORY_HOST
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def _validate_root_level(cls, v: str) -> str:
        import logging

        if v.upper() not in logging._nameToLevel:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()

    @field_validator("log_level_per_module")
    @classmethod
    def _validate_per_module(cls, v: str | None) -> str | None:
        if v is None:
            return None
        for pair in v.split(","):
            if "=" not in pair:
                raise ValueError("LOG_LEVEL_PER_MODULE must be 'module=LEVEL' list")
        return v

    # ------------------------------------------------------------------
    # Post‑processing
    # ------------------------------------------------------------------

    @property
    def module_log_levels(self) -> Dict[str, str]:
        if not self.log_level_per_module:
            return {}
        result: Dict[str, str] = {}
        for pair in self.log_level_per_module.split(","):
            mod, lvl = pair.split("=", 1)
            result[mod.strip()] = lvl.strip().upper()
        return result

    # ------------------------------------------------------------------
    # Customise sources – TOML/YAML support
    # ------------------------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type["Settings"],
        init_settings: Callable[[BaseSettings], Dict[str, Any]],
        env_settings: Callable[[BaseSettings], Dict[str, Any]],
        file_secret_settings: Callable[[BaseSettings], Dict[str, Any]],
    ) -> Tuple[Callable[[BaseSettings], Dict[str, Any]], ...]:
        """Provide extra sources: default.toml, external toml/yaml."""

        def _load_default(_: BaseSettings) -> Dict[str, Any]:
            path = Path(__file__).with_suffix(".toml").with_name(DEFAULT_SETTINGS_FILE)
            return _toml_loader(path)

        def _load_external(_: BaseSettings) -> Dict[str, Any]:
            # Priority: explicit env var → cwd lookup
            explicit = os.getenv(ENV_SETTINGS_PATH_KEY)
            if explicit:
                p = Path(explicit)
                if p.suffix in {".yaml", ".yml"}:
                    return _yaml_loader(p)
                if p.suffix == ".toml":
                    return _toml_loader(p)
            # Iterate default names
            for name in EXTERNAL_SETTINGS_FILE_NAMES:
                p = Path.cwd() / name
                if p.suffix == ".toml":
                    data = _toml_loader(p)
                else:
                    data = _yaml_loader(p)
                if data:
                    return data
            return {}

        # Order: default.toml < external file < .env / env vars < init kwargs
        return (_load_default, _load_external, env_settings, init_settings, file_secret_settings)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:  # noqa: D401 – short docstring is fine
    """Return *cached* settings instance (singleton)."""

    return Settings()  # type: ignore[arg-type]


# Convenience – configure logging right away when imported as early as possible
if "LOG_CFG_INITIALISED" not in os.environ:
    import logging.config
    from importlib.resources import files

    LOG_CFG_INITIALISED = "LOG_CFG_INITIALISED"

    cfg_path = files(__package__).joinpath("../../logging.yaml").resolve()
    if cfg_path.exists():
        import yaml

        with open(cfg_path, "r") as fh:
            config_dict = yaml.safe_load(fh)
        logging.config.dictConfig(config_dict)

        # Apply per‑module overrides
        s = get_settings()
        for mod, lvl in s.module_log_levels.items():
            logging.getLogger(mod).setLevel(lvl)

        os.environ[LOG_CFG_INITIALISED] = "1"
        
