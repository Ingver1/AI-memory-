"""memory_system.config.settings
================================
Elegant, single-source runtime configuration for **AI-memory-**.

This module exposes a single :class:`Settings` object (based on
*Pydantic-Settings* v2) that transparently merges configuration from several
sources – highest priority first:

1. **Environment variables** with the ``AI_`` prefix (12-factor first)
2. Explicit keyword arguments when instantiating ``Settings(...)``
3. External YAML file pointed to by ``AI_SETTINGS_YAML``
4. External TOML file pointed to by ``AI_SETTINGS_TOML``
5. A local ``.env`` file in the project root

It also provides :func:`configure_logging` – a helper that selects either the
*plaintext* or *JSON* root handler depending on the ``LOG_JSON`` environment
variable.

"""

from __future__ import annotations

import json
import logging.config
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Optional config-file parsers                                                #
# --------------------------------------------------------------------------- #

try:  # TOML support: stdlib on 3.11+, otherwise fall back to tomli
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

try:  # YAML support (optional)
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore

# --------------------------------------------------------------------------- #
# Pydantic                                                                    #
# --------------------------------------------------------------------------- #

from pydantic import Field, PositiveInt, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Helpers to load external config files                                       #
# --------------------------------------------------------------------------- #


def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fp:  # tomllib expects *binary* mode
        return tomllib.load(fp) or {}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


# --------------------------------------------------------------------------- #
# Base settings class                                                         #
# --------------------------------------------------------------------------- #


class UnifiedSettings(BaseSettings):
    """Application settings loaded from env, optional files and defaults."""

    # ------------------------------------------------------------------ Core #
    host: str = Field("0.0.0.0", description="Bind address for Uvicorn")
    port: PositiveInt = Field(8000, description="Listening port")
    reload: bool = Field(
        False, description="Reload on code changes (development only)"
    )

    # -------------------------------------------------------------- Database #
    sqlite_dsn: str = Field("sqlite:///memory.db", description="SQLite DSN")
    pool_size: PositiveInt = Field(
        5, description="`aiosqlite` connection-pool size"
    )

    # ------------------------------------------------------------- Security #
    jwt_secret: SecretStr = Field(
        ..., description="JWT signing / verification key"
    )

    # ----------------------------------------------------------- Version 🏷️ #
    version: str = Field(
        "0.8.0a0",  # PEP-440
        description="Application version",
    )

    # ----------------------------------------------------------- Logging 🪵 #
    log_level: str = Field("INFO", description="Root log level")
    log_level_per_module: str = Field(
        "",
        description="Comma-separated overrides, e.g. "
        "'memory_system.core=DEBUG,uvicorn.error=WARNING'",
    )

    # ------------------------------------------------------ OpenTelemetry 🌐 #
    otlp_endpoint: str = Field(
        "", description="OTLP collector endpoint (empty → disabled)"
    )

    # ---------------------------------------------------------------- model #
    model_config = SettingsConfigDict(env_prefix="AI_", env_file=".env")

    # -------------------------------------------------------- Custom parser #
    @field_validator("log_level_per_module", mode="before")
    @classmethod
    def _parse_levels(cls, raw: str | List[str]) -> Dict[str, str]:
        pairs: list[str] = raw.split(",") if isinstance(raw, str) else raw
        out: dict[str, str] = {}
        for pair in pairs:
            if "=" not in pair:
                continue
            module, level = (s.strip() for s in pair.split("=", 1))
            if module and level:
                out[module] = level.upper()
        return out


# --------------------------------------------------------------------------- #
# Helper to pick JSON vs plaintext logging                                    #
# --------------------------------------------------------------------------- #


def configure_logging(settings: Settings) -> None:
    """Apply ``logging.yaml`` and switch to JSON if ``LOG_JSON=1``."""
    from importlib import resources

    import yaml  # safe: we only reach here if pyyaml is installed

    cfg_path = (
        resources.files("memory_system")
        / "config"
        / "logging.yaml"  # type: ignore[arg-type]
    )
    with cfg_path.open("r", encoding="utf-8") as fp:
        logging_cfg = yaml.safe_load(fp)

    if os.getenv("LOG_JSON") == "1":
        # Swap handlers & formatter to JSON
        logging_cfg["root"]["handlers"] = ["json_console"]

    # Apply per-module overrides
    for module, level in settings.log_level_per_module.items():
        logging_cfg.setdefault("loggers", {}).setdefault(module, {})[
            "level"
        ] = level

    logging.config.dictConfig(logging_cfg)


# --------------------------------------------------------------------------- #
# Public accessor (singleton-style, lazy)                                     #
# --------------------------------------------------------------------------- #

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (lazy singleton)."""
    global _settings
    if _settings is None:
        # Customise source order: env → kwargs → YAML → TOML → .env
        def _settings_customise_sources(_: Any) -> Tuple[Callable[..., Dict[str, Any]], ...]:  # noqa: ANN401
            return (
                BaseSettings.SettingsConfigDict.env_settings,
                BaseSettings.SettingsConfigDict.init_settings,
                _yaml_source,
                _toml_source,
                BaseSettings.SettingsConfigDict.file_secret_settings,
            )

        def _yaml_source(_: Settings) -> Dict[str, Any]:
            path = Path(os.getenv("AI_SETTINGS_YAML", ""))
            return _load_yaml(path)

        def _toml_source(_: Settings) -> Dict[str, Any]:
            path = Path(os.getenv("AI_SETTINGS_TOML", ""))
            return _load_toml(path)

        Settings.settings_customise_sources = _settings_customise_sources  # type: ignore[attr-defined]
        _settings = UnifiedSettings()  # type: ignore[call-arg]

    return _settings
