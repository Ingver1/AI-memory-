"""Configuration module for Unified Memory System.

This module contains all configuration-related classes and utilities.
"""

from __future__ import annotations

__all__ = [
    "UnifiedSettings",
    "get_settings",
    "DatabaseConfig",
    "ModelConfig", 
    "SecurityConfig",
    "PerformanceConfig",
    "ReliabilityConfig",
    "APIConfig",
    "MonitoringConfig",
]

# Lazy imports to avoid heavy dependencies during package import
def __getattr__(name: str):
    if name in ("UnifiedSettings", "get_settings"):
        from memory_system.config.settings import UnifiedSettings, get_settings
        return locals()[name]
    elif name in (
        "DatabaseConfig",
        "ModelConfig",
        "SecurityConfig", 
        "PerformanceConfig",
        "ReliabilityConfig",
        "APIConfig",
        "MonitoringConfig",
    ):
        from memory_system.config.settings import (
            DatabaseConfig,
            ModelConfig,
            SecurityConfig,
            PerformanceConfig,
            ReliabilityConfig,
            APIConfig,
            MonitoringConfig,
        )
        return locals()[name]
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
