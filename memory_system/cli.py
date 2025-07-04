#!/usr/bin/env python3
"""Command-line interface for Unified Memory System v0.8-alpha."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from memory_system.config.settings import UnifiedSettings
from memory_system.core.store import EnhancedMemoryStore

log = logging.getLogger("ums.cli")
console = Console()

app_cli = typer.Typer(
    name="unified-memory",
    help="Unified Memory System CLI",
    add_completion=False,
    rich_markup_mode="rich",
)

VERSION = "0.8-alpha"

# ────────────────────────── Utility Functions ──────────────────────────

def _get_settings(env: Optional[str] = None) -> UnifiedSettings:
    """Load UnifiedSettings for the specified environment profile."""
    if env:
        os.environ["ENVIRONMENT"] = env
    return UnifiedSettings()

def _print_version() -> None:
    """Print version information to the console."""
    console.print(f"[bold green]Unified Memory System v{VERSION}[/bold green]")
    console.print(f"Python {sys.version}")
    console.print(f"Platform: {sys.platform}")

def _print_settings_summary(settings: UnifiedSettings) -> None:
    """Print a summary of the current configuration settings."""
    table = Table(title="Configuration Summary")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Environment", settings.profile)
    table.add_row("Database Path", str(settings.database.db_path))
    table.add_row("API Port", str(settings.api.port))
    table.add_row("Workers", str(settings.performance.max_workers))
    table.add_row("Cache Size", str(settings.performance.cache_size))
    table.add_row("Metrics Enabled", str(settings.monitoring.enable_metrics))
    table.add_row("Encryption", str(settings.security.encrypt_at_rest))
    table.add_row("PII Filtering", str(settings.security.filter_pii))
    console.print(table)

# ────────────────────────── Server Commands ───────────────────────────

@app_cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host address to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve on"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload for development"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    env: Optional[str] = typer.Option(None, "--env", "-e", help="Environment (development/production/testing)"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Log level for the server"),
) -> None:
    """Start the Unified Memory System API server."""
    settings = _get_settings(env)
    console.print(f"[bold green]Starting Unified Memory System v{VERSION}[/bold green]")
    _print_settings_summary(settings)

    # Import here to avoid circular imports
    from memory_system.api.app import create_app

    # Override settings if provided via CLI options
    if host and host != "0.0.0.0":
        settings.api.host = host
    if port and port != 8000:
        settings.api.port = port

    app = create_app()

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level.lower(),
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise typer.Exit(code=1)

# ──────────────────────── Health and Diagnostics ───────────────────────

@app_cli.command()
def health(
    env: Optional[str] = typer.Option(None, "--env", "-e", help="Environment profile for settings"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed health info"),
) -> None:
    """Run health checks and connectivity diagnostics."""
    settings = _get_settings(env)
    console.print("[bold blue]Running health checks...[/bold blue]")

    async def _health_check() -> None:
        try:
            store = EnhancedMemoryStore(settings)
            health_status = await store.get_health()
            if health_status.healthy:
                console.print("[green]✓ System is healthy[/green]")
                if verbose:
                    table = Table(title="Health Details")
                    table.add_column("Component", style="cyan")
                    table.add_column("Status", style="green")
                    for component, ok in health_status.checks.items():
                        status_str = "OK" if ok else "FAIL"
                        table.add_row(component, status_str)
                    console.print(table)
            else:
                console.print("[yellow]System is degraded[/yellow]")
                for comp, ok in health_status.checks.items():
                    if not ok:
                        console.print(f"[red]✗ {comp} not healthy[/red]")
        except Exception as e:
            console.print(f"[red]Health check failed: {e}[/red]")
            raise typer.Exit(code=1)
        finally:
            # Ensure store is closed to release resources
            await store.close()

    # Run the asynchronous health check
    asyncio.run(_health_check())
