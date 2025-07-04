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


# ────────────────────────────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────────────────────────────
def _get_settings(env: Optional[str] = None) -> UnifiedSettings:
    """Get settings for the specified environment."""
    if env:
        os.environ["ENVIRONMENT"] = env
    return UnifiedSettings()


def _print_version():
    """Print version information."""
    console.print(f"[bold green]Unified Memory System v{VERSION}[/bold green]")
    console.print(f"Python {sys.version}")
    console.print(f"Platform: {sys.platform}")


def _print_settings_summary(settings: UnifiedSettings):
    """Print a summary of current settings."""
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


# ────────────────────────────────────────────────────────────────────────
# Server commands
# ────────────────────────────────────────────────────────────────────────
@app_cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    env: Optional[str] = typer.Option(None, "--env", "-e", help="Environment (development/production/testing)"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Log level"),
):
    """Start the Unified Memory System API server."""
    settings = _get_settings(env)
    
    console.print(f"[bold green]Starting Unified Memory System v{VERSION}[/bold green]")
    _print_settings_summary(settings)
    
    # Import here to avoid circular imports
    from memory_system.api.app import create_app
    
    # Override settings if provided
    if host != "0.0.0.0":
        settings.api.host = host
    if port != 8000:
        settings.api.port = port
    
    app = create_app()
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise typer.Exit(1)


# ────────────────────────────────────────────────────────────────────────
# Health and diagnostics
# ────────────────────────────────────────────────────────────────────────
@app_cli.command()
def health(
    env: Optional[str] = typer.Option(None, "--env", "-e", help="Environment"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Check system health and connectivity."""
    settings = _get_settings(env)
    
    console.print("[bold blue]Running health checks...[/bold blue]")
    
    async def _health_check():
        try:
            store = EnhancedMemoryStore(settings)
            health = await store.get_health()
            
            if health.healthy:
                console.print("[green]✓ System is healthy[/green]")
                if verbose:
                    table = Table(title="Health Details")
                    table.add_column("Component", style="cyan")
                    table.add_column("Status", style="green")
                    
                    for component, status in health.checks.items():
                        status_str = "✓ OK" if status else "✗ FAIL"
                        table.add_row(component, status_str)
                    
                    console.print(table)
                    console.print(f"Uptime: {health.uptime} seconds")
                
                await store.close()
                return 0
            else:
                console.print("[red]✗ System is unhealthy[/red]")
                console.print(f"Message: {health.message}")
                await store.close()
                return 1
                
        except Exception as e:
            console.print(f"[red]Health check failed: {e}[/red]")
            return 1
    
    exit_code = asyncio.run(_health_check())
    raise typer.Exit(exit_code)


@app_cli.command()
def stats(
    env: Optional[str] = typer.Option(None, "--env", "-e", help="Environment"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table/json)"),
):
    """Display system statistics."""
    settings = _get_settings(env)
    
    async def _get_stats():
        try:
            store = EnhancedMemoryStore(settings)
            stats = await store.get_stats()
            
            if format == "json":
                import json
                console.print(json.dumps(stats, indent=2))
            else:
                table = Table(title="System Statistics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                
                for key, value in stats.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            table.add_row(f"{key}.{sub_key}", str(sub_value))
                    else:
                        table.add_row(key, str(value))
                
                console.print(table)
            
            await store.close()
            
        except Exception as e:
            console.print(f"[red]Failed to get stats: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_get_stats())


# ────────────────────────────────────────────────────────────────────────
# Configuration commands
# ────────────────────────────────────────────────────────────────────────
@app_cli.command()
def config(
    env: Optional[str] = typer.Option(None, "--env", "-e", help="Environment"),
    show_secrets: bool = typer.Option(False, "--show-secrets", help="Show sensitive values"),
):
    """Display current configuration."""
    settings = _get_settings(env)
    
    console.print(f"[bold blue]Configuration for {settings.profile} environment[/bold blue]")
    _print_settings_summary(settings)
    
    if show_secrets:
        console.print("\n[yellow]Sensitive Configuration:[/yellow]")
        console.print(f"API Token: {settings.security.api_token}")
        console.print(f"Encryption Key: {'Set' if settings.security.encryption_key else 'Not set'}")


@app_cli.command()
def validate_config(
    env: Optional[str] = typer.Option(None, "--env", "-e", help="Environment"),
):
    """Validate configuration for production readiness."""
    settings = _get_settings(env)
    
    console.print("[bold blue]Validating configuration...[/bold blue]")
    
    issues = settings.validate_production_ready()
    
    if not issues:
        console.print("[green]✓ Configuration is production-ready[/green]")
    else:
        console.print("[red]✗ Configuration has issues:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
        raise typer.Exit(1)


# ────────────────────────────────────────────────────────────────────────
# Database commands
# ────────────────────────────────────────────────────────────────────────
@app_cli.command()
def init_db(
    env: Optional[str] = typer.Option(None, "--env", "-e", help="Environment"),
    force: bool = typer.Option(False, "--force", help="Force initialization"),
):
    """Initialize the database."""
    settings = _get_settings(env)
    
    console.print("[bold blue]Initializing database...[/bold blue]")
    
    if settings.database.db_path.exists() and not force:
        console.print("[yellow]Database already exists. Use --force to reinitialize.[/yellow]")
        return
    
    try:
        # Create database directory
        settings.database.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        async def _init_db():
            store = EnhancedMemoryStore(settings)
            await store.close()
        
        asyncio.run(_init_db())
        
        console.print(f"[green]✓ Database initialized at {settings.database.db_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to initialize database: {e}[/red]")
        raise typer.Exit(1)


# ────────────────────────────────────────────────────────────────────────
# Development commands
# ────────────────────────────────────────────────────────────────────────
@app_cli.command()
def dev(
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Enable auto-reload"),
):
    """Start development server with optimal settings."""
    console.print("[bold green]Starting development server...[/bold green]")
    
    serve(
        host="127.0.0.1",
        port=port,
        reload=reload,
        workers=1,
        env="development",
        log_level="debug",
    )


@app_cli.command()
def test(
    coverage: bool = typer.Option(False, "--coverage", "-c", help="Run with coverage"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run the test suite."""
    import subprocess
    
    console.print("[bold blue]Running test suite...[/bold blue]")
    
    cmd = ["pytest"]
    if coverage:
        cmd.extend(["--cov=memory_system", "--cov-report=term-missing"])
    if verbose:
        cmd.append("-v")
    
    try:
        result = subprocess.run(cmd, check=True)
        console.print("[green]✓ Tests passed[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Tests failed with exit code {e.returncode}[/red]")
        raise typer.Exit(e.returncode)


# ────────────────────────────────────────────────────────────────────────
# Version and info commands
# ────────────────────────────────────────────────────────────────────────
@app_cli.command()
def version():
    """Show version information."""
    _print_version()


@app_cli.command()
def info():
    """Show system information."""
    _print_version()
    
    console.print("\n[bold blue]System Information:[/bold blue]")
    
    # Check dependencies
    deps_table = Table(title="Dependencies")
    deps_table.add_column("Package", style="cyan")
    deps_table.add_column("Status", style="green")
    
    required_packages = [
        "fastapi", "uvicorn", "pydantic", "numpy", 
        "sentence_transformers", "faiss", "cryptography"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            status = "✓ Available"
        except ImportError:
            status = "✗ Missing"
        
        deps_table.add_row(package, status)
    
    console.print(deps_table)


# ────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────
def main():
    """Main entry point for the CLI."""
    app_cli()


if __name__ == "__main__":
    main()
