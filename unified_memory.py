#!/usr/bin/env python3
"""unified_memory.py — CLI entry point for Unified Memory System

Version: 0.8‑alpha

This single file provides a minimal yet extensible command‑line interface around
*Unified Memory System*.  Typical usage::

    # Start API server on default host/port (127.0.0.1:8000)
    $ python unified_memory.py serve

    # Start API server in production mode (gunicorn‑style workers)
    $ python unified_memory.py serve --production --workers 4 --host 0.0.0.0 --port 80

    # Run a lightweight HTTP health probe on :8080
    $ python unified_memory.py health --port 8080

The file intentionally avoids deep business logic; anything heavy lives inside
``memory_system/`` modules.  This wrapper only wires them up, performs basic
parameter validation and surfaces helpful log messages.
"""
from __future__ import annotations

import argparse
import asyncio as _asyncio
import importlib
import logging
import os
import signal
import sys
from typing import Any, Dict, List

import uvicorn

# ---------------------------------------------------------------------------
# Version & logging
# ---------------------------------------------------------------------------
VERSION = "0.8-alpha"
logging.basicConfig(
    level=os.getenv("UMS_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("ums.cli")

# ---------------------------------------------------------------------------
# Dependency verifier
# ---------------------------------------------------------------------------


def _check_dependencies() -> List[str]:
    """Return a list of missing *critical* Python modules."""
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "numpy": "numpy",
        "sentence_transformers": "sentence-transformers",
        "faiss": "faiss-cpu or faiss-gpu",
    }

    missing: List[str] = []
    for module_name, pip_name in required.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(pip_name)
    return missing


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _setup_signal_handlers() -> None:
    """Intercept SIGINT / SIGTERM to shut down gracefully."""

    def _graceful_exit(signum: int, _frame):
        log.info("Received signal %s — exiting…", signum)
        sys.exit(0)

    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)


# ---------------------------------------------------------------------------
# Sub-command: health
# ---------------------------------------------------------------------------


async def _run_health_server(port: int) -> None:
    """Serve a minimal JSON health response on */health*."""
    from fastapi import FastAPI

    app = FastAPI(title="UMS Health", version=VERSION)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "healthy", "version": VERSION}

    log.info("Starting lightweight health server on 0.0.0.0:%d", port)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


# ---------------------------------------------------------------------------
# Sub-command: serve
# ---------------------------------------------------------------------------


def _serve_api(args: argparse.Namespace) -> None:
    """Start the full FastAPI application via Uvicorn."""
    from memory_system.api.app import create_app

    app = create_app()

    if args.production:
        log.info(
            "Launching API server (production) on %s:%d with %d workers",
            args.host,
            args.port,
            args.workers,
        )
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="info",
        )
    else:
        log.info(
            "Launching API server (development) on %s:%d (reload=%s)",
            args.host,
            args.port,
            bool(args.reload),
        )
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="debug",
        )


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unified-memory",
        description=f"Unified Memory System {VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:\n  ums serve --host 0.0.0.0 --port 80 --production --workers 4\n  ums health --port 8080""",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # serve
    p_serve = sub.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port")
    p_serve.add_argument("--reload", action="store_true", help="Enable code reload")
    p_serve.add_argument("--production", action="store_true", help="Production mode")
    p_serve.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    # health
    p_health = sub.add_parser("health", help="Run a lightweight health probe server")
    p_health.add_argument("--port", type=int, default=8080, help="Health server port")

    return parser


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    _setup_signal_handlers()

    missing = _check_dependencies()
    if missing:
        log.error("Missing dependencies: %s", ", ".join(missing))
        log.error("Install them with: pip install %s", " ".join(missing))
        sys.exit(1)

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        _serve_api(args)
    elif args.command == "health":
        _asyncio.run(_run_health_server(args.port))
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


# Allow `python -m unified_memory` execution
if __name__ == "__main__":
    main()
