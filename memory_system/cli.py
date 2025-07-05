"""cli.py — Command‑line interface for **AI‑memory‑**
===================================================

This module exposes a small **Typer** application that lets you perform
basic operations against the running memory store from the shell.  It is
kept separate from the core library so heavy CLI‑only dependencies do
not pollute production runtimes.

Installation
------------
The extra "cli" introduces the ``typer`` and ``rich`` tread‑offs::

    pip install "ai-memory[cli]"

Usage examples
--------------
::

    # Add a memory snippet
    ai-memory add "Sky above Port‑Arthur was the color of television." \
                 --metadata '{"source": "book", "title": "Neuromancer"}'

    # Search for a fragment
    ai-memory search "television" --limit 3

    # Delete a memory by its UUID
    ai-memory delete 1e638967-0f94-47ad-8f98-e9a3b61b62d3

    # Bulk‑import from a NDJSON file
    ai-memory import-json ./snippets.ndjson

Notes
-----
* All commands run **asynchronously** using ``asyncio.run``.
* The SQLite database location and pool size are read from
  :pymod:`memory_system.config.settings` – override via environment
  variables or config files.
* Errors are rendered with coloured tracebacks using *rich*.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.traceback import install as rich_tb_install

from memory_system.config.settings import get_settings
from memory_system.core.store import Memory, SQLiteMemoryStore

# pretty tracebacks for CLI users
rich_tb_install(show_locals=False)

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------
app = typer.Typer(add_completion=False, rich_markup_mode="rich", no_args_is_help=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_store() -> SQLiteMemoryStore:
    """Create a temporary store instance based on current Settings."""

    settings = get_settings()
    return SQLiteMemoryStore(
        dsn=settings.sqlite_dsn,
        pool_size=settings.sqlite_pool_size,
    )


async def _safe_close(store: SQLiteMemoryStore) -> None:
    """Ensure the store is closed even if an exception bubbles up."""

    try:
        await store.close()
    except Exception:  # pragma: no cover – log & swallow in CLI context
        rprint("[yellow]Warning:[/] failed to close store cleanly.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command(help="Add a single memory snippet to the store.")
def add(
    text: str = typer.Argument(..., help="Textual content of the memory"),
    metadata: Optional[str] = typer.Option(
        None,
        "--metadata",
        "-m",
        help="JSON string with arbitrary key/values.",
    ),
):
    """Insert new :class:`~memory_system.core.store.Memory`."""

    async def _run() -> None:
        store = _get_store()
        data = json.loads(metadata) if metadata else None
        await store.add_memory(text, data)
        await _safe_close(store)
        rprint("[green]✓ Added memory.")

    asyncio.run(_run())


@app.command(help="Perform semantic search across stored memories.")
def search(
    query: str = typer.Argument(..., help="Search query text"),
    limit: int = typer.Option(5, "--limit", "-l", help="Max hits to return"),
):
    """Return up to *limit* memory rows sorted by similarity."""

    async def _run() -> None:
        store = _get_store()
        results = await store.search_memory(query, limit)
        await _safe_close(store)
        if not results:
            rprint("[yellow]No matches found.[/]")
            return
        for mem, score in results:
            rprint(f"[bold]{score:.3f}[/] {mem.text} [dim]{mem.id}[/]")

    asyncio.run(_run())


@app.command(help="Delete a memory row by its UUID.")
def delete(memory_id: str = typer.Argument(..., help="Memory UUID")) -> None:
    async def _run() -> None:
        store = _get_store()
        await store.delete_memory(memory_id)
        await _safe_close(store)
        rprint("[green]✓ Deleted.")

    asyncio.run(_run())


@app.command("import-json", help="Bulk‑import memories from newline‑delimited JSON file.")
def import_json(
    path: Path = typer.Argument(..., exists=True, readable=True, help="NDJSON file path"),
):
    async def _run() -> None:
        store = _get_store()
        imported = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    await store.add_memory(obj["text"], obj.get("metadata"))
                    imported += 1
                except Exception as exc:  # pragma: no cover
                    rprint(f"[red]Skipping line:[/] {exc}")
        await _safe_close(store)
        rprint(f"[green]✓ Imported {imported} memories.[/]")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """CLI entry‑point used by `python -m memory_system.cli`."""

    try:
        app()
    except Exception as exc:
        rprint(f"[red]Error:[/] {exc}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
