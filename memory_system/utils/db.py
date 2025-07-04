"""db.py — SQLite connection pool, query helpers, and migration utilities."""

from __future__ import annotations
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from memory_system.utils.metrics import LAT_DB_QUERY, MET_POOL_EXHAUSTED

__all__ = ["DatabaseError", "ConnectionPool", "QueryBuilder", "DatabaseMigration"]

log = logging.getLogger(__name__)

# Exceptions

class DatabaseError(Exception):
    """Generic database-layer exception."""

# Connection Pool

class ConnectionPool:
    """Thread-safe SQLite connection pool with basic health checking."""
    
    def __init__(self, db_path: Path, pool_size: int = 10, timeout: float = 30.0) -> None:
        if pool_size <= 0:
            raise ValueError("Pool size must be positive")
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout

        self._pool: List[sqlite3.Connection] = []
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

        # Connection statistics
        self._stats: Dict[str, int] = {
            "created": 0,
            "borrowed": 0,
            "returned": 0,
            "errors": 0,
            "health_checks": 0,
            "timeouts": 0,
            "recycled": 0,
            "transaction_rollbacks": 0,
            "deadlocks": 0,
        }
        self._connection_ages: Dict[int, float] = {}
        self._connection_usage: Dict[int, int] = {}
        self._connection_transactions: Dict[int, int] = {}

        self._max_connection_age = 3600  # seconds
        self._max_connection_uses = 1000
        self._max_connection_transactions = 100

        # Ensure database file directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Pre-populate the pool
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Create initial SQLite connections to fill the pool."""
        for _ in range(self.pool_size):
            conn = self._create_connection()
            self._pool.append(conn)
            self._stats["created"] += 1
            self._track_connection(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with appropriate pragmas."""
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=self.timeout,
                isolation_level=None,  # autocommit mode
            )
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                PRAGMA cache_size=10000;
                PRAGMA temp_store=MEMORY;
                PRAGMA mmap_size=268435456;
                PRAGMA foreign_keys=ON;
                PRAGMA busy_timeout=30000;
                PRAGMA wal_autocheckpoint=1000;
                PRAGMA journal_size_limit=67108864;
                PRAGMA secure_delete=ON;
                PRAGMA cache_spill=OFF;
                PRAGMA locking_mode=NORMAL;
                PRAGMA case_sensitive_like=ON;
                """
            )
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            self._stats["errors"] += 1
            raise DatabaseError(f"Failed to create DB connection: {e}") from e

    def _track_connection(self, conn: sqlite3.Connection) -> None:
        """Track a new connection's usage stats."""
        cid = id(conn)
        self._connection_ages[cid] = time.monotonic()
        self._connection_usage[cid] = 0
        self._connection_transactions[cid] = 0

    def _untrack_connection(self, conn: sqlite3.Connection) -> None:
        """Remove tracking of a connection (when closed)."""
        cid = id(conn)
        self._connection_ages.pop(cid, None)
        self._connection_usage.pop(cid, None)
        self._connection_transactions.pop(cid, None)

    def _should_recycle_connection(self, conn: sqlite3.Connection) -> bool:
        """Check if a connection should be recycled (aged out or overused)."""
        cid = id(conn)
        now = time.monotonic()
        if now - self._connection_ages.get(cid, 0.0) > self._max_connection_age:
            return True
        if self._connection_usage.get(cid, 0) > self._max_connection_uses:
            return True
        if self._connection_transactions.get(cid, 0) > self._max_connection_transactions:
            return True
        return False

    def _validate_connection(self, conn: sqlite3.Connection) -> bool:
        """Perform a simple health check on a connection."""
        try:
            conn.execute("SELECT 1").fetchone()
            self._stats["health_checks"] += 1
            return True
        except Exception:
            return False

    @contextmanager
    def get_connection(self, *, use_transaction: bool = False) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager to borrow a SQLite connection from the pool.
        Automatically returns the connection to the pool after use.
        If use_transaction is True, begin a transaction.
        """
        start = time.monotonic()
        conn: Optional[sqlite3.Connection] = None
        try:
            with self._condition:
                deadline = start + self.timeout
                # Wait for available connection
                while not self._pool and time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    self._condition.wait(timeout=remaining)
                if self._pool:
                    conn = self._pool.pop()
                    self._stats["borrowed"] += 1
                else:
                    # Pool exhausted: create a new connection (and mark pool exhaustion)
                    MET_POOL_EXHAUSTED.inc()
                    self._stats["timeouts"] += 1
                    conn = self._create_connection()
                    self._stats["created"] += 1
                    self._track_connection(conn)
            # Optionally start a transaction
            if use_transaction and conn:
                conn.isolation_level = None  # autocommit off if needed
                conn.execute("BEGIN")
            yield conn
            # If we used a transaction, commit it
            if use_transaction and conn:
                conn.commit()
        except Exception as e:
            if conn:
                # On exception, attempt rollback if transaction was active
                try:
                    conn.rollback()
                    self._stats["transaction_rollbacks"] += 1
                except Exception:
                    pass
            log.error("Database error during connection usage: %s", e)
            raise
        finally:
            if conn:
                # Determine if connection should be recycled
                if self._should_recycle_connection(conn):
                    try:
                        conn.close()
                    except Exception:
                        pass
                    finally:
                        self._stats["recycled"] += 1
                        self._untrack_connection(conn)
                        # Replace with a fresh connection
                        conn = self._create_connection()
                        self._stats["created"] += 1
                        self._track_connection(conn)
                # Return connection to pool
                with self._condition:
                    self._pool.append(conn)
                    self._stats["returned"] += 1
                    self._condition.notify()
    
    # Additional methods for executing queries, migrations, etc., would go here.
    # e.g., execute_query, execute_many, and migration handling.
