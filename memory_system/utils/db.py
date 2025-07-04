"""db.py — SQLite connection pool, query helpers and migration utilities

Version: 0.8‑alpha
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memory_system.utils.metrics import LAT_DB_QUERY, MET_POOL_EXHAUSTED

__all__ = [
    "DatabaseError",
    "ConnectionPool",
    "QueryBuilder",
    "DatabaseMigration",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#                               EXCEPTIONS                                    #
# --------------------------------------------------------------------------- #
class DatabaseError(Exception):
    """Generic database‑layer exception."""


# --------------------------------------------------------------------------- #
#                               CONNECTION POOL                               #
# --------------------------------------------------------------------------- #
class ConnectionPool:
    """Thread‑safe SQLite connection pool with basic health checking."""

    def __init__(self, db_path: Path, pool_size: int = 10, timeout: float = 30.0):
        if pool_size <= 0:
            raise ValueError("Pool size must be positive")

        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout

        self._pool: List[sqlite3.Connection] = []
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

        # Stats counters
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
        self._max_connection_uses = 1_000
        self._max_connection_transactions = 100

        # Make sure directory exists before opening SQLite file.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_pool()

    # ------------------------------------------------------------------ #
    #                         INTERNAL HELPERS                           #
    # ------------------------------------------------------------------ #
    def _initialize_pool(self) -> None:
        for _ in range(self.pool_size):
            conn = self._create_connection()
            self._pool.append(conn)
            self._stats["created"] += 1
            self._track_connection(conn)

    def _create_connection(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=self.timeout,
                isolation_level=None,  # autocommit
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
        except Exception as e:  # noqa: BLE001
            self._stats["errors"] += 1
            raise DatabaseError(f"Failed to create DB connection: {e}") from e

    def _track_connection(self, conn: sqlite3.Connection) -> None:
        cid = id(conn)
        self._connection_ages[cid] = time.monotonic()
        self._connection_usage[cid] = 0
        self._connection_transactions[cid] = 0

    def _untrack_connection(self, conn: sqlite3.Connection) -> None:
        cid = id(conn)
        self._connection_ages.pop(cid, None)
        self._connection_usage.pop(cid, None)
        self._connection_transactions.pop(cid, None)

    def _should_recycle_connection(self, conn: sqlite3.Connection) -> bool:
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
        try:
            conn.execute("SELECT 1").fetchone()
            self._stats["health_checks"] += 1
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ #
    #                         PUBLIC INTERFACE                           #
    # ------------------------------------------------------------------ #
    @contextmanager
    def get_connection(self, *, use_transaction: bool = False):  # noqa: D401
        start = time.monotonic()
        conn: Optional[sqlite3.Connection] = None
        in_tx = False
        try:
            with self._condition:
                deadline = start + self.timeout
                while not self._pool and time.monotonic() < deadline:
                    self._condition.wait(deadline - time.monotonic())

                if self._pool:
                    conn = self._pool.pop()
                    self._stats["borrowed"] += 1
                else:
                    MET_POOL_EXHAUSTED.inc()
                    self._stats["timeouts"] += 1
                    conn = self._create_connection()
                    self._stats["created"] += 1
                    self._track_connection(conn)

            if not self._validate_connection(conn) or self._should_recycle_connection(conn):
                try:
                    conn.close()
                finally:
                    self._untrack_connection(conn)
                conn = self._create_connection()
                self._stats["created"] += 1
                self._stats["recycled"] += 1
                self._track_connection(conn)

            cid = id(conn)
            self._connection_usage[cid] += 1

            if use_transaction:
                conn.execute("BEGIN IMMEDIATE")
                in_tx = True
                self._connection_transactions[cid] += 1

            yield conn

            if in_tx:
                conn.execute("COMMIT")
        except Exception as e:  # noqa: BLE001
            if in_tx and conn is not None:
                try:
                    conn.execute("ROLLBACK")
                    self._stats["transaction_rollbacks"] += 1
                except Exception:  # noqa: BLE001
                    pass
            if isinstance(e, sqlite3.OperationalError):
                self._stats["deadlocks"] += 1
            self._stats["errors"] += 1
            raise DatabaseError(f"Database error: {e}") from e
        finally:
            LAT_DB_QUERY.observe(time.monotonic() - start)
            if conn is None:
                return
            with self._condition:
                if (
                    len(self._pool) < self.pool_size
                    and self._validate_connection(conn)
                    and not self._should_recycle_connection(conn)
                ):
                    self._pool.append(conn)
                    self._stats["returned"] += 1
                    self._condition.notify()
                else:
                    try:
                        conn.close()
                    finally:
                        self._untrack_connection(conn)

    # ---------- high‑level helpers ------------------------------------ #
    def execute_query(self, query: str, params: Tuple | None = None) -> List[sqlite3.Row]:
        if not query.strip():
            raise DatabaseError("Empty query")
        with self.get_connection() as conn:
            try:
                cur = conn.execute(query, params or ())
                rows = cur.fetchall()
                cur.close()
                return rows
            except Exception as e:  # noqa: BLE001
                raise DatabaseError(f"Query failed: {e}") from e

    def execute_update(self, query: str, params: Tuple | None = None) -> int:
        if not query.strip():
            raise DatabaseError("Empty query")
        with self.get_connection(use_transaction=True) as conn:
            try:
                cur = conn.execute(query, params or ())
                affected = cur.rowcount
                cur.close()
                return affected
            except Exception as e:  # noqa: BLE001
                raise DatabaseError(f"Update failed: {e}") from e

    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        if not query.strip():
            raise DatabaseError("Empty query")
        if not params_list:
            return 0
        with self.get_connection(use_transaction=True) as conn:
            try:
                conn.executemany(query, params_list)
                return len(params_list)  # SQLite rowcount unreliable
            except Exception as e:  # noqa: BLE001
                raise DatabaseError(f"Batch failed: {e}") from e

    def execute_script(self, script: str) -> None:
        if not script.strip():
            raise DatabaseError("Empty script")
        with self.get_connection(use_transaction=True) as conn:
            try:
                conn.executescript(script)
            except Exception as e:  # noqa: BLE001
                raise DatabaseError(f"Script failed: {e}") from e

    # ---------- stats / health --------------------------------------- #
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            avg_age = (
                sum(now - ts for ts in self._connection_ages.values()) / len(self._connection_ages)
                if self._connection_ages
                else 0.0
            )
            avg_use = (
                sum(self._connection_usage.values()) / len(self._connection_usage)
                if self._connection_usage
                else 0.0
            )
            avg_tx = (
                sum(self._connection_transactions.values()) / len(self._connection_transactions)
                if self._connection_transactions
                else 0.0
            )
            return {
                "pool_size": len(self._pool),
                "max_pool_size": self.pool_size,
                "avg_connection_age_seconds": avg_age,
                "avg_connection_usage": avg_use,
                "avg_connection_transactions": avg_tx,
                "active_connections": self.pool_size - len(self._pool),
                **self._stats,
            }

    def health_check(self) -> Dict[str, Any]:
        try:
            with self.get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
                integrity = conn.execute("PRAGMA quick_check(1)").fetchone()[0]
                wal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            return {
                "healthy": integrity == "ok",
                "message": "ok" if integrity == "ok" else integrity,
                "wal_mode": wal_mode,
                "pool_stats": self.get_stats(),
            }
        except Exception as e:  # noqa: BLE001
            return {"healthy": False, "message": str(e), "pool_stats": self.get_stats()}

    def close_all(self) -> None:
        with self._lock:
            for c in self._pool:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
            self._pool.clear()
            self._connection_ages.clear()
            self._connection_usage.clear()
            self._connection_transactions.clear()


# --------------------------------------------------------------------------- #
#                              QUERY BUILDER                                  #
# --------------------------------------------------------------------------- #
class QueryBuilder:
    """Lightweight SQL builder with whitelisting."""

    ALLOWED_FILTER_COLUMNS: Dict[str, str] = {
        "role": "role",
        "activity": "activity",
        "language": "language",
        "importance": "importance",
        "tags": "tags",
        "created_at": "created_at",
        "updated_at": "updated_at",
        "deleted": "del",
    }

    ALLOWED_OPERATORS = {"=", "!=", "<", ">", "<=", ">=", "LIKE", "IN", "NOT IN"}

    @staticmethod
    def _escape_like(text: str) -> str:
        return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @staticmethod
    def _valid_name(name: str) -> bool:
        import re

        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))

    @classmethod
    def build_search_query(
        cls,
        base_query: str,
        base_params: Tuple[Any, ...],
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Any]]:
        params: List[Any] = list(base_params)
        conds: List[str] = []
        if filters:
            for key, value in filters.items():
                if key not in cls.ALLOWED_FILTER_COLUMNS:
                    continue
                col = cls.ALLOWED_FILTER_COLUMNS[key]
                if not cls._valid_name(col):
                    continue
                if key in {"role", "activity", "language"} and isinstance(value, str):
                    conds.append(f"AND {col} = ?")
                    params.append(value.strip()[:100])
                elif key == "importance" and isinstance(value, (int, float)) and 0 <= value <= 10:
                    conds.append(f"AND {col} >= ?")
                    params.append(value)
                elif key == "tags" and isinstance(value, list):
                    safe_tags = [t.strip()[:50] for t in value if isinstance(t, str)][:10]
                    tag_conds = []
                    for t in safe_tags:
                        tag_conds.append("tags LIKE ? ESCAPE '\\'")
                        params.append(f"%{cls._escape_like(t)}%")
                    if tag_conds:
                        conds.append("AND (" + " OR ".join(tag_conds) + ")")
        return f"{base_query} {' '.join(conds)}", params

    @staticmethod
    def build_pagination(
        query: str, params: List[Any], offset: int = 0, limit: int = 50
    ) -> Tuple[str, List[Any]]:
        off = max(0, min(offset, 1_000_000))
        lim = max(1, min(limit, 1_000))
        return f"{query} LIMIT ? OFFSET ?", params + [lim, off]

    @staticmethod
    def build_count_query(base_query: str) -> str:
        up = base_query.upper()
        for kw in ("ORDER BY", "LIMIT"):
            pos = up.find(kw)
            if pos != -1:
                base_query = base_query[:pos]
                up = base_query.upper()
        return f"SELECT COUNT(*) AS total FROM ({base_query.strip()})"

    @staticmethod
    def build_safe_update(
        table: str,
        updates: Dict[str, Any],
        where: Dict[str, Any],
    ) -> Tuple[str, List[Any]]:
        if not QueryBuilder._valid_name(table):
            raise DatabaseError("Invalid table name")
        if not updates:
            raise DatabaseError("No updates specified")
        if not where:
            raise DatabaseError("WHERE clause required")

        set_parts, where_parts, params = [], [], []
        for col, val in updates.items():
            if not QueryBuilder._valid_name(col):
                raise DatabaseError(f"Bad column: {col}")
            set_parts.append(f"{col} = ?")
            params.append(val)
        for col, val in where.items():
            if not QueryBuilder._valid_name(col):
                raise DatabaseError(f"Bad column: {col}")
            where_parts.append(f"{col} = ?")
            params.append(val)
        sql = f"UPDATE {table} SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}"
        return sql, params


# --------------------------------------------------------------------------- #
#                              MIGRATION HELPER                               #
# --------------------------------------------------------------------------- #
class DatabaseMigration:
    """Very simple SQL migration runner."""

    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        self._migs: List[Dict[str, Any]] = []

    def add(self, version: str, up_sql: str, down_sql: str = "", desc: str = "") -> None:
        if not version or not up_sql.strip():
            raise DatabaseError("Version and up_sql required")
        self._migs.append({"version": version, "up": up_sql, "down": down_sql, "desc": desc})

    # -- internal helpers -------------------------------------------------
    def _current(self) -> Optional[str]:
        try:
            with self.pool.get_connection() as conn:
                row = conn.execute(
                    "SELECT value FROM system_info WHERE key='schema_version'"
                ).fetchone()
                return row["value"] if row else None
        except Exception:
            return None

    def _set_version(self, v: str) -> None:
        with self.pool.get_connection(use_transaction=True) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_info(key,value,updated_at) VALUES (?,?,?)",
                ("schema_version", v, time.time()),
            )

    # -- public -----------------------------------------------------------
    def migrate_up(self, target: Optional[str] = None) -> List[str]:
        cur = self._current()
        applied: List[str] = []
        for mig in sorted(self._migs, key=lambda m: m["version"]):
            if cur and mig["version"] <= cur:
                continue
            if target and mig["version"] > target:
                break
            with self.pool.get_connection(use_transaction=True) as conn:
                conn.executescript(mig["up"])
            self._set_version(mig["version"])
            applied.append(mig["version"])
            log.info("Applied migration %s", mig["version"])
        return applied

    def migrate_down(self, target: str) -> List[str]:
        cur = self._current()
        if not cur:
            return []
        reverted: List[str] = []
        for mig in sorted(self._migs, key=lambda m: m["version"], reverse=True):
            if mig["version"] > cur:
                continue
            if mig["version"] <= target:
                break
            if not mig["down"]:
                continue
            with self.pool.get_connection(use_transaction=True) as conn:
                conn.executescript(mig["down"])
            reverted.append(mig["version"])
            log.info("Reverted migration %s", mig["version"])
        self._set_version(target)
        return reverted
