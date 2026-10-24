# tools/dbtools.py
"""
AMB-Sentinel Database Tools — Production v1.0.0
Author: AMB Team

Overview
--------
This module implements a DatabaseTools system suitable for use as tools registered
into the ToolRegistry used by ai_agent.AIModelService. It is designed to:

- Support multiple database backends (preferred: PostgreSQL with psycopg2; fallback: SQLAlchemy; minimal: sqlite3)
- Manage multiple named connections via DatabaseManager (thread-safe singleton)
- Provide safe, parameterized execution helpers for SELECT/INSERT/UPDATE/DELETE
- Provide DDL helpers: create_table, get_schema, list_tables, get_structure
- Provide health check, transaction management, and arbitrary SQL execution
- Return consistent dict responses like other tools: {"ok": True/False, "result": ..., "error": ...}
- Be resilient: retries, timeouts, detailed errors, minimal attack surface (avoid string formatting raw SQL)
- Easy registration into tools.ToolRegistry via register_db_tools(tool_registry)

Default Database: PostgreSQL (with psycopg2 as primary driver)
Extended Support: MySQL (pymysql), Oracle (cx_Oracle), IBM DB2 (ibm_db_sa)
Fallback: SQLAlchemy for unified interface
Minimal: sqlite3 for development/testing

Notes on safety
----------------
- Use parameterized queries. Never format user input into SQL strings.
- Prefer PostgreSQL with psycopg2 as default. Falls back to SQLAlchemy for other databases.
- Install required drivers: psycopg2-binary (PostgreSQL), pymysql (MySQL), cx_Oracle (Oracle), ibm_db_sa (IBM DB2)
- For production, ensure proper connection pooling and timeout configurations.

Usage (concept)
----------------
from amb_sentinel.tools.dbtools import DatabaseManager, register_db_tools

# create a named connection
DatabaseManager().add_connection("primary", "postgresql://user:pass@host:5432/dbname")

# use tool wrappers registered with ToolRegistry or call DatabaseManager directly:
dbm = DatabaseManager()
dbm.execute("primary", "INSERT INTO foo (a, b) VALUES (%s, %s)", (1, 2))
res = dbm.query("primary", "SELECT * FROM foo WHERE a = %s", (1,))

Author: AMB Team
copyright 2025

"""

from __future__ import annotations
import threading
import time
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import functools
import json

logger = logging.getLogger("DBTools")

# Try primary driver: PostgreSQL with psycopg2
try:
    import psycopg2
    from psycopg2 import sql as _psql_sql, extras
    from psycopg2.pool import ThreadedConnectionPool

    PSYCOPG2_AVAILABLE = True
except Exception:
    PSYCOPG2_AVAILABLE = False
    _psql_sql = None
    extras = None

# Try secondary driver: SQLAlchemy (for unified interface)
try:
    import sqlalchemy
    from sqlalchemy import text, Table, MetaData
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import SQLAlchemyError

    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False
    sqlalchemy = None
    text = None
    Table = None
    MetaData = None
    Engine = None
    SQLAlchemyError = Exception

# Try MySQL driver
try:
    import pymysql

    MYSQL_AVAILABLE = True
except Exception:
    MYSQL_AVAILABLE = False
    pymysql = None

# Try Oracle driver
try:
    import cx_Oracle

    ORACLE_AVAILABLE = True
except Exception:
    try:
        import oracledb as cx_Oracle

        ORACLE_AVAILABLE = True
    except Exception:
        ORACLE_AVAILABLE = False
        cx_Oracle = None

# Try IBM DB2 driver
try:
    import ibm_db_sa
    import ibm_db

    DB2_AVAILABLE = True
except Exception:
    DB2_AVAILABLE = False
    ibm_db_sa = None
    ibm_db = None

# Minimal drivers
import sqlite3


def _mask_uri(uri: str) -> str:
    """Mask password in database URI for safe logging.
    
    postgresql://user:secret@host:5432/db  →  postgresql://user:***@host:5432/db
    """
    if not uri:
        return uri
    return re.sub(r'(://[^:@/]+:)[^@]+(@)', r'\1***\2', uri)


DBTOOLS_AVAILABLE = any([
    PSYCOPG2_AVAILABLE,
    SQLALCHEMY_AVAILABLE,
    MYSQL_AVAILABLE,
    ORACLE_AVAILABLE,
    DB2_AVAILABLE,
    True  # sqlite3 is always available as it's built-in
])

# Minimal retry decorator used in db ops
def retry(times: int = 3, delay: float = 0.5, exceptions: Tuple[type, ...] = (Exception,)):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < times:
                        wait = delay * (2 ** (attempt - 1))
                        logger.warning(f"Retry {attempt}/{times} in {wait:.2f}s for {func.__name__}: {e}")
                        time.sleep(wait)
                    else:
                        logger.error(f"All {times} attempts failed for {func.__name__}: {e}", exc_info=True)
                        raise
            raise last_exc

        return wrapper

    return deco




import hashlib
import collections as _collections


class _QueryCache:
    """Thread-safe LRU query result cache for SELECT queries.
    
    - Only caches SELECT/WITH/EXPLAIN queries (never DML/DDL)
    - Cache key = SHA-256(connection_name + sql + repr(params))
    - Evicts expired entries on access (TTL-based)
    - Thread-safe with RLock
    """

    def __init__(self, maxsize: int = 512, default_ttl: float = 60.0):
        self._maxsize = maxsize
        self._ttl = default_ttl
        self._cache: "_collections.OrderedDict[str, tuple]" = _collections.OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def cache_key(conn: str, sql: str, params) -> str:
        raw = f"{conn}|{sql}|{repr(params)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # Write keywords that must never be cached, even when the top-level
    # statement is SELECT or WITH. PostgreSQL CTEs allow
    # `WITH x AS (DELETE FROM y RETURNING *) SELECT * FROM x` — the
    # top-level form is a SELECT but the CTE body mutates data.
    # `EXPLAIN ANALYZE DELETE ...` similarly executes the statement.
    _WRITE_KEYWORDS = re.compile(
        r"\b(INSERT|UPDATE|DELETE|MERGE|TRUNCATE|DROP|ALTER|CREATE"
        r"|GRANT|REVOKE|CALL|DO|COPY|VACUUM|REINDEX)\b"
    )
    # Match a single-quoted literal, honouring the '' escape form.
    _SQ_LITERAL = re.compile(r"'(?:[^']|'')*'")

    @classmethod
    def is_cacheable(cls, sql: str) -> bool:
        s = sql.strip()
        if not s:
            return False
        up = s.upper()
        if not up.startswith(("SELECT", "WITH", "EXPLAIN")):
            return False
        # Strip string literals first so `WHERE content ILIKE '%DELETE%'`
        # does not trip the write-keyword scan.
        cleaned = cls._SQ_LITERAL.sub("''", s).upper()
        if cls._WRITE_KEYWORDS.search(cleaned):
            return False
        return True

    def get(self, key: str):
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            value, expire_at = self._cache[key]
            if time.monotonic() > expire_at:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value, ttl: float = None) -> None:
        with self._lock:
            exp = time.monotonic() + (ttl if ttl is not None else self._ttl)
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, exp)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate_all(self) -> int:
        with self._lock:
            n = len(self._cache)
            self._cache.clear()
            return n

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


class _QueryMetrics:
    """Per-connection query performance counters."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.errors = 0
        self.total_ms = 0.0
        self._window: _collections.deque = _collections.deque(maxlen=100)

    def record(self, latency_s: float, error: bool = False) -> None:
        with self._lock:
            self.calls += 1
            if error:
                self.errors += 1
            ms = latency_s * 1000
            self.total_ms += ms
            self._window.append(ms)

    def summary(self) -> dict:
        with self._lock:
            n = self.calls
            win = sorted(self._window)
            return {
                "calls": n,
                "errors": self.errors,
                "error_rate": round(self.errors / n, 4) if n else 0.0,
                "avg_ms": round(self.total_ms / n, 2) if n else 0.0,
                "p95_ms": round(win[int(len(win) * 0.95)], 2) if win else 0.0,
            }

class DatabaseConnectionError(Exception):
    pass



# ─────────────────────────────────────────────────────────────────────────────
# SQL IDENTIFIER SAFETY  (v5.0 security hardening)
# ─────────────────────────────────────────────────────────────────────────────

_IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_$]{0,62}$')


def _safe_ident(name: str) -> str:
    """Return a safely double-quoted PostgreSQL identifier.

    Validates that *name* contains only alphanumeric characters, underscores,
    and dollar signs (PostgreSQL's allowed unquoted identifier characters),
    then wraps it in double-quotes for safe interpolation into SQL strings.

    Raises ``ValueError`` for names that could be used for SQL injection.
    """
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError(
            f"Unsafe SQL identifier rejected: {name!r}. "
            "Only letters, digits, underscores and $ are permitted."
        )
    # Double any embedded double-quotes (defence-in-depth; regex above prevents them)
    return '"' + name.replace('"', '""') + '"'


def _safe_filter_sql(filter_sql: Optional[str]) -> Optional[str]:
    """Validate a WHERE-clause fragment supplied to vector_search().

    Only allows simple equality/comparison patterns against metadata fields:
      metadata->>'key' = 'value'
      metadata->>'key' ILIKE '%value%'
      score > 0.5
    Raises ``ValueError`` if any SQL-injection patterns are detected.
    """
    if not filter_sql:
        return None
    # Block dangerous keywords
    _BLOCKED = re.compile(
        r'\b(DROP|DELETE|INSERT|UPDATE|TRUNCATE|CREATE|ALTER|EXEC|EXECUTE'
        r'|UNION|SELECT|INTO|COPY|\\|GRANT|REVOKE)\b',
        re.IGNORECASE
    )
    if _BLOCKED.search(filter_sql):
        raise ValueError(
            f"Potentially unsafe filter_sql rejected: {filter_sql!r}"
        )
    # Max length guard
    if len(filter_sql) > 500:
        raise ValueError("filter_sql exceeds maximum allowed length of 500 characters")
    return filter_sql


class DatabaseManager:
    """
    Thread-safe singleton manager for multiple database connections.

    Responsibilities:
    - Maintain named connections (engines) and their metadata
    - Provide high-level methods for executing parameterized queries
    - Provide schema introspection utilities
    - Provide health checks and lifecycle management (connect, close)

    How it stores connections:
    - Primary: PostgreSQL with psycopg2 (preferred)
    - Secondary: SQLAlchemy Engine objects for unified interface
    - Fallback: Direct DB-API connections for MySQL, Oracle, IBM DB2, sqlite

    Public methods:
    - add_connection(name, uri, **kwargs)
    - remove_connection(name)
    - close_connection(name)
    - execute(name, sql, params=None, fetch="none")  -> fetch="one"|"all"|"none"
    - query(name, sql, params=None) -> {"ok": True, "rows": [...]}
    - insert/update/delete helpers
    - create_table(name, table_sql or columns spec)
    - list_tables(name)
    - get_schema(name, table)
    - get_db_type(name)
    - health_check(name)
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self):
        self._conns: Dict[str, Dict[str, Any]] = {}
        self._manager_lock = threading.RLock()
        # Production additions: query cache + per-connection metrics
        self._query_cache = _QueryCache(maxsize=512, default_ttl=60.0)
        self._conn_metrics: Dict[str, "_QueryMetrics"] = {}

    def _parse_uri(self, uri: str) -> Dict[str, Any]:
        """Parse URI and return connection parameters"""
        # Default to PostgreSQL if no scheme specified
        if not uri.startswith(('postgresql://', 'postgres://', 'mysql://',
                               'oracle://', 'db2://', 'sqlite://')):
            # Assume PostgreSQL
            uri = f"postgresql://{uri}"

        return {"uri": uri, "parsed": True}

    # -----------------------
    # Connection management
    # -----------------------
    def add_connection(self, name: str, uri: str, connect_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a named connection.

        Defaults to PostgreSQL with psycopg2. Supports:
        - PostgreSQL: postgresql://user:pass@host:5432/db
        - MySQL: mysql://user:pass@host:3306/db
        - Oracle: oracle://user:pass@host:1521/service
        - IBM DB2: db2://user:pass@host:50000/db
        - SQLite: sqlite:///path.db or sqlite:///:memory:

        Arguments:
            name: friendly connection name
            uri: connection URI
            connect_kwargs: optional dict passed to driver connect

        Returns:
            {"ok": True} or {"ok": False, "error": "..."}
        """
        connect_kwargs = connect_kwargs or {}
        with self._manager_lock:
            if name in self._conns:
                return {"ok": False, "error": f"connection '{name}' already exists"}

            try:
                parsed = self._parse_uri(uri)
                uri = parsed["uri"]

                # PostgreSQL (primary default)
                if uri.startswith(('postgresql://', 'postgres://')):
                    if PSYCOPG2_AVAILABLE:
                        # Use psycopg2 with connection pooling
                        if "min_conn" in connect_kwargs or "max_conn" in connect_kwargs:
                            min_conn = connect_kwargs.pop("min_conn", 1)
                            max_conn = connect_kwargs.pop("max_conn", 10)
                            pool = ThreadedConnectionPool(
                                min_conn, max_conn, dsn=uri, **connect_kwargs
                            )
                            self._conns[name] = {
                                "type": "postgresql",
                                "pool": pool,
                                "uri": uri,
                                "conn_params": connect_kwargs
                            }
                        else:
                            # Single connection for now (will be pooled on demand)
                            conn = psycopg2.connect(dsn=uri, **connect_kwargs)
                            self._conns[name] = {
                                "type": "postgresql",
                                "conn": conn,
                                "uri": uri,
                                "conn_params": connect_kwargs
                            }
                        logger.info(f"Added PostgreSQL connection '{name}' -> {_mask_uri(uri)}")
                        self._conn_metrics[name] = _QueryMetrics()
                        return {"ok": True}
                    elif SQLALCHEMY_AVAILABLE:
                        # Fallback to SQLAlchemy
                        engine = sqlalchemy.create_engine(uri, pool_pre_ping=True, **connect_kwargs)
                        with engine.connect() as conn:
                            pass
                        self._conns[name] = {"type": "sqlalchemy", "engine": engine, "uri": uri}
                        logger.info(f"Added PostgreSQL via SQLAlchemy connection '{name}' -> {_mask_uri(uri)}")
                        return {"ok": True}
                    else:
                        return {"ok": False, "error": "psycopg2 or SQLAlchemy required for PostgreSQL"}

                # MySQL
                elif uri.startswith('mysql://'):
                    if MYSQL_AVAILABLE and pymysql:
                        conn = pymysql.connect(
                            host=uri.split('@')[1].split(':')[0] if '@' in uri else 'localhost',
                            user=uri.split('://')[1].split(':')[0] if '://' in uri else 'root',
                            password=uri.split('://')[1].split(':')[1].split('@')[0] if '://' in uri and ':' in
                                                                                        uri.split('://')[1] else '',
                            database=uri.split('/')[-1] if '/' in uri else None,
                            **connect_kwargs
                        )
                        self._conns[name] = {"type": "mysql", "conn": conn, "uri": uri}
                        logger.info(f"Added MySQL connection '{name}' -> {_mask_uri(uri)}")
                        return {"ok": True}
                    elif SQLALCHEMY_AVAILABLE:
                        engine = sqlalchemy.create_engine(uri, pool_pre_ping=True, **connect_kwargs)
                        with engine.connect() as conn:
                            pass
                        self._conns[name] = {"type": "sqlalchemy", "engine": engine, "uri": uri}
                        logger.info(f"Added MySQL via SQLAlchemy connection '{name}' -> {_mask_uri(uri)}")
                        return {"ok": True}
                    else:
                        return {"ok": False, "error": "pymysql or SQLAlchemy required for MySQL"}

                # Oracle
                elif uri.startswith('oracle://'):
                    if ORACLE_AVAILABLE and cx_Oracle:
                        # Parse Oracle connection string
                        parts = uri.replace('oracle://', '').split('@')
                        if len(parts) == 2:
                            user_pass = parts[0].split(':')
                            host_service = parts[1]
                            if len(user_pass) == 2:
                                dsn = cx_Oracle.makedsn(
                                    host_service.split(':')[0] if ':' in host_service else host_service,
                                    int(host_service.split(':')[1]) if ':' in host_service else 1521,
                                    service_name=host_service.split('/')[1] if '/' in host_service else host_service
                                )
                                conn = cx_Oracle.connect(
                                    user=user_pass[0],
                                    password=user_pass[1],
                                    dsn=dsn,
                                    **connect_kwargs
                                )
                                self._conns[name] = {"type": "oracle", "conn": conn, "uri": uri}
                                logger.info(f"Added Oracle connection '{name}' -> {_mask_uri(uri)}")
                                return {"ok": True}
                    elif SQLALCHEMY_AVAILABLE:
                        engine = sqlalchemy.create_engine(uri, pool_pre_ping=True, **connect_kwargs)
                        with engine.connect() as conn:
                            pass
                        self._conns[name] = {"type": "sqlalchemy", "engine": engine, "uri": uri}
                        logger.info(f"Added Oracle via SQLAlchemy connection '{name}' -> {_mask_uri(uri)}")
                        return {"ok": True}
                    return {"ok": False, "error": "cx_Oracle/oracledb or SQLAlchemy required for Oracle"}

                # IBM DB2
                elif uri.startswith('db2://'):
                    if DB2_AVAILABLE and ibm_db:
                        conn = ibm_db.connect(uri, "", "", **connect_kwargs)
                        self._conns[name] = {"type": "db2", "conn": conn, "uri": uri}
                        logger.info(f"Added IBM DB2 connection '{name}' -> {_mask_uri(uri)}")
                        return {"ok": True}
                    elif SQLALCHEMY_AVAILABLE:
                        engine = sqlalchemy.create_engine(uri, pool_pre_ping=True, **connect_kwargs)
                        with engine.connect() as conn:
                            pass
                        self._conns[name] = {"type": "sqlalchemy", "engine": engine, "uri": uri}
                        logger.info(f"Added IBM DB2 via SQLAlchemy connection '{name}' -> {_mask_uri(uri)}")
                        return {"ok": True}
                    return {"ok": False, "error": "ibm_db or SQLAlchemy required for IBM DB2"}

                # SQLite
                elif uri.startswith('sqlite://'):
                    if uri == "sqlite:///:memory:" or uri.endswith(":memory:"):
                        dbpath = ":memory:"
                    else:
                        dbpath = uri.split("://", 1)[1]
                    conn = sqlite3.connect(dbpath, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    self._conns[name] = {"type": "sqlite", "conn": conn, "uri": uri}
                    logger.info(f"Added SQLite connection '{name}' -> {_mask_uri(uri)}")
                    return {"ok": True}

                # Default: Try SQLAlchemy if available
                elif SQLALCHEMY_AVAILABLE:
                    engine = sqlalchemy.create_engine(uri, pool_pre_ping=True, **connect_kwargs)
                    with engine.connect() as conn:
                        pass
                    self._conns[name] = {"type": "sqlalchemy", "engine": engine, "uri": uri}
                    logger.info(f"Added generic SQLAlchemy connection '{name}' -> {_mask_uri(uri)}")
                    return {"ok": True}

                else:
                    return {"ok": False, "error": f"Unsupported database URI: {uri}. Install required drivers."}

            except Exception as e:
                logger.exception("Failed to add connection")
                return {"ok": False, "error": str(e)}

    def remove_connection(self, name: str) -> Dict[str, Any]:
        with self._manager_lock:
            if name not in self._conns:
                return {"ok": False, "error": f"connection '{name}' not found"}
            try:
                self.close_connection(name)
                del self._conns[name]
                return {"ok": True}
            except Exception as e:
                logger.exception("remove_connection failed")
                return {"ok": False, "error": str(e)}

    def close_connection(self, name: str) -> Dict[str, Any]:
        with self._manager_lock:
            if name not in self._conns:
                return {"ok": False, "error": f"connection '{name}' not found"}

            info = self._conns[name]
            try:
                conn_type = info.get("type")

                if conn_type == "postgresql":
                    if "pool" in info:
                        info["pool"].closeall()
                    elif "conn" in info:
                        info["conn"].close()
                elif conn_type == "sqlalchemy":
                    info["engine"].dispose()
                elif conn_type == "mysql" and "conn" in info:
                    info["conn"].close()
                elif conn_type == "oracle" and "conn" in info:
                    info["conn"].close()
                elif conn_type == "db2" and "conn" in info:
                    info["conn"].close()
                elif conn_type == "sqlite" and "conn" in info:
                    info["conn"].close()
                else:
                    logger.warning(f"Unknown connection type during close: {conn_type}")

                return {"ok": True}
            except Exception as e:
                logger.exception("close_connection failed")
                return {"ok": False, "error": str(e)}

    def close_all(self) -> Dict[str, Any]:
        """Close every registered connection. Safe to call repeatedly."""
        with self._manager_lock:
            names = list(self._conns.keys())
        results: Dict[str, Any] = {}
        for n in names:
            try:
                results[n] = self.close_connection(n)
            except Exception as exc:
                logger.warning("close_all: %s failed: %s", n, exc)
                results[n] = {"ok": False, "error": str(exc)}
        # Also invalidate the query cache so a partial close cannot leave
        # stale SELECT results behind a fresh run.
        try:
            self._query_cache.invalidate_all()
        except Exception:
            pass
        return {
            "ok": all(r.get("ok") for r in results.values()),
            "closed": results,
        }
    
    # -----------------------
    # Introspection helpers
    # -----------------------
    def get_db_type(self, name: str) -> Dict[str, Any]:
        if name not in self._conns:
            return {"ok": False, "error": f"connection '{name}' not found"}
        # Never return raw URI (contains password) — return masked version only
        raw_uri = self._conns[name].get("uri", "")
        return {"ok": True, "db_type": self._conns[name]["type"], "masked_uri": _mask_uri(raw_uri)}

    # -----------------------
    # Core execution helpers (now using context manager)
    # -----------------------
    class _ConnectionContext:
        """
        Context manager that acquires a raw DB-API connection and returns it
        to the pool (or closes it) on exit.
        """
        def __init__(self, manager: "DatabaseManager", name: str):
            self._manager = manager
            self._name    = name
            self._conn    = None

        def __enter__(self):
            self._conn = self._manager._get_connection(self._name)
            if self._conn is None:
                raise DatabaseConnectionError(
                    f"Could not acquire connection for '{self._name}'"
                )
            return self._conn

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._conn is not None:
                try:
                    if exc_type is None:
                        self._conn.commit()
                    else:
                        self._conn.rollback()
                except Exception:
                    pass
                self._manager._release_connection(self._name, self._conn)
            return False  # do not suppress exceptions

    def connection(self, name: str) -> "_ConnectionContext":
        """
        Return a context manager that yields a raw DB-API connection.
        """
        return self._ConnectionContext(self, name)

    def _get_connection(self, name: str):
        """Get or create a database connection (internal)."""
        if name not in self._conns:
            return None

        info = self._conns[name]
        conn_type = info.get("type")

        if conn_type == "postgresql":
            if "pool" in info:
                return info["pool"].getconn()
            elif "conn" in info:
                return info["conn"]
        elif conn_type in ["mysql", "oracle", "db2", "sqlite"] and "conn" in info:
            return info["conn"]
        elif conn_type == "sqlalchemy":
            return info["engine"].connect()

        return None

    def _release_connection(self, name: str, conn):
        """Release a connection back to pool (internal)."""
        if name not in self._conns:
            return

        info = self._conns[name]
        if info.get("type") == "postgresql" and "pool" in info:
            info["pool"].putconn(conn)

    def execute(self, name: str, sql_text: str,
                params: Optional[Union[Dict[str, Any], List[Any], Tuple[Any, ...]]] = None,
                fetch: str = "none", timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a SQL statement safely using parameterization.

        Retries up to 3 times on transient DB errors (connection drops, deadlocks).
        Returns a consistent dict: {"ok": True/False, "result": ..., "error": ...}

        Security Measures Applied:
        1. Strict input validation
        2. SQL injection prevention through proper parameter binding
        3. SQL statement whitelisting for DDL operations
        4. Restricted SQL keywords in user input
        5. Prepared statements where supported
        6. Timeout enforcement
        7. Connection isolation per thread

        Parameter style varies by database:
        - PostgreSQL: %s
        - MySQL: %s
        - Oracle: :name
        - SQLite: ? or :name
        - IBM DB2: ?

        Arguments:
            sql_text: SQL statement
            params: dict, list, or tuple of values to bind
            fetch: "none" (default), "one", "all"
            timeout: execution timeout in seconds

        Returns:
            {"ok": True, "result": <rows/list/rowcount>} or {"ok": False, "error": "..."}
        """

        # ✅ SECURITY: Validate and sanitize input
        if not sql_text or not isinstance(sql_text, str):
            return {"ok": False, "error": "SQL text must be a non-empty string"}

        # ✅ SECURITY: Strip leading/trailing whitespace and limit SQL length
        sql_text = sql_text.strip()
        if len(sql_text) > 1000000:  # 1MB limit
            return {"ok": False, "error": "SQL text exceeds maximum allowed length"}

        # ✅ SECURITY: Validate fetch parameter
        if fetch not in ("none", "one", "all"):
            return {"ok": False, "error": "fetch must be 'none', 'one', or 'all'"}

        # ✅ SECURITY: Validate timeout
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            return {"ok": False, "error": "timeout must be a positive number"}

        # Validate and sanitize params
        if params:
            if not isinstance(params, (dict, list, tuple)):
                return {"ok": False, "error": "params must be dict, list, or tuple"}

            # Coerce (not strip) param values — see _sanitize_param_value docstring
            if isinstance(params, dict):
                # Validate dict keys are safe Python identifiers
                sanitized_params: Dict[str, Any] = {}
                for key, value in params.items():
                    if not isinstance(key, str) or not key.isidentifier():
                        return {"ok": False, "error": f"Invalid parameter name: {key!r}"}
                    sanitized_params[key] = self._sanitize_param_value(value)
                params = sanitized_params
            else:
                # list / tuple — preserve as list for consistent indexing
                params = [self._sanitize_param_value(p) for p in params]
        else:
            # No params supplied — keep as None so each driver skips param binding
            params = None

        if name not in self._conns:
            return {"ok": False, "error": f"connection '{name}' not found"}

        info = self._conns[name]
        conn_type = info.get("type")

        # ✅ SECURITY: Restrict dangerous SQL operations based on connection type
        # Cache lookup for SELECT queries
        cache_key = None
        if _QueryCache.is_cacheable(sql_text) and fetch in ("one", "all"):
            cache_key = _QueryCache.cache_key(name, sql_text, params)
            import copy
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                return copy.deepcopy(cached)

        if not self._is_safe_sql(sql_text, conn_type):
            return {"ok": False, "error": "Potentially dangerous SQL operation detected"}

        # Use the connection context manager to guarantee release.
        # Retry up to 3 times on transient errors (connection drops, deadlocks).
        _MAX_RETRIES = 3
        _last_err: Exception = Exception("unknown")
        for _attempt in range(1, _MAX_RETRIES + 1):
            t_start = time.time()
            try:
                with self.connection(name) as conn:
                    # Delegate to type-specific execution method
                    if conn_type == "postgresql":
                        result = self._execute_postgresql(conn, sql_text, params, fetch, timeout)
                    elif conn_type == "mysql":
                        result = self._execute_mysql(conn, sql_text, params, fetch, timeout)
                    elif conn_type == "oracle":
                        result = self._execute_oracle(conn, sql_text, params, fetch, timeout)
                    elif conn_type == "db2":
                        result = self._execute_db2(conn, sql_text, params, fetch, timeout)
                    elif conn_type == "sqlite":
                        result = self._execute_sqlite(conn, sql_text, params, fetch, timeout)
                    elif conn_type == "sqlalchemy":
                        result = self._execute_sqlalchemy(conn, sql_text, params, fetch, timeout)
                    else:
                        return {"ok": False, "error": f"unsupported connection type: {conn_type}"}

                # Record latency on success
                if name in self._conn_metrics:
                    self._conn_metrics[name].record(time.time() - t_start)
                # Store in cache if eligible
                if result.get("ok") and cache_key and fetch in ("one", "all"):
                    self._query_cache.put(cache_key, result)
                return result

            except Exception as e:
                _last_err = e
                latency = time.time() - t_start
                if name in self._conn_metrics:
                    self._conn_metrics[name].record(latency, error=True)
                if _attempt < _MAX_RETRIES:
                    wait = 0.5 * (2 ** (_attempt - 1))   # 0.5s, 1s, 2s
                    logger.warning(
                        f"DB execute attempt {_attempt}/{_MAX_RETRIES} failed for "
                        f"'{name}' — retrying in {wait:.1f}s: {e}"
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"DB execute failed after {_MAX_RETRIES} attempts for '{name}': {e}",
                        exc_info=True,
                    )

        return {"ok": False, "error": str(_last_err)}

    # ✅ SECURITY: Add helper methods for each database type (now receiving connection from context)
    def _execute_postgresql(self, conn, sql_text: str, params, fetch: str, timeout: Optional[float]) -> Dict[str, Any]:
        """Execute SQL for PostgreSQL with security enhancements."""
        cur = conn.cursor()
        try:
            if timeout:
                cur.execute("SET statement_timeout = %s", (int(timeout * 1000),))

            if params is None:
                cur.execute(sql_text)
            elif isinstance(params, dict):
                param_names = set(re.findall(r'%\((\w+)\)s', sql_text))
                if param_names and set(params.keys()) != param_names:
                    missing = param_names - set(params.keys())
                    extra = set(params.keys()) - param_names
                    return {
                        "ok": False,
                        "error": f"Parameter mismatch. Missing: {missing}, Extra: {extra}"
                    }
                cur.execute(sql_text, params)
            else:
                # Positional params — validate count
                param_count = sql_text.count('%s')
                if param_count != len(params):
                    return {
                        "ok": False,
                        "error": f"Expected {param_count} positional parameters, got {len(params)}"
                    }
                cur.execute(sql_text, params)

            return self._handle_fetch_result(cur, fetch, conn)

        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _execute_mysql(self, conn, sql_text: str, params, fetch: str, timeout: Optional[float]) -> Dict[str, Any]:
        cur = conn.cursor()
        try:
            if params is None:
                cur.execute(sql_text)
            elif isinstance(params, dict):
                # Convert named %(key)s dict to ordered positional list for MySQL
                param_names = [m.group(1) for m in re.finditer(r'%\((\w+)\)s', sql_text)]
                ordered: list = []
                for pname in param_names:
                    if pname not in params:
                        return {"ok": False, "error": f"Missing parameter: {pname!r}"}
                    ordered.append(params[pname])
                cur.execute(sql_text, ordered)
            else:
                cur.execute(sql_text, params)

            if fetch == "one":
                row = cur.fetchone()
                if row and cur.description:
                    cols = [d[0] for d in cur.description]
                    return {"ok": True, "result": dict(zip(cols, row, strict=False))}
                return {"ok": True, "result": row}
            elif fetch == "all":
                rows = cur.fetchall()
                if rows and cur.description:
                    cols = [d[0] for d in cur.description]
                    return {"ok": True, "result": [dict(zip(cols, row, strict=False)) for row in rows]}
                return {"ok": True, "result": rows or []}
            else:
                conn.commit()
                return {"ok": True, "result": {"rowcount": cur.rowcount}}

        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _execute_oracle(self, conn, sql_text: str, params, fetch: str, timeout: Optional[float]) -> Dict[str, Any]:
        cur = conn.cursor()
        try:
            if params is None:
                cur.execute(sql_text)
            elif isinstance(params, dict):
                param_names = set(re.findall(r':(\w+)', sql_text))
                if param_names and not all(n in params for n in param_names):
                    missing = param_names - set(params.keys())
                    return {"ok": False, "error": f"Missing Oracle parameters: {missing}"}
                cur.execute(sql_text, params)
            else:
                # Oracle does not support positional (?) params natively; pass as sequence
                cur.execute(sql_text, list(params))

            if fetch == "one":
                row = cur.fetchone()
                if row and cur.description:
                    cols = [d[0] for d in cur.description]
                    return {"ok": True, "result": dict(zip(cols, row, strict=False))}
                return {"ok": True, "result": row}
            elif fetch == "all":
                rows = cur.fetchall()
                if rows and cur.description:
                    cols = [d[0] for d in cur.description]
                    return {"ok": True, "result": [dict(zip(cols, row, strict=False)) for row in rows]}
                return {"ok": True, "result": rows or []}
            else:
                conn.commit()
                return {"ok": True, "result": {"rowcount": cur.rowcount}}

        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _execute_db2(self, conn, sql_text: str, params, fetch: str, timeout: Optional[float]) -> Dict[str, Any]:
        try:
            # DB2 uses ? for positional parameters
            if isinstance(params, dict):
                # Convert dict to ordered list
                params = list(params.values())

            stmt = ibm_db.prepare(conn, sql_text)
            if params:
                # ✅ SECURITY: Bind parameters to prevent injection
                for i, param in enumerate(params, 1):
                    ibm_db.bind_param(stmt, i, param)

            if not ibm_db.execute(stmt):
                error = ibm_db.stmt_error()
                return {"ok": False, "error": error}

            if fetch == "one":
                result = ibm_db.fetch_assoc(stmt)
                return {"ok": True, "result": result if result else None}
            elif fetch == "all":
                results = []
                result = ibm_db.fetch_assoc(stmt)
                while result:
                    results.append(result)
                    result = ibm_db.fetch_assoc(stmt)
                return {"ok": True, "result": results}
            else:
                return {"ok": True, "result": {"rowcount": ibm_db.num_rows(stmt)}}

        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _execute_sqlite(self, conn, sql_text: str, params, fetch: str, timeout: Optional[float]) -> Dict[str, Any]:
        cur = conn.cursor()
        try:
            if params is None:
                cur.execute(sql_text)
            elif isinstance(params, dict):
                param_names = set(re.findall(r':(\w+)', sql_text))
                if param_names and not all(n in params for n in param_names):
                    missing = param_names - set(params.keys())
                    return {"ok": False, "error": f"Missing parameters: {missing}"}
                cur.execute(sql_text, params)
            else:
                param_count = sql_text.count('?')
                if param_count != len(params):
                    return {
                        "ok": False,
                        "error": f"Expected {param_count} positional parameters, got {len(params)}"
                    }
                cur.execute(sql_text, params)

            if fetch == "one":
                row = cur.fetchone()
                return {"ok": True, "result": dict(row) if row else None}
            elif fetch == "all":
                rows = cur.fetchall()
                return {"ok": True, "result": [dict(r) for r in rows]}
            else:
                conn.commit()
                return {"ok": True, "result": {"rowcount": cur.rowcount}}

        except Exception as e:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _execute_sqlalchemy(self, conn, sql_text: str, params, fetch: str, timeout: Optional[float]) -> Dict[str, Any]:
        try:
            stmt = text(sql_text)
            execution_options: dict = {}
            if timeout:
                execution_options["timeout"] = timeout

            conn_exec = conn.execution_options(**execution_options) if execution_options else conn
            # SQLAlchemy text() expects a dict or nothing — never a list
            if params is None:
                result = conn_exec.execute(stmt)
            elif isinstance(params, dict):
                result = conn_exec.execute(stmt, params)
            else:
                # Convert positional list → dict using :p0, :p1, ... style
                named = {f"p{i}": v for i, v in enumerate(params)}
                # Rewrite %s placeholders to :p0, :p1 ...
                import re as _re2
                counter = [0]
                def _repl(_m):
                    k = f":p{counter[0]}"
                    counter[0] += 1
                    return k
                rewritten = _re2.sub(r'%s', _repl, sql_text)
                result = conn_exec.execute(text(rewritten), named)

            if fetch == "one":
                row = result.fetchone()
                if row is None:
                    return {"ok": True, "result": None}
                return {"ok": True, "result": dict(row._mapping) if hasattr(row, "_mapping") else dict(row)}
            elif fetch == "all":
                rows = [dict(r._mapping) if hasattr(r, "_mapping") else dict(r) for r in result.fetchall()]
                return {"ok": True, "result": rows}
            else:
                return {"ok": True, "result": {"rowcount": result.rowcount}}

        except Exception:
            conn.rollback()
            raise

    # ✅ SECURITY: Helper method to handle fetch results consistently
    def _handle_fetch_result(self, cursor, fetch: str, connection) -> Dict[str, Any]:
        """Handle fetch results for PostgreSQL with security."""
        try:
            if fetch == "one":
                row = cursor.fetchone()
                if row:
                    if hasattr(cursor, 'description') and cursor.description:
                        cols = [desc[0] for desc in cursor.description]
                        result = dict(zip(cols, row, strict=False))
                    else:
                        result = dict(enumerate(row))
                    return {"ok": True, "result": result}
                return {"ok": True, "result": None}

            elif fetch == "all":
                rows = cursor.fetchall()
                if rows and hasattr(cursor, 'description') and cursor.description:
                    cols = [desc[0] for desc in cursor.description]
                    result = [dict(zip(cols, row, strict=False)) for row in rows]
                else:
                    result = [dict(enumerate(row)) for row in rows] if rows else []
                return {"ok": True, "result": result}

            else:
                connection.commit()
                return {"ok": True, "result": {"rowcount": cursor.rowcount}}

        except Exception as e:
            connection.rollback()
            raise

    # ✅ SECURITY: Add parameter sanitization method
    def _sanitize_param_value(self, value: Any) -> Any:
        """Validate and type-coerce a bound parameter value.

        IMPORTANT: Parameterized queries (psycopg2 %s, sqlite ?, etc.) already
        prevent SQL injection at the driver level — the database driver escapes
        every bound value before it reaches the SQL engine.  Stripping keywords
        like '--', 'DROP', ';' from string values is therefore not only
        unnecessary but actively harmful: it silently corrupts legitimate data
        such as URLs (``https://example.com/path--id``), article text, or any
        content that legitimately contains those character sequences.

        This method therefore only performs safe, non-destructive coercions:
        - None / scalars (int, float, bool) → passed through unchanged
        - str  → truncated at 1 MB (prevents accidental OOM, not injection)
        - bytes → passed through (binary columns)
        - list / tuple → recursively coerced element-wise
        - dict  → recursively coerced value-wise
        - anything else → str(value) (safe fallback for dates, UUIDs, etc.)
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            # Only enforce a generous size cap to prevent accidental OOM.
            # 1 MB is far above any sane DB row value.
            if len(value) > 1_048_576:
                return value[:1_048_576]
            return value
        if isinstance(value, bytes):
            return value
        if isinstance(value, (list, tuple)):
            return [self._sanitize_param_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._sanitize_param_value(v) for k, v in value.items()}
        # Dates, UUIDs, Decimal, etc. — let the driver handle them natively
        return value

    # ── literal-aware SQL safety ──────────────────────────────────────────
    _LITERAL_RE = re.compile(
        r"'(?:[^']|'')*'"                       # 'string literal' ('' = escaped quote)
        r"|\$\$.*?\$\$"                          # $$dollar quoted$$
        r"|\$([A-Za-z_]\w*)\$.*?\$\1\$"          # $tag$dollar quoted$tag$
        r"|--[^\n]*"                             # line comment
        r"|/\*.*?\*/",                           # block comment
        re.S,
    )
    _ESCAPE_STRING_RE = re.compile(r"\b[eE]'")

    @classmethod
    def _strip_sql_literals(cls, sql_text: str) -> str:
        """Blank out string literals and comments so the guard inspects
        SQL *code*, not *data*.

        Without this, a legitimate INSERT whose article text contains e.g.
        ``; police updated`` or ``xp_`` is rejected as an injection.
        Postgres escape strings (E'...\\'...') use backslash rules this
        lexer does not model, so for those we return unchanged and let the
        strict behaviour apply.
        """
        if cls._ESCAPE_STRING_RE.search(sql_text):
            return sql_text
        return cls._LITERAL_RE.sub(" ", sql_text)

    # ✅ SECURITY: Add SQL safety check
    def _is_safe_sql(self, sql_text: str, conn_type: str) -> bool:
        """Check if SQL statement is safe to execute."""
        sql_upper = self._strip_sql_literals(sql_text).upper().strip()

        # Block dangerous SQL patterns
        dangerous_patterns = [
            # Multiple statements (except for PostgreSQL DO blocks)
            (r';.*\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b', 'Multiple statements detected'),
            (r'\bXP_\w+', 'Extended stored procedure'),
            # SQL injection attempts
            (r'UNION\s+SELECT', 'UNION SELECT injection attempt'),
            (r'EXEC\s*\(', 'EXEC() function call'),
            # Schema modification without proper authorization
            (r'DROP\s+(DATABASE|SCHEMA|TABLE|INDEX|VIEW)\s+', 'DROP operation'),
            (r'TRUNCATE\s+TABLE\s+', 'TRUNCATE operation'),
            # replace the ALTER pattern:
            (r'ALTER\s+(DATABASE|SCHEMA)\s+', 'ALTER DATABASE/SCHEMA operation'),
            (r'ALTER\s+TABLE\s+.+\s+DROP\s+COLUMN', 'DROP COLUMN operation'),
            (r'ALTER\s+TABLE\s+.+\s+DROP\s+CONSTRAINT', 'DROP CONSTRAINT operation'),
            # System table access
            #(r'FROM\s+(pg_catalog|information_schema|sys\\.)', 'System catalog access'),
            # File system access
            (r'\b(LOAD_FILE|INTO\s+OUTFILE|INTO\s+DUMPFILE)\b', 'File system access'),
            # Network access
            (r'\b(LOAD_DATA_INFILE|SELECT\s+\.\.\.\s+INTO\s+OUTFILE)\b', 'Network/file access'),
        ]

        import re
        for pattern, reason in dangerous_patterns:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                logger.warning(f"Blocked potentially dangerous SQL: {reason}")
                return False

        # Allow specific safe patterns
        safe_patterns = [
            r'^SELECT\s+',
            r'^INSERT\s+INTO\s+',
            r'^UPDATE\s+',
            r'^DELETE\s+FROM\s+',
            r'^CREATE\s+TABLE\s+',
            r'^CREATE\s+INDEX\s+',
            r'^DROP\s+TABLE\s+',  # Only if explicitly allowed by policy
            r'^ALTER\s+TABLE\s+',  # Only if explicitly allowed by policy
            r'^WITH\s+',  # CTE
            r'^EXPLAIN\s+',
            r'^BEGIN\s*;',
            r'^COMMIT\s*;',
            r'^ROLLBACK\s*;',
            r'^SET\s+',
        ]

        # Check if SQL starts with a safe pattern
        for pattern in safe_patterns:
            if re.match(pattern, sql_upper, re.IGNORECASE):
                return True

        # For DDL operations, require explicit authorization
        ddl_keywords = ['CREATE ', 'DROP ', 'ALTER ', 'TRUNCATE ']
        if any(keyword in sql_upper for keyword in ddl_keywords):
            logger.warning("DDL operation requires explicit authorization")
            return False

        return True  # Default to allowing if no dangerous patterns found

    # Convenience wrappers (unchanged from original)
    def query(self, name: str, sql_text: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.execute(name, sql_text, params=params, fetch="all")

    def query_one(self, name: str, sql_text: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.execute(name, sql_text, params=params, fetch="one")

    def run(self, name: str, sql_text: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.execute(name, sql_text, params=params, fetch="none")

    # -----------------------
    # DDL & Introspection (also use connection context)
    # -----------------------
    def list_tables(self, name: str) -> Dict[str, Any]:
        if name not in self._conns:
            return {"ok": False, "error": "connection not found"}
        info = self._conns[name]
        conn_type = info.get("type")

        try:
            with self.connection(name) as conn:
                if conn_type == "postgresql":
                    cur = conn.cursor()
                    cur.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname NOT IN ('pg_catalog','information_schema')")
                    rows = cur.fetchall()
                    cur.close()
                    return {"ok": True, "result": [row[0] for row in rows]}

                elif conn_type == "mysql":
                    cur = conn.cursor()
                    cur.execute("SHOW TABLES")
                    rows = cur.fetchall()
                    cur.close()
                    return {"ok": True, "result": [row[0] for row in rows]}

                elif conn_type == "oracle":
                    cur = conn.cursor()
                    cur.execute("SELECT table_name FROM user_tables")
                    rows = cur.fetchall()
                    cur.close()
                    return {"ok": True, "result": [row[0] for row in rows]}

                elif conn_type == "db2":
                    cur = conn.cursor()
                    cur.execute("SELECT TABNAME FROM SYSCAT.TABLES WHERE TYPE='T' AND TABSCHEMA = CURRENT SCHEMA")
                    rows = cur.fetchall()
                    cur.close()
                    return {"ok": True, "result": [row[0] for row in rows]}

                elif conn_type == "sqlite":
                    cur = conn.cursor()
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    rows = cur.fetchall()
                    cur.close()
                    return {"ok": True, "result": [row["name"] for row in rows]}

                elif conn_type == "sqlalchemy":
                    inspector = sqlalchemy.inspect(conn)
                    return {"ok": True, "result": inspector.get_table_names()}

                else:
                    return {"ok": False, "error": "unsupported connection type"}
        except Exception as e:
            logger.exception("list_tables failed")
            return {"ok": False, "error": str(e)}

    def get_schema(self, name: str, table: str) -> Dict[str, Any]:
        """Return column metadata for a table"""
        if name not in self._conns:
            return {"ok": False, "error": "connection not found"}
        info = self._conns[name]
        conn_type = info.get("type")

        try:
            with self.connection(name) as conn:
                if conn_type == "sqlalchemy":
                    meta = MetaData()
                    tbl = Table(table, meta, autoload_with=conn)
                    cols = []
                    for c in tbl.columns:
                        cols.append({
                            "name": c.name,
                            "type": str(c.type),
                            "nullable": c.nullable,
                            "default": str(c.default) if c.default else None
                        })
                    return {"ok": True, "result": cols}

                elif conn_type == "postgresql":
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns 
                        WHERE table_name = %s
                        ORDER BY ordinal_position
                    """, (table,))
                    rows = cur.fetchall()
                    cur.close()
                    cols = []
                    for row in rows:
                        cols.append({
                            "name": row[0],
                            "type": row[1],
                            "nullable": row[2] == 'YES',
                            "default": row[3]
                        })
                    return {"ok": True, "result": cols}

                elif conn_type == "mysql":
                    try:
                        safe_tbl = _safe_ident(table)
                    except ValueError as exc:
                        return {"ok": False, "error": str(exc)}
                    cur = conn.cursor()
                    cur.execute(f"DESCRIBE {safe_tbl}")
                    rows = cur.fetchall()
                    cur.close()
                    cols = []
                    for row in rows:
                        cols.append({
                            "name": row[0],
                            "type": row[1],
                            "nullable": row[2] == 'YES',
                            "default": row[4]
                        })
                    return {"ok": True, "result": cols}

                elif conn_type == "oracle":
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT column_name, data_type, nullable, data_default
                        FROM user_tab_columns
                        WHERE table_name = :table
                        ORDER BY column_id
                    """, {"table": table.upper()})
                    rows = cur.fetchall()
                    cur.close()
                    cols = []
                    for row in rows:
                        cols.append({
                            "name": row[0],
                            "type": row[1],
                            "nullable": row[2] == 'Y',
                            "default": row[3]
                        })
                    return {"ok": True, "result": cols}

                elif conn_type == "db2":
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT COLNAME, TYPENAME, NULLS, DEFAULT
                        FROM SYSCAT.COLUMNS
                        WHERE TABNAME = ? AND TABSCHEMA = CURRENT SCHEMA
                        ORDER BY COLNO
                    """, (table.upper(),))
                    rows = cur.fetchall()
                    cur.close()
                    cols = []
                    for row in rows:
                        cols.append({
                            "name": row[0],
                            "type": row[1],
                            "nullable": row[2] == 'Y',
                            "default": row[3]
                        })
                    return {"ok": True, "result": cols}

                elif conn_type == "sqlite":
                    try:
                        safe_tbl = _safe_ident(table)
                    except ValueError as exc:
                        return {"ok": False, "error": str(exc)}
                    cur = conn.cursor()
                    cur.execute(f"PRAGMA table_info({safe_tbl})")
                    rows = cur.fetchall()
                    cur.close()
                    cols = []
                    for row in rows:
                        cols.append({
                            "name": row["name"],
                            "type": row["type"],
                            "nullable": not row["notnull"],
                            "default": row["dflt_value"]
                        })
                    return {"ok": True, "result": cols}

                else:
                    # Generic fallback
                    cur = conn.cursor()
                    cur.execute(f"SELECT * FROM {_safe_ident(table)} LIMIT 0")
                    col_names = [desc[0] for desc in cur.description] if cur.description else []
                    cur.close()
                    return {"ok": True, "result": {"note": "column names only", "columns": col_names}}

        except Exception as e:
            logger.exception("get_schema failed")
            return {"ok": False, "error": str(e)}

    def create_table(self, name: str, ddl_sql: str) -> Dict[str, Any]:
        """
        Create table via DDL SQL. ddl_sql must be a complete CREATE TABLE ... statement.
        Use with caution: do not build DDL by concatenating user input.
        """
        return self.run(name, ddl_sql)

    # -----------------------
    # Transaction helpers (these already use connection context via execute)
    # -----------------------
    def begin(self, name: str) -> Dict[str, Any]:
        # Not needed for DB-API (autocommit is off by default)
        return {"ok": True}

    def commit(self, name: str) -> Dict[str, Any]:
        # The connection context commits automatically on exit, but we provide explicit commit for clarity
        return self.run(name, "COMMIT")

    def rollback(self, name: str) -> Dict[str, Any]:
        return self.run(name, "ROLLBACK")

    # -----------------------
    # Utility / diagnostic
    # -----------------------
    def health_check(self, name: str) -> Dict[str, Any]:
        if name not in self._conns:
            return {"ok": False, "error": "connection not found"}
        try:
            info = self._conns[name]
            conn_type = info.get("type")
            if conn_type == "postgresql":
                r = self.run(name, "SELECT 1")
            elif conn_type == "mysql":
                r = self.run(name, "SELECT 1")
            elif conn_type == "oracle":
                r = self.run(name, "SELECT 1 FROM DUAL")
            elif conn_type == "db2":
                r = self.run(name, "SELECT 1 FROM SYSIBM.SYSDUMMY1")
            elif conn_type in ("sqlite", "sqlalchemy"):
                r = self.run(name, "SELECT 1")
            else:
                return {"ok": False, "error": f"unsupported connection type: {conn_type}"}
            if r.get("ok"):
                return {"ok": True, "db_type": conn_type}
            return {"ok": False, "error": r.get("error")}
        except Exception as e:
            logger.exception("health_check failed")
            return {"ok": False, "error": str(e)}

    # -----------------------
    # pgvector helpers
    # -----------------------

    def create_vector_extension(self, name: str) -> Dict[str, Any]:
        """Enable pgvector extension on the connection (PostgreSQL only)."""
        return self.run(name, "CREATE EXTENSION IF NOT EXISTS vector")

    def create_vector_table(
        self,
        name: str,
        table: str,
        dims: int,
        id_type: str = "TEXT",
        extra_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a pgvector table with an HNSW index.

        Args:
            name:       Connection name.
            table:      Table name.
            dims:       Vector dimensionality (e.g. 1536 for OpenAI ada-002).
            id_type:    SQL type for the id column (default TEXT).
            extra_cols: Optional list of raw SQL column definitions, e.g.
                        ["metadata JSONB", "source TEXT"].
        """
        # Validate identifiers before interpolating into DDL
        try:
            safe_table = _safe_ident(table)
            safe_idx   = _safe_ident(f"{table}_embedding_idx")
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        # id_type must be a plain SQL type keyword — whitelist the common ones
        _ALLOWED_ID_TYPES = {
            "TEXT", "VARCHAR(255)", "UUID", "BIGINT", "INTEGER", "INT", "SERIAL", "BIGSERIAL"
        }
        if id_type.upper() not in _ALLOWED_ID_TYPES:
            return {"ok": False, "error": f"Unsupported id_type {id_type!r}. Use one of: {_ALLOWED_ID_TYPES}"}

        # dims must be a positive integer
        try:
            dims = int(dims)
            if dims <= 0:
                raise ValueError("dims must be positive")
        except (ValueError, TypeError) as exc:
            return {"ok": False, "error": f"Invalid dims: {exc}"}

        extra = ""
        if extra_cols:
            extra = ",\n    " + ",\n    ".join(extra_cols)
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {safe_table} (\n"
            f"    id {id_type} PRIMARY KEY,\n"
            f"    embedding vector({dims}),\n"
            f"    document TEXT{extra}\n"
            f")"
        )
        r = self.run(name, ddl)
        if not r.get("ok"):
            return r
        # Create HNSW index for fast ANN search
        idx_sql = (
            f"CREATE INDEX IF NOT EXISTS {safe_idx} "
            f"ON {safe_table} USING hnsw (embedding vector_cosine_ops)"
        )
        self.run(name, idx_sql)   # best-effort index creation
        return {"ok": True, "table": table, "dims": dims}

    def upsert_vector(
        self,
        name: str,
        table: str,
        id_: str,
        embedding: List[float],
        document: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Insert or update a vector row.

        The table must have columns: id, embedding, document, and optionally metadata (JSONB).
        """
        import json as _json

        has_meta = metadata is not None
        if has_meta:
            sql = (
                f"INSERT INTO {_safe_ident(table)} (id, embedding, document, metadata) "
                "VALUES (%s, %s::vector, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "embedding = EXCLUDED.embedding, document = EXCLUDED.document, "
                "metadata = EXCLUDED.metadata"
            )
            params = (id_, str(embedding), document, _json.dumps(metadata or {}))
        else:
            sql = (
                f"INSERT INTO {_safe_ident(table)} (id, embedding, document) "
                "VALUES (%s, %s::vector, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "embedding = EXCLUDED.embedding, document = EXCLUDED.document"
            )
            params = (id_, str(embedding), document)

        return self.run(name, sql, params)

    def vector_search(
        self,
        name: str,
        table: str,
        query_embedding: List[float],
        top_k: int = 10,
        distance_metric: str = "cosine",
        filter_sql: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Approximate nearest-neighbour search using pgvector.

        Args:
            name:             Connection name.
            table:            Table to search.
            query_embedding:  Query vector (must match table dims).
            top_k:            Number of results to return.
            distance_metric:  "cosine" | "l2" | "inner_product".
            filter_sql:       Optional WHERE clause fragment (parameterized by caller).

        Returns:
            dict with "rows": [{"id", "document", "score", ...}, ...]
        """
        ops = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}
        op  = ops.get(distance_metric, "<=>")
        where = f"WHERE {_safe_filter_sql(filter_sql)}" if filter_sql else ""
        sql = (
            f"SELECT id, document, (embedding {op} %s::vector) AS score "
            f"FROM {_safe_ident(table)} "
            f"{where} "
            f"ORDER BY score "
            f"LIMIT {int(top_k)}"
        )
        return self.query(name, sql, (str(query_embedding),))



    # ─────────────────────────────────────────────────────────────────────
    # PRODUCTION ADDITIONS (v5.0)
    # ─────────────────────────────────────────────────────────────────────

    def bulk_insert(
        self,
        name: str,
        table: str,
        columns: List[str],
        rows: List,
        page_size: int = 1000,
        on_conflict: str = "",
    ) -> Dict[str, Any]:
        """High-throughput multi-row insert.
        
        Uses psycopg2.extras.execute_values() for PostgreSQL (fastest path,
        ~10x faster than individual INSERTs). Falls back to executemany()
        for other backends.
        
        Args:
            name: Connection name
            table: Target table name  
            columns: List of column names
            rows: List of row tuples/lists
            page_size: Rows per batch (default 1000)
            on_conflict: Optional ON CONFLICT suffix
            
        Returns:
            {"ok": True, "inserted": N, "latency_ms": N}
        """
        if name not in self._conns:
            return {"ok": False, "error": f"connection '{name}' not found"}
        if not columns or not rows:
            return {"ok": False, "error": "columns and rows must be non-empty"}
        
        rows_list = list(rows) if not isinstance(rows, list) else rows
        info = self._conns[name]
        ct = info.get("type")
        t0 = time.time()
        
        try:
            if ct == "postgresql" and PSYCOPG2_AVAILABLE and extras:
                col_sql = ", ".join(_safe_ident(c) for c in columns)
                sql = f'INSERT INTO {_safe_ident(table)} ({col_sql}) VALUES %s {on_conflict}'
                with self.connection(name) as conn:
                    cur = conn.cursor()
                    extras.execute_values(cur, sql, rows_list, page_size=page_size)
                    total = cur.rowcount if cur.rowcount >= 0 else len(rows_list)
                    cur.close()
            else:
                col_sql = ", ".join(_safe_ident(c) for c in columns)
                ph = ", ".join(["%s"] * len(columns))
                sql = f'INSERT INTO {_safe_ident(table)} ({col_sql}) VALUES ({ph}) {on_conflict}'
                with self.connection(name) as conn:
                    cur = conn.cursor()
                    cur.executemany(sql, rows_list)
                    total = cur.rowcount if cur.rowcount >= 0 else len(rows_list)
                    cur.close()
            
            # Invalidate cache for this table
            self._query_cache.invalidate_all()
            latency = (time.time() - t0) * 1000
            if name in self._conn_metrics:
                self._conn_metrics[name].record(latency / 1000)
            return {"ok": True, "inserted": total, "latency_ms": round(latency, 2)}
        except Exception as e:
            logger.exception("bulk_insert failed")
            return {"ok": False, "error": str(e)}

    def get_metrics(self, name: str = None) -> Dict[str, Any]:
        """Return query performance metrics for one or all connections.
        
        Returns per-connection stats: call count, error rate, avg/p95 latency.
        Also returns cache hit rate.
        """
        result = {
            "cache": self._query_cache.stats(),
        }
        if name:
            m = self._conn_metrics.get(name)
            result["connections"] = {name: m.summary() if m else {}}
        else:
            result["connections"] = {n: m.summary() for n, m in self._conn_metrics.items()}
        result["ok"] = True
        return result

    def invalidate_cache(self) -> Dict[str, Any]:
        """Flush the entire query result cache."""
        n = self._query_cache.invalidate_all()
        return {"ok": True, "invalidated": n}

    def stream_query(
        self,
        name: str,
        sql_text: str,
        params=None,
        chunk_size: int = 500,
    ):
        """Generator yielding result rows in chunks for large queries.
        
        Uses server-side cursor on PostgreSQL for true streaming (avoids
        loading all rows into memory). Falls back to batch slicing for
        other backends.
        
        Usage::
        
            for chunk in dbm.stream_query("primary", "SELECT * FROM raw_data"):
                for row in chunk:
                    process(row)
        """
        if name not in self._conns:
            raise DatabaseConnectionError(f"connection '{name}' not found")
        
        info = self._conns[name]
        ct = info.get("type")
        params = params or ()
        
        if ct == "postgresql" and PSYCOPG2_AVAILABLE:
            cursor_name = f"stream_{id(self)}_{int(time.time())}"
            with self.connection(name) as conn:
                with conn.cursor(name=cursor_name) as cur:
                    cur.itersize = chunk_size
                    cur.execute(sql_text, params)
                    cols = [d[0] for d in cur.description] if cur.description else []
                    while True:
                        batch = cur.fetchmany(chunk_size)
                        if not batch:
                            break
                        yield [dict(zip(cols, row, strict=False)) for row in batch]
        else:
            # Fallback: execute all, yield in slices
            result = self.execute(name, sql_text, params=params, fetch="all")
            if not result.get("ok"):
                return
            data = result.get("result") or []
            for i in range(0, len(data), chunk_size):
                yield data[i : i + chunk_size]


# -----------------------
# Tool registration helper
# -----------------------
def register_db_tools(tool_registry, manager: Optional[DatabaseManager] = None,
                      config: Optional[Dict[str, Any]] = None):
    """
    Register a set of database tools into a ToolRegistry.

    Tools registered:
      - db_add_connection(name, uri, connect_kwargs=None)
      - db_remove_connection(name)
      - db_close_connection(name)
      - db_get_db_type(name)
      - db_execute(name, sql, params=None, fetch='none')
      - db_query(name, sql, params=None)
      - db_query_one(name, sql, params=None)
      - db_list_tables(name)
      - db_get_schema(name, table)
      - db_create_table(name, ddl_sql)
      - db_begin(name)
      - db_commit(name)
      - db_rollback(name)
      - db_health_check(name)

    Each tool returns {"ok": True/False, "result": ... } consistent with other tools.

    Parameters
    ----------
    tool_registry : ToolRegistry
    manager : DatabaseManager, optional
        Reuse an existing manager. Defaults to the process singleton.
    config : dict, optional
        The parsed config.yaml. When provided, forwards
        `data_analyzer.raw_data_table` and `ai_service.store.min_chars`
        into the `store_raw_item` closure so the tool reads its
        thresholds from configuration rather than defaults.
    """
    manager = manager or DatabaseManager()

    # Apply config *only* when the caller passed it in — no silent
    # singleton mutation from other modules.
    if config:
        da = (config.get("data_analyzer") or {})
        ai = (config.get("ai_service") or {})
        manager.raw_data_table = da.get("raw_data_table", "raw_data")
        manager.raw_min_chars = int((ai.get("store") or {}).get("min_chars", 1))

    def _wrap(f):
        @functools.wraps(f)
        def _inner(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                logger.exception("db tool wrapper error")
                return {"ok": False, "error": str(e)}

        return _inner

    @_wrap
    def db_add_connection(name: str, uri: str, connect_kwargs: Optional[Dict[str, Any]] = None):
        return manager.add_connection(name, uri, connect_kwargs=connect_kwargs)

    @_wrap
    def db_remove_connection(name: str):
        return manager.remove_connection(name)

    @_wrap
    def db_close_connection(name: str):
        return manager.close_connection(name)

    @_wrap
    def db_get_db_type(name: str):
        return manager.get_db_type(name)

    @_wrap
    def db_execute(name: str, sql: str, params: Optional[Dict[str, Any]] = None, fetch: str = "none",
                   timeout: Optional[float] = None):
        return manager.execute(name, sql, params=params, fetch=fetch, timeout=timeout)

    @_wrap
    def db_query(name: str, sql: str, params: Optional[Dict[str, Any]] = None):
        return manager.query(name, sql, params=params)

    @_wrap
    def db_query_one(name: str, sql: str, params: Optional[Dict[str, Any]] = None):
        return manager.query_one(name, sql, params=params)

    @_wrap
    def db_list_tables(name: str):
        return manager.list_tables(name)

    @_wrap
    def db_get_schema(name: str, table: str):
        return manager.get_schema(name, table)

    @_wrap
    def db_create_table(name: str, ddl_sql: str):
        return manager.create_table(name, ddl_sql)

    @_wrap
    def db_begin(name: str):
        return manager.begin(name)

    @_wrap
    def db_commit(name: str):
        return manager.commit(name)

    @_wrap
    def db_rollback(name: str):
        return manager.rollback(name)

    @_wrap
    def db_health_check(name: str):
        return manager.health_check(name)

    # Register into registry
    tool_registry.register_function(db_add_connection, name="db_add_connection",
                                    description="Add a named DB connection (uri)", timeout=30, retries=1)
    tool_registry.register_function(db_remove_connection, name="db_remove_connection",
                                    description="Remove a named DB connection", timeout=30, retries=1)
    tool_registry.register_function(db_close_connection, name="db_close_connection",
                                    description="Close a named DB connection", timeout=30, retries=1)
    tool_registry.register_function(db_get_db_type, name="db_get_db_type",
                                    description="Get DB type and uri for connection", timeout=5, retries=0)
    tool_registry.register_function(db_execute, name="db_execute",
                                    description="Execute SQL (parameterized). fetch='none'|'one'|'all'", timeout=300,
                                    retries=1)
    tool_registry.register_function(db_query, name="db_query",
                                    description="Execute SELECT and return all rows", timeout=300, retries=1)
    tool_registry.register_function(db_query_one, name="db_query_one",
                                    description="Execute SELECT and return one row", timeout=300, retries=1)
    tool_registry.register_function(db_list_tables, name="db_list_tables",
                                    description="List tables in the database", timeout=10, retries=0)
    tool_registry.register_function(db_get_schema, name="db_get_schema",
                                    description="Get schema/columns for a table", timeout=10, retries=0)
    tool_registry.register_function(db_create_table, name="db_create_table",
                                    description="Run CREATE TABLE DDL", timeout=60, retries=0)
    tool_registry.register_function(db_begin, name="db_begin",
                                    description="Begin transaction (SQLAlchemy engines only)", timeout=5, retries=0)
    tool_registry.register_function(db_commit, name="db_commit",
                                    description="Commit transaction", timeout=5, retries=0)
    tool_registry.register_function(db_rollback, name="db_rollback",
                                    description="Rollback transaction", timeout=5, retries=0)
    tool_registry.register_function(db_health_check, name="db_health_check",
                                    description="Simple DB health check (SELECT 1) + metrics", timeout=5, retries=0)

    # ── New production tools ──────────────────────────────────────────────

    @_wrap
    def store_raw_item(
        content: str,
        source_type: str = "news",
        source_url: str = "",
        title: str = "",
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None,
        name: str = "primary",
    ):
        """Store ONE collected item in raw_data (analyzed=FALSE).

        Computes sha256 of content, skips exact duplicates, tags the row
        with the current agent run id. Returns
        ``{"ok", "inserted": 1|0, "duplicate": bool, "id", "content_hash", "rows"}``.

        Pass content as a handle like ``"$r4.text_content"`` — the agent
        runtime resolves handles before calling.
        """
        import hashlib as _hl
        from amb_sentinel.observability import current_run_id
        table = getattr(manager, "raw_data_table", "raw_data")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(table)):
            return {"ok": False, "error": f"unsafe raw_data table name {table!r}"}
        text = (content or "").strip()
        min_chars = int(getattr(manager, "raw_min_chars", 1))
        if len(text) < max(1, min_chars):
            return {
                "ok": False,
                "error": f"content too short ({len(text)} < {max(1, min_chars)} chars)",
                "inserted": 0,
            }
        digest = _hl.sha256(text.encode("utf-8")).hexdigest()
        run_id = current_run_id() or None
        meta = dict(metadata or {})
        if title:
            meta["title"] = title
        r = manager.execute(
            name,
            f'INSERT INTO "{table}" '
            f'(source_type, source_url, content_type, content, content_hash, '
            f'metadata, collected_by_agent, agent_execution_id, '
            f'collection_method, file_size) '
            f'VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s) '
            f'ON CONFLICT (content_hash) DO NOTHING RETURNING id',
            (
                str(source_type)[:100], source_url or None, str(content_type)[:100],
                text, digest, json.dumps(meta, default=str),
                getattr(manager, "collected_by", "amb-sentinel"), run_id,
                "agent", len(text.encode("utf-8")),
            ),
            fetch="one",
        )
        if not r.get("ok"):
            return {
                "ok": False, "error": r.get("error", "insert failed"),
                "inserted": 0,
            }
        row = r.get("result")
        if row:
            return {
                "ok": True, "inserted": 1, "duplicate": False,
                "id": row.get("id"), "content_hash": digest, "rows": 1,
            }
        ex = manager.execute(
            name,
            f'SELECT id FROM "{table}" WHERE content_hash = %s',
            (digest,), fetch="one",
        )
        return {
            "ok": True, "inserted": 0, "duplicate": True,
            "id": (ex.get("result") or {}).get("id") if ex.get("ok") else None,
            "content_hash": digest, "rows": 1,
        }

    tool_registry.register_function(
        store_raw_item, name="store_raw_item",
        description=(
            "Store one collected item in raw_data (hashes + de-duplicates; "
            "no SQL needed)"
        ),
        timeout=30, retries=0,
    )

    @_wrap
    def db_bulk_insert(name: str, table: str, columns: List[str], rows: List,
                       page_size: int = 1000, on_conflict: str = ""):
        """High-throughput multi-row insert using execute_values."""
        return manager.bulk_insert(name, table, columns, rows, page_size=page_size, on_conflict=on_conflict)

    @_wrap
    def db_get_metrics(name: Optional[str] = None):
        """Query performance metrics + cache hit rate for connections."""
        return manager.get_metrics(name)

    @_wrap
    def db_invalidate_cache():
        """Flush the entire query result cache immediately."""
        return manager.invalidate_cache()

    @_wrap
    def db_stream_query(name: str, sql: str, params: Optional[Dict] = None, chunk_size: int = 500):
        """Stream large result sets; returns all rows merged from chunks."""
        chunks = list(manager.stream_query(name, sql, params=params, chunk_size=chunk_size))
        all_rows = [r for chunk in chunks for r in chunk]
        return {"ok": True, "result": all_rows, "total": len(all_rows)}

    tool_registry.register_function(db_bulk_insert, name="db_bulk_insert",
                                    description="High-throughput multi-row insert (execute_values)", timeout=300, retries=1)
    tool_registry.register_function(db_get_metrics, name="db_get_metrics",
                                    description="Query performance metrics and cache stats", timeout=5, retries=0)
    tool_registry.register_function(db_invalidate_cache, name="db_invalidate_cache",
                                    description="Flush the query result cache", timeout=5, retries=0)
    tool_registry.register_function(db_stream_query, name="db_stream_query",
                                    description="Stream large result sets in chunks (avoids OOM)", timeout=300, retries=1)

    logger.info("Registered database tools in ToolRegistry")
    # at end of register_db_tools, before return
    globals().setdefault("store_raw_item", store_raw_item)
    return {"ok": True}
