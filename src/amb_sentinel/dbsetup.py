"""amb-db-setup — idempotent, config-driven PostgreSQL (+pgvector) and Redis setup.

Run explicitly after ``pip install``::

    amb-db-setup                 # create/repair everything, then verify
    amb-db-setup --check         # verify only, change nothing
    amb-db-setup --json          # machine-readable report

Every value comes from (highest priority first):
  1. CLI flags      2. process environment / .env      3. config.yaml

Works on Linux / macOS / Windows: pure Python (psycopg2), no shell, no sudo.
Creating the role / database / extension needs *admin* credentials
(AMB_PG_ADMIN_USER / AMB_PG_ADMIN_PASSWORD, optional AMB_PG_ADMIN_DB).
Without them the tool still creates tables and tells you the exact one-liner
an admin must run for the rest.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_TYPE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_ ]*(\(\s*\d+(\s*,\s*\d+)?\s*\))?(\[\])?$"
)
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

KNOWN_EMBEDDING_DIMS = {
    "all-minilm-l6-v2": 384, "all-minilm-l12-v2": 384,
    "all-mpnet-base-v2": 768, "bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768, "bge-large-en-v1.5": 1024,
    "e5-small-v2": 384, "e5-base-v2": 768,
    "text-embedding-3-small": 1536,
}


# ═════════════════════════════════════════════════════════════════════════════
# Config / env
# ═════════════════════════════════════════════════════════════════════════════
def expand_vars(obj: Any, env: Optional[Dict[str, str]] = None) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` in every string."""
    env = os.environ if env is None else env
    if isinstance(obj, str):
        return _VAR.sub(
            lambda m: env.get(
                m.group(1),
                m.group(2) if m.group(2) is not None else "",
            ),
            obj,
        )
    if isinstance(obj, dict):
        return {k: expand_vars(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_vars(v, env) for v in obj]
    return obj


def load_env_files(extra: Optional[str] = None) -> List[str]:
    """Load .env files WITHOUT overriding variables already in the environment."""
    from amb_sentinel import paths
    try:
        from dotenv import load_dotenv
    except Exception:
        return []
    cands = [
        extra,
        os.environ.get("AMB_ENV_FILE"),
        paths.config_dir() / ".env",
        Path.cwd() / ".env",
    ]
    loaded: List[str] = []
    for c in cands:
        if c and Path(c).is_file() and str(c) not in loaded:
            load_dotenv(c, override=False)
            loaded.append(str(c))
    return loaded


def load_config(path: Optional[str]) -> tuple[Dict[str, Any], Optional[Path]]:
    from amb_sentinel import paths
    p = paths.find_config(path)
    if p is None:
        return {}, None
    return expand_vars(yaml.safe_load(p.read_text(encoding="utf-8")) or {}), p


# ═════════════════════════════════════════════════════════════════════════════
# Connection target
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class PgTarget:
    host: str = "localhost"
    port: int = 5432
    database: str = "amb_sentinel"
    user: str = ""
    password: str = ""
    sslmode: str = "prefer"
    connect_timeout: int = 10
    sources: Dict[str, str] = field(default_factory=dict)

    def dsn_kwargs(self) -> Dict[str, Any]:
        kw = dict(
            host=self.host, port=self.port, dbname=self.database,
            user=self.user, sslmode=self.sslmode,
            connect_timeout=self.connect_timeout,
        )
        if self.password:
            kw["password"] = self.password
        return kw

    def safe(self) -> str:
        return (
            f"{self.user}@{self.host}:{self.port}/{self.database} "
            f"(sslmode={self.sslmode})"
        )


def _pick(
    name: str,
    env_names: tuple,
    cfg_val: Any,
    default: Any,
    src: Dict[str, str],
):
    for e in env_names:
        v = os.environ.get(e)
        if v not in (None, ""):
            src[name] = f"env:{e}"
            return v
    if cfg_val not in (None, "", "<PROMPT>"):
        src[name] = "config.yaml"
        return cfg_val
    src[name] = "default"
    return default


def target_from_config(
    cfg: Dict[str, Any],
    *,
    allow_prompt: bool = True,
) -> PgTarget:
    pg = ((cfg.get("database") or {}).get("postgresql")) or {}
    src: Dict[str, str] = {}
    t = PgTarget(
        host=str(_pick(
            "host", ("AMB_PG_HOST", "PGHOST"),
            pg.get("host"), "localhost", src,
        )),
        port=int(_pick(
            "port", ("AMB_PG_PORT", "PGPORT"),
            pg.get("port"), 5432, src,
        )),
        database=str(_pick(
            "database", ("AMB_PG_DATABASE", "PGDATABASE"),
            pg.get("database"), "amb_sentinel", src,
        )),
        user=str(_pick(
            "user", ("AMB_PG_USER", "PGUSER"),
            pg.get("user"), "", src,
        )),
        sslmode=str(_pick(
            "sslmode", ("AMB_PG_SSLMODE", "PGSSLMODE"),
            pg.get("sslmode"), "prefer", src,
        )),
        connect_timeout=int(_pick(
            "connect_timeout", ("AMB_PG_CONNECT_TIMEOUT",),
            pg.get("timeout"), 10, src,
        )),
        sources=src,
    )
    pw = _pick(
        "password",
        ("AMB_PG_PASSWORD", "PGPASSWORD", "POSTGRES_PASSWORD", "DB_PASSWORD"),
        pg.get("password"), "", src,
    )
    if not pw and allow_prompt and sys.stdin.isatty():
        pw = getpass.getpass(f"PostgreSQL password for {t.user}@{t.host}: ")
        src["password"] = "prompt"
    t.password = str(pw)
    return t


def admin_from_env(
    target: PgTarget,
    args: argparse.Namespace,
) -> Optional[PgTarget]:
    user = args.admin_user or os.environ.get("AMB_PG_ADMIN_USER")
    if not user:
        return None
    return PgTarget(
        host=target.host,
        port=target.port,
        database=args.admin_db or os.environ.get("AMB_PG_ADMIN_DB", "postgres"),
        user=user,
        password=args.admin_password or os.environ.get("AMB_PG_ADMIN_PASSWORD", ""),
        sslmode=target.sslmode,
        connect_timeout=target.connect_timeout,
        sources={"admin": "flag/env"},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    changed: bool = False
    skipped: bool = False


class Report:
    def __init__(self) -> None:
        self.checks: List[Check] = []

    def add(
        self,
        name: str,
        ok: bool,
        detail: str = "",
        fix: str = "",
        changed: bool = False,
        skipped: bool = False,
    ) -> Check:
        c = Check(name, ok, detail, fix, changed, skipped)
        self.checks.append(c)
        return c

    @property
    def ok(self) -> bool:
        return all(c.ok or c.skipped for c in self.checks)

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "checks": [c.__dict__ for c in self.checks]}

    def render(self) -> str:
        out = []
        for c in self.checks:
            mark = "SKIP" if c.skipped else ("OK  " if c.ok else "FAIL")
            chg = " (changed)" if c.changed else ""
            out.append(f"[{mark}] {c.name}{chg}" + (f" — {c.detail}" if c.detail else ""))
            if not c.ok and c.fix:
                out.append(f"       fix: {c.fix}")
        out.append("RESULT: " + ("READY" if self.ok else "NOT READY"))
        return "\n".join(out)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _connect(t: PgTarget):
    import psycopg2
    return psycopg2.connect(**t.dsn_kwargs())


def _service_hint() -> str:
    s = platform.system()
    if s == "Linux":
        return "sudo systemctl start postgresql   (or: sudo service postgresql start)"
    if s == "Darwin":
        return "brew services start postgresql@16"
    if s == "Windows":
        return "Start the 'postgresql-x64-16' service (services.msc) or: net start postgresql-x64-16"
    return "start the PostgreSQL service for your OS"


def _ident(name: str, what: str) -> str:
    if not _IDENT.match(str(name or "")):
        raise ValueError(f"unsafe {what} identifier: {name!r}")
    return name


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _dim_cfg(cfg: Dict[str, Any]) -> tuple[int, int]:
    pgv = (((cfg.get("database") or {}).get("postgresql") or {}).get("pgvector")) or {}
    return int(pgv.get("dimension", 384)), int(pgv.get("lists", 100))


def _analyzer_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    da = cfg.get("data_analyzer") or {}
    return {
        "raw": _ident(da.get("raw_data_table", "raw_data"), "raw_data_table"),
        "id": _ident(da.get("raw_data_id_col", "id"), "raw_data_id_col"),
        "content": _ident(da.get("raw_data_content_col", "content"), "raw_data_content_col"),
        "analyzed": _ident(da.get("raw_data_analyzed_col", "analyzed"), "raw_data_analyzed_col"),
        "source": _ident(da.get("raw_data_source_col", "source_type"), "raw_data_source_col"),
        "results": _ident(da.get("analysis_results_table", "analysis_results"), "analysis_results_table"),
        "schema": da.get("output_schema") or {},
    }


def _columns(cur, table: str) -> Dict[str, str]:
    cur.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = %s",
        (table,),
    )
    return {r[0]: r[1] for r in cur.fetchall()}


# ═════════════════════════════════════════════════════════════════════════════
# Steps
# ═════════════════════════════════════════════════════════════════════════════
def step_bootstrap(
    rep: Report,
    target: PgTarget,
    admin: Optional[PgTarget],
    check_only: bool,
) -> bool:
    """Make sure the app role + database exist. Returns True if app can connect."""
    import psycopg2
    try:
        _connect(target).close()
        rep.add("connect (app role)", True, target.safe())
        return True
    except psycopg2.OperationalError as e:
        msg = str(e).strip().splitlines()[0]
        refused = (
            "refused" in msg.lower()
            or "could not connect" in msg.lower()
            or "timeout" in msg.lower()
        )
        if refused and "does not exist" not in msg:
            rep.add("connect (app role)", False, msg, fix=_service_hint())
            return False
        missing = "does not exist" in msg or "authentication failed" in msg.lower()
        if not missing or admin is None or check_only:
            hint = (
                "Provide admin credentials (AMB_PG_ADMIN_USER / "
                "AMB_PG_ADMIN_PASSWORD) so this tool can create the role "
                "and database, or run as a superuser: "
                f"CREATE ROLE {target.user} LOGIN PASSWORD '***'; "
                f"CREATE DATABASE {target.database} OWNER {target.user};"
            )
            rep.add("connect (app role)", False, msg, fix=hint)
            return False

    try:
        _ident(target.user, "user")
        _ident(target.database, "database")
        conn = _connect(admin)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (target.user,))
        created = []
        if not cur.fetchone():
            from psycopg2 import sql
            cur.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD %s").format(
                    sql.Identifier(target.user),
                ),
                (target.password,),
            )
            created.append("role")
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (target.database,),
        )
        if not cur.fetchone():
            from psycopg2 import sql
            cur.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(target.database),
                    sql.Identifier(target.user),
                )
            )
            created.append("database")
        conn.close()
        rep.add(
            "bootstrap role/database (admin)", True,
            "created: " + (", ".join(created) or "nothing, already present"),
            changed=bool(created),
        )
        _connect(target).close()
        rep.add("connect (app role)", True, target.safe())
        return True
    except Exception as e:      # noqa: BLE001
        rep.add(
            "bootstrap role/database (admin)", False,
            str(e).strip().splitlines()[0],
            fix="check AMB_PG_ADMIN_* credentials and that the admin may CREATE ROLE / CREATE DATABASE",
        )
        return False


def step_extension(
    rep: Report,
    target: PgTarget,
    admin: Optional[PgTarget],
    check_only: bool,
) -> bool:
    conn = _connect(target)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
    row = cur.fetchone()
    if row:
        rep.add("pgvector extension", True, f"version {row[0]}")
        conn.close()
        return True
    if check_only:
        rep.add(
            "pgvector extension", False, "not installed",
            fix="run amb-db-setup without --check",
        )
        conn.close()
        return False

    err = None
    from dataclasses import replace
    # Extension must be created INSIDE the application database.
    admin_in_db = (
        replace(admin, database=target.database) if admin is not None else None
    )
    for who, tgt in (("app role", target), ("admin", admin_in_db)):
        if tgt is None:
            continue
        try:
            c2 = _connect(tgt)
            c2.autocommit = True
            c2.cursor().execute("CREATE EXTENSION IF NOT EXISTS vector")
            c2.close()
            rep.add("pgvector extension", True, f"created by {who}", changed=True)
            conn.close()
            return True
        except Exception as e:      # noqa: BLE001
            err = str(e).strip().splitlines()[0]

    conn.close()
    rep.add(
        "pgvector extension", False, err or "no privileges",
        fix=(
            f"install the pgvector package for your server first, then as "
            f"a superuser run: \\c {target.database}  "
            f"CREATE EXTENSION vector;  "
            f"(Debian/Ubuntu: apt install postgresql-<ver>-pgvector; "
            f"macOS: brew install pgvector; Windows: see "
            f"github.com/pgvector/pgvector#windows)"
        ),
    )
    return False


def step_schema(
    rep: Report,
    cfg: Dict[str, Any],
    target: PgTarget,
    check_only: bool,
) -> None:
    from amb_sentinel.db import render_schema
    dim, lists = _dim_cfg(cfg)
    conn = _connect(target)
    cur = conn.cursor()
    cur.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema=current_schema() "
        "AND table_name IN ('agent_executions','raw_data','analysis_results',"
        "'tool_executions','embeddings','agent_events')"
    )
    before = cur.fetchone()[0]
    if check_only:
        rep.add(
            "schema tables", before >= 6,
            f"{before}/6 core tables present",
            fix="run amb-db-setup without --check",
        )
        conn.close()
        return
    try:
        cur.execute(render_schema(dim, lists))
        conn.commit()
        cur.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema=current_schema() "
            "AND table_name IN ('agent_executions','raw_data','analysis_results',"
            "'tool_executions','embeddings','agent_events')"
        )
        after = cur.fetchone()[0]
        rep.add(
            "schema tables", after >= 6,
            f"{after}/6 core tables present (dim={dim}, lists={lists})",
            changed=after != before,
        )
    except Exception as e:      # noqa: BLE001
        conn.rollback()
        rep.add(
            "schema tables", False, str(e).strip().splitlines()[0],
            fix="fix the error above and re-run; the schema is applied in one transaction",
        )
    conn.close()


def step_config_alignment(
    rep: Report,
    cfg: Dict[str, Any],
    target: PgTarget,
    check_only: bool,
) -> None:
    """The analyzer's table/column names and output_schema must exist in the DB."""
    a = _analyzer_cfg(cfg)
    conn = _connect(target)
    cur = conn.cursor()
    problems: List[str] = []
    changed = False

    raw_cols = _columns(cur, a["raw"])
    if not raw_cols:
        problems.append(f"table {a['raw']} missing")
    else:
        for key in ("id", "content", "analyzed", "source"):
            if a[key] not in raw_cols:
                problems.append(
                    f"{a['raw']}.{a[key]} missing (data_analyzer.raw_data_*_col)"
                )

    res_cols = _columns(cur, a["results"])
    if not res_cols:
        problems.append(f"table {a['results']} missing")
    else:
        for col, spec in a["schema"].items():
            _ident(col, "output_schema column")
            if col in res_cols:
                continue
            typ = str((spec or {}).get("type", "TEXT"))
            if not _TYPE.match(typ):
                problems.append(f"output_schema.{col}: unsafe type {typ!r}")
                continue
            if check_only:
                problems.append(f"{a['results']}.{col} missing")
                continue
            default = (spec or {}).get("default")
            ddl = (
                f"ALTER TABLE {_q(a['results'])} "
                f"ADD COLUMN IF NOT EXISTS {_q(col)} {typ}"
            )
            if default and re.match(r"^[A-Za-z0-9_().' :]+$", str(default)):
                ddl += f" DEFAULT {default}"
            cur.execute(ddl)
            changed = True
        conn.commit()
    conn.close()
    rep.add(
        "config ↔ database alignment",
        not problems,
        "; ".join(problems) if problems else
        "analyzer tables/columns and output_schema all present",
        fix=(
            "run amb-db-setup (adds missing output_schema columns) or "
            "correct data_analyzer.* names in config.yaml"
        ),
        changed=changed,
    )


def step_dimension(
    rep: Report,
    cfg: Dict[str, Any],
    target: PgTarget,
    fix_dim: bool,
    check_only: bool,
) -> None:
    dim, lists = _dim_cfg(cfg)
    conn = _connect(target)
    cur = conn.cursor()
    cur.execute(
        "SELECT a.atttypmod FROM pg_attribute a "
        "JOIN pg_class c ON a.attrelid=c.oid "
        "WHERE c.relname='embeddings' AND a.attname='embedding' "
        "AND NOT a.attisdropped"
    )
    row = cur.fetchone()
    actual = row[0] if row else None
    model = str(
        ((cfg.get("ai_service") or {}).get("embeddings") or {}).get("model", "")
    ).lower().split("/")[-1]
    model_dim = KNOWN_EMBEDDING_DIMS.get(model)
    notes = []
    if model_dim and model_dim != dim:
        notes.append(
            f"config dimension {dim} != embedding model "
            f"'{model}' output {model_dim}"
        )
    if actual in (None, -1):
        rep.add(
            "embeddings dimension", False,
            "embeddings.embedding column not found",
            fix="run amb-db-setup",
        )
    elif actual == dim and not notes:
        rep.add("embeddings dimension", True, f"{actual} (matches config)")
    elif actual != dim and fix_dim and not check_only:
        cur.execute("SELECT count(*) FROM embeddings")
        n = cur.fetchone()[0]
        if n:
            rep.add(
                "embeddings dimension", False,
                f"table has {n} rows at dim {actual}; refusing to drop",
                fix="export/clear embeddings manually, then re-run with --fix-dimension",
            )
        else:
            cur.execute("DROP TABLE embeddings")
            conn.commit()
            from amb_sentinel.db import render_schema
            cur.execute(render_schema(dim, lists))
            conn.commit()
            rep.add(
                "embeddings dimension", True,
                f"recreated empty embeddings table at dim {dim}",
                changed=True,
            )
    else:
        if actual != dim:
            notes.append(f"database dimension {actual} != config {dim}")
        rep.add(
            "embeddings dimension", False, "; ".join(notes),
            fix=(
                "set database.postgresql.pgvector.dimension to the "
                "embedding model's output size, then re-run with "
                "--fix-dimension (only allowed while the table is empty)"
            ),
        )
    conn.close()


def step_roundtrip(
    rep: Report,
    cfg: Dict[str, Any],
    target: PgTarget,
) -> None:
    """Prove write+read works on every table in a transaction that is always rolled back."""
    dim, _ = _dim_cfg(cfg)
    conn = _connect(target)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO raw_data "
            "(source_type, content, content_hash, collected_by_agent) "
            "VALUES ('selftest', 'selftest', repeat('0', 64), 'amb-db-setup') "
            "RETURNING id"
        )
        rid = cur.fetchone()[0]
        cur.execute(
            "SELECT count(*) FROM raw_data WHERE id=%s AND analyzed = FALSE",
            (rid,),
        )
        assert cur.fetchone()[0] == 1
        cur.execute(
            "INSERT INTO tool_executions "
            "(tool_name, arguments, success, duration) "
            "VALUES ('selftest', '{}'::jsonb, TRUE, 0)"
        )
        cur.execute(
            "INSERT INTO agent_executions "
            "(execution_id, conversation_id, instruction) "
            "VALUES ('selftest', 'selftest', 'selftest')"
        )
        vec = "[" + ",".join(["0.001"] * dim) + "]"
        cur.execute(
            "INSERT INTO embeddings (id, embedding, payload) "
            "VALUES ('selftest', %s::vector, '{}'::jsonb)",
            (vec,),
        )
        cur.execute(
            "SELECT id FROM embeddings ORDER BY embedding <=> %s::vector LIMIT 1",
            (vec,),
        )
        assert cur.fetchone() is not None
        rep.add(
            "write/read round-trip", True,
            "raw_data, tool_executions, agent_executions, embeddings (rolled back)",
        )
    except Exception as e:      # noqa: BLE001
        rep.add(
            "write/read round-trip", False,
            str(e).strip().splitlines()[0],
            fix=(
                "the role needs INSERT/SELECT on all tables; re-run "
                "amb-db-setup or GRANT ALL ON ALL TABLES"
            ),
        )
    finally:
        conn.rollback()
        conn.close()


def step_redis(rep: Report, cfg: Dict[str, Any]) -> None:
    r = ((cfg.get("database") or {}).get("redis")) or {}
    if not r:
        rep.add("redis", True, "not configured", skipped=True)
        return
    src: Dict[str, str] = {}
    host = str(_pick("host", ("AMB_REDIS_HOST",), r.get("host"), "localhost", src))
    port = int(_pick("port", ("AMB_REDIS_PORT",), r.get("port"), 6379, src))
    pw = _pick(
        "password", ("AMB_REDIS_PASSWORD", "REDIS_PASSWORD"),
        r.get("password"), "", src,
    )
    try:
        import redis
        c = redis.Redis(
            host=host, port=port, db=int(r.get("db", 0)),
            password=pw or None,
            socket_connect_timeout=float(r.get("socket_connect_timeout", 5)),
        )
        c.ping()
        rep.add("redis", True, f"{host}:{port} PING ok")
    except Exception as e:      # noqa: BLE001
        rep.add(
            "redis", False, str(e).splitlines()[0][:160],
            fix="start Redis, or remove database.redis from config.yaml if you do not use it",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
def run(args: argparse.Namespace) -> Report:
    rep = Report()
    loaded = load_env_files(args.env_file)
    cfg, cfg_path = load_config(args.config)
    rep.add(
        "config", cfg_path is not None,
        f"{cfg_path} (.env: {', '.join(loaded) or 'none'})",
        fix="pass --config /path/config.yaml or set AMB_CONFIG",
    )
    if cfg_path is None:
        return rep

    try:
        target = target_from_config(cfg, allow_prompt=not args.no_prompt)
    except Exception as e:      # noqa: BLE001
        rep.add("target", False, str(e))
        return rep

    if not target.user:
        rep.add(
            "target", False,
            "no database user (database.postgresql.user / AMB_PG_USER)",
        )
        return rep
    if not target.password:
        rep.add(
            "password", False, "empty",
            fix="export AMB_PG_PASSWORD or put it in .env",
        )
        return rep

    rep.add(
        "target", True,
        target.safe() + "  sources=" +
        ",".join(f"{k}:{v}" for k, v in target.sources.items() if k != "password") +
        f"  password:{target.sources.get('password')}",
    )
    admin = admin_from_env(target, args)

    try:
        if not step_bootstrap(rep, target, admin, args.check):
            return rep
        if not step_extension(rep, target, admin, args.check):
            return rep
        step_schema(rep, cfg, target, args.check)
        step_config_alignment(rep, cfg, target, args.check)
        step_dimension(rep, cfg, target, args.fix_dimension, args.check)
        step_roundtrip(rep, cfg, target)
    except ValueError as e:
        rep.add(
            "config validation", False, str(e),
            fix="correct the identifier in config.yaml",
        )
    except Exception as e:      # noqa: BLE001
        rep.add(
            "unexpected error", False,
            f"{type(e).__name__}: {str(e)[:200]}",
        )
    step_redis(rep, cfg)
    return rep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="amb-db-setup", description=__doc__.split("\n\n")[0],
    )
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--env-file", default=None, help="extra .env file to load")
    p.add_argument("--check", action="store_true", help="verify only; change nothing")
    p.add_argument(
        "--fix-dimension", action="store_true",
        help="recreate EMPTY embeddings table at configured dimension",
    )
    p.add_argument("--admin-user", default=None)
    p.add_argument("--admin-password", default=None)
    p.add_argument("--admin-db", default=None)
    p.add_argument("--no-prompt", action="store_true", help="never prompt for a password")
    p.add_argument("--json", action="store_true", help="print JSON report")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rep = run(args)
    print(json.dumps(rep.as_dict(), indent=2) if args.json else rep.render())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
