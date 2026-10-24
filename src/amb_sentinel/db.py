"""Database DDL and status. This module and `data/schema.sql` are the ONLY
places DDL may live. `ai_agent.py::_create_database_tables` is replaced by a
call to `apply_schema` once that edit is applied (see phase 3).
"""
from __future__ import annotations

import re
from importlib.resources import files
from typing import Any, Dict

_SCHEMA_RE = re.compile(r"\{\{(VECTOR_DIM|IVF_LISTS)\}\}")


def render_schema(pgvector_dimension: int = 384, pgvector_lists: int = 100) -> str:
    """Read data/schema.sql and render placeholders."""
    text = (files("amb_sentinel.data") / "schema.sql").read_text(encoding="utf-8")
    return _SCHEMA_RE.sub(
        lambda m: str(pgvector_dimension if m.group(1) == "VECTOR_DIM" else pgvector_lists),
        text,
    )


def apply_schema(db_manager, cfg: Dict[str, Any], connection: str = "primary") -> Dict[str, Any]:
    """Enable pgvector and apply data/schema.sql. Requires superuser or the
    vector extension already installed. Never raises on missing extension —
    returns {"ok": False, "hint": ...} so callers can log it."""
    pgv = (
        cfg.get("database", {}).get("postgresql", {}).get("pgvector", {}) or {}
    )
    dim = int(pgv.get("dimension", 384))
    lists = int(pgv.get("lists", 100))

    # Trusted DDL bypasses DatabaseManager.execute(), whose _is_safe_sql
    # check exists to sanitise LLM-issued SQL and blocks CREATE EXTENSION.
    # Use a raw connection via db_manager.connection() so the extension and
    # the rendered schema both reach the server.
    sql = render_schema(dim, lists)
    try:
        with db_manager.connection(connection) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                except Exception as e:
                    conn.rollback()
                    return {
                        "ok": False,
                        "error": str(e),
                        "hint": (
                            "As a Postgres superuser run: "
                            "CREATE EXTENSION IF NOT EXISTS vector;"
                        ),
                    }
                cur.execute(sql)
    except Exception as e:
        return {"ok": False, "error": str(e), "hint": None}
    return {"ok": True, "dimension": dim, "lists": lists}


def status(db_manager, cfg: Dict[str, Any], connection: str = "primary") -> Dict[str, Any]:
    """Report tables present, vector extension, embedding dim vs config."""
    expected_tables = (
        "agent_executions", "raw_data", "analysis_results",
        "tool_executions", "embeddings",
    )
    info: Dict[str, Any] = {
        "ok": True,
        "tables": {},
        "vector_extension": False,
        "dimension_match": True,
        "dimension_expected": 384,
        "dimension_actual": None,
    }

    r = db_manager.execute(
        connection,
        "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public';",
        fetch="all",
    )
    present = set()
    if r.get("ok"):
        for row in r.get("result", []):
            name = list(row.values())[0] if isinstance(row, dict) else row[0]
            present.add(name)
    for t in expected_tables:
        info["tables"][t] = t in present

    r = db_manager.execute(
        connection,
        "SELECT 1 FROM pg_extension WHERE extname='vector';",
        fetch="all",
    )
    info["vector_extension"] = bool(r.get("ok") and r.get("result"))

    pgv = (
        cfg.get("database", {}).get("postgresql", {}).get("pgvector", {}) or {}
    )
    info["dimension_expected"] = int(pgv.get("dimension", 384))

    if info["vector_extension"]:
        r = db_manager.execute(
            connection,
            "SELECT a.atttypmod FROM pg_attribute a "
            "JOIN pg_class c ON a.attrelid = c.oid "
            "WHERE c.relname='embeddings' AND a.attname='embedding';",
            fetch="all",
        )
        if r.get("ok") and r.get("result"):
            row = r["result"][0]
            typmod = list(row.values())[0] if isinstance(row, dict) else row[0]
            if isinstance(typmod, int) and typmod > 0:
                info["dimension_actual"] = typmod
                info["dimension_match"] = (typmod == info["dimension_expected"])

    if not all(info["tables"].values()) or not info["vector_extension"] or not info["dimension_match"]:
        info["ok"] = False
    return info
