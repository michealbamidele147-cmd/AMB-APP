"""`sentinel db init` / `sentinel db status`."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml


def _resolve_password(pw: str, env_names: tuple[str, ...]) -> str:
    """`<PROMPT>` (or empty) → look up env vars; else use the literal value."""
    if pw and pw != "<PROMPT>":
        return pw
    for name in env_names:
        v = os.environ.get(name)
        if v:
            return v
    return ""


def _open_manager(config_path: str | None):
    from amb_sentinel import paths
    from amb_sentinel.tools.dbtools import DatabaseManager

    cfg_path = paths.find_config(config_path) or (Path(config_path) if config_path else None)
    if cfg_path is None or not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    pg = cfg.get("database", {}).get("postgresql", {}) or {}
    pw = _resolve_password(
        str(pg.get("password", "")),
        ("PGPASSWORD", "POSTGRES_PASSWORD", "DB_PASSWORD", "AMB_PG_PASSWORD"),
    )
    from urllib.parse import quote as _q
    uri = (
        f"postgresql://{_q(str(pg.get('user','')), safe='')}:{_q(pw, safe='')}"
        f"@{pg.get('host','localhost')}:{pg.get('port',5432)}"
        f"/{_q(str(pg.get('database','amb_sentinel')), safe='')}"
    )
    dbm = DatabaseManager()
    r = dbm.add_connection("primary", uri, {
        "min_conn": int(pg.get("min_connections", 1)),
        "max_conn": int(pg.get("max_connections", 10)),
        "sslmode":  pg.get("sslmode", "prefer"),
        "connect_timeout": int(pg.get("timeout", 10)),
    })
    if not r.get("ok"):
        raise SystemExit(f"connect failed: {r.get('error')}")
    return cfg, dbm


def _cmd_init(args) -> int:
    from amb_sentinel import db as db_mod
    cfg, dbm = _open_manager(args.config)
    result = db_mod.apply_schema(dbm, cfg)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def _cmd_status(args) -> int:
    from amb_sentinel import db as db_mod
    cfg, dbm = _open_manager(args.config)
    result = db_mod.status(dbm, cfg)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def add_subparser(sub) -> None:
    p = sub.add_parser("db", help="Database schema and status")
    sp = p.add_subparsers(dest="db_cmd")

    init_p = sp.add_parser("init", help="Create tables and enable pgvector")
    init_p.add_argument("--config", default=None)
    init_p.set_defaults(func=_cmd_init)

    st_p = sp.add_parser("status", help="Report tables, extension, dimension match")
    st_p.add_argument("--config", default=None)
    st_p.set_defaults(func=_cmd_status)

    p.set_defaults(func=lambda _a: p.print_help() or 0)
