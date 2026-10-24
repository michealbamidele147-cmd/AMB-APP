"""`sentinel config path|show|validate`, `sentinel init`."""
from __future__ import annotations

import os
import shutil
from importlib.resources import files

import yaml

_SECRET_KEYS = {"password", "api_key", "secret", "token"}


def _mask(v):
    if v is None:
        return None
    return "<redacted>" if v else ""


def _walk_mask(node):
    if isinstance(node, dict):
        return {k: (_mask(v) if k.lower() in _SECRET_KEYS else _walk_mask(v))
                for k, v in node.items()}
    if isinstance(node, list):
        return [_walk_mask(x) for x in node]
    return node


def _cmd_init(args) -> int:
    from amb_sentinel import paths
    paths.ensure_dirs()
    dst = paths.config_dir() / "config.yaml"
    if dst.exists() and not args.force:
        print(f"already exists: {dst}  (use --force to overwrite)")
        return 1
    src = files("amb_sentinel.data") / "config.example.yaml"
    shutil.copy(str(src), str(dst))
    os.chmod(dst, 0o600)
    print(f"wrote {dst}")
    print(f"Edit it, then run:  sentinel config validate")
    return 0


def _cmd_path(args) -> int:
    from amb_sentinel import paths
    p = paths.find_config(args.config)
    if p:
        print(p)
        return 0
    print("(no config.yaml found; run `sentinel init`)")
    return 1


def _cmd_show(args) -> int:
    from amb_sentinel import paths
    p = paths.find_config(args.config)
    if not p:
        print("(no config.yaml found; run `sentinel init`)")
        return 1
    data = yaml.safe_load(p.read_text()) or {}
    print(yaml.safe_dump(_walk_mask(data), sort_keys=False))
    return 0


def _cmd_validate(args) -> int:
    from amb_sentinel import paths
    p = paths.find_config(args.config)
    if not p:
        print("config not found")
        return 1
    try:
        cfg = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        print(f"YAML error: {e}")
        return 1

    issues = []
    if not cfg.get("ai_service"):
        issues.append("missing 'ai_service' section")
    elif not cfg["ai_service"].get("models"):
        issues.append("'ai_service.models' is empty")
    elif not any(m.get("enabled") for m in cfg["ai_service"]["models"]):
        issues.append("no model with enabled: true")

    for section in ("postgresql", "redis"):
        if section in (cfg.get("database") or {}):
            sec = cfg["database"][section]
            if sec.get("password") == "<PROMPT>" and not os.environ.get(
                    f"AMB_{section.upper()}_PASSWORD"):
                issues.append(f"database.{section}.password is <PROMPT> but env not set")

    if issues:
        for i in issues:
            print(f"  ✗ {i}")
        return 1
    print(f"  ✓ {p} looks valid")
    return 0


def add_subparser(sub) -> None:
    # sentinel init  (top-level shorthand)
    init_p = sub.add_parser("init", help="Create config_dir()/config.yaml from the example")
    init_p.add_argument("--force", action="store_true")
    init_p.set_defaults(func=_cmd_init)

    p = sub.add_parser("config", help="Inspect or validate config.yaml")
    sp = p.add_subparsers(dest="config_cmd")

    path_p = sp.add_parser("path", help="Print the resolved config.yaml path")
    path_p.add_argument("--config", default=None)
    path_p.set_defaults(func=_cmd_path)

    show_p = sp.add_parser("show", help="Print config with secrets masked")
    show_p.add_argument("--config", default=None)
    show_p.set_defaults(func=_cmd_show)

    val_p = sp.add_parser("validate", help="Basic schema and secret checks")
    val_p.add_argument("--config", default=None)
    val_p.set_defaults(func=_cmd_validate)

    p.set_defaults(func=lambda _a: p.print_help() or 0)
