"""`sentinel run` / `once` / `health` — thin wrappers over ai_agent.main(argv)."""
from __future__ import annotations

import argparse
import asyncio
import sys


def _resolve_config(args) -> str | None:
    """Return a concrete config path or None after printing an error."""
    from amb_sentinel import paths
    cfg = paths.find_config(getattr(args, "config", None))
    if cfg is None:
        print(
            "sentinel: no config.yaml found — run `sentinel init` first",
            file=sys.stderr,
        )
        return None
    return str(cfg)


def _dispatch(argv: list[str]) -> int:
    from amb_sentinel.agent import ai_agent
    try:
        return asyncio.run(ai_agent.main(argv)) or 0
    except KeyboardInterrupt:
        return 130


def _cmd_run(args) -> int:
    cfg = _resolve_config(args)
    if cfg is None:
        return 1
    argv = ["--config", cfg]
    if args.no_agent:
        argv.append("--no-agent")
    argv += ["--log-level", args.log_level]
    return _dispatch(argv)


def _cmd_once(args) -> int:
    cfg = _resolve_config(args)
    if cfg is None:
        return 1
    return _dispatch(["--config", cfg, "--run-agent-once",
                      "--log-level", args.log_level])


def _cmd_health(args) -> int:
    cfg = _resolve_config(args)
    if cfg is None:
        return 1
    return _dispatch(["--config", cfg, "--health-check",
                      "--log-level", args.log_level])


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])


def add_subparsers(sub) -> None:
    run_p = sub.add_parser("run", help="Start the service (scheduler + agent + analyzer)")
    _common(run_p)
    run_p.add_argument("--no-agent", action="store_true",
                       help="Disable automatic agent execution")
    run_p.set_defaults(func=_cmd_run)

    once_p = sub.add_parser("once", help="Run the agent once and exit")
    _common(once_p)
    once_p.set_defaults(func=_cmd_once)

    health_p = sub.add_parser("health", help="Check components and exit (0 = healthy)")
    _common(health_p)
    health_p.set_defaults(func=_cmd_health)
