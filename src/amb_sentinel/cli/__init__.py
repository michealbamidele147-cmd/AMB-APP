"""`sentinel` / `amb` — CLI entry point."""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from amb_sentinel import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sentinel",
        description="AMB-Sentinel — autonomous local-first AI agent service",
    )
    p.add_argument("--version", action="version", version=f"amb-sentinel {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    from amb_sentinel.cli import run as run_mod
    run_mod.add_subparsers(sub)

    # `logs` is handled specially in main() before argparse so any subcommand
    # argument reaches the logs CLI untouched.
    logs_p = sub.add_parser("logs", help="Monitor log files (same as `sentinel-logs`)")
    logs_p.set_defaults(func=lambda _a: 0)

    from amb_sentinel.cli import config_cmd
    config_cmd.add_subparser(sub)

    from amb_sentinel.cli import db as db_mod
    db_mod.add_subparser(sub)

    from amb_sentinel.cli import service as svc_mod
    svc_mod.add_subparser(sub)

    from amb_sentinel.cli import doctor
    doctor.add_subparser(sub)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    _argv = list(argv) if argv is not None else sys.argv[1:]

    # Configure logging once (idempotent). ai_agent.main will re-configure
    # with the user's --log-level if the run subcommand runs.
    try:
        from amb_sentinel.logging_setup import configure
        configure("INFO")
    except Exception:
        pass

    # `sentinel logs ...` → hand off verbatim
    if _argv and _argv[0] == "logs":
        from amb_sentinel.logmon.cli import main as logs_main
        return logs_main(_argv[1:])

    parser = _build_parser()
    args = parser.parse_args(_argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"sentinel: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
