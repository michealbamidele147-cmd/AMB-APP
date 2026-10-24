"""Command line entry point (``sentinel-logs`` / ``amb-logs``)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .hub import Hub
from .ingest import Ingestor
from .model import LEVEL_NUM, LEVEL_SHORT, Query, bucket
from .providers import clock

ENV_VARS = ("AMB_SENTINEL_LOG_DIR", "AMB_LOG_DIR")


def candidate_dirs() -> list:
    """Return candidate log directories in priority order.

    The current working directory's ``logs/`` folder wins — that is what
    you expect when you ``cd`` into the repo and run ``amb-logs``. If it
    has no ``*.log`` files (or does not exist), fall back to the env-var
    overrides, then ``amb_sentinel.paths.log_dir()`` — which honours
    ``$AMB_HOME`` and the installer's pointer file at
    ``/etc/amb-sentinel/paths.json`` — then the legacy platform default
    locations.
    """
    candidates: list = []

    # 1. ./logs relative to the working directory.
    candidates.append(str(Path.cwd() / "logs"))

    # 2. Explicit env-var overrides.
    for var in ENV_VARS:
        v = os.environ.get(var)
        if v:
            candidates.append(v)

    # 3. The service's own log directory. Resolves via AMB_HOME env var,
    #    the pointer file, or platformdirs — the same logic the running
    #    service uses. Only tried when the package is importable; a
    #    pure-CLI install (no service) simply skips this candidate.
    try:
        from amb_sentinel import paths as _paths
        candidates.append(str(_paths.log_dir()))
    except Exception:
        pass

    # 4. Legacy/platform defaults for non-repo installs.
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    candidates.append(str(Path(state) / "amb-sentinel" / "logs"))
    candidates.append("/var/log/amb-sentinel")
    candidates.append("/var/lib/amb-sentinel/logs")

    # Deduplicate preserving order.
    seen: set = set()
    out: list = []
    for item in candidates:
        resolved = str(Path(item).expanduser())
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out

_LEGACY_SUBCOMMANDS = frozenset({
    "follow", "list", "show", "grep", "errors", "path", "clear",
})


def _reject_legacy_subcommand(first: str) -> None:
    """Emit a clear error when the pre-1.0 subcommand syntax is used."""
    hints = {
        "follow": "amb-logs                    # TUI (auto-detects ./logs)\n"
                  "    amb-logs <dir> -f --plain  # follow merged plain text",
        "list":   "amb-logs --list <dir>",
        "show":   "amb-logs <dir> --plain -n 200",
        "grep":   "amb-logs <dir> --plain --grep 'pattern'",
        "errors": "amb-logs <dir> --plain --level ERROR",
        "path":   "python -c 'from amb_sentinel import paths; print(paths.log_dir())'",
        "clear":  "(no direct equivalent; truncate the file you want to clear)",
    }
    print(
        f"amb-logs: the '{first}' subcommand is gone (pre-1.0 syntax).\n"
        f"  Try:\n    {hints.get(first, 'amb-logs --help')}\n"
        f"  Full syntax: amb-logs --help",
        file=sys.stderr,
    )
    raise SystemExit(2)


def resolve_targets(args) -> list:
    if args.paths:
        first = args.paths[0]
        if first in _LEGACY_SUBCOMMANDS:
            _reject_legacy_subcommand(first)

        resolved: list = []
        for raw in args.paths:
            path = Path(raw).expanduser()
            if path.exists():
                resolved.append(path)
            else:
                print(
                    f"amb-logs: path does not exist: {path.resolve()}",
                    file=sys.stderr,
                )
        if resolved:
            return resolved
        # Every explicit path was bogus — fall through to auto-detect.

    tried = candidate_dirs()
    for d in tried:
        p = Path(d).expanduser()
        if p.is_dir() and any(p.glob("*.log")):
            return [p]

    print(
        "amb-logs: no log directory found.\n"
        "  looked in:\n    "
        + "\n    ".join(tried)
        + "\n\n"
        "  Pass one explicitly:   amb-logs /path/to/logs\n"
        "  Or set env var:        export AMB_SENTINEL_LOG_DIR=/path/to/logs\n"
        "  Or install the service: sentinel service install --system",
        file=sys.stderr,
    )
    return []

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="amb-logs", description="Live multi-file log monitor (TUI).")
    p.add_argument("paths", nargs="*", help="log files or directories (default: auto-detect)")
    p.add_argument("--glob", action="append", default=None, help="file pattern inside directories (default *.log; repeatable)")
    p.add_argument("-r", "--recursive", action="store_true", help="descend into sub-directories")
    p.add_argument("--only", action="append", default=[], metavar="NAME", help="only files whose name matches (repeatable)")
    p.add_argument("--exclude", action="append", default=[], metavar="NAME", help="skip files whose name matches (repeatable)")
    p.add_argument("--backfill-kb", type=int, default=256, help="history read per file at start, in KiB (default 256)")
    p.add_argument("--ring", type=int, default=5000, help="records kept per file in memory (default 5000, ~0.5 KB each)")
    p.add_argument("--merged", type=int, default=30000, help="records kept in the merged timeline (default 30000)")
    p.add_argument("--poll-min", type=float, default=0.2, help="fastest poll interval for active files, s")
    p.add_argument("--poll-max", type=float, default=2.0, help="slowest poll interval for idle files, s")
    p.add_argument("--tags", action="store_true", help="treat leading [tag] as the source (for pasted/merged dumps)")
    p.add_argument("--plain", action="store_true", help="no TUI: print the merged timeline (use with -f to follow)")
    p.add_argument("-f", "--follow", action="store_true", help="plain mode: keep following")
    p.add_argument("--level", default=None, help="plain mode: minimum level (DEBUG/INFO/WARNING/ERROR)")
    p.add_argument("--grep", default="", help="plain mode: filter expression (see ? in the TUI)")
    p.add_argument("-n", "--lines", type=int, default=200, help="plain mode: lines of history to print")
    p.add_argument("--list", action="store_true", help="list discovered log files and exit")
    return p


def _make_ingestor(args, hub: Hub, targets: list) -> Ingestor:
    return Ingestor(hub, targets, patterns=tuple(args.glob or ["*.log"]), recursive=args.recursive,
                    include=args.only, exclude=args.exclude, backfill_bytes=args.backfill_kb * 1024,
                    poll_min=args.poll_min, poll_max=args.poll_max, tag_sources=args.tags,
                    from_start=args.tags)


def _fmt(rec, width: int = 0) -> str:
    return f"{clock(rec.ts)} {LEVEL_SHORT[bucket(rec.level)]} {rec.source[:12]:<12} {rec.msg}"


def run_plain(args, hub: Hub, ing: Ingestor) -> int:
    ing.start()
    ing.ready.wait(30)
    q = Query(((args.grep or "") + (f" lvl:{args.level}" if args.level else "")).strip())
    recs = [r for r in hub.merged_snapshot() if q.match(r)][-args.lines:]
    last = recs[-1].key if recs else (0.0, 0)
    for r in recs:
        print(_fmt(r))
        for c in r.cont:
            print("      │ " + c)
    if not args.follow:
        ing.stop()
        return 0
    seen_rev = hub.rev
    try:
        while True:
            time.sleep(0.25)
            if hub.rev == seen_rev:
                continue
            seen_rev = hub.rev
            for r in hub.merged_snapshot():
                if r.key > last and q.match(r):
                    print(_fmt(r))
                    for c in r.cont:
                        print("      │ " + c)
            snap = hub.merged_tail(1)
            if snap:
                last = max(last, snap[-1].key)
            sys.stdout.flush()
    except KeyboardInterrupt:
        ing.stop()
        return 0


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    targets = resolve_targets(args)
    if not targets:
        return 2
    hub = Hub(ring=args.ring, merged_cap=args.merged)
    ing = _make_ingestor(args, hub, targets)
    if args.list:
        for p, name in ing.discover():
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            print(f"{name:<32} {size:>12}  {p}")
        return 0
    if args.plain or not sys.stdout.isatty():
        return run_plain(args, hub, ing)
    from .tui import LogMonApp  # imported lazily: Textual is only needed for the TUI

    ing.start()
    app = LogMonApp(hub, ing)
    try:
        app.run()
    finally:
        ing.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
