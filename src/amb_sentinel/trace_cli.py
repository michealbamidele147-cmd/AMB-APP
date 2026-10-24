"""amb-trace — watch what the agent is really doing.

    amb-trace                      # live feed of every event
    amb-trace --run 3fa9c1d2       # only one run
    amb-trace --kind tool_result   # only tool calls (repeatable)
    amb-trace runs                 # last runs from PostgreSQL: llm calls, tool calls, rows
    amb-trace show 3fa9c1d2        # step-by-step timeline of a finished run
    amb-trace verify 3fa9c1d2      # exit 0 only if real work happened
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from amb_sentinel.observability import format_human

_COLORS = {
    "gate_reject": "red", "run_end": "bold", "llm_response": "cyan",
    "tool_result": "green", "step_start": "yellow",
    "run_start": "bold magenta", "action": "blue",
}


def default_events_file() -> Path:
    from amb_sentinel import paths
    return paths.log_dir() / "agent_events.jsonl"


def iter_events(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield events from the active file and its rotated siblings (oldest first)."""
    files = sorted(
        path.parent.glob(path.name + ".*"),
        key=lambda p: p.suffix, reverse=True,
    ) + [path]
    for f in files:
        if not f.exists():
            continue
        with f.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue


def _match(
    ev: Dict[str, Any],
    run: Optional[str],
    kinds: Optional[List[str]],
) -> bool:
    return (
        (not run or ev.get("run_id") == run)
        and (not kinds or ev.get("kind") in kinds)
    )


def _emit(ev: Dict[str, Any], raw: bool, console: Any) -> None:
    if raw:
        print(json.dumps(ev, ensure_ascii=False))
        return
    line = format_human(ev)
    if console is not None:
        style = _COLORS.get(ev["kind"], "")
        if ev["kind"] == "tool_result" and ev["payload"].get("ok") is False:
            style = "red"
        stamp = time.strftime("%H:%M:%S", time.localtime(ev["ts"]))
        if style:
            console.print(
                f"[dim]{stamp}[/] [{style}]{line}[/]",
                highlight=False, markup=True,
            )
        else:
            console.print(
                f"[dim]{stamp}[/] {line}",
                highlight=False, markup=True,
            )
    else:
        print(f"{time.strftime('%H:%M:%S', time.localtime(ev['ts']))} {line}")


def follow(
    path: Path,
    run: Optional[str],
    kinds: Optional[List[str]],
    tail: int,
    raw: bool,
    once: bool,
) -> int:
    try:
        from rich.console import Console
        console: Any = (
            Console(highlight=False) if sys.stdout.isatty() and not raw else None
        )
    except Exception:
        console = None

    backlog = [e for e in iter_events(path) if _match(e, run, kinds)]
    for e in (backlog[-tail:] if tail else []):
        _emit(e, raw, console)
    if once:
        return 0

    pos = path.stat().st_size if path.exists() else 0
    try:
        while True:
            if path.exists():
                size = path.stat().st_size
                if size < pos:      # rotated
                    pos = 0
                if size > pos:
                    with path.open("r", encoding="utf-8", errors="replace") as fh:
                        fh.seek(pos)
                        for line in fh:
                            try:
                                e = json.loads(line)
                            except ValueError:
                                continue
                            if _match(e, run, kinds):
                                _emit(e, raw, console)
                        pos = fh.tell()
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


# ═════════════════════════════════════════════════════════════════════════════
# Analysis
# ═════════════════════════════════════════════════════════════════════════════
def summarize_run(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    evs = list(events)
    s: Dict[str, Any] = {
        "events": len(evs), "llm_calls": 0, "tool_calls": 0,
        "tool_ok": 0, "tool_failed": 0, "gate_rejects": 0,
        "rows_written": 0, "steps": {}, "success": None, "tools": {},
    }
    for e in evs:
        k = e["kind"]
        p = e.get("payload", {})
        st = e.get("step")
        if k == "llm_response":
            s["llm_calls"] += 1
        elif k == "tool_result":
            s["tool_calls"] += 1
            ok = bool(p.get("ok"))
            s["tool_ok" if ok else "tool_failed"] += 1
            t = s["tools"].setdefault(p.get("tool"), {"ok": 0, "failed": 0})
            t["ok" if ok else "failed"] += 1
            if ok and p.get("tool") in (
                "store_raw_item", "db_execute", "db_bulk_insert",
            ) and isinstance(p.get("rows"), int):
                s["rows_written"] += p["rows"]
        elif k == "gate_reject":
            s["gate_rejects"] += 1
        elif k == "step_end":
            s["steps"][st] = p.get("status")
        elif k == "run_end":
            s["success"] = p.get("success")
    return s


def verdict(s: Dict[str, Any]) -> tuple[bool, str]:
    if s["success"] is None:
        return False, "run has no run_end event (still running, crashed, or killed)"
    if s["tool_ok"] == 0:
        return False, "NO EVIDENCE: zero successful tool calls — the model only talked"
    if s["gate_rejects"]:
        return bool(s["success"]), (
            f"{s['gate_rejects']} premature 'done' claim(s) were rejected by the gate"
        )
    return bool(s["success"]), (
        "tools ran and the run reports success" if s["success"] else "run failed"
    )


# ═════════════════════════════════════════════════════════════════════════════
# DB-backed commands
# ═════════════════════════════════════════════════════════════════════════════
def _db(args: argparse.Namespace):
    from amb_sentinel import dbsetup
    dbsetup.load_env_files()
    cfg, _ = dbsetup.load_config(args.config)
    t = dbsetup.target_from_config(cfg, allow_prompt=False)
    return dbsetup._connect(t)


def cmd_runs(args: argparse.Namespace) -> int:
    conn = _db(args)
    cur = conn.cursor()
    cur.execute(
        "SELECT run_id, started, llm_calls, tool_calls, tool_ok, "
        "gate_rejects, raw_rows_written "
        "FROM v_agent_run_summary ORDER BY started DESC LIMIT %s",
        (args.limit,),
    )
    rows = cur.fetchall()
    conn.close()
    print(
        f"{'run_id':<10} {'started':<20} {'llm':>4} {'tools':>5} "
        f"{'ok':>4} {'rejects':>7} {'rows_in_db':>10}"
    )
    for r in rows:
        print(
            f"{r[0]:<10} {r[1]:%Y-%m-%d %H:%M:%S}  "
            f"{r[2]:>4} {r[3]:>5} {r[4]:>4} {r[5]:>7} {r[6]:>10}"
        )
    if not rows:
        print("(no runs recorded yet)")
    return 0


def _events_for(args: argparse.Namespace, run: str) -> List[Dict[str, Any]]:
    evs = [
        e for e in iter_events(
            Path(args.file) if args.file else default_events_file()
        ) if e.get("run_id") == run
    ]
    if evs:
        return evs
    try:
        conn = _db(args)
        cur = conn.cursor()
        cur.execute(
            "SELECT extract(epoch from ts), run_id, seq, kind, step, payload "
            "FROM agent_events WHERE run_id=%s ORDER BY seq",
            (run,),
        )
        out = [
            {
                "ts": float(r[0]), "run_id": r[1], "seq": r[2],
                "kind": r[3], "step": r[4], "payload": r[5] or {},
            }
            for r in cur.fetchall()
        ]
        conn.close()
        return out
    except Exception:
        return []


def cmd_show(args: argparse.Namespace) -> int:
    evs = _events_for(args, args.run)
    if not evs:
        print(f"no events for run {args.run}")
        return 1
    for e in evs:
        _emit(e, False, None)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    evs = _events_for(args, args.run)
    if not evs:
        print(f"no events for run {args.run}")
        return 1
    s = summarize_run(evs)
    ok, why = verdict(s)
    print(json.dumps(
        {
            "run": args.run,
            "verdict": "REAL WORK" if ok else "NOT VERIFIED",
            "reason": why, **s,
        },
        indent=2, default=str,
    ))
    return 0 if ok else 1


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="amb-trace", description=__doc__.split("\n\n")[0],
    )
    ap.add_argument(
        "--file", default=None,
        help="agent_events.jsonl (default: <log_dir>/agent_events.jsonl)",
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--kind", action="append", default=None)
    ap.add_argument(
        "-n", "--tail", type=int, default=50,
        help="backlog lines before following",
    )
    ap.add_argument("--raw", action="store_true", help="print raw JSON lines")
    ap.add_argument("--no-follow", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("runs")
    r.add_argument("--limit", type=int, default=20)
    sh = sub.add_parser("show")
    sh.add_argument("run")
    v = sub.add_parser("verify")
    v.add_argument("run")
    a = ap.parse_args(argv)
    if a.cmd == "runs":
        return cmd_runs(a)
    if a.cmd == "show":
        return cmd_show(a)
    if a.cmd == "verify":
        return cmd_verify(a)
    return follow(
        Path(a.file) if a.file else default_events_file(),
        a.run, a.kind, a.tail, a.raw, a.no_follow,
    )


if __name__ == "__main__":
    raise SystemExit(main())
