"""Agent run observability — nothing the agent does is invisible.

One ``RunRecorder`` per agent run. Every observable action becomes an event:

    run_start  step_start  llm_request  llm_response  action  tool_start
    tool_result  gate_accept  gate_reject  step_end  checkpoint_resume
    db_write  run_end

Each event goes to four sinks (each can fail without affecting the agent):

  1. ``<log_dir>/agent_events.jsonl``     machine-readable, rotating
  2. ``<log_dir>/agent_trace.log``        one human line per event (sentinel-logs TUI)
  3. PostgreSQL ``agent_events`` / ``tool_executions`` / ``agent_executions``
  4. an in-memory ring buffer (``recorder.recent()``) for live GUIs

Secrets are redacted and long values truncated before anything is written.
All DB I/O runs on a background thread so a dead database cannot stall
the agent loop.
"""
from __future__ import annotations

import contextvars
import json
import logging
import queue
import re
import threading
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

# ═════════════════════════════════════════════════════════════════════════════
# Run-id propagation across threads
# ═════════════════════════════════════════════════════════════════════════════
CURRENT_RUN: contextvars.ContextVar = contextvars.ContextVar(
    "amb_current_run", default=None,
)


def current_run_id() -> str:
    """Return the current run's id, or '' when no recorder is active."""
    rec = CURRENT_RUN.get()
    return getattr(rec, "run_id", "") or ""


# ═════════════════════════════════════════════════════════════════════════════
# Redaction
# ═════════════════════════════════════════════════════════════════════════════
_SECRET = re.compile(
    r"pass(word)?|secret|token|api[_-]?key|authorization|cookie|credential",
    re.I,
)


def redact(obj: Any, limit: int = 300, _depth: int = 0) -> Any:
    """Redact secret-looking keys and truncate long strings, recursively."""
    if _depth > 6:
        return "…"
    if isinstance(obj, dict):
        return {
            str(k): (
                "***" if _SECRET.search(str(k))
                else redact(v, limit, _depth + 1)
            )
            for k, v in list(obj.items())[:40]
        }
    if isinstance(obj, (list, tuple)):
        head = [redact(v, limit, _depth + 1) for v in list(obj)[:10]]
        if len(obj) > 10:
            head.append(f"…(+{len(obj) - 10} more)")
        return head
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + f"…(+{len(obj) - limit} chars)"
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return redact(str(obj), limit, _depth + 1)


# ═════════════════════════════════════════════════════════════════════════════
# Human formatting
# ═════════════════════════════════════════════════════════════════════════════
ICONS = {
    "run_start": "▶", "run_end": "■", "step_start": "→", "step_end": "←",
    "llm_request": "…", "llm_response": "🧠", "action": "◆",
    "tool_start": "⚙", "tool_result": "🔧",
    "gate_accept": "✔", "gate_reject": "✖",
    "checkpoint_resume": "↻", "db_write": "🗄", "note": "•",
}


def format_human(ev: Dict[str, Any]) -> str:
    """One compact, greppable line per event (also used by amb-trace)."""
    k = ev["kind"]
    p = ev.get("payload", {})
    step = f" s{ev['step']}" if ev.get("step") else ""
    head = f"{ICONS.get(k, '•')} [{ev['run_id']}{step} #{ev['seq']}] {k}"

    if k == "run_start":
        return (
            f"{head} model={p.get('model')} ctx={p.get('n_ctx')} "
            f"size_tier={p.get('size_tier')} steps={p.get('steps')} "
            f"resumed={p.get('resumed')}"
        )
    if k == "step_start":
        return f"{head} {p.get('title', '')!r} tools={p.get('tools')}"
    if k == "llm_request":
        return (
            f"{head} prompt≈{p.get('est_prompt_tokens')}tok "
            f"max_out={p.get('max_tokens')} ctx={p.get('n_ctx')}"
        )
    if k == "llm_response":
        return (
            f"{head} in={p.get('prompt_tokens')} out={p.get('completion_tokens')} "
            f"{p.get('elapsed_s')}s finish={p.get('finish')} "
            f"→ {p.get('text_preview')!r}"
        )
    if k == "action":
        return (
            f"{head} {p.get('action')} tool={p.get('tool')} "
            f"args={json.dumps(p.get('args'), default=str)[:200]} "
            f"errors={p.get('errors')}"
        )
    if k == "tool_start":
        return f"{head} {p.get('tool')}({json.dumps(p.get('args'), default=str)[:200]})"
    if k == "tool_result":
        return (
            f"{head} {p.get('tool')} ok={p.get('ok')} {p.get('duration_s')}s "
            f"rows={p.get('rows')} → {p.get('summary')!r}"
        )
    if k in ("gate_accept", "gate_reject"):
        return f"{head} {p.get('reason')} evidence={p.get('evidence')}"
    if k == "step_end":
        return (
            f"{head} status={p.get('status')} "
            f"llm_calls={p.get('llm_calls')} tool_calls={p.get('tool_calls')} "
            f"{p.get('duration_s')}s"
        )
    if k == "run_end":
        return (
            f"{head} success={p.get('success')} verified={p.get('verified')} "
            f"llm_calls={p.get('llm_calls')} tool_calls={p.get('tool_calls')} "
            f"rows_written={p.get('rows_written')} steps={p.get('step_status')}"
        )
    return f"{head} {json.dumps(p, default=str)[:300]}"


# ═════════════════════════════════════════════════════════════════════════════
# Background DB sink
# ═════════════════════════════════════════════════════════════════════════════
class _DbSink:
    """Background writer — a slow/dead DB can never stall the agent."""

    def __init__(
        self,
        db: Any,
        conn: str,
        on_error: Callable[[str], None],
        max_queue: int = 10_000,
    ) -> None:
        self.db = db
        self.conn = conn
        self.on_error = on_error
        self.q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self.written = 0
        self.broken = False
        self.t = threading.Thread(
            target=self._loop, name="amb-obs-db", daemon=True,
        )
        self.t.start()

    def put(self, sql: str, params: tuple) -> None:
        if self.broken:
            self.dropped += 1
            return
        try:
            self.q.put_nowait((sql, params))
        except queue.Full:
            self.dropped += 1

    def _loop(self) -> None:
        consecutive_failures = 0
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            try:
                r = self.db.execute(self.conn, item[0], item[1], fetch="none")
                if isinstance(r, dict) and r.get("ok") is False:
                    consecutive_failures += 1
                    self.on_error(str(r.get("error"))[:200])
                else:
                    consecutive_failures = 0
                    self.written += 1
            except Exception as e:      # noqa: BLE001
                consecutive_failures += 1
                self.on_error(f"{type(e).__name__}: {str(e)[:200]}")
            finally:
                self.q.task_done()

            # Give up after 20 consecutive failures — the DB is down, not slow.
            if consecutive_failures >= 20 and not self.broken:
                self.broken = True
                self.on_error(
                    "DB sink disabled after 20 consecutive failures; "
                    "further events go to file only"
                )

    def flush(self, timeout: float = 10.0) -> None:
        end = time.time() + timeout
        while self.q.unfinished_tasks and time.time() < end:
            time.sleep(0.02)

    def close(self) -> None:
        self.flush()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# File logger factory
# ═════════════════════════════════════════════════════════════════════════════
def _file_logger(
    name: str,
    path: Path,
    fmt: str,
    max_bytes: int,
    backups: int,
    datefmt: str = "",
) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    target = path.resolve()
    if not any(
        isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == target
        for h in lg.handlers
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        h = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8",
        )
        formatter = logging.Formatter(fmt)
        if datefmt:
            formatter.datefmt = datefmt
        h.setFormatter(formatter)
        lg.addHandler(h)
    return lg


# ═════════════════════════════════════════════════════════════════════════════
# RunRecorder
# ═════════════════════════════════════════════════════════════════════════════
class RunRecorder:
    """Emit every agent decision to four sinks. Never raises."""

    MAX_SINK_ERRORS = 50

    def __init__(
        self,
        run_id: str,
        *,
        log_dir: Optional[Path] = None,
        db: Any = None,
        db_conn: str = "primary",
        instruction: str = "",
        model: str = "",
        jsonl: bool = True,
        human: bool = True,
        ring: int = 500,
        max_bytes: int = 20_000_000,
        backups: int = 5,
        preview_chars: int = 300,
    ) -> None:
        self.run_id = run_id
        self.preview = preview_chars
        self.seq = 0
        self._lock = threading.Lock()
        self._ring: Deque[Dict[str, Any]] = deque(maxlen=ring)
        self.sink_errors: List[str] = []
        self.tool_calls = 0
        self.llm_calls = 0
        self.rows_written = 0
        self._jsonl: Optional[logging.Logger] = None
        self._human: Optional[logging.Logger] = None
        self._instruction = instruction or ""
        self._model = model or ""

        if log_dir is None:
            try:
                from amb_sentinel import paths
                log_dir = paths.log_dir()
            except Exception:
                log_dir = Path.cwd() / "logs"
        log_dir = Path(log_dir)

        try:
            if jsonl:
                self._jsonl = _file_logger(
                    "amb.events.jsonl",
                    log_dir / "agent_events.jsonl",
                    "%(message)s",
                    max_bytes, backups,
                )
            if human:
                self._human = _file_logger(
                    "amb.events.human",
                    log_dir / "agent_trace.log",
                    "%(asctime)s.%(msecs)03d | %(levelname)-5s | AgentTrace | %(message)s",
                    max_bytes, backups,
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
        except Exception as e:      # noqa: BLE001
            self._note_error(f"file sinks disabled: {e}")

        self._db = (
            _DbSink(db, db_conn, self._note_error)
            if db is not None else None
        )
        self._token = None

    # ── context manager ────────────────────────────────────────────────────
    def __enter__(self) -> "RunRecorder":
        self._token = CURRENT_RUN.set(self)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            CURRENT_RUN.reset(self._token)
        self.close()

    # ── introspection ──────────────────────────────────────────────────────
    def recent(self, n: int = 100, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        evs = list(self._ring)
        if kind:
            evs = [e for e in evs if e["kind"] == kind]
        return evs[-n:]

    # ── emit ───────────────────────────────────────────────────────────────
    def emit(self, kind: str, step: Optional[str] = None, **payload: Any) -> Dict[str, Any]:
        with self._lock:
            self.seq += 1
            ev = {
                "ts": time.time(),
                "run_id": self.run_id,
                "seq": self.seq,
                "kind": kind,
                "step": step,
                "payload": redact(payload, self.preview),
            }
        if kind == "llm_response":
            self.llm_calls += 1

        self._ring.append(ev)

        try:
            if self._jsonl:
                self._jsonl.info(json.dumps(ev, default=str, ensure_ascii=False))
            if self._human:
                lvl = (
                    logging.WARNING
                    if kind in ("gate_reject",) or payload.get("ok") is False
                    else logging.INFO
                )
                self._human.log(lvl, format_human(ev))
        except Exception as e:      # noqa: BLE001
            self._note_error(f"file sink: {e}")

        if self._db is not None:
            try:
                self._to_db(ev, payload)
            except Exception as e:      # noqa: BLE001
                self._note_error(f"db sink enqueue: {e}")

        return ev

    def _to_db(self, ev: Dict[str, Any], raw: Dict[str, Any]) -> None:
        d = self._db
        if d is None:
            return

        # Universal event row
        d.put(
            "INSERT INTO agent_events (run_id, seq, kind, step, payload) "
            "VALUES (%s, %s, %s, %s, %s::jsonb)",
            (
                self.run_id, ev["seq"], ev["kind"], ev["step"],
                json.dumps(ev["payload"], default=str),
            ),
        )

        k = ev["kind"]
        if k == "run_start":
            d.put(
                "INSERT INTO agent_executions "
                "(execution_id, conversation_id, instruction, status, metadata) "
                "VALUES (%s, %s, %s, 'running', %s::jsonb) "
                "ON CONFLICT (execution_id) DO UPDATE SET "
                "status='running', updated_at=CURRENT_TIMESTAMP",
                (
                    self.run_id, self.run_id, self._instruction[:2000],
                    json.dumps(ev["payload"], default=str),
                ),
            )
        elif k == "tool_result":
            p = ev["payload"]
            d.put(
                "INSERT INTO tool_executions "
                "(tool_name, execution_id, arguments, result, success, "
                " duration, error_message) "
                "VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)",
                (
                    str(p.get("tool")), self.run_id,
                    json.dumps(p.get("args") or {}, default=str),
                    json.dumps(
                        {"summary": p.get("summary"), "rows": p.get("rows")},
                        default=str,
                    ),
                    bool(p.get("ok")),
                    float(p.get("duration_s") or 0),
                    None if p.get("ok") else str(p.get("summary"))[:500],
                ),
            )
        elif k == "run_end":
            p = ev["payload"]
            d.put(
                "UPDATE agent_executions SET "
                "status=%s, success=%s, end_time=CURRENT_TIMESTAMP, "
                "iterations=%s, steps_count=%s, tools_used=%s::jsonb, "
                "result=%s::jsonb, updated_at=CURRENT_TIMESTAMP "
                "WHERE execution_id=%s",
                (
                    "completed" if p.get("success") else "failed",
                    bool(p.get("success")),
                    int(p.get("llm_calls") or 0),
                    len(p.get("step_status") or {}),
                    json.dumps(p.get("tools_used") or []),
                    json.dumps(p, default=str),
                    self.run_id,
                ),
            )

    # ── helpers ────────────────────────────────────────────────────────────
    def tool_timer(self, step: Optional[str], tool: str, args: Dict[str, Any]) -> "_ToolTimer":
        return _ToolTimer(self, step, tool, args)

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception:      # noqa: BLE001
                pass
        for lg in (self._jsonl, self._human):
            for h in (lg.handlers if lg else []):
                try:
                    h.flush()
                except Exception:      # noqa: BLE001
                    pass

    def _note_error(self, msg: str) -> None:
        if len(self.sink_errors) < self.MAX_SINK_ERRORS:
            self.sink_errors.append(msg)


class _ToolTimer:
    def __init__(self, rec: RunRecorder, step: Optional[str], tool: str, args: Dict[str, Any]):
        self.rec = rec
        self.step = step
        self.tool = tool
        self.args = args
        self.t0 = time.monotonic()
        rec.emit("tool_start", step, tool=tool, args=args)

    def done(self, ok: bool, result: Any, summary: str) -> Dict[str, Any]:
        from amb_sentinel.agent.step_gate import extract_rows
        rows = extract_rows(result)
        self.rec.tool_calls += 1

        # Count actual insertions, not duplicate-hit rows. store_raw_item
        # returns {"inserted": 1, "rows": 1} on a real insert and
        # {"inserted": 0, "rows": 1, "duplicate": true} on a duplicate.
        if ok and self.tool in ("db_execute", "db_bulk_insert", "store_raw_item"):
            if self.tool == "store_raw_item" and isinstance(result, dict):
                ins = result.get("inserted")
                if isinstance(ins, int):
                    self.rec.rows_written += ins
            elif rows is not None:
                self.rec.rows_written += int(rows)

        return self.rec.emit(
            "tool_result", self.step,
            tool=self.tool, args=self.args, ok=bool(ok),
            duration_s=round(time.monotonic() - self.t0, 3),
            rows=rows, summary=summary,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Null fallback
# ═════════════════════════════════════════════════════════════════════════════
class NullRecorder:
    """Drop-in used when observability is disabled or fails to initialise."""
    run_id = ""
    tool_calls = llm_calls = rows_written = 0
    sink_errors: List[str] = []

    def emit(self, *a, **k): return {}
    def recent(self, *a, **k): return []
    def close(self): pass
    def tool_timer(self, step, tool, args): return _NullTimer()
    def __enter__(self): return self
    def __exit__(self, *e): return None


class _NullTimer:
    def done(self, ok, result, summary): return {}
