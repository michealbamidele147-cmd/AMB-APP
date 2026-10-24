"""Background ingestion: discovers log files, tails them with adaptive polling, assembles
records and feeds the :class:`~.hub.Hub`.

One thread serves every file. Idle files back off to ``poll_max`` seconds; busy files are
read in bounded chunks so a single noisy file cannot starve the others or the UI.
"""
from __future__ import annotations

import fnmatch
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from .hub import Hub
from .model import Assembler, Record
from .tail import FileTailer


class _Entry:
    __slots__ = ("name", "tailer", "asm", "interval", "due", "had_data")

    def __init__(self, name, tailer, asm, now):
        self.name, self.tailer, self.asm = name, tailer, asm
        self.interval = 0.0
        self.due = now
        self.had_data = False


def default_name(path: Path, root: Optional[Path]) -> str:
    try:
        rel = path.relative_to(root) if root else Path(path.name)
    except ValueError:
        rel = Path(path.name)
    s = str(rel.with_suffix("")) if rel.suffix in (".log", ".txt", ".jsonl", ".out") else str(rel)
    return s.replace(os.sep, "/")


class Ingestor(threading.Thread):
    def __init__(self, hub: Hub, targets: Iterable, *, patterns=("*.log",), recursive: bool = False,
                 include: Optional[list] = None, exclude: Optional[list] = None,
                 backfill_bytes: int = 256 * 1024, poll_min: float = 0.2, poll_max: float = 2.0,
                 rescan: float = 3.0, max_open: int = 256, tag_sources: bool = False,
                 from_start: bool = False):
        super().__init__(name="logmon-ingest", daemon=True)
        self.hub = hub
        self.targets = [Path(t).expanduser() for t in targets]
        self.patterns = tuple(patterns)
        self.recursive = recursive
        self.include = [x.lower() for x in (include or [])]
        self.exclude = [x.lower() for x in (exclude or [])]
        self.backfill_bytes = backfill_bytes
        self.poll_min, self.poll_max = poll_min, poll_max
        self.rescan = rescan
        self.max_open = max_open
        self.tag_sources = tag_sources
        self.from_start = from_start
        self.ready = threading.Event()
        self._stop_evt = threading.Event()
        self._entries: dict = {}
        self._next_scan = 0.0
        self.errors = 0

    # ------------------------------------------------------------------ discovery
    def discover(self) -> list:
        found: list = []
        for t in self.targets:
            try:
                if t.is_file():
                    found.append((t, None))
                elif t.is_dir():
                    for pat in self.patterns:
                        it = t.rglob(pat) if self.recursive else t.glob(pat)
                        for p in it:
                            if p.is_file():
                                found.append((p, t))
            except OSError:
                continue
        out, seen = [], set()
        for p, root in sorted(found, key=lambda x: str(x[0])):
            if p in seen:
                continue
            seen.add(p)
            name = default_name(p, root)
            low = name.lower()
            if self.include and not any(fnmatch.fnmatch(low, i) or i in low for i in self.include):
                continue
            if any(fnmatch.fnmatch(low, e) or e in low for e in self.exclude):
                continue
            out.append((p, name))
        return out

    def _scan(self, initial: bool) -> list:
        added = []
        now = time.monotonic()
        for p, name in self.discover():
            key = str(p)
            if key in self._entries:
                continue
            transient = len(self._entries) >= self.max_open
            tailer = FileTailer(p, backfill_bytes=self.backfill_bytes,
                                from_start=self.from_start or not initial, transient=transient)
            asm = Assembler(name, tag_as_source=self.tag_sources)
            e = _Entry(name, tailer, asm, now)
            self._entries[key] = e
            if not self.tag_sources:
                self.hub.ensure_source(name, key)
            added.append(e)
        return added

    # ------------------------------------------------------------------ main loop
    def run(self):
        try:
            self._initial()
        except Exception:  # never let the monitor thread die silently
            self.errors += 1
        finally:
            self.ready.set()
        while not self._stop_evt.is_set():
            now = time.monotonic()
            if now >= self._next_scan:
                self._next_scan = now + self.rescan
                for e in self._scan(initial=False):
                    e.due = now
            nxt = now + self.poll_max
            for e in list(self._entries.values()):
                if e.due <= now:
                    self._poll(e, backfill=False)
                    now = time.monotonic()
                nxt = min(nxt, e.due)
            self._stop_evt.wait(max(0.02, min(self.poll_max, nxt - time.monotonic())))
        for e in self._entries.values():
            e.tailer.close()

    def _initial(self):
        entries = self._scan(initial=True)
        self._next_scan = time.monotonic() + self.rescan
        pending: list = []
        for e in entries:
            for _ in range(64):  # drain the backfill window
                batch = self._poll(e, backfill=True, sink=pending)
                if batch is None or not batch.more:
                    break
        self.hub.bulk(pending)

    def stop(self):
        self._stop_evt.set()

    # ------------------------------------------------------------------ one poll
    def _poll(self, e: _Entry, *, backfill: bool, sink: Optional[list] = None):
        batch = None
        try:
            batch = e.tailer.poll()
            recs, touched = e.asm.feed(batch.lines, now=time.time())
            if batch.reset and not backfill:
                marker = Record(time.time(), 20, "", e.name, "─── file rotated or truncated ───", synthetic=True)
                recs.insert(0, marker)
            if not self.tag_sources:
                self.hub.set_meta(e.name, size=batch.size, mtime=batch.mtime, missing=batch.missing,
                                  error=batch.error, rotated=batch.reset and not backfill)
            if backfill and sink is not None:
                sink.extend(recs)
                if touched:
                    self.hub.ingest([], True, live=False)
            else:
                self.hub.ingest(recs, touched, live=True)
            got = bool(batch.lines)
            if got:
                e.interval = self.poll_min
            else:
                e.interval = min(self.poll_max, max(self.poll_min, e.interval * 1.5))
            e.due = time.monotonic() + (0.0 if batch.more else e.interval)
        except Exception as exc:  # isolate one bad file from the rest
            self.errors += 1
            e.interval = self.poll_max
            e.due = time.monotonic() + self.poll_max
            if not self.tag_sources:
                self.hub.set_meta(e.name, error=f"{type(exc).__name__}: {exc}")
        return batch
