"""Shared state between the ingest thread and the UI: per-source ring buffers, a merged
time-ordered timeline, and an incident board that de-duplicates repeating errors.

Every public method takes ``hub.lock``; the UI never touches the containers directly.
"""
from __future__ import annotations

import bisect
import re
import threading
import time
import zlib
from collections import deque
from itertools import islice
from typing import Optional

from .model import LEVEL_NAME, Record, bucket

SPARK = "▁▂▃▄▅▆▇█"


# ----------------------------------------------------------------------------- per source
class SourceStore:
    def __init__(self, name: str, path: Optional[str], ring: int):
        self.name = name
        self.path = path
        self.records: deque = deque(maxlen=ring)
        self.counts = {10: 0, 20: 0, 30: 0, 40: 0, 50: 0}
        self.total = 0
        self.size = 0
        self.mtime = 0.0
        self.missing = False
        self.error: Optional[str] = None
        self.rotations = 0
        self.last_wall = 0.0       # wall-clock time of last live record
        self.last_warn_wall = 0.0
        self.last_err_wall = 0.0
        self.last_error: Optional[Record] = None
        self.rev = 0
        self._b = [0] * 60
        self._bs = [0] * 60

    def add(self, rec: Record, live: bool, now: float):
        self.records.append(rec)
        self.total += 1
        self.counts[bucket(rec.level)] += 1
        self.rev += 1
        if rec.synthetic and not rec.logger and rec.msg.startswith("───"):
            return
        # status uses *event time* (clipped to now) so errors found while backfilling
        # a log written a few seconds ago still light the panel up.
        ev = min(now, rec.ts)
        if ev > self.last_wall:
            self.last_wall = ev
        if live:
            s = int(now)
            i = s % 60
            if self._bs[i] != s:
                self._bs[i], self._b[i] = s, 0
            self._b[i] += 1
        if rec.level >= 40:
            if ev >= self.last_err_wall:
                self.last_err_wall = ev
                self.last_error = rec
        elif rec.level >= 30 and ev > self.last_warn_wall:
            self.last_warn_wall = ev

    def rate(self, now: float, window: int = 10) -> float:
        s = int(now)
        tot = sum(c for c, st in zip(self._b, self._bs) if s - window < st <= s)
        return tot / window

    def spark(self, now: float, n: int = 12, span: int = 60) -> str:
        s = int(now)
        step = span // n
        vals = []
        for k in range(n):
            hi = s - (n - 1 - k) * step
            lo = hi - step
            vals.append(sum(c for c, st in zip(self._b, self._bs) if lo < st <= hi))
        m = max(vals)
        if m == 0:
            return "·" * n
        return "".join(SPARK[min(7, int(v / m * 7.999))] if v else "·" for v in vals)

    def status(self, now: float) -> str:
        if self.missing:
            return "missing"
        if self.error:
            return "error"
        if self.last_err_wall and now - self.last_err_wall < 60:
            return "hot"
        if self.last_warn_wall and now - self.last_warn_wall < 60:
            return "warm"
        if self.last_wall and now - self.last_wall < 30:
            return "live"
        if self.total == 0:
            return "empty"
        return "idle"


# ----------------------------------------------------------------------------- incidents
_NORM = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0x#"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"), "<ts>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<uuid>"),
    (re.compile(r"(?:/[\w.\-@+~]+){2,}"), "<path>"),
    (re.compile(r"\d+(?:\.\d+)?"), "#"),
]
_EMOJI = re.compile(r"^[^\w\[\(<'\"]+")


def fingerprint(msg: str) -> str:
    """Stable signature of a message: numbers, paths, ids and timestamps are normalised."""
    s = _EMOJI.sub("", msg.strip())
    for rx, rep in _NORM:
        s = rx.sub(rep, s)
    return s[:200]


class Incident:
    __slots__ = ("sig", "level", "count", "first_ts", "last_ts", "last_wall", "sources", "sample", "recent", "id")

    def __init__(self, sig: str, rec: Record, now: float):
        self.sig = sig
        self.id = zlib.crc32(sig.encode()) | 0x40000000  # synthetic seq space
        self.level = rec.level
        self.count = 0
        self.first_ts = rec.ts
        self.last_ts = rec.ts
        self.last_wall = now
        self.sources: dict = {}
        self.sample = rec
        self.recent: deque = deque(maxlen=300)


class IncidentBoard:
    def __init__(self, cap: int = 1000):
        self.cap = cap
        self.items: dict = {}
        self.rev = 0

    def observe(self, rec: Record, now: float):
        if rec.level < 30 or rec.synthetic:
            return
        sig = fingerprint(rec.msg)
        if not sig:
            return
        inc = self.items.get(sig)
        if inc is None:
            if len(self.items) >= self.cap:
                oldest = min(self.items.values(), key=lambda i: i.last_wall)
                del self.items[oldest.sig]
            inc = self.items[sig] = Incident(sig, rec, now)
        inc.count += 1
        inc.level = max(inc.level, rec.level)
        inc.first_ts = min(inc.first_ts, rec.ts)
        if rec.ts >= inc.last_ts:
            inc.last_ts = rec.ts
            inc.sample = rec
        inc.last_wall = now
        inc.sources[rec.source] = inc.sources.get(rec.source, 0) + 1
        inc.recent.append(rec.ts)
        self.rev += 1

    def top(self, min_level: int = 30):
        return sorted((i for i in self.items.values() if i.level >= min_level),
                      key=lambda i: (i.last_ts, i.count), reverse=True)

    def as_records(self, min_level: int = 30) -> list:
        """Synthetic records (oldest first) so the generic log view can display incidents."""
        out = []
        for inc in reversed(self.top(min_level)):
            head = f"×{inc.count:<4} {inc.sample.msg}"
            span = f"first {time.strftime('%H:%M:%S', time.localtime(inc.first_ts))}  last {time.strftime('%H:%M:%S', time.localtime(inc.last_ts))}"
            srcs = ", ".join(f"{k}×{v}" for k, v in sorted(inc.sources.items(), key=lambda kv: -kv[1])[:6])
            cont = [f"{span}   sources: {srcs}", *inc.sample.cont[:60]]
            r = Record(inc.last_ts, inc.level, "", ",".join(list(inc.sources)[:3]), head, cont, seq=inc.id, synthetic=True)
            out.append(r)
        return out


# ----------------------------------------------------------------------------- hub
class Hub:
    def __init__(self, ring: int = 5000, merged_cap: int = 30000):
        self.lock = threading.RLock()
        self.ring = ring
        self.merged_cap = merged_cap
        self.sources: dict = {}
        self.merged: list = []
        self.merged_keys: list = []
        self.incidents = IncidentBoard()
        self.rev = 0
        self.struct = 0            # bumps when merged is edited anywhere except the tail
        self.new_sources: list = []
        self.started = time.time()

    # --- sources ---------------------------------------------------------
    def ensure_source(self, name: str, path: Optional[str] = None) -> SourceStore:
        with self.lock:
            st = self.sources.get(name)
            if st is None:
                st = self.sources[name] = SourceStore(name, path, self.ring)
                self.new_sources.append(name)
                self.rev += 1
            return st

    def pop_new_sources(self) -> list:
        with self.lock:
            out, self.new_sources = self.new_sources, []
            return out

    def set_meta(self, name: str, *, size=None, mtime=None, missing=None, error=None, rotated=False):
        with self.lock:
            st = self.sources.get(name)
            if st is None:
                return
            if size is not None:
                st.size = size
            if mtime is not None:
                st.mtime = mtime
            if missing is not None:
                st.missing = missing
            st.error = error
            if rotated:
                st.rotations += 1
            self.rev += 1

    # --- ingest ------------------------------------------------------------
    def ingest(self, recs: list, touched: bool = False, *, live: bool = True):
        now = time.time()
        with self.lock:
            for r in recs:
                st = self.sources.get(r.source) or self.ensure_source(r.source)
                st.add(r, live, now)
                self._merge(r)
                self.incidents.observe(r, now)
            if touched:
                self.struct += 1
            if recs or touched:
                self.rev += 1

    def bulk(self, recs: list):
        """Initial backfill: sort once, then keep only the newest ``merged_cap``."""
        now = time.time()
        with self.lock:
            recs = sorted(recs, key=lambda r: r.key)
            for r in recs:
                st = self.sources.get(r.source) or self.ensure_source(r.source)
                st.add(r, False, now)
                self.incidents.observe(r, now)
            merged = sorted(self.merged + recs, key=lambda r: r.key)[-self.merged_cap:]
            self.merged = merged
            self.merged_keys = [r.key for r in merged]
            self.struct += 1
            self.rev += 1

    def _merge(self, r: Record):
        k = r.key
        if not self.merged_keys or k >= self.merged_keys[-1]:
            self.merged.append(r)
            self.merged_keys.append(k)
        else:
            i = bisect.bisect_right(self.merged_keys, k)
            self.merged.insert(i, r)
            self.merged_keys.insert(i, k)
            self.struct += 1
        if len(self.merged) > self.merged_cap * 1.2:
            drop = len(self.merged) - self.merged_cap
            del self.merged[:drop]
            del self.merged_keys[:drop]
            self.struct += 1

    # --- reads (UI) ----------------------------------------------------------
    def tail(self, name: str, n: int) -> list:
        with self.lock:
            st = self.sources.get(name)
            if st is None:
                return []
            return list(islice(reversed(st.records), n))[::-1]

    def merged_tail(self, n: int) -> list:
        with self.lock:
            return self.merged[-n:]

    def snapshot(self, name: str) -> list:
        with self.lock:
            st = self.sources.get(name)
            return list(st.records) if st else []

    def merged_snapshot(self) -> list:
        with self.lock:
            return list(self.merged)

    def incident_tail(self, n: int) -> list:
        with self.lock:
            return self.incidents.as_records()[-n:]

    def totals(self, now: Optional[float] = None) -> dict:
        now = now or time.time()
        with self.lock:
            rate = sum(s.rate(now) for s in self.sources.values())
            states: dict = {}
            counts = {10: 0, 20: 0, 30: 0, 40: 0, 50: 0}
            for s in self.sources.values():
                states[s.status(now)] = states.get(s.status(now), 0) + 1
                for k, v in s.counts.items():
                    counts[k] += v
            return {"sources": len(self.sources), "rate": rate, "states": states, "counts": counts,
                    "incidents": len(self.incidents.items)}

    def level_name(self, lvl: int) -> str:
        return LEVEL_NAME.get(bucket(lvl), "INFO")
