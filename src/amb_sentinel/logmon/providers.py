"""Uniform data providers so panels and the full-screen viewer treat every view the same:
one log file, the merged timeline across all files, or the de-duplicated incident list."""
from __future__ import annotations

import time
from typing import Optional

from .hub import Hub
from .model import Record


class Provider:
    name = ""
    title = ""
    kind = "source"            # source | merged | incidents
    show_source = False        # prefix rows with the originating file
    expand_default = False     # show continuation lines by default
    incremental = True         # rows can be appended without a full rebuild

    def __init__(self, hub: Hub):
        self.hub = hub

    def sort_key(self, rec: Record):
        return rec.seq

    def tail(self, n: int) -> list:
        raise NotImplementedError

    def snapshot(self) -> list:
        raise NotImplementedError

    def token(self):
        """Changes whenever previously built rows may be stale (not on plain appends)."""
        return self.hub.struct

    def rev(self):
        raise NotImplementedError

    def info(self, now: float) -> dict:
        raise NotImplementedError


class SourceProvider(Provider):
    def __init__(self, hub: Hub, name: str):
        super().__init__(hub)
        self.name = name
        self.title = name

    def _st(self):
        return self.hub.sources.get(self.name)

    def tail(self, n):
        return self.hub.tail(self.name, n)

    def snapshot(self):
        return self.hub.snapshot(self.name)

    def rev(self):
        st = self._st()
        return (st.rev, st.error, st.missing, st.size) if st else 0

    def info(self, now):
        with self.hub.lock:
            st = self._st()
            if st is None:
                return {}
            return {
                "path": st.path, "size": st.size, "state": st.status(now), "counts": dict(st.counts),
                "total": st.total, "rate": st.rate(now), "spark": st.spark(now), "rotations": st.rotations,
                "error": st.error, "last_wall": st.last_wall, "mtime": st.mtime,
            }


class MergedProvider(Provider):
    name = "ALL"
    title = "⟡ ALL · merged timeline"
    kind = "merged"
    show_source = True

    def sort_key(self, rec):
        return rec.key

    def tail(self, n):
        return self.hub.merged_tail(n)

    def snapshot(self):
        return self.hub.merged_snapshot()

    def rev(self):
        return self.hub.rev

    def info(self, now):
        t = self.hub.totals(now)
        return {"path": f"{t['sources']} files", "size": 0, "state": "live", "counts": t["counts"],
                "total": sum(t["counts"].values()), "rate": t["rate"], "spark": "", "rotations": 0,
                "error": None, "last_wall": now, "mtime": 0}


class IncidentProvider(Provider):
    name = "INCIDENTS"
    title = "⚠ INCIDENTS · unique warnings/errors"
    kind = "incidents"
    show_source = True
    expand_default = True
    incremental = False

    def sort_key(self, rec):
        return rec.key

    def tail(self, n):
        return self.hub.incident_tail(n)

    def snapshot(self):
        with self.hub.lock:
            return self.hub.incidents.as_records()

    def token(self):
        return self.hub.incidents.rev

    def rev(self):
        return self.hub.incidents.rev

    def info(self, now):
        with self.hub.lock:
            items = self.hub.incidents.items.values()
            counts = {10: 0, 20: 0, 30: 0, 40: 0, 50: 0}
            for i in items:
                counts[40 if i.level >= 40 else 30] += 1
            return {"path": f"{len(self.hub.incidents.items)} unique signatures", "size": 0,
                    "state": "hot" if counts[40] else ("warm" if counts[30] else "idle"),
                    "counts": counts, "total": sum(counts.values()), "rate": 0.0, "spark": "",
                    "rotations": 0, "error": None, "last_wall": now, "mtime": 0}


def fmt_size(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return str(n)


def fmt_age(sec: Optional[float]) -> str:
    if sec is None or sec < 0:
        return "-"
    if sec < 90:
        return f"{int(sec)}s"
    if sec < 5400:
        return f"{int(sec // 60)}m"
    if sec < 172800:
        return f"{int(sec // 3600)}h"
    return f"{int(sec // 86400)}d"


_clock_cache: dict = {}


def clock(ts: float) -> str:
    s = int(ts)
    v = _clock_cache.get(s)
    if v is None:
        if len(_clock_cache) > 2048:
            _clock_cache.clear()
        v = _clock_cache[s] = time.strftime("%H:%M:%S", time.localtime(s))
    return v
