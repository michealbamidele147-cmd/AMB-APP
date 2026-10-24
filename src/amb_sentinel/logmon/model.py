"""Record model, multi-line record assembly, redaction and query filtering.

A *record* is one logical log event: a header line (``timestamp | LEVEL | logger | message``)
plus any continuation lines (tracebacks, wrapped output, dumps) that follow it.
Everything here is pure Python and thread-agnostic.
"""
from __future__ import annotations

import calendar
import itertools
import json
import re
import shlex
import time
from typing import Iterable, Optional

LEVEL_NUM = {
    "TRACE": 5, "DEBUG": 10, "INFO": 20, "NOTICE": 20, "WARN": 30, "WARNING": 30,
    "ERR": 40, "ERROR": 40, "CRITICAL": 50, "FATAL": 50,
}
LEVEL_NAME = {5: "TRACE", 10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}
LEVEL_SHORT = {5: "TRC", 10: "DBG", 20: "INF", 30: "WRN", 40: "ERR", 50: "CRT"}

_seq = itertools.count(1)


def bucket(level: int) -> int:
    """Collapse any numeric level into one of 10/20/30/40/50."""
    if level <= 10:
        return 10
    if level <= 20:
        return 20
    if level <= 30:
        return 30
    if level <= 40:
        return 40
    return 50


class Record:
    """One logical log event (header + continuation lines)."""

    __slots__ = ("seq", "ts", "level", "logger", "source", "msg", "cont", "dropped", "rev", "synthetic")

    def __init__(self, ts: float, level: int, logger: str, source: str, msg: str,
                 cont: Optional[list] = None, seq: Optional[int] = None, synthetic: bool = False):
        self.seq = next(_seq) if seq is None else seq
        self.ts = ts
        self.level = level
        self.logger = logger
        self.source = source
        self.msg = msg
        self.cont = cont if cont is not None else []
        self.dropped = 0
        self.rev = 0
        self.synthetic = synthetic

    @property
    def key(self):
        return (self.ts, self.seq)

    def text(self) -> str:
        """Full plain-text form (header + continuation)."""
        return "\n".join([self.msg, *self.cont])

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Record({LEVEL_SHORT.get(self.level, self.level)} {self.source}: {self.msg[:60]!r} +{len(self.cont)})"


# --------------------------------------------------------------------------- timestamps
_sec_cache: dict = {}
_FRAC = re.compile(r"\d+")
_TZ = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def parse_ts(s: str) -> Optional[float]:
    """Parse ``YYYY-MM-DD[ T]HH:MM:SS[.fff][Z|+hh:mm]`` to epoch seconds.

    Naive timestamps are interpreted as local time. Per-second results are cached because
    log files contain thousands of lines inside the same second.
    """
    if len(s) < 19:
        return None
    tz = None
    m = _TZ.search(s, 19)
    if m:
        tz = m.group(1)
        s = s[: m.start()]
    key = s[:19]
    base = _sec_cache.get(key)
    if base is None:
        try:
            tm = time.strptime(key.replace("T", " "), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        base = float(calendar.timegm(tm)) if tz else time.mktime(tm)
        if len(_sec_cache) > 4096:
            _sec_cache.clear()
        _sec_cache[key] = base
    frac = 0.0
    if len(s) > 20 and s[19] in ".,":
        fm = _FRAC.match(s, 20)
        if fm:
            frac = int(fm.group()) / (10 ** len(fm.group()))
    if tz and tz != "Z":
        sign = 1 if tz[0] == "+" else -1
        digits = tz[1:].replace(":", "")
        base -= sign * (int(digits[:2]) * 3600 + int(digits[2:]) * 60)
    return base + frac


# --------------------------------------------------------------------------- redaction
_REDACT = [
    (re.compile(r"(?i)\b(password|passwd|pwd|api[_-]?key|secret|token)\b(\s*[=:]\s*)(?!\*\*\*)(\S+)"), r"\1\2***"),
    (re.compile(r"(://[^:/\s@]+:)([^@\s]+)(@)"), r"\1***\3"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-~+/=]{8,}"), r"\1 ***"),
]
_REDACT_HINT = re.compile(r"(?i)pass|pwd|key|secret|token|bearer|://")


def redact(s: str) -> str:
    """Defence in depth: mask credentials even if the producer forgot to."""
    if not _REDACT_HINT.search(s):
        return s
    for rx, rep in _REDACT:
        s = rx.sub(rep, s)
    return s


# --------------------------------------------------------------------------- assembly
_STD = re.compile(
    r"^(?:\[(?P<tag>[\w.\-]+)\]\s+)?"
    r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?)"
    r"\s*\|\s*(?P<lvl>[A-Za-z]+)\s*\|\s*(?P<logger>[^|]*?)\s*\|\s?(?P<msg>.*)$"
)
_TAG = re.compile(r"^\[([\w.\-]+)\] ?")


class Assembler:
    """Turns raw lines into Records; continuation lines are attached to the previous record."""

    def __init__(self, source: str, *, tag_as_source: bool = False, max_cont: int = 500, max_line: int = 8192):
        self.source = source
        self.tag_as_source = tag_as_source
        self.max_cont = max_cont
        self.max_line = max_line
        self._cur: Optional[Record] = None
        self._tagged = False

    def feed(self, lines: Iterable[str], now: Optional[float] = None):
        """Returns ``(new_records, touched_existing)``; ``touched`` means a continuation line was
        appended to an already-emitted record."""
        new: list = []
        touched = False
        cur_new = False  # True while self._cur was created inside this call
        for raw in lines:
            line = raw.rstrip("\r\n")
            if len(line) > self.max_line:
                line = line[: self.max_line] + f" …[+{len(raw) - self.max_line} chars]"
            rec = self._header(line, now)
            if rec is not None:
                self._cur = rec
                cur_new = True
                new.append(rec)
                continue
            if self._tagged:
                line = _TAG.sub("", line, count=1)
            if not line.strip():
                continue
            line = redact(line)
            cur = self._cur
            if cur is None:
                cur = Record(now or time.time(), 20, "", self.source, line, synthetic=True)
                self._cur = cur
                cur_new = True
                new.append(cur)
                continue
            if len(cur.cont) < self.max_cont:
                cur.cont.append(line)
            else:
                cur.dropped += 1
            cur.rev += 1
            if not cur_new:
                touched = True
        return new, touched

    # ------------------------------------------------------------------
    def _header(self, line: str, now: Optional[float]) -> Optional[Record]:
        if not line:
            return None
        c = line[0]
        if c == "{":
            return self._json(line, now)
        if not (c.isdigit() or c == "["):
            return None
        m = _STD.match(line)
        if not m:
            return None
        ts = parse_ts(m.group("ts"))
        if ts is None:
            return None
        tag = m.group("tag")
        if tag:
            self._tagged = True
        src = tag if (tag and self.tag_as_source) else self.source
        level = LEVEL_NUM.get(m.group("lvl").upper())
        if level is None:
            return None  # e.g. "2026-10-21 12:00:00 | something | x | y" is not a log header
        return Record(ts, level, m.group("logger"), src, redact(m.group("msg")))

    def _json(self, line: str, now: Optional[float]) -> Optional[Record]:
        try:
            d = json.loads(line)
        except ValueError:
            return None
        if not isinstance(d, dict):
            return None
        raw_ts = d.get("ts") or d.get("time") or d.get("timestamp") or d.get("asctime") or d.get("@timestamp")
        ts = None
        if isinstance(raw_ts, (int, float)):
            ts = float(raw_ts) / (1000.0 if raw_ts > 1e11 else 1.0)
        elif isinstance(raw_ts, str):
            ts = parse_ts(raw_ts)
        if ts is None:
            ts = now or time.time()
        lvl = d.get("level") or d.get("levelname") or d.get("severity") or "INFO"
        level = lvl if isinstance(lvl, int) else LEVEL_NUM.get(str(lvl).upper(), 20)
        msg = d.get("message") or d.get("msg") or d.get("event") or ""
        logger = d.get("logger") or d.get("name") or ""
        rec = Record(ts, level, str(logger), self.source, redact(str(msg)))
        exc = d.get("exc_info") or d.get("exception") or d.get("stack")
        if exc:
            rec.cont = [redact(x) for x in str(exc).splitlines()[: self.max_cont]]
        return rec


# --------------------------------------------------------------------------- query
class Query:
    """Filter language used by the search bar.

    ``word``       substring (case-insensitive, searches message + continuation lines)
    ``-word``      must NOT contain word
    ``"a phrase"`` quoted phrase
    ``src:name``   record source contains name (repeatable = OR)
    ``lvl:error``  minimum level
    ``re:pattern`` regular expression (repeatable = AND)
    """

    def __init__(self, text: str = ""):
        self.text = text.strip()
        self.terms: list = []
        self.neg: list = []
        self.src: list = []
        self.regex: list = []
        self.min_level = 0
        self.error: Optional[str] = None
        self._parse()

    def _parse(self):
        if not self.text:
            return
        try:
            toks = shlex.split(self.text)
        except ValueError:
            toks = self.text.split()
        for t in toks:
            low = t.lower()
            if low.startswith("src:") and len(t) > 4:
                self.src.append(low[4:])
            elif low.startswith("lvl:") and len(t) > 4:
                self.min_level = LEVEL_NUM.get(t[4:].upper(), 0)
            elif low.startswith("re:") and len(t) > 3:
                try:
                    self.regex.append(re.compile(t[3:], re.I))
                except re.error as e:
                    self.error = f"bad regex: {e}"
            elif t.startswith("-") and len(t) > 1:
                self.neg.append(low[1:])
            else:
                self.terms.append(low)

    @property
    def active(self) -> bool:
        return bool(self.terms or self.neg or self.src or self.regex or self.min_level)

    def match(self, rec: Record) -> bool:
        if rec.level < self.min_level:
            return False
        if self.src:
            s = rec.source.lower()
            if not any(x in s for x in self.src):
                return False
        if not (self.terms or self.neg or self.regex):
            return True
        hay = rec.text() if rec.cont else rec.msg
        low = hay.lower()
        for t in self.terms:
            if t not in low:
                return False
        for t in self.neg:
            if t in low:
                return False
        for rx in self.regex:
            if not rx.search(hay):
                return False
        return True
