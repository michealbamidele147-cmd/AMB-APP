"""Incremental, rotation-safe file tailing.

Handles: first-open backfill (last N bytes only, never the whole file), partial trailing
lines, rename rotation (drains the old handle first), copytruncate/truncation, files that
disappear and reappear, bounded reads per poll, and optional transient (open-per-poll) mode
so hundreds of files cannot exhaust file descriptors.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

MAX_PARTIAL = 1 << 20  # a "line" longer than this without newline is force-flushed


@dataclass
class TailBatch:
    lines: list = field(default_factory=list)
    reset: bool = False       # file rotated / truncated since last poll
    missing: bool = False
    more: bool = False        # more data is waiting; poll again immediately
    size: int = 0
    mtime: float = 0.0
    error: Optional[str] = None


class FileTailer:
    def __init__(self, path, *, backfill_bytes: int = 256 * 1024, chunk: int = 1 << 20,
                 from_start: bool = False, transient: bool = False, partial_flush_s: float = 1.0):
        self.path = os.fspath(path)
        self.backfill_bytes = backfill_bytes
        self.chunk = chunk
        self.from_start = from_start
        self.transient = transient
        self.partial_flush_s = partial_flush_s
        self._fh = None
        self._ident = None
        self._pos = 0
        self._buf = b""
        self._buf_since = 0.0
        self._opened = False

    # ------------------------------------------------------------------ public
    def poll(self) -> TailBatch:
        out = TailBatch()
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            if self._fh is not None:
                out.lines += self._drain()
                self._close()
            out.missing = True
            out.lines += self._flush_partial(force=True)
            if self._opened:
                self._ident, self._pos = None, 0      # whatever reappears is a new file
            return out
        except OSError as e:
            out.error = f"{type(e).__name__}: {e}"
            return out

        out.size, out.mtime = st.st_size, st.st_mtime
        ident = (st.st_dev, st.st_ino)
        try:
            if self._opened and (self._ident is None or (ident[1] != 0 and ident != self._ident)):
                # rename-rotation: finish the old file, then start the new one from byte 0
                if self._fh is not None:
                    out.lines += self._drain()
                out.lines += self._flush_partial(force=True)
                self._close()
                self._pos, self._ident, out.reset = 0, ident, True
            elif self._opened and st.st_size < self._pos:
                # copytruncate / truncation
                out.lines += self._flush_partial(force=True)
                self._close()
                self._pos, out.reset = 0, True

            if not self._opened or st.st_size > self._pos:
                out.lines += self._read_new(st, out)
            else:
                out.lines += self._flush_partial(force=False)
        except OSError as e:
            out.error = f"{type(e).__name__}: {e}"
            self._close()
        return out

    def close(self):
        self._close()

    # ------------------------------------------------------------------ internals
    def _close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def _open(self, st):
        fh = open(self.path, "rb")
        if not self._opened:
            start = 0 if self.from_start else max(0, st.st_size - self.backfill_bytes)
            if start > 0:
                fh.seek(start - 1)
                head = fh.read(min(65536, st.st_size - start + 1))
                if head[:1] == b"\n":
                    start = start            # we landed exactly on a line boundary
                else:
                    nl = head.find(b"\n", 1)
                    start = start + nl if nl >= 0 else st.st_size
            fh.seek(start)
            self._pos = start
            self._opened = True
        else:
            fh.seek(self._pos)
        self._fh = fh
        self._ident = (st.st_dev, st.st_ino)

    def _read_new(self, st, out: TailBatch):
        if self._fh is None:
            self._open(st)
        want = max(0, st.st_size - self._pos)
        data = self._fh.read(min(want, self.chunk)) if want else b""
        self._pos += len(data)
        out.more = self._pos < st.st_size
        lines = self._split(data)
        if self.transient:
            self._close()
        return lines

    def _drain(self) -> list:
        """Read everything left in the currently open handle (used on rotation/removal)."""
        lines: list = []
        total = 0
        while self._fh is not None and total < (16 << 20):
            data = self._fh.read(self.chunk)
            if not data:
                break
            total += len(data)
            lines += self._split(data)
        return lines

    def _split(self, data: bytes) -> list:
        if not data:
            return self._flush_partial(force=False)
        buf = self._buf + data
        parts = buf.split(b"\n")
        self._buf = parts.pop()
        lines = [p.decode("utf-8", "replace") for p in parts]
        if self._buf:
            if not self._buf_since:
                self._buf_since = time.monotonic()
            if len(self._buf) > MAX_PARTIAL:
                lines += self._flush_partial(force=True)
        else:
            self._buf_since = 0.0
        return lines

    def _flush_partial(self, force: bool) -> list:
        if not self._buf:
            return []
        if not force and (time.monotonic() - self._buf_since) < self.partial_flush_s:
            return []
        line = self._buf.decode("utf-8", "replace")
        self._buf, self._buf_since = b"", 0.0
        return [line]
