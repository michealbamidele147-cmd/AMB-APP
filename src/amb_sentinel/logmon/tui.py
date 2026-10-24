"""Textual user interface.

Main screen  : paginated grid with one live panel per log file (+ merged ALL and INCIDENTS).
Click a panel: opens it full screen (virtualised, searchable, follow/pause, folding).
Esc / q / click the top bar: back to the grid.
"""
from __future__ import annotations

import math
import re
import time
import zlib
from typing import Optional

from rich.cells import cell_len, chop_cells
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.screen import ModalScreen, Screen
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Footer, Input, Static

from .hub import Hub
from .model import LEVEL_SHORT, Query, Record, bucket
from .providers import (IncidentProvider, MergedProvider, Provider, SourceProvider, clock,
                        fmt_age, fmt_size)

SRC_W = 12
MIN_W, MIN_H = 46, 8
LVL_STYLE = {10: "dim", 20: "", 30: "yellow", 40: "bold red", 50: "bold white on red"}
STATE_STYLE = {"hot": "bold red", "warm": "yellow", "live": "green", "idle": "dim", "empty": "dim",
               "missing": "bold magenta", "error": "bold magenta"}
PALETTE = ["cyan", "green", "magenta", "blue", "yellow", "bright_cyan", "bright_green", "bright_magenta"]
_EXC = re.compile(r"^\s*[\w.]*(Error|Exception|Unavailable|Timeout|Exit|Failure)\b")


def src_style(name: str) -> str:
    return PALETTE[zlib.crc32(name.encode()) % len(PALETTE)]


def panel_id(name: str) -> str:
    """Valid, collision-free DOM id for a source name (names may contain '/', '.', etc.)."""
    return f"p-{re.sub(r'[^A-Za-z0-9_-]', '_', name)}-{zlib.crc32(name.encode()) & 0xFFFF:04x}"


def counts_text(counts: dict) -> Text:
    t = Text()
    e, w, i = counts.get(40, 0) + counts.get(50, 0), counts.get(30, 0), counts.get(20, 0)
    t.append(f"E{e}", "bold red" if e else "dim")
    t.append(" ")
    t.append(f"W{w}", "yellow" if w else "dim")
    t.append(" ")
    t.append(f"I{i}", "dim")
    return t


def layout_for(k: int, width: int, height: int, max_cols: int, max_rows: int):
    """Pick (cols, rows) so cells are close to a comfortable ~2.4:1 character aspect."""
    if k <= 0:
        return 1, 1
    best, best_score = None, 1e9
    for cols in range(1, min(k, max_cols) + 1):
        rows = math.ceil(k / cols)
        if rows > max_rows:
            continue
        cw, ch = width / cols, height / rows
        score = abs(math.log(max(cw, 1) / max(ch, 1) / 2.4)) + (rows * cols - k) * 0.15
        if score < best_score:
            best, best_score = (cols, rows), score
    if best is None:
        cols = max_cols
        best = (cols, math.ceil(k / cols))
    return best


# =============================================================================== panel
class LogPanel(Widget, can_focus=True):
    DEFAULT_CSS = """
    LogPanel { border: round $primary-darken-2; padding: 0 1; width: 100%; height: 100%;
               text-wrap: nowrap; text-overflow: ellipsis; }
    LogPanel:focus { border: heavy $accent; }
    LogPanel.live { border: round $success-darken-1; }
    LogPanel.warm { border: round $warning; }
    LogPanel.hot { border: round $error; }
    LogPanel.missing, LogPanel.error { border: round $secondary; }
    LogPanel.idle, LogPanel.empty { border: round $primary-darken-3; color: $text-muted; }
    LogPanel:focus.hot { border: heavy $error; }
    """
    STATES = ("live", "warm", "hot", "idle", "empty", "missing", "error")

    def __init__(self, provider: Provider, **kw):
        super().__init__(**kw)
        self.provider = provider
        self._rev = None
        self._state = None
        self._tick_sec = -1

    def render(self) -> Text:
        h = max(1, self.size.height)
        w = max(8, self.size.width)
        recs = self.provider.tail(h)
        t = Text(no_wrap=True, overflow="ellipsis")
        if not recs:
            info = self.provider.info(time.time())
            msg = "file missing" if info.get("state") == "missing" else (info.get("error") or "no records yet")
            t.append(msg, "dim italic")
            return t
        show_src = self.provider.show_source
        for n, r in enumerate(recs):
            t.append(clock(r.ts) + " ", "dim")
            t.append(LEVEL_SHORT[bucket(r.level)] + " ", LVL_STYLE[bucket(r.level)])
            if show_src:
                t.append(f"{r.source[:8]:<8} ", src_style(r.source.split(",")[0]))
            t.append(r.msg, LVL_STYLE[bucket(r.level)] if r.level >= 30 else "")
            if r.cont:
                t.append(f" ↳{len(r.cont)}", "dim")
            if n < len(recs) - 1:
                t.append("\n")
        return t

    def sync(self, now: float, force: bool = False):
        prov = self.provider
        info = prov.info(now)
        if not info:
            return
        state = info["state"]
        if state != self._state:
            if self._state:
                self.remove_class(self._state)
            self.add_class(state)
            self._state = state
        sec = int(now)
        rev = prov.rev()
        title = Text(f" {prov.title} ", "bold")
        width = self.size.width + 2
        sub = Text(" ")
        sub.append_text(counts_text(info["counts"]))
        sub.append(f" · {info['rate']:.1f}/s")
        if prov.kind == "source":
            lw = info.get("last_wall") or 0
            sub.append(" · ")
            sub.append(state if state != "idle" or not lw else f"idle {fmt_age(now - lw)}",
                       STATE_STYLE.get(state, ""))
            if width >= 52:
                sub.append(f" · {fmt_size(info['size'])}")
        if info.get("spark") and width >= 62:
            sub.append(" " + info["spark"], "cyan")
        sub.append(" ")
        self.border_title = title
        self.border_subtitle = sub
        if force or rev != self._rev or sec != self._tick_sec and state in ("hot", "warm", "live"):
            self._rev, self._tick_sec = rev, sec
            self.refresh()

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.focus()
        self.app.open_view(self.provider)  # type: ignore[attr-defined]

    def key_enter(self) -> None:
        self.app.open_view(self.provider)  # type: ignore[attr-defined]


# =============================================================================== log view
class LogView(ScrollView, can_focus=True):
    DEFAULT_CSS = "LogView { height: 1fr; width: 100%; background: $surface; }"
    BINDINGS = [
        Binding("home,g", "top", "Top", show=False),
        Binding("end,G", "bottom", "Bottom", show=False),
    ]

    def __init__(self, provider: Provider, **kw):
        super().__init__(**kw)
        self.provider = provider
        self.query_obj = Query("")
        self.min_level = 0
        self.follow = True
        self.wrap = True
        self.show_ts = True
        self.expand_all = provider.expand_default
        self.paused = False
        self._expanded: set = set()
        self._rows: list = []
        self._token = None
        self._rev = None
        self._last_key = None
        self._need_full = True
        self._built_w = 0
        self._maxw = 0
        self._follow_y = 0
        self._landed = False
        self.matches = 0

    # ---------------------------------------------------------------- configuration
    def set_query(self, text: str):
        self.query_obj = Query(text)
        self._need_full = True

    def _prefix_len(self) -> int:
        n = 4
        if self.show_ts:
            n += 9
        if self.provider.show_source:
            n += SRC_W + 1
        return n

    def _passes(self, rec: Record) -> bool:
        if rec.level < self.min_level:
            return False
        q = self.query_obj
        return q.match(rec) if q.active else True

    def _is_expanded(self, rec: Record) -> bool:
        return bool(rec.cont) and (self.expand_all or rec.seq in self._expanded)

    def _text_of(self, rec: Record, k: int, expanded: bool) -> str:
        if k == 0:
            if rec.cont and not expanded:
                return f"{rec.msg}  ▸+{len(rec.cont)}{'+' if rec.dropped else ''}"
            return rec.msg
        if k <= len(rec.cont):
            return rec.cont[k - 1]
        return f"… {rec.dropped} more lines not stored"

    # ---------------------------------------------------------------- row building
    def _cw(self) -> int:
        """Usable content width (excludes scrollbars; ``size`` includes them)."""
        return self.scrollable_content_region.width

    def _avail(self) -> int:
        return max(20, self._cw() - self._prefix_len() - 1)

    def _rows_for(self, rec: Record, avail: int, out: list):
        exp = self._is_expanded(rec)
        n_lines = 1 + ((len(rec.cont) + (1 if rec.dropped else 0)) if exp else 0)
        pre = self._prefix_len()
        for k in range(n_lines):
            text = self._text_of(rec, k, exp)
            width = cell_len(text)
            if self.wrap:
                chunks = max(1, -(-width // avail))
            else:
                chunks = 1
                if width + pre > self._maxw:
                    self._maxw = width + pre
            for i in range(chunks):
                out.append((rec, k, i))

    def _maintain_follow(self):
        """Run every tick (even without new data): detect user scroll-away, re-apply the
        bottom position when layout was not ready yet, resume following at the bottom."""
        max_y = int(self.max_scroll_y)
        if self.follow:
            if self.scroll_y >= self._follow_y - 1:
                self._landed = True
            elif self._landed:
                self.follow = False            # the user scrolled up
        elif max_y > 0 and self.scroll_y >= max_y:
            self.follow, self._landed = True, True   # user scrolled back to the bottom
        if self.follow and self.scroll_y < max_y - 1 and not self._landed:
            self._follow_y = max_y
            self.scroll_to(y=max_y, animate=False, immediate=True)

    def refresh_data(self, force: bool = False):
        if self.paused and not force:
            return
        prov = self.provider
        w = self._cw()
        if w <= 0:
            return
        self._maintain_follow()

        token, rev = prov.token(), prov.rev()
        full = force or self._need_full or token != self._token or w != self._built_w
        if not full and rev == self._rev:
            return
        snap = prov.snapshot()
        recs: list = []
        if not full and prov.incremental:
            key = self._last_key
            new: list = []
            for r in reversed(snap):
                if key is not None and prov.sort_key(r) <= key:
                    break
                new.append(r)
            new.reverse()
            recs = new
            if len(self._rows) > 4 * max(len(snap), 500):
                full = True
        if full:
            self._rows = []
            self._maxw = 0
            recs = snap
            self._landed = False
        avail = self._avail()
        for r in recs:
            if self._passes(r):
                self._rows_for(r, avail, self._rows)
        if snap:
            self._last_key = prov.sort_key(snap[-1])
        self._token, self._rev, self._built_w, self._need_full = token, rev, w, False
        self.virtual_size = Size(w if self.wrap else max(w, self._maxw), len(self._rows))
        if self.follow:
            self._follow_y = int(self.max_scroll_y)
            self.scroll_to(y=self._follow_y, animate=False, immediate=True)
        self.refresh()

    # ---------------------------------------------------------------- rendering
    def _styled(self, rec: Record, k: int, i: int, avail: int) -> Text:
        t = Text(no_wrap=True, end="", overflow="crop")
        pre = self._prefix_len()
        b = bucket(rec.level)
        if k == 0 and i == 0:
            if self.show_ts:
                t.append(clock(rec.ts) + " ", "dim")
            t.append(LEVEL_SHORT[b] + " ", LVL_STYLE[b])
            if self.provider.show_source:
                t.append(f"{rec.source[:SRC_W]:<{SRC_W}} ", src_style(rec.source.split(",")[0]))
        else:
            t.append(" " * (pre - 2) + ("│ " if k > 0 else "  "))
        exp = self._is_expanded(rec)
        text = self._text_of(rec, k, exp)
        if self.wrap:
            parts = chop_cells(text, avail) or [""]
            text = parts[i] if i < len(parts) else ""
        if k == 0:
            style = LVL_STYLE[b] if rec.level >= 30 else ""
        elif _EXC.match(text):
            style = "bold red"
        else:
            style = "dim"
        t.append(text, style)
        for term in self.query_obj.terms:
            t.highlight_words([term], "reverse", case_sensitive=False)
        return t

    def render_line(self, y: int) -> Strip:
        width = self._cw()
        sx, sy = self.scroll_offset
        idx = sy + y
        if idx >= len(self._rows) or width <= 0:
            return Strip.blank(width, self.rich_style)
        rec, k, i = self._rows[idx]
        t = self._styled(rec, k, i, self._avail())
        strip = Strip(list(t.render(self.app.console)))
        return strip.crop_extend(sx, sx + width, self.rich_style)

    # ---------------------------------------------------------------- actions
    def action_top(self):
        self.follow = False
        self.scroll_to(y=0, animate=False, immediate=True)

    def action_bottom(self):
        self.follow = True
        self._landed = False
        self._follow_y = int(self.max_scroll_y)
        self.scroll_to(y=self._follow_y, animate=False, immediate=True)

    def jump_error(self, direction: int):
        cur = int(self.scroll_y)
        rng = range(cur + 1, len(self._rows)) if direction > 0 else range(cur - 1, -1, -1)
        for idx in rng:
            rec, k, i = self._rows[idx]
            if k == 0 and i == 0 and rec.level >= 40:
                self.follow = False
                self.scroll_to(y=idx, animate=False, immediate=True)
                return True
        return False

    def on_click(self, event: events.Click) -> None:
        idx = self.scroll_offset.y + event.y
        if 0 <= idx < len(self._rows):
            rec = self._rows[idx][0]
            if rec.cont:
                if rec.seq in self._expanded:
                    self._expanded.discard(rec.seq)
                else:
                    self._expanded.add(rec.seq)
                self._need_full = True
                self.refresh_data(force=True)

    def on_resize(self, event: events.Resize) -> None:
        self._need_full = True
        self._landed = False


# =============================================================================== screens
class BackBar(Static):
    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.screen.action_back()  # type: ignore[attr-defined]


class HelpScreen(ModalScreen):
    BINDINGS = [Binding("escape,q,question_mark,enter,space", "dismiss", "Close")]
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > Static { width: 78; height: auto; border: round $accent; padding: 1 2; background: $panel; }
    """
    TEXT = (
        "[b]Grid[/b]   click/Enter open panel · arrows/Tab move · [ ] or PgUp/PgDn page\n"
        "       a merged ALL · i INCIDENTS · / filter panels · s sort · h hide idle · p problems only\n"
        "       r rescan files · q quit\n\n"
        "[b]Full view[/b]   Esc/q/click top bar = back · / filter · c clear · l min level\n"
        "       f follow · space pause · w wrap · t timestamps · e expand tracebacks\n"
        "       n/N next/prev error · g/G top/bottom · click a row = fold/unfold its traceback\n\n"
        "[b]Filter syntax[/b]   word  -exclude  \"a phrase\"  src:name  lvl:error  re:pattern\n"
        "[b]Incidents[/b]       identical errors (numbers/paths normalised) are grouped with a count"
    )

    def compose(self) -> ComposeResult:
        yield Static(self.TEXT)


class FullView(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "back", "Back", show=False),
        Binding("slash", "search", "Filter"),
        Binding("c", "clear", "Clear"),
        Binding("l", "level", "Level"),
        Binding("f", "follow", "Follow"),
        Binding("space", "pause", "Pause"),
        Binding("w", "wrap", "Wrap"),
        Binding("t", "ts", "Time", show=False),
        Binding("e", "expand", "Expand"),
        Binding("n", "next_err", "Next err"),
        Binding("N", "prev_err", "Prev err", show=False),
        Binding("question_mark", "help", "Help", show=False),
    ]
    DEFAULT_CSS = """
    FullView #fv-back { height: 1; background: $accent; color: $text; text-style: bold; padding: 0 1; }
    FullView #fv-info { height: 1; padding: 0 1; background: $boost; }
    FullView #fv-q { display: none; dock: bottom; margin-bottom: 1; }
    FullView #fv-q.-open { display: block; }
    """

    def __init__(self, provider: Provider):
        super().__init__()
        self.provider = provider
        self.view = LogView(provider, id="fv-log")

    def compose(self) -> ComposeResult:
        yield BackBar(f"◀ BACK to all panels   (Esc / q / click here)      {self.provider.title}", id="fv-back")
        yield Static(id="fv-info")
        yield self.view
        yield Input(placeholder='filter:  words  -exclude  src:name  lvl:error  re:pattern   (Enter keeps, Esc closes)',
                    id="fv-q", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.view.focus()
        self.set_interval(0.2, self._tick)
        self.call_after_refresh(lambda: self.view.refresh_data(force=True))

    # ---------------------------------------------------------------- periodic
    def _tick(self):
        try:
            self.view.refresh_data()
            self._info()
        except Exception as exc:  # keep the viewer alive
            self.query_one("#fv-info", Static).update(f"view error: {exc!r}")

    def _info(self):
        now = time.time()
        info = self.provider.info(now)
        v = self.view
        t = Text()
        t.append(self.provider.name, "bold")
        if info:
            t.append(f"  {info.get('path') or ''}", "dim")
            if self.provider.kind == "source":
                t.append(f"  {fmt_size(info['size'])}", "dim")
            t.append("  ")
            t.append(info["state"], STATE_STYLE.get(info["state"], ""))
            t.append("  ")
            t.append_text(counts_text(info["counts"]))
            t.append(f"  {info['rate']:.1f}/s {info.get('spark', '')}", "cyan")
            if info.get("rotations"):
                t.append(f"  rot×{info['rotations']}", "magenta")
            if info.get("error"):
                t.append(f"  ERR {info['error']}", "bold red")
        t.append(f"   rows {len(v._rows)}", "dim")
        t.append("  FOLLOW" if v.follow else "  scroll", "green" if v.follow else "yellow")
        if v.paused:
            t.append("  PAUSED", "bold yellow")
        if v.min_level:
            t.append(f"  ≥{LEVEL_SHORT[bucket(v.min_level)]}", "bold")
        if v.query_obj.text:
            t.append(f"  filter: {v.query_obj.text}", "bold cyan")
            if v.query_obj.error:
                t.append(f"  ({v.query_obj.error})", "red")
        self.query_one("#fv-info", Static).update(t)

    # ---------------------------------------------------------------- actions
    def action_back(self):
        q = self.query_one("#fv-q", Input)
        if q.has_class("-open"):
            q.remove_class("-open")
            q.disabled = True
            self.view.focus()
            return
        self.app.pop_screen()

    def action_search(self):
        q = self.query_one("#fv-q", Input)
        q.disabled = False
        q.value = self.view.query_obj.text
        q.add_class("-open")
        self.call_after_refresh(q.focus)

    def action_clear(self):
        self.view.min_level = 0
        self.view.set_query("")
        self.query_one("#fv-q", Input).value = ""
        self.view.refresh_data(force=True)

    def action_level(self):
        order = [0, 20, 30, 40]
        cur = self.view.min_level
        self.view.min_level = order[(order.index(cur) + 1) % len(order)] if cur in order else 0
        self.view._need_full = True
        self.view.refresh_data(force=True)

    def action_follow(self):
        self.view.follow = not self.view.follow
        if self.view.follow:
            self.view.action_bottom()

    def action_pause(self):
        self.view.paused = not self.view.paused
        if not self.view.paused:
            self.view.refresh_data(force=True)

    def action_wrap(self):
        self.view.wrap = not self.view.wrap
        self.view.scroll_to(x=0, animate=False)
        self.view._need_full = True
        self.view.refresh_data(force=True)

    def action_ts(self):
        self.view.show_ts = not self.view.show_ts
        self.view._need_full = True
        self.view.refresh_data(force=True)

    def action_expand(self):
        self.view.expand_all = not self.view.expand_all
        self.view._need_full = True
        self.view.refresh_data(force=True)

    def action_next_err(self):
        self.view.jump_error(1)

    def action_prev_err(self):
        self.view.jump_error(-1)

    def action_help(self):
        self.app.push_screen(HelpScreen())

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "fv-q":
            self.view.set_query(event.value)
            self.view.refresh_data(force=True)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "fv-q":
            event.input.remove_class("-open")
            event.input.disabled = True
            self.view.focus()


# =============================================================================== app
class LogMonApp(App):
    TITLE = "AMB-Sentinel · log monitor"
    CSS = """
    #top { height: 1; padding: 0 1; background: $boost; }
    #grid { layout: grid; grid-size: 1 1; grid-gutter: 0; height: 1fr; }
    #pf { display: none; dock: bottom; margin-bottom: 1; }
    #pf.-open { display: block; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("pagedown,right_square_bracket", "page(1)", "Next page"),
        Binding("pageup,left_square_bracket", "page(-1)", "Prev page"),
        Binding("a", "open_all", "ALL"),
        Binding("i", "open_incidents", "Incidents"),
        Binding("slash", "pfilter", "Filter"),
        Binding("s", "sort", "Sort"),
        Binding("h", "hide_idle", "Hide idle"),
        Binding("p", "problems", "Problems"),
        Binding("r", "rescan", "Rescan", show=False),
        Binding("question_mark", "help", "Help"),
        Binding("left", "move(-1,0)", show=False),
        Binding("right", "move(1,0)", show=False),
        Binding("up", "move(0,-1)", show=False),
        Binding("down", "move(0,1)", show=False),
    ]

    def __init__(self, hub: Hub, ingestor=None):
        super().__init__()
        self.hub = hub
        self.ingestor = ingestor
        self.panels: dict = {}
        self.pinned = ["ALL", "INCIDENTS"]
        self._sort = "name"
        self._hide_idle = False
        self._problems = False
        self._pfilter = ""
        self._page = 0
        self._pages = 1
        self._cols = 1
        self._order: list = []
        self._page_names: list = []
        self._last_sort = 0.0
        self._dirty_layout = True
        self.error_msg = ""

    # ---------------------------------------------------------------- compose / mount
    def compose(self) -> ComposeResult:
        yield Static(id="top")
        yield Grid(id="grid")
        yield Input(placeholder="filter panels by name (substring, or re:pattern) — Enter keeps, Esc clears",
                    id="pf", disabled=True)
        yield Footer()

    async def on_mount(self) -> None:
        grid = self.query_one("#grid", Grid)
        await grid.mount(LogPanel(MergedProvider(self.hub), id="p-ALL"),
                         LogPanel(IncidentProvider(self.hub), id="p-INCIDENTS"))
        self.panels["ALL"] = grid.query_one("#p-ALL", LogPanel)
        self.panels["INCIDENTS"] = grid.query_one("#p-INCIDENTS", LogPanel)
        self.set_interval(0.25, self._tick)
        self.call_after_refresh(self._first_layout)

    def _first_layout(self):
        self._relayout()
        self.panels["ALL"].focus()

    def on_resize(self, event: events.Resize) -> None:
        self._dirty_layout = True

    # ---------------------------------------------------------------- panel set
    async def _adopt(self):
        names = self.hub.pop_new_sources()
        if not names:
            return
        grid = self.query_one("#grid", Grid)
        new = []
        for n in names:
            if n in self.panels:
                continue
            p = LogPanel(SourceProvider(self.hub, n), id=panel_id(n))
            p.display = False
            self.panels[n] = p
            new.append(p)
        if new:
            await grid.mount(*new)
            self._dirty_layout = True

    def _status_of(self, name: str, now: float) -> str:
        st = self.hub.sources.get(name)
        return st.status(now) if st else "empty"

    def _visible_names(self, now: float) -> list:
        names = [n for n in self.panels if n not in self.pinned]
        f = self._pfilter.strip()
        if f:
            if f.lower().startswith("re:"):
                try:
                    rx = re.compile(f[3:], re.I)
                    names = [n for n in names if rx.search(n)]
                except re.error:
                    pass
            else:
                fl = f.lower()
                names = [n for n in names if fl in n.lower()]
        if self._hide_idle:
            names = [n for n in names if self._status_of(n, now) not in ("idle", "empty")]
        if self._problems:
            names = [n for n in names if self._status_of(n, now) in ("hot", "warm", "error", "missing")]
        if self._sort == "name":
            names.sort(key=str.lower)
        elif self._sort == "activity":
            names.sort(key=lambda n: -(self.hub.sources[n].last_wall if n in self.hub.sources else 0))
        else:
            names.sort(key=lambda n: -(self.hub.sources[n].last_err_wall if n in self.hub.sources else 0))
        return self.pinned + names

    def _relayout(self):
        grid = self.query_one("#grid", Grid)
        W, H = grid.size.width, grid.size.height
        if W <= 0 or H <= 0:
            return
        now = time.time()
        order = self._visible_names(now)
        max_cols, max_rows = max(1, W // MIN_W), max(1, H // MIN_H)
        per_page = max_cols * max_rows
        self._pages = max(1, math.ceil(len(order) / per_page))
        self._page = max(0, min(self._page, self._pages - 1))
        page = order[self._page * per_page:(self._page + 1) * per_page]
        cols, rows = layout_for(len(page), W, H, max_cols, max_rows)
        self._cols = cols
        grid.styles.grid_size_columns = cols
        grid.styles.grid_size_rows = rows
        grid.styles.grid_rows = "1fr"
        grid.styles.grid_columns = "1fr"
        desired = page + [n for n in self.panels if n not in page]
        if desired != self._order:
            for n in reversed(desired):
                p = self.panels.get(n)
                if p is not None and p.is_attached:
                    grid.move_child(p, before=0)
            self._order = desired
        shown = set(page)
        for n, p in self.panels.items():
            p.display = n in shown
        self._page_names = page
        self._dirty_layout = False
        for n in page:
            self.panels[n].sync(now, force=True)

    # ---------------------------------------------------------------- tick
    async def _tick(self):
        try:
            await self._adopt()
            now = time.time()
            if self._sort != "name" and now - self._last_sort > 3:
                self._last_sort = now
                self._dirty_layout = True
            if self._dirty_layout:
                self._relayout()
            if len(self.screen_stack) <= 1:
                for n in self._page_names:
                    p = self.panels.get(n)
                    if p is not None:
                        p.sync(now)
            self._status(now)
        except Exception as exc:  # the monitor itself must not die
            self.error_msg = f"ui error: {exc!r}"
            self.query_one("#top", Static).update(Text(self.error_msg, "bold red"))

    def _status(self, now: float):
        t = self.hub.totals(now)
        s = t["states"]
        line = Text()
        line.append("AMB-Sentinel logs ", "bold")
        line.append(f"│ files {t['sources']} ")
        line.append(f"live {s.get('live', 0)} ", "green")
        line.append(f"warm {s.get('warm', 0)} ", "yellow")
        line.append(f"hot {s.get('hot', 0)} ", "bold red" if s.get("hot") else "dim")
        line.append(f"idle {s.get('idle', 0) + s.get('empty', 0)} ", "dim")
        if s.get("missing") or s.get("error"):
            line.append(f"missing/err {s.get('missing', 0) + s.get('error', 0)} ", "magenta")
        line.append(f"│ {t['rate']:.1f} lines/s │ ")
        line.append_text(counts_text(t["counts"]))
        line.append(f" │ incidents {t['incidents']}")
        line.append(f" │ page {self._page + 1}/{self._pages} │ sort:{self._sort}")
        if self._hide_idle:
            line.append(" hide-idle", "cyan")
        if self._problems:
            line.append(" problems-only", "cyan")
        if self._pfilter:
            line.append(f" filter:{self._pfilter}", "cyan")
        self.query_one("#top", Static).update(line)

    # ---------------------------------------------------------------- actions
    def open_view(self, provider: Provider):
        self.push_screen(FullView(provider))

    def action_open_all(self):
        self.open_view(self.panels["ALL"].provider)

    def action_open_incidents(self):
        self.open_view(self.panels["INCIDENTS"].provider)

    def action_page(self, delta: int):
        self._page = max(0, min(self._pages - 1, self._page + delta))
        self._dirty_layout = True
        self._relayout()

    def action_sort(self):
        self._sort = {"name": "activity", "activity": "errors", "errors": "name"}[self._sort]
        self._page = 0
        self._dirty_layout = True

    def action_hide_idle(self):
        self._hide_idle = not self._hide_idle
        self._page = 0
        self._dirty_layout = True

    def action_problems(self):
        self._problems = not self._problems
        self._page = 0
        self._dirty_layout = True

    def action_rescan(self):
        if self.ingestor is not None:
            self.ingestor._next_scan = 0.0

    def action_help(self):
        self.push_screen(HelpScreen())

    def _open_input(self, q: Input, value: str):
        q.disabled = False
        q.value = value
        q.add_class("-open")
        self.call_after_refresh(q.focus)

    def _close_input(self, q: Input):
        q.remove_class("-open")
        q.disabled = True

    def action_pfilter(self):
        self._open_input(self.query_one("#pf", Input), self._pfilter)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "pf":
            self._pfilter = event.value
            self._page = 0
            self._relayout()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "pf":
            self._close_input(event.input)
            self._focus_first_panel()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            q = self.query_one("#pf", Input)
            if q.has_class("-open"):
                self._close_input(q)
                if self._pfilter:
                    q.value = ""
                    self._pfilter = ""
                    self._relayout()
                self._focus_first_panel()
                event.stop()

    def _focus_first_panel(self):
        if self._page_names:
            self.call_after_refresh(self.panels[self._page_names[0]].focus)

    def action_move(self, dx: int, dy: int):
        if len(self.screen_stack) > 1 or not self._page_names:
            return
        names = self._page_names
        foc = self.focused
        cur = -1
        for i, n in enumerate(names):
            if self.panels.get(n) is foc:
                cur = i
        if cur < 0:
            nxt = 0
        else:
            nxt = max(0, min(len(names) - 1, cur + dx + dy * self._cols))
        self.panels[names[nxt]].focus()
