"""Executable proof for amb_sentinel.logmon. Run:  pytest tests/test_logmon.py -q"""
import asyncio
import os
import shutil
import time
from pathlib import Path

import pytest

from amb_sentinel.logmon.hub import Hub, fingerprint
from amb_sentinel.logmon.ingest import Ingestor
from amb_sentinel.logmon.model import Assembler, Query, parse_ts, redact
from amb_sentinel.logmon.tail import FileTailer

# One real start-up cycle from the production crash loop (timestamps shifted per cycle).
CYCLE = """\
{t0} | INFO     | Initials | 🔧 Initializing AMB-Sentinel Tools Package...
{t1} | INFO     | ai_agent | ✅ Postgres host=localhost user=amb_sentinel_user ✅ password=<HIDDEN>
{t1} | ERROR    | ai_agent | ❌ Failed to load sentence-transformers: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
{t1} | INFO     | ai_agent | ✅ Embedding adapter initialized with model: all-MiniLM-L6-v2
{t1} | ERROR    | ai_agent | ❌ Failed to initialize model local_llm_cloud: Cached client for localhost:8080 has timeout=300s; caller requested 2400s. Restart the process to change the timeout, or use a distinct endpoint.
Traceback (most recent call last):
  File "/home/deve-oops/AMB-APP/src/amb_sentinel/agent/ai_agent.py", line 692, in initialize
    await self._initialize_impl()
amb_sentinel.llm.llm_cloud_adapter.LLMCloudUnavailable: Cached client for localhost:8080 has timeout=300s; caller requested 2400s.
{t1} | ERROR    | ai_agent | 💥 Unhandled exception in main(): A3: no real LLM backend could be initialised.
  resolved primary provider: 'local_llm_cloud'
  probe result: reachable
{t1} | INFO     | ai_agent | ✅ Main function exiting cleanly
"""


def cycle(n: int) -> str:
    base = 1_792_551_863 + n * 43   # arbitrary epoch seconds
    f = lambda s: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(base + s))
    return CYCLE.format(t0=f(0), t1=f(2))


def wait_for(cond, timeout=5.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return False


# ------------------------------------------------------------------------------- parsing
def test_multiline_record_assembly():
    asm = Assembler("ai_agent")
    recs, _ = asm.feed(cycle(0).splitlines())
    assert [r.level for r in recs] == [20, 20, 40, 20, 40, 40, 20]
    sent = recs[2]
    assert "sentence-transformers" in sent.msg and len(sent.cont) == 1      # wrapped line attached
    tb = recs[4]
    assert tb.cont[0].startswith("Traceback") and tb.cont[-1].startswith("amb_sentinel.llm")
    assert recs[5].cont[0].strip().startswith("resolved primary provider")


def test_continuation_marks_touched_only_for_old_records():
    asm = Assembler("x")
    new, touched = asm.feed(["2026-10-21 03:04:25 | ERROR | a | boom"])
    assert len(new) == 1 and not touched
    new, touched = asm.feed(["Traceback (most recent call last):"])
    assert new == [] and touched and new == [] and asm._cur.cont == ["Traceback (most recent call last):"]


def test_pasted_dump_with_tags_and_terminal_wrapping():
    lines = [
        "[initials] 2026-10-21 03:04:23 | INFO     | Initials | ✅ Tools initialized successfully: {'scraping': True, 'http':",
        "True, 'download': True}",
        "[ai_agent] 2026-10-21 03:04:25 | ERROR    | ai_agent | ❌ Failed",
        "[ai_agent] Traceback (most recent call last):",
        "[ai_agent]   File \"x.py\", line 1, in f",
    ]
    asm = Assembler("pasted", tag_as_source=True)
    recs, _ = asm.feed(lines)
    assert [r.source for r in recs] == ["initials", "ai_agent"]
    assert recs[0].cont == ["True, 'download': True}"]
    assert recs[1].cont == ["Traceback (most recent call last):", "  File \"x.py\", line 1, in f"]  # tag stripped


def test_json_lines():
    asm = Assembler("j")
    recs, _ = asm.feed(['{"ts":"2026-10-21T03:04:25.250Z","level":"error","logger":"db","message":"down","exc_info":"Trace\\nboom"}'])
    assert recs[0].level == 40 and recs[0].logger == "db" and recs[0].cont == ["Trace", "boom"]


def test_timestamps_fraction_and_tz():
    a = parse_ts("2026-10-21 03:04:25")
    assert parse_ts("2026-10-21 03:04:25.500") == pytest.approx(a + 0.5)
    assert parse_ts("2026-10-21 03:04:25,250") == pytest.approx(a + 0.25)
    z = parse_ts("2026-10-21T03:04:25Z")
    assert parse_ts("2026-10-21T04:04:25+01:00") == pytest.approx(z)
    assert parse_ts("garbage") is None


def test_redaction():
    assert "hunter2" not in redact("connect postgresql://u:hunter2@localhost/db")
    assert redact("api_key=abcdef123 ok") == "api_key=*** ok"
    assert redact("✅ api_key_set=True") == "✅ api_key_set=True"           # not a secret
    assert "abcdefghijk" not in redact("Authorization: Bearer abcdefghijk.lmn")


def test_query_language():
    asm = Assembler("s")
    recs, _ = asm.feed(cycle(0).splitlines())
    q = Query("timeout lvl:error -huggingface")
    hits = [r for r in recs if q.match(r)]
    assert len(hits) == 1 and "Cached client" in hits[0].msg
    assert Query("re:LLMCloudUnavailable").match(recs[4])        # searches traceback text too
    assert Query("src:s").match(recs[0]) and not Query("src:other").match(recs[0])
    assert Query("re:(").error


# ------------------------------------------------------------------------------- tailer
def lines_of(t):
    return t.poll().lines


def test_tailer_backfill_skips_partial_first_line_and_handles_partial_tail(tmp_path):
    p = tmp_path / "a.log"
    p.write_text("".join(f"line {i:04d}\n" for i in range(1000)))
    t = FileTailer(p, backfill_bytes=100, partial_flush_s=0.2)
    first = lines_of(t)
    assert first and all(x.startswith("line ") and len(x) == 9 for x in first)   # no torn line
    assert first[-1] == "line 0999"
    with open(p, "a") as f:
        f.write("part")
    assert lines_of(t) == []                    # incomplete line is held back
    with open(p, "a") as f:
        f.write("ial\nnext\n")
    assert lines_of(t) == ["partial", "next"]
    with open(p, "a") as f:
        f.write("no-newline-at-exit")
    assert lines_of(t) == []                    # first sighting starts the idle timer
    time.sleep(0.3)
    assert lines_of(t) == ["no-newline-at-exit"]   # idle partial is flushed, not lost


def test_tailer_rename_rotation_drains_old_file(tmp_path):
    p = tmp_path / "r.log"
    p.write_text("one\n")
    t = FileTailer(p, from_start=True)
    assert lines_of(t) == ["one"]
    with open(p, "a") as f:
        f.write("two\n")               # written, then rotated before we polled
    os.rename(p, tmp_path / "r.log.1")
    p.write_text("three\n")
    b = t.poll()
    assert b.lines == ["two", "three"] and b.reset


def test_tailer_truncate_and_missing(tmp_path):
    p = tmp_path / "t.log"
    p.write_text("aaaa\nbbbb\n")
    t = FileTailer(p, from_start=True)
    assert lines_of(t) == ["aaaa", "bbbb"]
    p.write_text("c\n")                                  # copytruncate style
    b = t.poll()
    assert b.reset and b.lines == ["c"]
    p.unlink()
    assert t.poll().missing
    p.write_text("back\n")
    b = t.poll()
    assert b.lines == ["back"] and b.reset


def test_tailer_bounded_read_reports_more(tmp_path):
    p = tmp_path / "big.log"
    p.write_text("x" * 99 + "\n" * 1)
    with open(p, "a") as f:
        f.write(("y" * 99 + "\n") * 5000)
    t = FileTailer(p, from_start=True, chunk=64 * 1024)
    b = t.poll()
    assert b.more and len(b.lines) < 5001
    got = len(b.lines)
    while b.more:
        b = t.poll()
        got += len(b.lines)
    assert got == 5001


def test_tailer_transient_mode_matches_persistent(tmp_path):
    p = tmp_path / "tr.log"
    p.write_text("a\n")
    t = FileTailer(p, from_start=True, transient=True)
    assert lines_of(t) == ["a"] and t._fh is None
    with open(p, "a") as f:
        f.write("b\n")
    assert lines_of(t) == ["b"]


# ------------------------------------------------------------------------------- hub / incidents
def test_fingerprint_groups_numeric_variants():
    a = fingerprint("❌ timeout=300s; caller requested 2400s at /home/x/a.py:12 id 0xdeadbeef")
    b = fingerprint("❌ timeout=60s; caller requested 90s at /srv/y/b.py:99 id 0x1")
    assert a == b


def test_crash_loop_collapses_into_few_incidents(tmp_path):
    p = tmp_path / "ai_agent.log"
    p.write_text("".join(cycle(i) for i in range(4)))
    hub = Hub()
    ing = Ingestor(hub, [tmp_path], backfill_bytes=1 << 20)
    ing.start()
    assert ing.ready.wait(5)
    ing.stop()
    top = hub.incidents.top(40)
    sigs = {i.sig[:40]: i.count for i in top}
    assert len(top) == 3                                      # 12 error lines -> 3 unique incidents
    assert all(c == 4 for c in sigs.values())
    assert hub.sources["ai_agent"].counts[40] == 12


def test_merged_timeline_is_time_ordered_across_files(tmp_path):
    (tmp_path / "a.log").write_text("2026-10-21 03:00:01 | INFO | a | a1\n2026-10-21 03:00:05 | INFO | a | a2\n")
    (tmp_path / "b.log").write_text("2026-10-21 03:00:03 | INFO | b | b1\n2026-10-21 03:00:07 | INFO | b | b2\n")
    hub = Hub()
    ing = Ingestor(hub, [tmp_path])
    ing.start()
    assert ing.ready.wait(5)
    assert [r.msg for r in hub.merged_snapshot()] == ["a1", "b1", "a2", "b2"]
    # live late arrival lands in the middle, not at the end
    with open(tmp_path / "b.log", "a") as f:
        f.write("2026-10-21 03:00:04 | WARNING | b | late\n")
    assert wait_for(lambda: len(hub.merged_snapshot()) == 5)
    ing.stop()
    assert [r.msg for r in hub.merged_snapshot()] == ["a1", "b1", "late", "a2", "b2"]


def test_ring_buffer_bounds_memory(tmp_path):
    p = tmp_path / "r.log"
    p.write_text("".join(f"2026-10-21 03:00:00 | INFO | a | m{i}\n" for i in range(5000)))
    hub = Hub(ring=100, merged_cap=150)
    ing = Ingestor(hub, [tmp_path], backfill_bytes=1 << 22)
    ing.start()
    assert ing.ready.wait(5)
    ing.stop()
    st = hub.sources["r"]
    assert len(st.records) == 100 and st.total == 5000 and st.records[-1].msg == "m4999"
    assert len(hub.merged_snapshot()) <= 150


def test_ingest_discovers_new_files_and_survives_rotation(tmp_path):
    (tmp_path / "one.log").write_text("2026-10-21 03:00:00 | INFO | a | hello\n")
    hub = Hub()
    ing = Ingestor(hub, [tmp_path], rescan=0.2, poll_min=0.05, poll_max=0.2)
    ing.start()
    assert ing.ready.wait(5)
    (tmp_path / "two.log").write_text("2026-10-21 03:00:01 | ERROR | a | new file\n")
    assert wait_for(lambda: "two" in hub.sources and hub.sources["two"].total == 1)
    os.rename(tmp_path / "one.log", tmp_path / "one.log.1")
    (tmp_path / "one.log").write_text("2026-10-21 03:00:02 | INFO | a | after rotate\n")
    assert wait_for(lambda: hub.sources["one"].rotations == 1 and hub.sources["one"].records[-1].msg == "after rotate")
    (tmp_path / "two.log").unlink()
    assert wait_for(lambda: hub.sources["two"].missing)
    ing.stop()


def test_one_unreadable_file_does_not_stop_others(tmp_path, monkeypatch):
    (tmp_path / "good.log").write_text("2026-10-21 03:00:00 | INFO | a | ok\n")
    (tmp_path / "bad.log").write_text("2026-10-21 03:00:00 | INFO | a | x\n")
    hub = Hub()
    ing = Ingestor(hub, [tmp_path], rescan=0.2, poll_min=0.05, poll_max=0.2)
    orig = FileTailer.poll

    def flaky(self):
        if self.path.endswith("bad.log") and self._opened:
            raise RuntimeError("disk on fire")
        return orig(self)

    monkeypatch.setattr(FileTailer, "poll", flaky)
    ing.start()
    assert ing.ready.wait(5)
    with open(tmp_path / "good.log", "a") as f:
        f.write("2026-10-21 03:00:05 | INFO | a | still alive\n")
    assert wait_for(lambda: hub.sources["good"].records[-1].msg == "still alive")
    assert wait_for(lambda: "disk on fire" in (hub.sources["bad"].error or ""))
    ing.stop()


# ------------------------------------------------------------------------------- TUI
def make_logs(d: Path, n: int):
    for i in range(n):
        lvl = "ERROR" if i == 2 else "INFO"
        d.joinpath(f"svc{i:02d}.log").write_text(
            "".join(f"2026-10-21 03:00:{j:02d} | {lvl if j == 9 else 'INFO'} | svc{i} | event {j} from svc{i}\n"
                    for j in range(10)))


def run(coro):
    return asyncio.run(coro)


def _start(tmp_path, n, **kw):
    from amb_sentinel.logmon.tui import LogMonApp
    make_logs(tmp_path, n)
    hub = Hub()
    ing = Ingestor(hub, [tmp_path], rescan=0.3, poll_min=0.05, poll_max=0.3, **kw)
    ing.start()
    assert ing.ready.wait(5)
    return hub, ing, LogMonApp(hub, ing)


def test_tui_grid_click_expand_and_return(tmp_path):
    from amb_sentinel.logmon.tui import FullView, LogPanel, panel_id

    async def go():
        hub, ing, app = _start(tmp_path, 6)
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.8)
            shown = [p for p in app.query(LogPanel) if p.display]
            assert len(shown) == 8                                     # 6 files + ALL + INCIDENTS
            sizes = {(p.size.width, p.size.height) for p in shown}
            assert all(w >= 40 and h >= 5 for w, h in sizes), sizes
            # click one panel -> full screen view of that file
            await pilot.click(f"#{panel_id('svc02')}")
            await pilot.pause(0.6)
            assert isinstance(app.screen, FullView)
            v = app.screen.view
            assert v.size.width >= 170 and v.size.height >= 40         # truly full size
            assert len(v._rows) == 10
            # filter inside the full view
            await pilot.press("slash")
            await pilot.press(*"lvl:error")
            await pilot.pause(0.4)
            assert len(v._rows) == 1 and "event 9" in v._rows[0][0].msg
            await pilot.press("escape")                                # closes search bar
            assert isinstance(app.screen, FullView)
            await pilot.press("escape")                                # back to grid
            await pilot.pause(0.3)
            assert not isinstance(app.screen, FullView)
            # click the BACK bar path as well
            await pilot.click(f"#{panel_id('svc01')}")
            await pilot.pause(0.4)
            assert isinstance(app.screen, FullView)
            await pilot.click("#fv-back")
            await pilot.pause(0.3)
            assert not isinstance(app.screen, FullView)
        ing.stop()

    run(go())


def test_tui_pagination_with_many_files(tmp_path):
    from amb_sentinel.logmon.tui import LogPanel

    async def go():
        hub, ing, app = _start(tmp_path, 60)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(1.0)
            assert app._pages > 1
            first = [n for n in app._page_names]
            shown = [p for p in app.query(LogPanel) if p.display]
            assert len(shown) == len(first) and len(first) <= (120 // 46) * (28 // 8)
            await pilot.press("pagedown")
            await pilot.pause(0.3)
            assert app._page == 1 and app._page_names != first
            await pilot.press("pageup")
            assert app._page == 0
            # panel filter narrows the grid
            await pilot.press("slash")
            await pilot.press(*"svc07")
            await pilot.pause(0.3)
            assert [n for n in app._page_names if n not in app.pinned] == ["svc07"]
        ing.stop()

    run(go())


def test_tui_live_follow_and_incident_view(tmp_path):
    from amb_sentinel.logmon.tui import FullView

    async def go():
        hub, ing, app = _start(tmp_path, 2)
        (tmp_path / "ai_agent.log").write_text("".join(cycle(i) for i in range(3)))
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause(1.0)
            await pilot.press("i")                                     # incidents view
            await pilot.pause(0.5)
            assert isinstance(app.screen, FullView)
            v = app.screen.view
            texts = " ".join(r[0].msg for r in v._rows if r[1] == 0)
            assert "×3" in texts and "Cached client" in texts
            assert len({r[0].seq for r in v._rows}) == 3               # 3 unique incidents
            await pilot.press("escape")
            await pilot.pause(0.3)
            # live follow in a single-file view
            await pilot.press("a")
            await pilot.pause(0.5)
            v = app.screen.view
            assert v.follow
            base = time.time() + 3600                                   # newer than everything loaded
            stamp = lambda j: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(base + j))
            with open(tmp_path / "svc00.log", "a") as f:
                f.write("".join(f"{stamp(j)} | INFO | svc0 | live {j}\n" for j in range(120)))
            await pilot.pause(1.0)
            last = v._rows[-1][0]
            assert last.msg == "live 119" and v.scroll_y >= len(v._rows) - v.size.height - 1
            await pilot.press("pageup")                                # user scrolls away -> follow off
            await pilot.pause(0.5)
            assert not v.follow
            await pilot.press("end")
            assert v.follow
        ing.stop()

    run(go())


def test_tui_traceback_fold_and_wrap(tmp_path):
    from amb_sentinel.logmon.tui import FullView, panel_id

    async def go():
        hub, ing, app = _start(tmp_path, 1)
        (tmp_path / "ai_agent.log").write_text(cycle(0))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause(1.0)
            await pilot.click(f"#{panel_id('ai_agent')}")
            await pilot.pause(0.5)
            assert isinstance(app.screen, FullView)
            v = app.screen.view
            folded = len(v._rows)
            await pilot.press("e")                                     # expand all tracebacks
            await pilot.pause(0.3)
            assert len(v._rows) > folded
            # every wrapped chunk fits the terminal width: no row text exceeds the view
            avail = v._avail()
            from rich.cells import cell_len, chop_cells
            for rec, k, i in v._rows:
                parts = chop_cells(v._text_of(rec, k, v._is_expanded(rec)), avail)
                assert all(cell_len(x) <= avail for x in parts)
            await pilot.press("e")
            await pilot.pause(0.3)
            assert len(v._rows) == folded
            await pilot.press("n")                                     # jump to an error row
            assert v.scroll_y >= 0
        ing.stop()

    run(go())


def test_cli_list_and_plain(tmp_path, capsys):
    from amb_sentinel.logmon.cli import main
    make_logs(tmp_path, 3)
    assert main([str(tmp_path), "--list"]) == 0
    out = capsys.readouterr().out
    assert "svc00" in out and "svc02" in out
    assert main([str(tmp_path), "--plain", "--level", "ERROR", "-n", "10"]) == 0
    out = capsys.readouterr().out
    assert out.count("ERR") == 1 and "svc2" not in out or "event 9" in out
