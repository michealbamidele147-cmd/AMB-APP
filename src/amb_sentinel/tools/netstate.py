"""Tiny shared 'is the internet reachable' state. Thread-safe, no deps."""
from __future__ import annotations
import os
import socket
import threading
import time

_LOCK = threading.Lock()
_STATE = {"online": True, "checked": 0.0}
_REPROBE_S = 15.0


def _dns_ok(host: str, timeout: float = 3.0) -> bool:
    res: list = []

    def _t():
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            res.append(True)
        except OSError:
            res.append(False)

    th = threading.Thread(target=_t, daemon=True)
    th.start()
    th.join(timeout)
    return bool(res and res[0])


def is_dns_error(exc: BaseException) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    return (
        "nameresolution" in s
        or "name resolution" in s
        or "temporary failure in name" in s
        or "nodename nor servname" in s
    )


def mark_offline() -> None:
    with _LOCK:
        _STATE.update(online=False, checked=time.monotonic())


def known_offline() -> bool:
    """True if we recently saw DNS failure and a fresh probe still fails."""
    with _LOCK:
        if _STATE["online"]:
            return False
        if time.monotonic() - _STATE["checked"] < _REPROBE_S:
            return True
    ok = _dns_ok(os.environ.get("AMB_NET_PROBE_HOST", "one.one.one.one"))
    with _LOCK:
        _STATE.update(online=ok, checked=time.monotonic())
    return not ok


def wait_for_network(max_wait: float = 60.0, poll: float = 5.0) -> bool:
    end = time.monotonic() + max_wait
    host = os.environ.get("AMB_NET_PROBE_HOST", "one.one.one.one")
    while True:
        if _dns_ok(host):
            with _LOCK:
                _STATE.update(online=True, checked=time.monotonic())
            return True
        if time.monotonic() >= end:
            mark_offline()
            return False
        time.sleep(poll)
