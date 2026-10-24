# tools/http.py
"""
AMB-Sentinel HTTP — Production-grade HTTP client.

Version: 4.0.0
Author: AMB Team © 2025

Features:
- Stealth / WAF-bypass headers (browser-grade fingerprinting)
- Circuit breaker + token-bucket rate limiter
- Exponential / linear / Fibonacci retry with jitter
- True parallel batch execution (ThreadPoolExecutor, order-preserved)
- Response size limiting & gzip/deflate decompression
- Sync + async (graceful aiohttp fallback to threaded sync)
- Security: header injection prevention, TLS enforcement

Dependencies:
    requests>=2.31.0  (required)
    aiohttp           (optional)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import gzip
import json as _json_lib
import logging
import random
import threading
import time
import warnings
import zlib
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode, urlparse

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.exceptions import (
        RequestException, Timeout, ConnectionError as ReqConnError,
        HTTPError, TooManyRedirects, ChunkedEncodingError,
    )
    from urllib3.util.retry import Retry
    from urllib3.exceptions import InsecureRequestWarning
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn("requests not available")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore

try:
    from .utils import (
        SecurityLevel, SessionConfig, create_ssl_context,
        create_waf_headers, make_session, validate_url,
        sanitize_filename, DEFAULT_CACHE_DIR, get_random_user_agent,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    warnings.warn("utils module not available")

try:
    from . import netstate as _netstate
except ImportError:
    _netstate = None

logger = logging.getLogger("HTTP")


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class HttpMethod(Enum):
    GET     = "GET"
    POST    = "POST"
    PUT     = "PUT"
    DELETE  = "DELETE"
    PATCH   = "PATCH"
    HEAD    = "HEAD"
    OPTIONS = "OPTIONS"
    TRACE   = "TRACE"


class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR      = "linear"
    FIXED       = "fixed"
    FIBONACCI   = "fibonacci"
    NO_RETRY    = "no_retry"


class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class ContentType(Enum):
    JSON              = "application/json"
    FORM_URLENCODED   = "application/x-www-form-urlencoded"
    MULTIPART_FORM    = "multipart/form-data"
    XML               = "application/xml"
    TEXT              = "text/plain"
    HTML              = "text/html"
    BINARY            = "application/octet-stream"


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class HttpConfig:
    """HTTP request configuration."""
    timeout: int               = 30
    max_retries: int           = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_base_delay: float    = 1.0
    retry_max_delay: float     = 60.0
    human_mimic: bool          = False
    human_delay_min: float     = 0.3
    human_delay_max: float     = 2.5
    waf_bypass: bool           = True
    browser_fingerprint: bool  = True
    verify_ssl: bool           = True
    security_level: "SecurityLevel" = None  # resolved at runtime
    follow_redirects: bool     = True
    max_redirects: int         = 10
    stream: bool               = False
    decompress: bool           = True
    max_response_size: Optional[int] = 20 * 1024 * 1024  # 20 MB
    rate_limit_rpm: Optional[int]    = None
    circuit_breaker_failures: int    = 5
    circuit_breaker_reset: float     = 60.0
    cache_enabled: bool        = False
    cache_ttl: int             = 300
    certificate_pins: Optional[Dict[str, List[str]]] = None
    proxy: Optional[Dict[str, str]] = None


@dataclass
class HttpResponse:
    """Structured HTTP response."""
    ok: bool
    url: str
    status_code: int
    headers: Dict[str, str]
    text: Optional[str]              = None
    content: Optional[bytes]         = None
    json_data: Optional[Any]         = None
    size: int                        = 0
    content_type: Optional[str]      = None
    encoding: Optional[str]          = None
    elapsed: float                   = 0.0
    retries: int                     = 0
    error: Optional[str]             = None
    error_type: Optional[str]        = None
    redirected: bool                 = False
    redirect_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any]         = field(default_factory=dict)


# ──────────────────────────────────────────────
# Custom exceptions
# ──────────────────────────────────────────────

class HttpError(Exception):
    """Base HTTP error."""


class CircuitBreakerOpenError(HttpError):
    """Circuit breaker is open."""


class RateLimitExceededError(HttpError):
    """Rate limit exceeded."""


class SecurityViolationError(HttpError):
    """Security check failed."""


class ResponseTooLargeError(HttpError):
    """Response exceeds size limit."""


# ──────────────────────────────────────────────
# Rate limiter (token bucket, thread-safe)
# ──────────────────────────────────────────────



class _InFlightDeduplicator:
    """
    In-flight request deduplication — prevents hammering the same URL 
    from concurrent threads. If thread A is already fetching URL X, 
    thread B will block and share A's result instead of launching a duplicate.
    
    Uses a dict of threading.Event per URL key; when A finishes it broadcasts
    to all waiting threads and they pick up the cached result.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._in_flight: Dict[str, threading.Event] = {}
        self._results: Dict[str, Any] = {}

    def acquire(self, key: str) -> bool:
        """Try to become the owner of this request.
        
        Returns True if this caller should execute the request.
        Returns False if another caller is already executing it (caller should wait).
        """
        with self._lock:
            if key in self._in_flight:
                return False
            self._in_flight[key] = threading.Event()
            return True

    def wait(self, key: str, timeout: float = 60.0):
        """Wait for another thread to complete this key. Returns their result."""
        event = None
        with self._lock:
            event = self._in_flight.get(key)
        if event:
            event.wait(timeout=timeout)
        with self._lock:
            return self._results.get(key)

    def release(self, key: str, result) -> None:
        """Signal waiting threads and store result."""
        with self._lock:
            self._results[key] = result
            event = self._in_flight.pop(key, None)
        if event:
            event.set()
        # Clean up result after brief window
        def _cleanup():
            import time as _time
            _time.sleep(5.0)
            with self._lock:
                self._results.pop(key, None)
        import threading as _t
        _t.Thread(target=_cleanup, daemon=True).start()


# Module-level deduplicator
_deduplicator = _InFlightDeduplicator()


class HttpMetrics:
    """Lightweight per-instance HTTP performance counters."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests = 0
        self.successes = 0
        self.failures = 0
        self.cache_hits = 0
        self.total_bytes = 0
        self.total_ms = 0.0
        self._latencies: list = []

    def record(self, success: bool, latency_s: float, bytes_recv: int = 0,
               cache_hit: bool = False) -> None:
        with self._lock:
            self.requests += 1
            if success:
                self.successes += 1
            else:
                self.failures += 1
            if cache_hit:
                self.cache_hits += 1
            self.total_bytes += bytes_recv
            ms = latency_s * 1000
            self.total_ms += ms
            self._latencies.append(ms)
            if len(self._latencies) > 500:
                self._latencies.pop(0)

    def summary(self) -> dict:
        with self._lock:
            n = self.requests
            lats = sorted(self._latencies)
            return {
                "requests": n,
                "successes": self.successes,
                "failures": self.failures,
                "cache_hits": self.cache_hits,
                "success_rate": round(self.successes / n, 4) if n else 0.0,
                "avg_ms": round(self.total_ms / n, 2) if n else 0.0,
                "p95_ms": round(lats[int(len(lats) * 0.95)], 2) if lats else 0.0,
                "total_mb": round(self.total_bytes / 1_048_576, 3),
            }


# Module-level metrics
_http_metrics = HttpMetrics()


def get_http_metrics() -> dict:
    """Return aggregated HTTP performance counters."""
    return _http_metrics.summary()

class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    __slots__ = ("_rpm", "_tokens", "_last_update", "_lock")

    def __init__(self, requests_per_minute: int):
        self._rpm = max(1, requests_per_minute)
        self._tokens = float(self._rpm)
        self._last_update = time.monotonic()
        self._lock = Lock()

    def acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking acquire. Returns True if tokens available."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait(self, tokens: float = 1.0) -> float:
        """Block until tokens available. Returns wait duration."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            deficit = tokens - self._tokens
            wait_sec = deficit * 60.0 / self._rpm
            self._tokens = 0.0
        time.sleep(wait_sec)
        return wait_sec

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        self._tokens = min(self._rpm, self._tokens + elapsed * self._rpm / 60.0)


# ──────────────────────────────────────────────
# Circuit breaker
# ──────────────────────────────────────────────

class CircuitBreaker:
    """
    Per-domain circuit breaker.

    States: CLOSED (normal) → OPEN (failures) → HALF_OPEN (probe) → CLOSED
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._lock = RLock()

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._check()
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception:
                self._on_failure()
                raise
        return wrapper

    def _check(self) -> None:
        with self._lock:
            if self.state == CircuitState.OPEN:
                age = time.monotonic() - self.last_failure_time
                if age >= self.reset_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("CircuitBreaker → HALF_OPEN")
                else:
                    remaining = self.reset_timeout - age
                    raise CircuitBreakerOpenError(
                        f"Circuit OPEN — retry in {remaining:.1f}s"
                    )

    def _on_success(self) -> None:
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("CircuitBreaker → CLOSED")

    def _on_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning("CircuitBreaker → OPEN (%d failures)", self.failure_count)

    def reset(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = 0.0


# ──────────────────────────────────────────────
# Module-level rate limiter & circuit breaker caches
# ──────────────────────────────────────────────

_rate_limiters: Dict[str, RateLimiter] = {}
_rl_lock = Lock()


def _get_rate_limiter(domain: str, rpm: int) -> RateLimiter:
    with _rl_lock:
        if domain not in _rate_limiters:
            _rate_limiters[domain] = RateLimiter(rpm)
        return _rate_limiters[domain]


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

_RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520, 521, 522, 524}
_NO_RETRY_METHODS = {"POST", "PUT", "DELETE", "PATCH"}


def _should_retry(status_code: int, exc: Optional[Exception], method: str) -> bool:
    if method.upper() in _NO_RETRY_METHODS and status_code not in {429, 503}:
        return False
    if status_code in _RETRY_STATUS:
        return True
    if exc is not None:
        if isinstance(exc, (Timeout, ReqConnError, ChunkedEncodingError)):
            return True
        if isinstance(exc, Exception) and "timeout" in str(exc).lower():
            return True
    return False


def _retry_delay(attempt: int, strategy: RetryStrategy, base: float, cap: float) -> float:
    """Compute retry delay with ±25 % jitter."""
    if strategy == RetryStrategy.NO_RETRY:
        return 0.0
    if strategy == RetryStrategy.EXPONENTIAL:
        d = min(base * 2 ** (attempt - 1), cap)
    elif strategy == RetryStrategy.LINEAR:
        d = min(base * attempt, cap)
    elif strategy == RetryStrategy.FIXED:
        d = base
    elif strategy == RetryStrategy.FIBONACCI:
        a, b = 1, 1
        for _ in range(attempt - 1):
            a, b = b, a + b
        d = min(a * base, cap)
    else:
        d = base
    return d * random.uniform(0.75, 1.25)


def _human_delay(min_s: float = 0.3, max_s: float = 2.0) -> None:
    time.sleep(random.triangular(min_s, max_s, (min_s + max_s) / 2))


def _validate_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Sanitize headers; raise SecurityViolationError on injection attempts."""
    out: Dict[str, str] = {}
    for k, v in headers.items():
        if "\x00" in k or "\x00" in v:
            raise SecurityViolationError("Null byte in header")
        if "\r" in v or "\n" in v:
            raise SecurityViolationError("Header injection attempt")
        out[k.strip()] = v.strip()
    return out


def _decode_content(content: bytes, response: "requests.Response") -> str:
    """Best-effort decode of response bytes to str."""
    for enc in [response.encoding, "utf-8", "latin-1", "cp1252"]:
        if not enc:
            continue
        try:
            return content.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode("utf-8", errors="replace")


def _process_response(
    response: "requests.Response",
    max_size: Optional[int],
    decompress: bool,
) -> Tuple[Optional[str], bytes, Optional[Any]]:
    """Read, decompress, decode response. Enforces max_size."""
    raw = b""
    for chunk in response.iter_content(chunk_size=65536):
        raw += chunk
        if max_size and len(raw) > max_size:
            raise ResponseTooLargeError(
                f"Response > {max_size // 1024 // 1024} MB limit"
            )

    if decompress:
        enc = response.headers.get("Content-Encoding", "").lower()
        if enc == "gzip":
            try:
                raw = gzip.decompress(raw)
            except Exception:
                pass
        elif enc in ("deflate", "zlib"):
            try:
                raw = zlib.decompress(raw)
            except Exception:
                try:
                    raw = zlib.decompress(raw, -zlib.MAX_WBITS)
                except Exception:
                    pass
        elif enc == "br":
            try:
                import brotli  # type: ignore
                raw = brotli.decompress(raw)
            except Exception:
                pass

    ct = response.headers.get("Content-Type", "").lower()
    is_text = any(t in ct for t in ("text", "json", "xml", "html", "javascript", "css", "application/"))
    text: Optional[str] = _decode_content(raw, response) if raw and is_text else None

    json_data: Optional[Any] = None
    if "json" in ct and text:
        try:
            json_data = _json_lib.loads(text)
        except Exception:
            pass

    return text, raw, json_data


def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return "unknown"


# ──────────────────────────────────────────────
# Core fetch function
# ──────────────────────────────────────────────

def http_fetch(
    url: str,
    method: Union[str, HttpMethod] = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    data: Any = None,
    json: Optional[Any] = None,
    timeout: int = 30,
    max_retries: int = 3,
    retry_strategy: Union[str, RetryStrategy] = RetryStrategy.EXPONENTIAL,
    human_mimic: bool = False,
    waf_bypass: bool = True,
    verify_ssl: bool = True,
    follow_redirects: bool = True,
    max_redirects: int = 10,
    stream: bool = False,
    max_response_size: Optional[int] = 20 * 1024 * 1024,
    rate_limit_rpm: Optional[int] = None,
    session_config: Optional["SessionConfig"] = None,
    http_config: Optional[HttpConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Fetch a URL with enterprise-grade reliability.

    Returns a dict:
        ok, url, status_code, headers, text, content, json_data,
        size, content_type, encoding, elapsed, retries, error, error_type,
        redirected, redirect_history
    """
    if not REQUESTS_AVAILABLE:
        return {"ok": False, "error": "requests library not available", "url": url,
                "status_code": 0, "headers": {}, "text": None, "content": None,
                "json_data": None, "size": 0, "elapsed": 0.0, "retries": 0}

    t0 = time.monotonic()

    # Validate URL
    try:
        if UTILS_AVAILABLE:
            validate_url(url)
        else:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL: {url}")
    except Exception as exc:
        return {"ok": False, "error": str(exc), "error_type": "ValidationError",
                "url": url, "status_code": 0, "headers": {}, "text": None,
                "content": None, "json_data": None, "size": 0, "elapsed": 0.0, "retries": 0}

    if _netstate is not None and _netstate.known_offline():
        return {
            "ok": False, "url": url, "status_code": 0, "headers": {},
            "text": None, "content": None, "json_data": None, "size": 0,
            "content_type": None, "encoding": None, "elapsed": 0.0, "retries": 0,
            "error": "network_unreachable: DNS/connectivity probe failed",
            "error_type": "OfflineError", "offline": True,
            "redirected": False, "redirect_history": [],
        }

    # Build config
    cfg = http_config or HttpConfig(
        timeout=timeout, max_retries=max_retries,
        retry_strategy=RetryStrategy(retry_strategy) if isinstance(retry_strategy, str) else retry_strategy,
        human_mimic=human_mimic, waf_bypass=waf_bypass,
        verify_ssl=verify_ssl, follow_redirects=follow_redirects,
        max_redirects=max_redirects, stream=stream,
        max_response_size=max_response_size, rate_limit_rpm=rate_limit_rpm,
    )

    method_str = method.value if isinstance(method, HttpMethod) else method.upper()

    # Rate limiting
    if cfg.rate_limit_rpm:
        _get_rate_limiter(_get_domain(url), cfg.rate_limit_rpm).wait()

    # Build headers
    req_headers: Dict[str, str] = {}
    if cfg.waf_bypass and UTILS_AVAILABLE:
        req_headers.update(create_waf_headers(cfg.browser_fingerprint))
    elif cfg.waf_bypass:
        req_headers["User-Agent"] = random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        ])

    if headers:
        req_headers.update(_validate_headers(headers))

    # Prepare body
    body: Any = None
    if json is not None:
        body = _json_lib.dumps(json, ensure_ascii=False)
        req_headers.setdefault("Content-Type", ContentType.JSON.value)
    elif data is not None:
        if isinstance(data, dict):
            body = urlencode(data)
            req_headers.setdefault("Content-Type", ContentType.FORM_URLENCODED.value)
        else:
            body = data

    # Build session
    if UTILS_AVAILABLE and session_config:
        session = make_session(session_config)
    elif UTILS_AVAILABLE:
        s_cfg = session_config or __import__("tools.utils", fromlist=["SessionConfig"]).SessionConfig(
            timeout=cfg.timeout, max_retries=0, verify_ssl=cfg.verify_ssl,
            user_agent_rotation=cfg.waf_bypass, browser_fingerprinting=cfg.browser_fingerprint,
            proxy=cfg.proxy or {},
        )
        session = make_session(s_cfg)
    else:
        session = requests.Session()
        if not cfg.verify_ssl:
            import warnings as _w
            _w.filterwarnings("ignore", category=InsecureRequestWarning)
            session.verify = False
        session.headers.update(req_headers)

    attempt_count = 0
    last_exc: Optional[Exception] = None

    for attempt in range(cfg.max_retries + 1):
        if cfg.human_mimic:
            _human_delay(cfg.human_delay_min, cfg.human_delay_max)

        try:
            if circuit_breaker:
                _invoke = circuit_breaker(lambda: session.request(
                    method=method_str, url=url, params=params,
                    data=body, headers=req_headers,
                    timeout=(min(10, cfg.timeout // 3), cfg.timeout),
                    verify=cfg.verify_ssl,
                    allow_redirects=cfg.follow_redirects,
                    stream=True,
                ))
                resp = _invoke()
            else:
                resp = session.request(
                    method=method_str, url=url, params=params,
                    data=body, headers=req_headers,
                    timeout=(min(10, cfg.timeout // 3), cfg.timeout),
                    verify=cfg.verify_ssl,
                    allow_redirects=cfg.follow_redirects,
                    stream=True,
                )

            text, content, json_data = _process_response(resp, cfg.max_response_size, cfg.decompress)

            if attempt < cfg.max_retries and _should_retry(resp.status_code, None, method_str):
                d = _retry_delay(attempt + 1, cfg.retry_strategy, cfg.retry_base_delay, cfg.retry_max_delay)
                logger.warning("Retry %d/%d for %s (HTTP %d) in %.1fs",
                               attempt + 1, cfg.max_retries, url, resp.status_code, d)
                time.sleep(d)
                attempt_count += 1
                continue

            redirect_hist = [
                {"url": str(h.url), "status_code": h.status_code}
                for h in resp.history
            ]
            return {
                "ok": resp.ok,
                "url": str(resp.url),
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "text": text,
                "content": content,
                "json_data": json_data,
                "size": len(content) if content else 0,
                "content_type": resp.headers.get("Content-Type"),
                "encoding": resp.encoding,
                "elapsed": round(time.monotonic() - t0, 3),
                "retries": attempt_count,
                "error": None,
                "error_type": None,
                "redirected": bool(resp.history),
                "redirect_history": redirect_hist,
            }

        except ResponseTooLargeError:
            raise
        except CircuitBreakerOpenError:
            raise
        except Exception as exc:
            last_exc = exc
            attempt_count += 1
            if _netstate is not None and _netstate.is_dns_error(exc) and attempt >= 1:
                _netstate.mark_offline()
                break

            if attempt < cfg.max_retries and _should_retry(0, exc, method_str):
                d = _retry_delay(attempt + 1, cfg.retry_strategy, cfg.retry_base_delay, cfg.retry_max_delay)
                logger.warning("Retry %d/%d for %s (%s) in %.1fs",
                               attempt + 1, cfg.max_retries, url, type(exc).__name__, d)
                time.sleep(d)
            else:
                break

    elapsed = round(time.monotonic() - t0, 3)
    logger.error("http_fetch failed for %s after %d attempts: %s", url, attempt_count, last_exc)
    return {
        "ok": False, "url": url, "status_code": 0, "headers": {},
        "text": None, "content": None, "json_data": None, "size": 0,
        "content_type": None, "encoding": None, "elapsed": elapsed,
        "retries": attempt_count,
        "error": str(last_exc) if last_exc else "Unknown error",
        "error_type": type(last_exc).__name__ if last_exc else "Unknown",
        "redirected": False, "redirect_history": [],
        "offline": bool(last_exc and _netstate is not None and _netstate.is_dns_error(last_exc)),
    }

# ──────────────────────────────────────────────
# Async fetch (uses aiohttp or threaded fallback)
# ──────────────────────────────────────────────

async def http_fetch_async(
    url: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    data: Any = None,
    json: Optional[Any] = None,
    timeout: int = 30,
    max_retries: int = 3,
    waf_bypass: bool = True,
    verify_ssl: bool = True,
    max_response_size: Optional[int] = 20 * 1024 * 1024,
    **kwargs,
) -> Dict[str, Any]:
    """
    Async HTTP fetch.

    Uses aiohttp when available; falls back to running sync http_fetch
    in a thread pool.
    """
    if AIOHTTP_AVAILABLE:
        t0 = time.monotonic()
        h = {}
        if UTILS_AVAILABLE:
            h.update(create_waf_headers(True) if waf_bypass else {})
        if headers:
            h.update(headers)
        body = None
        if json is not None:
            body = _json_lib.dumps(json).encode()
            h.setdefault("Content-Type", "application/json")
        elif data is not None:
            body = data

        connector = aiohttp.TCPConnector(limit=100, ssl=not verify_ssl)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers=h,
        ) as sess:
            for attempt in range(max_retries + 1):
                try:
                    async with sess.request(
                        method.upper(), url, params=params, data=body,
                        allow_redirects=True,
                    ) as resp:
                        raw = await resp.read()
                        if max_response_size and len(raw) > max_response_size:
                            raise ResponseTooLargeError("Response too large")
                        ct = resp.headers.get("Content-Type", "")
                        try:
                            text = raw.decode(resp.charset or "utf-8", errors="replace") if raw else None
                        except Exception:
                            text = raw.decode("utf-8", errors="replace") if raw else None
                        return {
                            "ok": resp.status < 400, "url": str(resp.url),
                            "status_code": resp.status, "headers": dict(resp.headers),
                            "text": text, "content": raw, "json_data": None,
                            "size": len(raw), "content_type": ct,
                            "elapsed": round(time.monotonic() - t0, 3),
                            "retries": attempt, "error": None,
                        }
                except Exception as exc:
                    if attempt < max_retries:
                        await asyncio.sleep(_retry_delay(attempt + 1, RetryStrategy.EXPONENTIAL, 1.0, 30.0))
                    else:
                        return {"ok": False, "url": url, "error": str(exc),
                                "status_code": 0, "headers": {}, "text": None,
                                "content": None, "size": 0, "elapsed": round(time.monotonic() - t0, 3)}
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: http_fetch(
                url=url, method=method, params=params, headers=headers,
                data=data, json=json, timeout=timeout, max_retries=max_retries,
                waf_bypass=waf_bypass, verify_ssl=verify_ssl,
                max_response_size=max_response_size,
            ),
        )


# ──────────────────────────────────────────────
# Batch fetch (parallel, order-preserved)
# ──────────────────────────────────────────────

def http_fetch_batch(
    urls: List[str],
    method: str = "GET",
    params_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    headers_list: Optional[List[Optional[Dict[str, str]]]] = None,
    max_concurrent: int = 10,
    timeout: int = 30,
    waf_bypass: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Fetch multiple URLs concurrently with order-preserved results.

    All kwargs forwarded to http_fetch.
    """
    n = len(urls)
    results: List[Optional[Dict[str, Any]]] = [None] * n

    def _fetch_one(idx: int, url: str) -> Tuple[int, Dict[str, Any]]:
        p = params_list[idx] if params_list and idx < len(params_list) else None
        h = headers_list[idx] if headers_list and idx < len(headers_list) else None
        res = http_fetch(
            url=url, method=method, params=p, headers=h,
            timeout=timeout, waf_bypass=waf_bypass, **kwargs
        )
        return idx, res

    if max_concurrent <= 1 or n <= 1:
        for i, url in enumerate(urls):
            _, r = _fetch_one(i, url)
            results[i] = r
        return results  # type: ignore

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futs = {pool.submit(_fetch_one, i, url): i for i, url in enumerate(urls)}
        for fut in concurrent.futures.as_completed(futs):
            try:
                idx, res = fut.result()
                results[idx] = res
            except Exception as exc:
                orig_idx = futs[fut]
                results[orig_idx] = {
                    "ok": False, "url": urls[orig_idx],
                    "error": str(exc), "error_type": type(exc).__name__,
                    "status_code": 0, "headers": {}, "text": None,
                    "content": None, "size": 0, "elapsed": 0.0,
                }

    # Fill any None slots (shouldn't happen)
    for i in range(n):
        if results[i] is None:
            results[i] = {"ok": False, "url": urls[i], "error": "Not executed",
                          "status_code": 0, "headers": {}}
    return results  # type: ignore
