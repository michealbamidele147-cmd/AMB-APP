# tools/utils.py
"""
AMB-Sentinel Utils — Production-grade foundation utilities.

Version: 1.0.0
Author: AMB Team

Provides:
- Secure HTTP session management with WAF-bypass headers
- TLS-aware SSL context factory
- Safe filename generation and URL validation
- File integrity (multi-algorithm hashing)
- Concurrent-safe download streaming with resume support
- Directory cleanup helpers
- Thread-safe session pool

Dependencies:
    requests>=2.31.0   (required)
    urllib3>=2.0.0     (required)
    aiohttp            (optional — async features)
    filelock           (optional — safe partial-file locking)
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import random
import re
import ssl
import threading
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

# ──────────────────────────────────────────────
# Version
# ──────────────────────────────────────────────
__version__ = "1.0.0"

# ──────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────
logger = logging.getLogger("Utils")

# ──────────────────────────────────────────────
# Optional imports with graceful degradation
# ──────────────────────────────────────────────
try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout, TCPConnector
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.exceptions import RequestException
    from urllib3.util.retry import Retry
    from urllib3.exceptions import InsecureRequestWarning
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn("requests not available — HTTP features disabled")

try:
    from filelock import FileLock, Timeout as FileLockTimeout
    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False
    FileLock = None  # type: ignore
    FileLockTimeout = None  # type: ignore


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class SecurityLevel(Enum):
    """TLS / SSL security tier."""
    LOW = auto()       # No verification (dev only)
    MEDIUM = auto()    # Standard TLS 1.2+
    HIGH = auto()      # Strict cipher suite
    PARANOID = auto()  # TLS 1.3 only


class RetryStrategy(Enum):
    """Back-off strategy for HTTP retries."""
    EXPONENTIAL = "exponential"
    LINEAR      = "linear"
    FIXED       = "fixed"
    FIBONACCI   = "fibonacci"
    NO_RETRY    = "no_retry"


class DownloadStatus(Enum):
    """Download outcome codes."""
    SUCCESS          = "success"
    PARTIAL          = "partial"
    FAILED           = "failed"
    SIZE_EXCEEDED    = "size_exceeded"
    RESUMED          = "resumed"
    VALIDATION_FAILED = "validation_failed"
    TIMEOUT          = "timeout"


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class SessionConfig:
    """Full HTTP session configuration."""
    timeout: int = 30
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    verify_ssl: bool = True
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    pool_connections: int = 20
    pool_maxsize: int = 40
    proxy: Optional[Dict[str, str]] = None
    user_agent_rotation: bool = True
    browser_fingerprinting: bool = True
    dns_cache_ttl: int = 300
    keep_alive: bool = True
    compression: bool = True
    max_redirects: int = 10
    certificate_pins: Optional[Dict[str, List[str]]] = None
    connect_timeout: int = 10
    read_timeout: int = 30


@dataclass
class DownloadConfig:
    """File download configuration."""
    dest_dir: Union[str, Path] = None
    filename: Optional[str] = None
    timeout: int = 120
    max_bytes: Optional[int] = None
    chunk_size: int = 65536        # 64 KB
    resume: bool = True
    validate_hash: bool = False
    expected_hash: Optional[str] = None
    hash_algorithm: str = "sha256"
    remove_incomplete: bool = True
    overwrite: bool = False
    progress_interval: float = 0.5
    parallel_chunks: int = 4       # concurrent download chunks for large files
    min_parallel_size: int = 10_485_760  # 10 MB threshold for parallel


@dataclass
class DownloadResult:
    """Structured download result."""
    ok: bool
    status: DownloadStatus
    path: Optional[Path] = None
    size: int = 0
    downloaded: int = 0
    content_type: Optional[str] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    elapsed: float = 0.0
    resumed: bool = False
    hash: Optional[str] = None
    checksum_valid: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanupResult:
    """Cleanup operation result."""
    cleaned_files: int = 0
    cleaned_size_bytes: int = 0
    remaining_files: int = 0
    remaining_size_bytes: int = 0
    errors: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# Custom exceptions
# ──────────────────────────────────────────────

class SecurityError(Exception):
    """Security constraint violated."""


class DownloadError(Exception):
    """Download operation failed."""


class ValidationError(Exception):
    """Validation failed."""


# ──────────────────────────────────────────────
# Realistic browser fingerprints
# ──────────────────────────────────────────────

# Full user-agent strings covering major browsers / platforms
USER_AGENTS: List[str] = [
    # Chrome 124 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome 124 — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome 124 — Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Firefox 125 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Firefox 125 — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Firefox 125 — Linux
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Safari 17 — macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    # Edge 124 — Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    # Chrome Android
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.82 Mobile Safari/537.36",
    # Safari iOS
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1",
]

# Matching Accept-Language headers for regional diversity
_ACCEPT_LANGS = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-US,en;q=0.9,fr;q=0.8",
    "en-US,en;q=0.9,de;q=0.8",
    "en-US,en;q=0.9,es;q=0.7",
]

# Matching Sec-CH-UA strings for Chrome browsers
_SEC_CH_UA = [
    '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    '"Chromium";v="124", "Not A(Brand";v="24", "Google Chrome";v="124"',
]

# Default directories — single source of truth is amb_sentinel.paths.
# Dirs are created by paths.ensure_dirs() at startup; no import-time side
# effects here.
from amb_sentinel import paths as _paths

DEFAULT_DOWNLOAD_DIR = _paths.download_dir()
DEFAULT_CACHE_DIR    = _paths.cache_dir()
DEFAULT_LOG_DIR      = _paths.log_dir()

# ──────────────────────────────────────────────
# Thread-safe session pool
# ──────────────────────────────────────────────
_session_pool: Dict[str, "requests.Session"] = {}
_session_lock = threading.RLock()


def _session_cache_key(config: SessionConfig) -> str:
    """Deterministic cache key for a SessionConfig."""
    return (
        f"{config.timeout}:{config.max_retries}:{config.retry_strategy.value}:"
        f"{config.verify_ssl}:{config.security_level.value}:"
        f"{config.pool_connections}:{config.pool_maxsize}:"
        f"{str(config.proxy)}:{config.user_agent_rotation}:"
        f"{config.browser_fingerprinting}"
    )


# ──────────────────────────────────────────────
# URL / Path validation helpers
# ──────────────────────────────────────────────

def validate_url(url: str) -> bool:
    """
    Validate URL format with security checks.

    Raises ValueError/SecurityError on bad input; returns True on success.

    Security checks applied:
    - Scheme must be http/https/ftp/ftps  (blocks file://, data://, javascript://)
    - No path-traversal components        (blocks /../ sequences)
    - No SSRF targets: localhost, loopback (127.x, ::1), link-local (169.254.x — AWS
      metadata), private RFC-1918 ranges (10.x, 172.16-31.x, 192.168.x), and 0.0.0.0
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")
    url = url.strip()
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL (missing scheme or host): {url!r}")
        if parsed.scheme not in {"http", "https", "ftp", "ftps"}:
            raise ValueError(f"Unsupported scheme: {parsed.scheme!r}")
        path = PurePosixPath(parsed.path or "/")
        if ".." in path.parts:
            raise SecurityError(f"Path traversal detected in URL: {url!r}")

        # ── SSRF protection: block private/internal network targets ──────────
        # Extract hostname (strip port if present)
        hostname = parsed.hostname or ""
        hostname_lower = hostname.lower()

        # Reject loopback and localhost
        _BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}
        if hostname_lower in _BLOCKED_HOSTNAMES:
            raise SecurityError(f"SSRF: blocked loopback hostname {hostname!r}")

        # Reject IPv4 private/reserved ranges
        import ipaddress as _ipa
        try:
            addr = _ipa.ip_address(hostname)
            # Block: loopback, private (RFC-1918), link-local (169.254 — cloud metadata),
            # unspecified (0.0.0.0), multicast, reserved
            if (addr.is_loopback or addr.is_private or addr.is_link_local
                    or addr.is_unspecified or addr.is_multicast or addr.is_reserved):
                raise SecurityError(
                    f"SSRF: blocked private/reserved IP {hostname!r}")
        except ValueError:
            # Not a plain IP — check for encoded/octal/hex bypass attempts and
            # partial-match private-range patterns in the hostname string
            _SSRF_PATTERNS = [
                "127.", "169.254.", "10.", "0.0.0.0",
                "192.168.", "172.16.", "172.17.", "172.18.", "172.19.",
                "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                "172.30.", "172.31.",
            ]
            for pat in _SSRF_PATTERNS:
                if hostname_lower.startswith(pat):
                    raise SecurityError(
                        f"SSRF: blocked private-range hostname {hostname!r}") from None

        return True
    except (SecurityError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"URL validation error: {exc}") from exc


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Remove characters that are unsafe in file-system paths.

    Returns a safe, non-empty filename string.
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
    filename = filename.replace("\x00", "")
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    filename = "".join(c for c in filename if ord(c) >= 32)
    filename = filename.strip(". ")
    RESERVED = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if filename.upper() in RESERVED:
        filename = f"_{filename}"
    if not filename:
        filename = "unnamed_file"
    if len(filename) > max_length:
        stem, ext = os.path.splitext(filename)
        ext = ext[:20]
        filename = stem[: max_length - len(ext)] + ext
    return filename


def generate_safe_filename(url: str, content_type: Optional[str] = None) -> str:
    """Generate a safe filesystem name from a URL and optional Content-Type."""
    try:
        validate_url(url)
    except Exception:
        return f"download_{hashlib.sha256(url.encode()).hexdigest()[:12]}"
    parsed = urlparse(url)
    base = Path(parsed.path).name or "download"
    base = base.split("?")[0].split("#")[0]
    base = sanitize_filename(base) or "download"
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:10]
    ext = ""
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            ext = guessed
    if not ext:
        path_ext = Path(parsed.path).suffix
        if path_ext and 1 <= len(path_ext) <= 12:
            ext = path_ext
    stem = base if "." not in base else os.path.splitext(base)[0]
    return sanitize_filename(f"{stem}_{url_hash}{ext}")


# ──────────────────────────────────────────────
# SSL context factory
# ──────────────────────────────────────────────

def create_ssl_context(
    security_level: SecurityLevel = SecurityLevel.MEDIUM,
    cert_pins: Optional[Dict[str, List[str]]] = None,
) -> ssl.SSLContext:
    """
    Build an SSLContext appropriate for the given security level.

    Raises SecurityError on failure.
    """
    try:
        if security_level == SecurityLevel.LOW:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        elif security_level == SecurityLevel.MEDIUM:
            ctx = ssl.create_default_context()
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.set_ciphers(
                "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20"
            )

        elif security_level == SecurityLevel.HIGH:
            ctx = ssl.create_default_context()
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20")
            ctx.options |= ssl.OP_NO_COMPRESSION
            ctx.options |= ssl.OP_SINGLE_ECDH_USE
            ctx.options |= ssl.OP_SINGLE_DH_USE

        else:  # PARANOID
            ctx = ssl.create_default_context()
            try:
                ctx.minimum_version = ssl.TLSVersion.TLSv1_3
            except AttributeError:
                ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.options |= ssl.OP_NO_COMPRESSION
            ctx.options |= ssl.OP_SINGLE_ECDH_USE
            ctx.options |= ssl.OP_SINGLE_DH_USE

        return ctx
    except Exception as exc:
        raise SecurityError(f"SSL context creation failed: {exc}") from exc


# ──────────────────────────────────────────────
# WAF-bypass / browser-fingerprint headers
# ──────────────────────────────────────────────

def create_waf_headers(browser_fingerprint: bool = True) -> Dict[str, str]:
    """
    Generate realistic browser request headers.

    When browser_fingerprint=True, adds modern Sec-CH-UA / Sec-Fetch headers
    that closely mimic a real Chrome browser session.
    """
    ua = random.choice(USER_AGENTS)
    is_chrome = "Chrome" in ua and "Edg" not in ua

    headers: Dict[str, str] = {
        "User-Agent": ua,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": random.choice(_ACCEPT_LANGS),
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "DNT": "1",
    }

    if browser_fingerprint and is_chrome:
        headers.update({
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Sec-CH-UA": random.choice(_SEC_CH_UA),
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": random.choice(['"Windows"', '"macOS"', '"Linux"']),
        })

    return headers


# Weighted UA pool — Chrome dominant (matches real-world ~65% share)
_UA_WEIGHTS: List[float] = [
    0.25,  # Chrome 124 Windows
    0.18,  # Chrome 124 macOS
    0.08,  # Chrome 124 Linux
    0.12,  # Firefox 125 Windows
    0.06,  # Firefox 125 macOS
    0.04,  # Firefox 125 Linux
    0.07,  # Safari 17 macOS
    0.10,  # Edge 124 Windows
    0.06,  # Chrome Android
    0.04,  # Safari iOS
]


def get_random_user_agent() -> str:
    """Return a weighted-random modern browser User-Agent (matches real traffic distribution)."""
    try:
        return random.choices(USER_AGENTS, weights=_UA_WEIGHTS, k=1)[0]
    except (ValueError, IndexError):
        return random.choice(USER_AGENTS)


# ──────────────────────────────────────────────
# Session factory (thread-safe pool)
# ──────────────────────────────────────────────

def make_session(config: Optional[SessionConfig] = None) -> "requests.Session":
    """
    Return a cached, fully-configured requests.Session.

    Each unique SessionConfig gets its own session object (pooled by config key).
    Sessions are *not* shared across threads — callers must not hold them
    across concurrent tasks without external locking.
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library required: pip install requests")

    config = config or SessionConfig()
    key = _session_cache_key(config)

    with _session_lock:
        if key in _session_pool:
            return _session_pool[key]

        session = requests.Session()

        # ── Retry adapter ──────────────────────────────────────────────────
        _backoff = {
            RetryStrategy.EXPONENTIAL: 1.0,
            RetryStrategy.LINEAR:      0.3,
            RetryStrategy.FIXED:       0.0,
            RetryStrategy.FIBONACCI:   1.618,
            RetryStrategy.NO_RETRY:    0.0,
        }.get(config.retry_strategy, 1.0)

        total_retries = 0 if config.retry_strategy == RetryStrategy.NO_RETRY else config.max_retries
        retry = Retry(
            total=total_retries,
            read=total_retries,
            connect=total_retries,
            backoff_factor=_backoff,
            status_forcelist=[408, 429, 500, 502, 503, 504, 507, 520, 521, 522, 524],
            allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"],
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=config.pool_connections,
            pool_maxsize=config.pool_maxsize,
            pool_block=False,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # ── SSL ────────────────────────────────────────────────────────────
        if not config.verify_ssl:
            session.verify = False
            warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        else:
            session.verify = True

        # ── Headers ────────────────────────────────────────────────────────
        if config.user_agent_rotation or config.browser_fingerprinting:
            session.headers.update(create_waf_headers(config.browser_fingerprinting))
        else:
            session.headers.update({
                "User-Agent": f"AMB-Sentinel/{__version__}",
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate",
            })

        # ── Proxy ──────────────────────────────────────────────────────────
        if config.proxy:
            session.proxies.update(config.proxy)

        # ── Misc ───────────────────────────────────────────────────────────
        session.max_redirects = config.max_redirects
        if config.compression:
            session.headers["Accept-Encoding"] = "gzip, deflate, br"

        _session_pool[key] = session
        logger.debug(
            "Session created: timeout=%ds retries=%d security=%s",
            config.timeout, total_retries, config.security_level.name,
        )
        return session


def clear_session_pool() -> int:
    """Close and remove all cached sessions. Returns count cleared."""
    with _session_lock:
        count = len(_session_pool)
        for s in _session_pool.values():
            try:
                s.close()
            except Exception:
                pass
        _session_pool.clear()
    return count


# ──────────────────────────────────────────────
# File integrity
# ──────────────────────────────────────────────

_HASH_FACTORIES = {
    "md5":    hashlib.md5,
    "sha1":   hashlib.sha1,
    "sha256": hashlib.sha256,
    "sha512": hashlib.sha512,
    "sha3_256": lambda: hashlib.new("sha3_256"),
    "blake2b": lambda: hashlib.new("blake2b"),
}




# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION ADDITIONS v5.0
# ─────────────────────────────────────────────────────────────────────────────

class _ResponseCache:
    """
    Thread-safe LRU response cache with TTL expiry.
    
    Used by http_fetch to avoid hammering the same URL within a short window.
    Only caches GET responses with status 200 and Content-Type text/*.
    Cache key = SHA-256(method + url + sorted(headers subset)).
    """

    def __init__(self, maxsize: int = 256, default_ttl: float = 300.0):
        import collections
        self._store: "collections.OrderedDict[str, tuple]" = collections.OrderedDict()
        self._maxsize = maxsize
        self._ttl = default_ttl
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(method: str, url: str, headers: dict = None) -> str:
        import hashlib
        sig = f"{method.upper()}|{url}"
        if headers:
            sig += "|" + str(sorted(headers.items()))
        return hashlib.sha256(sig.encode()).hexdigest()

    def get(self, key: str):
        """Return cached value or None if missing/expired."""
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            value, exp = self._store[key]
            if time.time() > exp:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value, ttl: float = None) -> None:
        """Insert/update entry with TTL."""
        with self._lock:
            exp = time.time() + (ttl if ttl is not None else self._ttl)
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, exp)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def invalidate(self, url_pattern: str = None) -> int:
        """Remove entries matching url_pattern substring, or all if None."""
        with self._lock:
            if url_pattern is None:
                n = len(self._store)
                self._store.clear()
                return n
            keys_to_del = [k for k, (v, _) in self._store.items()
                          if isinstance(v, dict) and url_pattern in v.get("url", "")]
            for k in keys_to_del:
                del self._store[k]
            return len(keys_to_del)

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


# Global response cache (used by http_fetch wrappers)
_response_cache = _ResponseCache(maxsize=256, default_ttl=300.0)


class AdaptiveDelayMixin:
    """
    Mixin adding per-domain adaptive delay tracking.
    
    Learns the right inter-request delay for each domain by tracking
    the last response status. If we see 429/503 we back off exponentially;
    if responses are clean we slowly reduce delay back to minimum.
    """

    _domain_delays: Dict[str, float] = {}
    _domain_lock = threading.Lock()
    _MIN_DELAY = 0.5   # seconds
    _MAX_DELAY = 60.0  # seconds
    _BACKOFF = 2.0     # multiplier on 429/503
    _RECOVERY = 0.85   # decay factor on success

    @classmethod
    def get_delay(cls, domain: str) -> float:
        with cls._domain_lock:
            return cls._domain_delays.get(domain, cls._MIN_DELAY)

    @classmethod
    def record_response(cls, domain: str, status_code: int) -> None:
        with cls._domain_lock:
            current = cls._domain_delays.get(domain, cls._MIN_DELAY)
            if status_code in (429, 503, 520, 521, 522):
                new_delay = min(current * cls._BACKOFF, cls._MAX_DELAY)
            elif status_code < 400:
                new_delay = max(current * cls._RECOVERY, cls._MIN_DELAY)
            else:
                new_delay = current  # keep on 4xx client errors
            cls._domain_delays[domain] = new_delay

    @classmethod
    def reset_domain(cls, domain: str) -> None:
        with cls._domain_lock:
            cls._domain_delays.pop(domain, None)

    @classmethod
    def all_delays(cls) -> Dict[str, float]:
        with cls._domain_lock:
            return dict(cls._domain_delays)


def get_response_cache() -> "_ResponseCache":
    """Return the global response cache instance."""
    return _response_cache


def rotate_session_headers(session) -> None:
    """
    In-place rotate the User-Agent + Accept-Language on an existing session.
    Call this periodically (e.g., every N requests) to avoid fingerprinting.
    """
    ua = get_random_user_agent()
    session.headers.update(create_waf_headers(browser_fingerprinting=True))
    session.headers["User-Agent"] = ua


def get_domain_delay_stats() -> Dict[str, float]:
    """Return current adaptive delay table for all known domains."""
    return AdaptiveDelayMixin.all_delays()

def calculate_file_hash(
    file_path: Union[str, Path],
    algorithm: str = "sha256",
    chunk_size: int = 65536,
) -> str:
    """
    Compute hex digest of a file using the specified algorithm.

    Supports: md5, sha1, sha256, sha512, sha3_256, blake2b.
    Raises ValueError / FileNotFoundError / IOError.
    """
    algorithm = algorithm.lower()
    if algorithm not in _HASH_FACTORIES:
        raise ValueError(f"Unsupported algorithm: {algorithm!r}. Choose from {list(_HASH_FACTORIES)}")
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp}")
    if not fp.is_file():
        raise ValueError(f"Not a regular file: {fp}")
    h = _HASH_FACTORIES[algorithm]()
    try:
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:
        raise IOError(f"Hash computation failed for {fp}: {exc}") from exc


def validate_file_hash(
    file_path: Union[str, Path],
    expected_hash: str,
    algorithm: str = "sha256",
) -> bool:
    """Return True if file hash matches expected_hash."""
    try:
        return calculate_file_hash(file_path, algorithm).lower() == expected_hash.lower()
    except Exception as exc:
        raise ValidationError(f"Hash validation failed: {exc}") from exc


# ──────────────────────────────────────────────
# Progress tracker
# ──────────────────────────────────────────────

class DownloadProgress:
    """Thread-safe download progress tracker with speed estimation."""

    __slots__ = (
        "total", "downloaded", "start_time", "callback",
        "_lock", "_last_cb_time", "_speed_window",
    )

    def __init__(self, total: int, callback: Optional[Callable[[int, int, float], None]] = None):
        self.total = total
        self.downloaded = 0
        self.start_time = time.monotonic()
        self.callback = callback
        self._lock = threading.Lock()
        self._last_cb_time = self.start_time
        self._speed_window: List[Tuple[float, int]] = []  # (timestamp, bytes)

    def update(self, chunk_bytes: int) -> None:
        """Record chunk_bytes downloaded; fire callback if interval elapsed."""
        with self._lock:
            self.downloaded += chunk_bytes
            now = time.monotonic()

            # Rolling 5-second window for speed calculation
            self._speed_window.append((now, chunk_bytes))
            cutoff = now - 5.0
            self._speed_window = [(t, b) for t, b in self._speed_window if t >= cutoff]
            window_bytes = sum(b for _, b in self._speed_window)
            elapsed_window = max(now - self._speed_window[0][0], 0.001) if self._speed_window else 0.001
            speed = window_bytes / elapsed_window

            if self.callback and (now - self._last_cb_time >= 0.5 or self.downloaded >= self.total):
                self.callback(self.downloaded, self.total, speed)
                self._last_cb_time = now

    @property
    def percentage(self) -> float:
        return (self.downloaded / self.total * 100) if self.total > 0 else 0.0

    @property
    def speed(self) -> float:
        elapsed = max(time.monotonic() - self.start_time, 0.001)
        return self.downloaded / elapsed


# ──────────────────────────────────────────────
# Low-level streaming download
# ──────────────────────────────────────────────

def _stream_download_sync(
    session: "requests.Session",
    url: str,
    dest_path: Path,
    config: DownloadConfig,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> DownloadResult:
    """
    Stream-download *url* to *dest_path* using *session*.

    Supports:
    - Resume via Range header if partial file exists
    - max_bytes cap
    - Progress callbacks
    - Hash computation on the fly
    """
    start = time.monotonic()
    partial_path = dest_path.with_suffix(dest_path.suffix + ".part")
    resumed = False
    resume_offset = 0

    headers: Dict[str, str] = {}
    if config.resume and partial_path.exists():
        resume_offset = partial_path.stat().st_size
        if resume_offset > 0:
            headers["Range"] = f"bytes={resume_offset}-"
            resumed = True

    h_algo = config.hash_algorithm.lower()
    hasher = _HASH_FACTORIES.get(h_algo, hashlib.sha256)() if config.validate_hash else None

    try:
        resp = session.get(
            url,
            stream=True,
            timeout=(config.timeout // 3, config.timeout),
            headers=headers,
            allow_redirects=True,
        )
        status_code = resp.status_code

        if status_code == 416:
            # Range not satisfiable — server already has whole file
            if partial_path.exists():
                partial_path.rename(dest_path)
            return DownloadResult(
                ok=True, status=DownloadStatus.SUCCESS,
                path=dest_path, elapsed=time.monotonic() - start,
                status_code=status_code,
            )

        if status_code not in {200, 206}:
            return DownloadResult(
                ok=False, status=DownloadStatus.FAILED,
                error=f"HTTP {status_code}", status_code=status_code,
                elapsed=time.monotonic() - start,
            )

        content_length = int(resp.headers.get("Content-Length", 0))
        total_size = content_length + resume_offset
        content_type = resp.headers.get("Content-Type", "")

        progress = DownloadProgress(total_size or 0, progress_callback)
        if resume_offset:
            progress.downloaded = resume_offset

        mode = "ab" if resumed and status_code == 206 else "wb"
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        written = resume_offset if resumed else 0
        with open(partial_path, mode) as fh:
            for chunk in resp.iter_content(chunk_size=config.chunk_size):
                if not chunk:
                    continue
                if config.max_bytes and (written + len(chunk)) > config.max_bytes:
                    chunk = chunk[: config.max_bytes - written]
                    fh.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    written += len(chunk)
                    progress.update(len(chunk))
                    break
                fh.write(chunk)
                if hasher:
                    hasher.update(chunk)
                written += len(chunk)
                progress.update(len(chunk))
                if config.max_bytes and written >= config.max_bytes:
                    break

        partial_path.rename(dest_path)
        file_hash = hasher.hexdigest() if hasher else None
        checksum_ok = None
        if config.validate_hash and file_hash and config.expected_hash:
            checksum_ok = file_hash.lower() == config.expected_hash.lower()
            if not checksum_ok:
                if config.remove_incomplete:
                    dest_path.unlink(missing_ok=True)
                return DownloadResult(
                    ok=False, status=DownloadStatus.VALIDATION_FAILED,
                    error="Hash mismatch", hash=file_hash, checksum_valid=False,
                    elapsed=time.monotonic() - start,
                )

        return DownloadResult(
            ok=True,
            status=DownloadStatus.RESUMED if resumed else DownloadStatus.SUCCESS,
            path=dest_path,
            size=total_size or written,
            downloaded=written,
            content_type=content_type,
            status_code=status_code,
            elapsed=time.monotonic() - start,
            resumed=resumed,
            hash=file_hash,
            checksum_valid=checksum_ok,
            metadata={"url": url, "content_length": content_length},
        )

    except Exception as exc:
        if config.remove_incomplete and partial_path.exists():
            partial_path.unlink(missing_ok=True)
        return DownloadResult(
            ok=False, status=DownloadStatus.FAILED,
            error=str(exc), elapsed=time.monotonic() - start,
        )


# ──────────────────────────────────────────────
# Public download_file API
# ──────────────────────────────────────────────

def download_file(
    url: str,
    dest_dir: Union[str, Path] = DEFAULT_DOWNLOAD_DIR,
    filename: Optional[str] = None,
    timeout: int = 120,
    max_bytes: Optional[int] = None,
    max_retries: int = 3,
    resume: bool = True,
    verify_ssl: bool = True,
    waf_bypass: bool = True,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    session_config: Optional[SessionConfig] = None,
    download_config: Optional[DownloadConfig] = None,
) -> Dict[str, Any]:
    """
    Download a file from *url* to *dest_dir*.

    Returns a dict with keys: ok, status, path, size, downloaded,
    content_type, status_code, error, elapsed, resumed, hash,
    checksum_valid, metadata.
    """
    try:
        validate_url(url)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "status": "failed"}

    dl_cfg = download_config or DownloadConfig()
    dl_cfg.dest_dir = Path(dest_dir)
    dl_cfg.timeout = timeout
    dl_cfg.max_bytes = max_bytes
    dl_cfg.resume = resume

    # Determine filename
    if filename:
        fn = sanitize_filename(filename)
    else:
        try:
            sess_cfg = session_config or SessionConfig(
                timeout=15, max_retries=1, user_agent_rotation=waf_bypass
            )
            s = make_session(sess_cfg)
            head = s.head(url, timeout=10, allow_redirects=True)
            ct = head.headers.get("Content-Type")
            fn = generate_safe_filename(url, ct)
        except Exception:
            fn = generate_safe_filename(url)

    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir_path / fn

    if dest_path.exists() and not dl_cfg.overwrite:
        return {
            "ok": True, "status": "already_exists",
            "path": str(dest_path), "size": dest_path.stat().st_size,
        }

    sess_cfg = session_config or SessionConfig(
        timeout=timeout,
        max_retries=max_retries,
        verify_ssl=verify_ssl,
        user_agent_rotation=waf_bypass,
        browser_fingerprinting=waf_bypass,
    )
    session = make_session(sess_cfg)

    result = _stream_download_sync(session, url, dest_path, dl_cfg, progress_callback)

    return {
        "ok": result.ok,
        "status": result.status.value,
        "path": str(result.path) if result.path else None,
        "size": result.size,
        "downloaded": result.downloaded,
        "content_type": result.content_type,
        "status_code": result.status_code,
        "error": result.error,
        "elapsed": result.elapsed,
        "resumed": result.resumed,
        "hash": result.hash,
        "checksum_valid": result.checksum_valid,
        "metadata": result.metadata,
    }


# ──────────────────────────────────────────────
# Async download wrapper (if aiohttp available)
# ──────────────────────────────────────────────

async def download_file_async(
    url: str,
    dest_dir: Union[str, Path] = DEFAULT_DOWNLOAD_DIR,
    filename: Optional[str] = None,
    timeout: int = 120,
    max_bytes: Optional[int] = None,
    chunk_size: int = 65536,
    **kwargs,
) -> Dict[str, Any]:
    """
    Async download; falls back to sync if aiohttp not available.
    """
    if not AIOHTTP_AVAILABLE:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: download_file(url, dest_dir, filename, timeout, max_bytes),
        )

    start = time.monotonic()
    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    fn = filename or generate_safe_filename(url)
    dest_path = dest_dir_path / fn
    partial_path = dest_path.with_suffix(dest_path.suffix + ".part")

    try:
        headers = create_waf_headers(browser_fingerprint=True)
        connector = aiohttp.TCPConnector(limit=100, ssl=False)
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(
            connector=connector, timeout=client_timeout, headers=headers
        ) as session:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status not in (200, 206):
                    return {"ok": False, "error": f"HTTP {resp.status}", "status": "failed"}
                written = 0
                with open(partial_path, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        if max_bytes and written + len(chunk) > max_bytes:
                            chunk = chunk[: max_bytes - written]
                        fh.write(chunk)
                        written += len(chunk)
                        if max_bytes and written >= max_bytes:
                            break
        partial_path.rename(dest_path)
        return {
            "ok": True, "status": "success", "path": str(dest_path),
            "downloaded": written, "elapsed": time.monotonic() - start,
        }
    except Exception as exc:
        if partial_path.exists():
            partial_path.unlink(missing_ok=True)
        return {"ok": False, "error": str(exc), "status": "failed"}


# ──────────────────────────────────────────────
# Directory cleanup
# ──────────────────────────────────────────────

def cleanup_downloads(
    directory: Union[str, Path] = DEFAULT_DOWNLOAD_DIR,
    max_age_hours: float = 24.0,
    max_total_size_mb: Optional[float] = None,
    remove_partial: bool = True,
) -> Dict[str, Any]:
    """
    Remove stale / partial downloads from *directory*.

    Returns dict with cleaned_files, cleaned_size_bytes, remaining_files,
    remaining_size_bytes, errors.
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return {
            "cleaned_files": 0, "cleaned_size_bytes": 0,
            "remaining_files": 0, "remaining_size_bytes": 0,
            "errors": ["Directory does not exist or is not a directory"],
        }

    result = CleanupResult()
    now = time.time()
    max_age_sec = max_age_hours * 3600

    all_files = []
    for fp in dir_path.rglob("*"):
        if fp.is_file():
            try:
                st = fp.stat()
                all_files.append(
                    {"path": fp, "size": st.st_size, "mtime": st.st_mtime,
                     "age": now - st.st_mtime, "partial": fp.suffix == ".part"}
                )
            except OSError as exc:
                result.errors.append(str(exc))

    all_files.sort(key=lambda x: x["mtime"])

    for fi in all_files:
        if (remove_partial and fi["partial"]) or fi["age"] > max_age_sec:
            try:
                fi["path"].unlink()
                result.cleaned_files += 1
                result.cleaned_size_bytes += fi["size"]
                fi["removed"] = True
            except Exception as exc:
                result.errors.append(f"Remove {fi['path']}: {exc}")
        else:
            fi["removed"] = False

    # Enforce total-size cap (oldest first)
    if max_total_size_mb:
        max_bytes_cap = int(max_total_size_mb * 1024 * 1024)
        remaining = [f for f in all_files if not f.get("removed") and f["path"].exists()]
        current_size = sum(f["size"] for f in remaining)
        for fi in remaining:
            if current_size <= max_bytes_cap:
                break
            try:
                fi["path"].unlink()
                result.cleaned_files += 1
                result.cleaned_size_bytes += fi["size"]
                current_size -= fi["size"]
            except Exception as exc:
                result.errors.append(f"Remove {fi['path']}: {exc}")

    for fp in dir_path.rglob("*"):
        if fp.is_file():
            try:
                result.remaining_files += 1
                result.remaining_size_bytes += fp.stat().st_size
            except OSError:
                pass

    logger.info(
        "Cleanup: removed %d files (%.2f MB)",
        result.cleaned_files, result.cleaned_size_bytes / 1_048_576,
    )
    return {
        "cleaned_files": result.cleaned_files,
        "cleaned_size_bytes": result.cleaned_size_bytes,
        "cleaned_size_mb": result.cleaned_size_bytes / 1_048_576,
        "remaining_files": result.remaining_files,
        "remaining_size_bytes": result.remaining_size_bytes,
        "remaining_size_mb": result.remaining_size_bytes / 1_048_576,
        "errors": result.errors or None,
    }
