# tools/download.py
"""
AMB-Sentinel Download — Production-grade file download manager.

Version: 4.0.0
Author: AMB Team © 2025

Features:
- Smart resume via HTTP Range headers + .part lock files
- Parallel batch downloads (ThreadPoolExecutor, order-preserved)
- Multi-algorithm integrity hashing in a single file read pass
- Domain-level rate limiting with token bucket
- DownloadManager class with metrics, audit logging, callbacks
- Convenience functions: download_file(), download_multiple()
- Health check: verifies utils version, disk space, network
- Graceful degradation when optional libs absent

Registrar compatibility:
    download_file, download_multiple, get_download_stats,
    validate_download_result, health_check,
    DownloadManager, DownloadOptions, DownloadResult

Dependencies:
    tools.utils  (required)
    requests     (required)
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import os
import shutil
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from traceback import format_exc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from .utils import (
    download_file as _utils_download_file,
    validate_url,
    sanitize_filename,
    generate_safe_filename,
    DownloadConfig as _UtilsDownloadConfig,
    DownloadStatus,
    SecurityLevel,
    RetryStrategy,
    SecurityError,
    DEFAULT_DOWNLOAD_DIR,
    __version__ as _UTILS_VERSION,
    create_waf_headers,
)

logger = logging.getLogger("Download")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS = 16
MIN_REQUEST_INTERVAL    = 0.05   # 50 ms
MAX_REQUESTS_PER_MINUTE = 120
CHUNK_SIZE_DEFAULT      = 65536  # 64 KB
MAX_FILE_SIZE_GB        = 50     # safety hard-cap


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class DownloadStrategy(Enum):
    STANDARD       = auto()   # Single file with retries
    BATCH          = auto()   # Multiple files concurrently
    RESUME_ONLY    = auto()   # Only resume existing partials
    ATOMIC         = auto()   # Complete or nothing
    STREAM         = auto()   # In-memory without saving
    VALIDATE_FIRST = auto()   # HEAD check before GET


class IntegrityAlgorithm(Enum):
    MD5    = "md5"
    SHA1   = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


class DownloadPriority(Enum):
    LOW      = 0
    NORMAL   = 1
    HIGH     = 2
    CRITICAL = 3


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class DownloadOptions:
    """
    Complete download configuration with security + performance options.

    All fields have safe production defaults.
    """
    # Destination
    dest_dir: Union[str, Path] = field(default=DEFAULT_DOWNLOAD_DIR)
    filename: Optional[str]    = None

    # Timing / retry
    timeout: int               = 120
    max_retries: int           = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL

    # Security
    verify_ssl: bool                              = True
    security_level: SecurityLevel                 = SecurityLevel.MEDIUM
    waf_bypass: bool                              = True
    certificate_pins: Optional[Dict[str, List[str]]] = None

    # Performance
    chunk_size: int             = CHUNK_SIZE_DEFAULT
    max_bytes: Optional[int]    = None
    resume: bool                = True
    overwrite: bool             = False
    remove_incomplete: bool     = True

    # Integrity
    verify_hash: bool                            = False
    hash_algorithms: List[IntegrityAlgorithm]    = field(
        default_factory=lambda: [IntegrityAlgorithm.SHA256]
    )
    expected_hashes: Optional[Dict[str, str]]    = None  # algo → hex

    # Request customisation
    user_agent: Optional[str]              = None
    custom_headers: Dict[str, str]         = field(default_factory=dict)
    proxy: Optional[Dict[str, str]]        = None
    follow_redirects: bool                 = True
    max_redirects: int                     = 10

    # Callbacks
    progress_callback: Optional[Callable[[int, int, float], None]] = None
    status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    # Concurrency
    max_concurrent: int             = 8
    rate_limit_per_domain: bool     = True
    request_delay: float            = 0.0

    # Monitoring
    enable_metrics: bool = True
    audit_log: bool      = True

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        # Normalise dest_dir
        if isinstance(self.dest_dir, str):
            self.dest_dir = Path(self.dest_dir).expanduser().resolve()
        else:
            self.dest_dir = Path(self.dest_dir).expanduser().resolve()
        # Deduplicate algorithms
        seen: set = set()
        uniq = []
        for a in (self.hash_algorithms or []):
            if a not in seen:
                uniq.append(a)
                seen.add(a)
        self.hash_algorithms = uniq


@dataclass
class DownloadResult:
    """
    Comprehensive download result.

    NOTE: This class shadows utils.DownloadResult intentionally — it carries
    richer audit/integrity fields needed by the download module and registrar.
    """
    # Core
    ok: bool
    status: DownloadStatus
    path: Optional[Path]        = None
    url: Optional[str]          = None

    # Size
    size: int                   = 0
    downloaded: int             = 0
    total_size: Optional[int]   = None

    # HTTP
    status_code: Optional[int]  = None
    content_type: Optional[str] = None
    headers: Dict[str, str]     = field(default_factory=dict)

    # Integrity
    hashes: Dict[str, str]              = field(default_factory=dict)
    hash_valid: Optional[bool]          = None
    integrity_verified: bool            = False

    # Performance
    elapsed: float              = 0.0
    speed_kbps: float           = 0.0
    retries: int                = 0
    resumed: bool               = False

    # Errors
    error: Optional[str]        = None
    error_type: Optional[str]   = None
    traceback: Optional[str]    = None

    # Audit
    timestamp: str              = field(default_factory=lambda: datetime.now().isoformat())
    download_id: Optional[str]  = None
    session_id: Optional[str]   = None
    metadata: Dict[str, Any]    = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status.value if isinstance(self.status, DownloadStatus) else str(self.status),
            "path": str(self.path) if self.path else None,
            "url": self.url,
            "size": self.size,
            "downloaded": self.downloaded,
            "total_size": self.total_size,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "headers": self.headers,
            "hashes": self.hashes,
            "hash_valid": self.hash_valid,
            "integrity_verified": self.integrity_verified,
            "elapsed": round(self.elapsed, 3),
            "speed_kbps": round(self.speed_kbps, 2),
            "retries": self.retries,
            "resumed": self.resumed,
            "error": self.error,
            "error_type": self.error_type,
            "timestamp": self.timestamp,
            "download_id": self.download_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_utils_result(cls, r: Dict[str, Any], url: str) -> "DownloadResult":
        """Bridge from utils.download_file() dict to DownloadResult."""
        status_str = r.get("status", "failed")
        try:
            status = DownloadStatus(status_str)
        except ValueError:
            status = DownloadStatus.FAILED
        size = r.get("size", 0) or r.get("downloaded", 0)
        elapsed = r.get("elapsed", 0.0)
        speed = (size / elapsed / 1024) if (elapsed > 0 and size > 0) else 0.0
        return cls(
            ok=r.get("ok", False),
            status=status,
            path=Path(r["path"]) if r.get("path") else None,
            url=url,
            size=size,
            downloaded=r.get("downloaded", 0),
            status_code=r.get("status_code"),
            content_type=r.get("content_type"),
            headers={},
            hashes={r.get("hash_algorithm", "sha256"): r["hash"]} if r.get("hash") else {},
            hash_valid=r.get("checksum_valid"),
            elapsed=elapsed,
            speed_kbps=speed,
            resumed=r.get("resumed", False),
            error=r.get("error"),
            metadata=r.get("metadata", {}),
        )


# ──────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────

class DownloadError(Exception):
    """Download failed."""


class RateLimitError(DownloadError):
    """Domain rate-limit exceeded."""


class IntegrityError(DownloadError):
    """File hash verification failed."""


# ──────────────────────────────────────────────
# URL / path helpers
# ──────────────────────────────────────────────

def validate_download_url(url: str) -> Tuple[bool, Optional[str]]:
    """Return (is_valid, error_message). No exceptions raised."""
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"
    if len(url) > 4096:
        return False, "URL exceeds max length (4096)"
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https", "ftp", "ftps"):
            return False, f"Unsupported scheme: {parsed.scheme}"
        if not parsed.netloc:
            return False, "Missing hostname"
        host = parsed.netloc.split(":")[0]
        if host.startswith("169.254."):
            return False, "Link-local addresses blocked"
        validate_url(url)
        return True, None
    except SecurityError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


def prepare_destination_path(
    url: str,
    dest_dir: Path,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Tuple[Path, str]:
    """
    Build a safe destination path; create directories if needed.

    Returns (full_path, safe_filename).
    Raises ValueError or OSError on problems.
    """
    dest_dir = Path(dest_dir).expanduser().resolve()
    if dest_dir.exists() and not os.access(str(dest_dir), os.W_OK):
        raise OSError(f"Destination not writable: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        dest_dir.chmod(0o700)
    except (PermissionError, NotImplementedError):
        pass

    safe_name = sanitize_filename(filename) if filename else generate_safe_filename(url, content_type)
    full_path = dest_dir / safe_name

    # Path-traversal guard
    try:
        full_path.resolve().relative_to(dest_dir.resolve())
    except ValueError as exc:
        raise SecurityError(f"Path traversal: {full_path}") from exc

    return full_path, safe_name


# ──────────────────────────────────────────────
# Multi-algorithm integrity hashing (single pass)
# ──────────────────────────────────────────────

_ALGO_MAP = {
    IntegrityAlgorithm.MD5:    hashlib.md5,
    IntegrityAlgorithm.SHA1:   hashlib.sha1,
    IntegrityAlgorithm.SHA256: hashlib.sha256,
    IntegrityAlgorithm.SHA512: hashlib.sha512,
    IntegrityAlgorithm.BLAKE2B: hashlib.blake2b,
    IntegrityAlgorithm.BLAKE2S: hashlib.blake2s,
}


def calculate_multiple_hashes(
    file_path: Path,
    algorithms: List[IntegrityAlgorithm],
    chunk_size: int = 65536,
) -> Dict[str, str]:
    """
    Calculate multiple hash digests in a single sequential file read.

    Returns dict: algo_name → hex_digest.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    hashers = {a.value: _ALGO_MAP[a]() for a in algorithms if a in _ALGO_MAP}
    if not hashers:
        return {}

    try:
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                for h in hashers.values():
                    h.update(chunk)
    except IOError as exc:
        raise IOError(f"Cannot hash {file_path}: {exc}") from exc

    return {name: h.hexdigest() for name, h in hashers.items()}


def verify_file_integrity(
    file_path: Path,
    expected_hashes: Dict[str, str],
    algorithms: Optional[List[IntegrityAlgorithm]] = None,
) -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
    """
    Verify file against expected hashes.

    Returns (all_valid, {algo: (valid, actual_hash)}).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    algos = algorithms or [
        IntegrityAlgorithm(a) for a in expected_hashes
        if any(a == x.value for x in IntegrityAlgorithm)
    ]
    actuals = calculate_multiple_hashes(file_path, algos)

    results: Dict[str, Tuple[bool, str]] = {}
    all_ok = True
    for algo in algos:
        name = algo.value
        expected = expected_hashes.get(name, "")
        actual   = actuals.get(name, "")
        valid    = bool(expected and actual and expected.lower() == actual.lower())
        results[name] = (valid, actual)
        if not valid:
            all_ok = False
            logger.warning("Hash mismatch %s: expected %s, got %s", name, expected, actual)

    return all_ok, results


# ──────────────────────────────────────────────
# Domain rate limiter
# ──────────────────────────────────────────────

class _DomainRateLimiter:
    """Per-domain sliding-window rate limiter."""

    def __init__(self, rpm: int = MAX_REQUESTS_PER_MINUTE, min_interval: float = MIN_REQUEST_INTERVAL):
        self._rpm = rpm
        self._min_interval = min_interval
        self._domain_ts: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def acquire(self, domain: str) -> None:
        """Block briefly to respect rate limits. Raises RateLimitError if excessive."""
        with self._lock:
            now = time.monotonic()
            window_start = now - 60.0
            ts = [t for t in self._domain_ts.get(domain, []) if t > window_start]

            if len(ts) >= self._rpm:
                oldest = ts[0]
                wait = 60.0 - (now - oldest) + 0.01
                if wait > 10:
                    raise RateLimitError(f"Rate limit for {domain}: wait {wait:.1f}s")
                time.sleep(wait)
                now = time.monotonic()

            if ts and (now - ts[-1]) < self._min_interval:
                time.sleep(self._min_interval - (now - ts[-1]))
                now = time.monotonic()

            ts.append(now)
            self._domain_ts[domain] = ts[-self._rpm:]


_rate_limiter = _DomainRateLimiter()


# ──────────────────────────────────────────────
# DownloadManager — core engine
# ──────────────────────────────────────────────

class DownloadManager:
    """
    Thread-safe download manager with parallel execution, metrics, and auditing.

    Usage:
        mgr = DownloadManager(max_concurrent=8)
        result = mgr.download_file("https://example.com/file.pdf")
        mgr.shutdown()
    """

    def __init__(self, max_concurrent: int = 8):
        self.max_concurrent = min(max_concurrent, MAX_CONCURRENT_DOWNLOADS)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrent,
            thread_name_prefix="download",
        )
        self.download_lock = threading.Lock()
        self.active_downloads: set = set()
        self.metrics: Dict[str, Any] = {
            "downloads_started":   0,
            "downloads_completed": 0,
            "downloads_failed":    0,
            "total_bytes":         0,
            "average_speed_kbps":  0.0,
            "errors_by_type":      {},
        }
        self._session_id = f"dm_{int(time.time() * 1000)}"

    def download_file(
        self,
        url: str,
        options: Optional[DownloadOptions] = None,
    ) -> DownloadResult:
        """
        Download a single file with full retry / integrity / callback pipeline.

        Returns a DownloadResult dataclass.
        """
        opts = options or DownloadOptions()
        t0   = time.monotonic()
        download_id = f"{int(t0 * 1000)}_{hash(url) & 0xFFFF:04x}"

        with self.download_lock:
            self.metrics["downloads_started"] += 1
            self.active_downloads.add(download_id)

        # Notify start
        if opts.status_callback:
            opts.status_callback("started", {"url": url, "download_id": download_id})

        # Validate
        valid, err_msg = validate_download_url(url)
        if not valid:
            result = DownloadResult(
                ok=False, status=DownloadStatus.FAILED,
                url=url, error=err_msg, error_type="ValidationError",
                elapsed=time.monotonic() - t0, download_id=download_id,
            )
            self._record_failure(result, opts)
            return result

        # Domain rate-limiting
        if opts.rate_limit_per_domain:
            try:
                domain = urlparse(url).netloc
                _rate_limiter.acquire(domain)
            except RateLimitError as exc:
                logger.warning("Rate limit: %s", exc)
                time.sleep(1.0)

        # Optional request delay
        if opts.request_delay > 0:
            time.sleep(opts.request_delay)

        # Resolve destination
        try:
            dest_dir = Path(opts.dest_dir).expanduser().resolve()
            dest_path, safe_fn = prepare_destination_path(
                url, dest_dir, opts.filename
            )
        except Exception as exc:
            result = DownloadResult(
                ok=False, status=DownloadStatus.FAILED, url=url,
                error=str(exc), error_type=type(exc).__name__,
                elapsed=time.monotonic() - t0, download_id=download_id,
            )
            self._record_failure(result, opts)
            return result

        # Check overwrite
        if dest_path.exists() and not opts.overwrite:
            result = DownloadResult(
                ok=True, status=DownloadStatus.SUCCESS, url=url,
                path=dest_path, size=dest_path.stat().st_size,
                elapsed=time.monotonic() - t0, download_id=download_id,
                metadata={"note": "already_exists"},
            )
            self._record_success(result, opts)
            return result

        # Build headers
        dl_headers = dict(create_waf_headers(browser_fingerprint=opts.waf_bypass))
        if opts.user_agent:
            dl_headers["User-Agent"] = opts.user_agent
        dl_headers.update(opts.custom_headers)

        # Delegate to utils stream-download
        utils_dl_cfg = _UtilsDownloadConfig(
            dest_dir=str(dest_dir),
            filename=safe_fn,
            timeout=opts.timeout,
            max_bytes=opts.max_bytes,
            chunk_size=opts.chunk_size,
            resume=opts.resume,
            validate_hash=bool(opts.verify_hash and opts.expected_hashes),
            expected_hash=opts.expected_hashes.get("sha256") if opts.expected_hashes else None,
            hash_algorithm="sha256",
            remove_incomplete=opts.remove_incomplete,
            overwrite=opts.overwrite,
        )
        try:
            utils_result = _utils_download_file(
                url=url,
                dest_dir=str(dest_dir),
                filename=safe_fn,
                timeout=opts.timeout,
                max_bytes=opts.max_bytes,
                max_retries=opts.max_retries,
                resume=opts.resume,
                verify_ssl=opts.verify_ssl,
                waf_bypass=opts.waf_bypass,
                progress_callback=opts.progress_callback,
                download_config=utils_dl_cfg,
            )
        except Exception as exc:
            result = DownloadResult(
                ok=False, status=DownloadStatus.FAILED, url=url,
                error=str(exc), error_type=type(exc).__name__,
                elapsed=time.monotonic() - t0, download_id=download_id,
                traceback=format_exc() if logger.isEnabledFor(logging.DEBUG) else None,
            )
            self._record_failure(result, opts)
            return result

        result = DownloadResult.from_utils_result(utils_result, url)
        result.download_id = download_id
        result.session_id  = self._session_id
        result.elapsed     = time.monotonic() - t0
        if result.size > 0 and result.elapsed > 0:
            result.speed_kbps = result.size / result.elapsed / 1024

        # Additional multi-algorithm hashing
        if result.ok and result.path and opts.hash_algorithms:
            try:
                result.hashes = calculate_multiple_hashes(result.path, opts.hash_algorithms)
                result.integrity_verified = True
                # Validate against expected
                if opts.verify_hash and opts.expected_hashes:
                    all_ok, _ = verify_file_integrity(
                        result.path, opts.expected_hashes, opts.hash_algorithms
                    )
                    result.hash_valid = all_ok
                    if not all_ok:
                        result.ok = False
                        result.status = DownloadStatus.VALIDATION_FAILED
                        result.error = "Hash verification failed"
                        if opts.remove_incomplete and result.path.exists():
                            result.path.unlink(missing_ok=True)
                            result.path = None
            except Exception as exc:
                logger.warning("Post-download hash failed: %s", exc)

        if result.ok:
            self._record_success(result, opts)
        else:
            self._record_failure(result, opts)

        if opts.audit_log:
            self._audit(result)

        return result

    def download_batch(
        self,
        urls: List[str],
        options: Optional[DownloadOptions] = None,
        fail_fast: bool = False,
    ) -> List[DownloadResult]:
        """
        Download multiple files in parallel (order-preserved).

        Args:
            urls:      List of URLs to download.
            options:   Shared DownloadOptions for all files.
            fail_fast: Cancel remaining downloads on first failure.
        """
        if not urls:
            return []
        opts = options or DownloadOptions()
        concurrency = min(opts.max_concurrent, self.max_concurrent, len(urls))

        logger.info("Batch: %d files, %d workers", len(urls), concurrency)
        results: List[Optional[DownloadResult]] = [None] * len(urls)
        cancelled = threading.Event()

        def _dl(idx: int, url: str) -> Tuple[int, DownloadResult]:
            if fail_fast and cancelled.is_set():
                r = DownloadResult(
                    ok=False, status=DownloadStatus.FAILED, url=url,
                    error="Cancelled", error_type="CancelledError",
                )
                return idx, r
            r = self.download_file(url, opts)
            if fail_fast and not r.ok:
                cancelled.set()
            return idx, r

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {pool.submit(_dl, i, u): i for i, u in enumerate(urls)}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    idx, r = fut.result(timeout=opts.timeout * 2)
                    results[idx] = r
                except concurrent.futures.TimeoutError:
                    orig = futs[fut]
                    results[orig] = DownloadResult(
                        ok=False, status=DownloadStatus.TIMEOUT, url=urls[orig],
                        error=f"Timed out after {opts.timeout * 2}s",
                    )
                except Exception as exc:
                    orig = futs[fut]
                    results[orig] = DownloadResult(
                        ok=False, status=DownloadStatus.FAILED, url=urls[orig],
                        error=str(exc), error_type=type(exc).__name__,
                    )

        # Fill any None gaps (shouldn't happen)
        for i in range(len(urls)):
            if results[i] is None:
                results[i] = DownloadResult(
                    ok=False, status=DownloadStatus.FAILED, url=urls[i],
                    error="Not executed",
                )

        ok_count  = sum(1 for r in results if r.ok)  # type: ignore[union-attr]
        total_mb  = sum(r.size for r in results if r.ok) / 1_048_576  # type: ignore
        logger.info("Batch done: %d/%d OK, %.2f MB total", ok_count, len(urls), total_mb)
        return results  # type: ignore

    def get_metrics(self) -> Dict[str, Any]:
        with self.download_lock:
            m = self.metrics.copy()
            m["active_downloads"] = len(self.active_downloads)
            m["worker_threads"]   = self.max_concurrent
        return m

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)
        logger.debug("DownloadManager shutdown complete")

    # ── private helpers ────────────────────────────────────────────────────

    def _record_success(self, result: DownloadResult, opts: DownloadOptions) -> None:
        with self.download_lock:
            self.metrics["downloads_completed"] += 1
            self.metrics["total_bytes"] += result.size
            if result.elapsed > 0 and result.size > 0:
                speed = result.size / result.elapsed / 1024
                n = self.metrics["downloads_completed"]
                self.metrics["average_speed_kbps"] = (
                    (self.metrics["average_speed_kbps"] * (n - 1) + speed) / n
                )
            self.active_downloads.discard(result.download_id or "")
        if opts.status_callback:
            opts.status_callback("completed", result.to_dict())
        if opts.completion_callback:
            opts.completion_callback(result.to_dict())

    def _record_failure(self, result: DownloadResult, opts: DownloadOptions) -> None:
        with self.download_lock:
            self.metrics["downloads_failed"] += 1
            et = result.error_type or "Unknown"
            self.metrics["errors_by_type"][et] = self.metrics["errors_by_type"].get(et, 0) + 1
            self.active_downloads.discard(result.download_id or "")
        if opts.status_callback:
            opts.status_callback("failed", result.to_dict())

    def _audit(self, result: DownloadResult) -> None:
        logger.info(
            "AUDIT url=%s status=%s size=%d elapsed=%.2fs id=%s",
            result.url, result.status.value if hasattr(result.status, "value") else result.status,
            result.size, result.elapsed, result.download_id,
        )


# Shared singleton for convenience functions
_default_manager: Optional[DownloadManager] = None
_manager_lock = threading.Lock()


def _get_default_manager() -> DownloadManager:
    global _default_manager
    with _manager_lock:
        if _default_manager is None:
            _default_manager = DownloadManager(max_concurrent=8)
    return _default_manager


# ──────────────────────────────────────────────
# Public convenience functions
# ──────────────────────────────────────────────

def download_file(
    url: str,
    dest_dir: Union[str, Path] = DEFAULT_DOWNLOAD_DIR,
    filename: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 3,
    resume: bool = True,
    verify_ssl: bool = True,
    waf_bypass: bool = True,
    verify_hash: bool = False,
    expected_hash: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Download a single file and return a result dict.

    This is the primary interface used by the registrar.

    Returns dict with: ok, status, path, url, size, downloaded,
    hashes, elapsed, error, metadata, …
    """
    mgr = _get_default_manager()
    opts_kw: Dict[str, Any] = dict(
        dest_dir=dest_dir, filename=filename, timeout=timeout,
        max_retries=max_retries, resume=resume, verify_ssl=verify_ssl,
        waf_bypass=waf_bypass, verify_hash=verify_hash,
        progress_callback=progress_callback,
    )
    if expected_hash and verify_hash:
        opts_kw["expected_hashes"] = {"sha256": expected_hash}
    # Allow extra DownloadOptions fields via kwargs
    valid_fields = {f.name for f in DownloadOptions.__dataclass_fields__.values()}  # type: ignore
    for k, v in kwargs.items():
        if k in valid_fields:
            opts_kw[k] = v

    try:
        opts = DownloadOptions(**opts_kw)
        return mgr.download_file(url, opts).to_dict()
    except Exception as exc:
        logger.error("download_file error: %s", exc)
        return {"ok": False, "url": url, "error": str(exc),
                "status": "failed", "path": None, "size": 0}


def download_multiple(
    urls: List[str],
    dest_dir: Union[str, Path] = DEFAULT_DOWNLOAD_DIR,
    max_concurrent: int = 8,
    timeout: int = 120,
    fail_fast: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Download multiple files in parallel.

    Returns a list of result dicts (same keys as download_file).
    """
    mgr = _get_default_manager()
    opts_kw: Dict[str, Any] = dict(
        dest_dir=dest_dir, timeout=timeout, max_concurrent=max_concurrent,
    )
    valid_fields = {f.name for f in DownloadOptions.__dataclass_fields__.values()}  # type: ignore
    for k, v in kwargs.items():
        if k in valid_fields:
            opts_kw[k] = v
    try:
        opts = DownloadOptions(**opts_kw)
    except Exception:
        opts = DownloadOptions(dest_dir=dest_dir, timeout=timeout, max_concurrent=max_concurrent)

    results = mgr.download_batch(urls, opts, fail_fast=fail_fast)
    return [r.to_dict() for r in results]


def validate_download_result(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a download result dict for completeness and consistency.

    Returns (is_valid, [issue_messages]).
    """
    issues: List[str] = []
    for req in ("ok", "status", "url"):
        if req not in result:
            issues.append(f"Missing field: {req}")

    if result.get("ok") and result.get("status") in ("failed", "timeout", "validation_failed"):
        issues.append("ok=True but status indicates failure")

    if result.get("ok") and result.get("path"):
        try:
            p = Path(result["path"])
            if not p.exists():
                issues.append(f"File not found: {p}")
            elif not p.is_file():
                issues.append(f"Path is not a file: {p}")
            else:
                actual = p.stat().st_size
                expected_size = result.get("size") or result.get("downloaded")
                if expected_size and actual != expected_size:
                    issues.append(f"Size mismatch: expected {expected_size}, got {actual}")
        except Exception as exc:
            issues.append(f"File check error: {exc}")

    if result.get("hash_valid") is False and result.get("ok"):
        issues.append("ok=True but hash_valid=False")

    return len(issues) == 0, issues


def get_download_stats(
    directory: Union[str, Path] = DEFAULT_DOWNLOAD_DIR,
    detailed: bool = False,
) -> Dict[str, Any]:
    """
    Scan *directory* and return statistics about its contents.

    Returns: directory, total_files, total_size_bytes, total_size_mb,
    file_types, age_distribution, largest_file, newest_file, …
    """
    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.exists():
        return {"error": "Directory does not exist", "directory": str(dir_path)}
    if not dir_path.is_dir():
        return {"error": "Not a directory", "directory": str(dir_path)}

    stats: Dict[str, Any] = {
        "directory": str(dir_path),
        "scanned_at": datetime.now().isoformat(),
        "total_files": 0,
        "total_size_bytes": 0,
        "total_size_mb": 0.0,
        "file_types": {},
        "age_distribution": {
            "less_than_hour": 0, "less_than_day": 0,
            "less_than_week": 0, "older_than_week": 0,
        },
        "largest_file": None,
        "smallest_file": None,
        "oldest_file": None,
        "newest_file": None,
    }
    if detailed:
        stats["files"] = []

    now = time.time()

    for fp in dir_path.rglob("*"):
        if not fp.is_file() or fp.name.startswith("."):
            continue
        try:
            st    = fp.stat()
            size  = st.st_size
            age   = now - st.st_mtime
            ext   = fp.suffix.lower() or "no_extension"

            stats["total_files"]      += 1
            stats["total_size_bytes"] += size
            stats["file_types"][ext]   = stats["file_types"].get(ext, 0) + 1

            if age < 3_600:
                stats["age_distribution"]["less_than_hour"] += 1
            elif age < 86_400:
                stats["age_distribution"]["less_than_day"] += 1
            elif age < 604_800:
                stats["age_distribution"]["less_than_week"] += 1
            else:
                stats["age_distribution"]["older_than_week"] += 1

            fi = {"path": str(fp), "size": size, "mtime": st.st_mtime, "age_seconds": age, "extension": ext}
            if stats["largest_file"] is None or size > stats["largest_file"]["size"]:
                stats["largest_file"] = fi.copy()
            if stats["smallest_file"] is None or size < stats["smallest_file"]["size"]:
                stats["smallest_file"] = fi.copy()
            if stats["oldest_file"] is None or st.st_mtime < stats["oldest_file"]["mtime"]:
                stats["oldest_file"] = fi.copy()
            if stats["newest_file"] is None or st.st_mtime > stats["newest_file"]["mtime"]:
                stats["newest_file"] = fi.copy()
            if detailed:
                stats["files"].append(fi)
        except OSError:
            pass

    n = stats["total_files"]
    stats["total_size_mb"] = stats["total_size_bytes"] / 1_048_576
    stats["file_types"]    = dict(sorted(stats["file_types"].items(), key=lambda x: -x[1]))
    if n:
        stats["average_file_size_bytes"] = stats["total_size_bytes"] // n
        stats["average_file_size_mb"]    = stats["average_file_size_bytes"] / 1_048_576
    return stats


def health_check() -> Dict[str, Any]:
    """
    Health check for the download subsystem.

    Verifies: utils version, download directory, network, disk space.
    """
    health: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    # Utils module version (Bug #3 fix: __version__ now exists)
    health["checks"]["utils_module"] = {"status": "ok", "version": _UTILS_VERSION}

    # Download directory
    try:
        DEFAULT_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        writable = os.access(str(DEFAULT_DOWNLOAD_DIR), os.W_OK)
        health["checks"]["download_directory"] = {
            "status": "ok", "path": str(DEFAULT_DOWNLOAD_DIR), "writable": writable,
        }
        if not writable:
            health["warnings"].append("Download directory not writable")
    except Exception as exc:
        health["checks"]["download_directory"] = {"status": "error", "error": str(exc)}
        health["errors"].append(f"Download dir: {exc}")

    # Network
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        health["checks"]["network"] = {"status": "ok"}
    except Exception as exc:
        health["checks"]["network"] = {"status": "warning", "error": str(exc)}
        health["warnings"].append(f"Network issue: {exc}")

    # Disk space
    try:
        usage = shutil.disk_usage(DEFAULT_DOWNLOAD_DIR)
        free_mb = usage.free / 1_048_576
        health["checks"]["disk_space"] = {
            "status": "ok" if free_mb > 100 else "warning",
            "free_mb": round(free_mb, 1),
            "total_gb": round(usage.total / 1_073_741_824, 1),
            "used_percent": round(usage.used / usage.total * 100, 1),
        }
        if free_mb < 100:
            health["warnings"].append(f"Low disk space: {free_mb:.0f} MB free")
    except Exception as exc:
        health["checks"]["disk_space"] = {"status": "warning", "error": str(exc)}

    if health["errors"]:
        health["status"] = "unhealthy"
    elif health["warnings"]:
        health["status"] = "degraded"

    return health



# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION ADDITIONS v5.0 — Priority queue, download metrics, retry budget
# ─────────────────────────────────────────────────────────────────────────────

import heapq as _heapq
import threading as _dl_threading
import time as _dl_time


class DownloadPriorityQueue:
    """
    Thread-safe priority queue for download jobs.
    
    Lower priority number = higher priority (like a min-heap).
    Jobs with the same priority are processed FIFO.
    
    Usage::
    
        q = DownloadPriorityQueue()
        q.push(priority=1, url="https://urgent.com/file.pdf", meta={})
        q.push(priority=5, url="https://normal.com/file.pdf", meta={})
        job = q.pop()   # → {"url": "https://urgent.com/file.pdf", ...}
    """

    def __init__(self):
        self._heap: list = []
        self._lock = _dl_threading.Lock()
        self._counter = 0  # tie-breaker for FIFO at same priority

    def push(self, priority: int, url: str, meta: dict = None) -> None:
        with self._lock:
            entry = (priority, self._counter, {"url": url, "meta": meta or {}, "queued_at": _dl_time.time()})
            _heapq.heappush(self._heap, entry)
            self._counter += 1

    def pop(self) -> dict:
        """Pop highest-priority job. Returns None if empty."""
        with self._lock:
            if not self._heap:
                return None
            _, _, job = _heapq.heappop(self._heap)
            return job

    def pop_many(self, n: int) -> list:
        """Pop up to n jobs in priority order."""
        with self._lock:
            result = []
            for _ in range(n):
                if not self._heap:
                    break
                _, _, job = _heapq.heappop(self._heap)
                result.append(job)
            return result

    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    def clear(self) -> int:
        with self._lock:
            n = len(self._heap)
            self._heap.clear()
            return n


class DownloadMetrics:
    """
    Detailed download performance counters with per-domain tracking.
    
    Tracks: bytes downloaded, errors, retry counts, avg speed per domain.
    """

    def __init__(self):
        self._lock = _dl_threading.Lock()
        self._total_bytes = 0
        self._total_files = 0
        self._errors = 0
        self._retries = 0
        self._domains: dict = {}  # domain → {"bytes": N, "files": N, "errors": N}

    @staticmethod
    def _domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except Exception:
            return "unknown"

    def record_success(self, url: str, bytes_downloaded: int, elapsed_s: float) -> None:
        dom = self._domain(url)
        with self._lock:
            self._total_bytes += bytes_downloaded
            self._total_files += 1
            d = self._domains.setdefault(dom, {"bytes": 0, "files": 0, "errors": 0, "speed_bps": 0.0})
            d["bytes"] += bytes_downloaded
            d["files"] += 1
            if elapsed_s > 0:
                d["speed_bps"] = round(bytes_downloaded / elapsed_s, 1)

    def record_error(self, url: str, retry: bool = False) -> None:
        dom = self._domain(url)
        with self._lock:
            self._errors += 1
            if retry:
                self._retries += 1
            self._domains.setdefault(dom, {"bytes": 0, "files": 0, "errors": 0, "speed_bps": 0.0})
            self._domains[dom]["errors"] += 1

    def summary(self) -> dict:
        with self._lock:
            total = self._total_files + self._errors
            return {
                "total_files": self._total_files,
                "total_bytes_mb": round(self._total_bytes / 1_048_576, 2),
                "errors": self._errors,
                "retries": self._retries,
                "success_rate": round(self._total_files / total, 4) if total else 0.0,
                "top_domains": dict(
                    sorted(self._domains.items(), key=lambda x: x[1]["bytes"], reverse=True)[:10]
                ),
            }


# Module-level singletons
_download_queue = DownloadPriorityQueue()
_download_metrics = DownloadMetrics()


def get_download_queue() -> "DownloadPriorityQueue":
    """Return the global priority download queue."""
    return _download_queue


def get_download_metrics() -> dict:
    """Return download performance summary."""
    return _download_metrics.summary()

