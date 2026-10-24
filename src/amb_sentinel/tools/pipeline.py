# tools/pipeline.py
"""
AMB-Sentinel Pipeline — Production v1.0.0
Author: AMB Team

Orchestrates the full Ingest → Transcribe/Extract → Chunk → Embed → Index
pipeline. Supports local files, HTTP URLs, and video platform URLs.

Key design decisions
────────────────────
- **No pydantic dependency** — PipelineConfig is a plain dataclass; the original
  required pydantic which is not installed in the runtime environment.
- **Dual async/sync API** — PipelineManager.process() is async for composability
  inside event loops; transcribe_and_index() / transcribe_url_to_vectorstore()
  are synchronous wrappers safe to call from regular (non-async) agent code.
- **Robust event-loop bridge** — sync wrappers detect an existing running loop
  (common in Jupyter / threaded servers) and fall back to a fresh thread with
  its own loop rather than failing with RuntimeError.
- **Parallel chunk indexing** — IndexingProcessor fans out embed+upsert calls
  across an async Semaphore (≤5 concurrent), preventing vector-store saturation.
- **Graceful degradation** — every stage is wrapped independently; a failed
  download does not prevent local-path fallback; a failed upsert is recorded
  but does not abort the batch.

Registrar-expected exports
──────────────────────────
  register_pipeline_tools, transcribe_url_to_vectorstore, transcribe_and_index,
  PipelineManager, PipelineConfig, ProcessingStage, SourceType, ProcessingStatus,
  ProcessingError, ProcessingResult, PipelineMetrics,
  SecurityValidator, DownloadProcessor, TranscriptionProcessor, IndexingProcessor
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import threading
import time
import traceback
import urllib.parse
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Local imports
from .transcribe import transcribe_audio_or_video
from .extract import (
    extract_text_from_file,
    chunk_text,
    _make_embedding_and_upsert,
)
from .download import download_file
from .utils import DEFAULT_DOWNLOAD_DIR

logger = logging.getLogger("Pipeline")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_AUDIO_VIDEO_EXTS = {
    ".mp4", ".mkv", ".wav", ".mp3", ".mov", ".webm", ".avi",
    ".flac", ".ogg", ".aac", ".m4a", ".wma", ".ts", ".opus",
}
SUPPORTED_DOCUMENT_EXTS = {
    ".pdf", ".docx", ".xls", ".xlsx", ".csv", ".txt", ".md",
    ".json", ".xml", ".html", ".htm", ".rtf", ".odt", ".pptx", ".ppt",
}
MAX_TEXT_LENGTH_FOR_DB    = 10_000
DEFAULT_CHUNK_SIZE        = 1_000
DEFAULT_CHUNK_OVERLAP     = 200
DEFAULT_MODEL_NAME        = "small"
MAX_RETRIES               = 3
RETRY_DELAY               = 1.0
MAX_CONCURRENT_TASKS      = 5
MAX_FILE_SIZE_MB          = 500
RATE_LIMIT_RPM            = 60
MAX_PATH_LENGTH           = 4_096
ALLOWED_URL_SCHEMES       = {"http", "https"}
BLACKLISTED_EXTENSIONS    = {".exe", ".bat", ".sh", ".js", ".vbs", ".jar"}


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class ProcessingStage(str, Enum):
    INITIALIZATION = "initialization"
    DOWNLOAD       = "download"
    TRANSCRIPTION  = "transcription"
    TEXT_EXTRACTION = "text_extraction"
    CHUNKING       = "chunking"
    INDEXING       = "indexing"
    PERSISTENCE    = "persistence"
    COMPLETION     = "completion"
    VALIDATION     = "validation"


class SourceType(str, Enum):
    URL      = "url"
    LOCAL    = "local"
    WEB      = "web"
    AUDIO    = "audio"
    VIDEO    = "video"
    DOCUMENT = "document"
    UNKNOWN  = "unknown"


class ProcessingStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    SUCCESS    = "success"
    FAILED     = "failed"
    PARTIAL    = "partial"
    CANCELLED  = "cancelled"


# ──────────────────────────────────────────────────────────────────────────────
# Configuration  (dataclass, NOT pydantic — pydantic not available at runtime)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Pipeline configuration.

    All fields have safe defaults.  Validation is performed in ``__post_init__``.
    """
    chunk_size:           int   = DEFAULT_CHUNK_SIZE
    chunk_overlap:        int   = DEFAULT_CHUNK_OVERLAP
    model_name:           str   = DEFAULT_MODEL_NAME
    max_retries:          int   = MAX_RETRIES
    persist_db:           bool  = False
    max_concurrent_tasks: int   = MAX_CONCURRENT_TASKS
    rate_limit:           int   = RATE_LIMIT_RPM
    max_file_size_mb:     int   = MAX_FILE_SIZE_MB

    def __post_init__(self) -> None:
        if not (100 <= self.chunk_size <= 10_000):
            raise ValueError(f"chunk_size must be 100–10 000, got {self.chunk_size}")
        if not (0 <= self.chunk_overlap < self.chunk_size):
            self.chunk_overlap = min(self.chunk_overlap, self.chunk_size // 5)
        if not (0 <= self.max_retries <= 10):
            raise ValueError(f"max_retries must be 0–10, got {self.max_retries}")
        if not (1 <= self.max_concurrent_tasks <= 50):
            self.max_concurrent_tasks = max(1, min(50, self.max_concurrent_tasks))

    def dict(self) -> Dict[str, Any]:   # pydantic-compat alias
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Error / Result data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessingError:
    """Structured error report — one per failure event in the pipeline."""
    stage:       ProcessingStage
    error_type:  str
    message:     str
    details:     Optional[Dict[str, Any]] = None
    timestamp:   str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    retryable: bool  = True
    severity:  str   = "error"   # debug | info | warning | error | critical

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingResult:
    """Aggregated result from a full pipeline run."""
    success:        bool
    source:         str                      = ""
    source_type:    SourceType               = SourceType.UNKNOWN
    status:         ProcessingStatus         = ProcessingStatus.PENDING
    text:           str                      = ""
    chunks_indexed: int                      = 0
    processing_time: float                   = 0.0
    errors:         List[ProcessingError]    = field(default_factory=list)
    metadata:       Dict[str, Any]           = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.errors:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_type"] = self.source_type.value
        d["status"]      = self.status.value
        d["errors"]      = [e.to_dict() if hasattr(e, "to_dict") else asdict(e)
                             for e in self.errors]
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class PipelineMetrics:
    """Thread-safe pipeline metrics collector."""

    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self.total       = 0
        self.successes   = 0
        self.failures    = 0
        self.total_chunks = 0
        self.total_time  = 0.0
        self.errors_by_type: Dict[str, int] = {}

    def record_success(self, processing_time: float, chunks: int) -> None:
        with self._lock:
            self.total       += 1
            self.successes   += 1
            self.total_chunks += chunks
            self.total_time  += processing_time

    def record_failure(self, stage: ProcessingStage, error_type: str) -> None:
        with self._lock:
            self.total    += 1
            self.failures += 1
            key = f"{stage.value}:{error_type}"
            self.errors_by_type[key] = self.errors_by_type.get(key, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            avg_time = self.total_time / self.successes if self.successes else 0.0
            return {
                "total":                self.total,
                "successes":            self.successes,
                "failures":             self.failures,
                "total_chunks_indexed": self.total_chunks,
                "average_time_seconds": round(avg_time, 3),
                "errors_by_type":       dict(self.errors_by_type),
            }


# ──────────────────────────────────────────────────────────────────────────────
# Security validator
# ──────────────────────────────────────────────────────────────────────────────

class SecurityValidator:
    """Validate URLs and file paths before processing."""

    _PRIVATE_RANGES = re.compile(
        r"^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|127\.|0\.|169\.254\.|::1|localhost)",
        re.IGNORECASE,
    )

    @staticmethod
    def validate_url(url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ALLOWED_URL_SCHEMES:
                return False
            if not parsed.netloc:
                return False
            host = parsed.netloc.split(":")[0]
            if SecurityValidator._PRIVATE_RANGES.match(host):
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def validate_file_path(file_path: str) -> Tuple[bool, Optional[str]]:
        if not file_path:
            return False, "File path is empty"
        if len(file_path) > MAX_PATH_LENGTH:
            return False, "Path exceeds maximum length"
        if "\x00" in file_path:
            return False, "Null byte in path"
        try:
            path = Path(file_path)
            if not path.exists():
                return False, f"File does not exist: {path}"
            if not path.is_file():
                return False, f"Path is not a file: {path}"
            ext = path.suffix.lower()
            if ext in BLACKLISTED_EXTENSIONS:
                return False, f"Blocked extension: {ext}"
            size_mb = path.stat().st_size / 1_048_576
            if size_mb > MAX_FILE_SIZE_MB:
                return False, f"File too large: {size_mb:.1f} MB"
        except Exception as exc:
            return False, str(exc)
        return True, None

    @staticmethod
    def sanitize_filename(name: str) -> str:
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
        safe = safe.strip(". ")
        return safe[:255] or "unnamed_file"


# ──────────────────────────────────────────────────────────────────────────────
# Rate limiter (async token-bucket)
# ──────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Async token-bucket rate limiter (per-instance)."""

    def __init__(self, requests_per_minute: int = RATE_LIMIT_RPM) -> None:
        self._rpm      = requests_per_minute
        self._interval = 60.0 / max(requests_per_minute, 1)
        self._last_ts  = 0.0
        self._lock     = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now  = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = asyncio.get_event_loop().time()


# ──────────────────────────────────────────────────────────────────────────────
# Async circuit breaker
# ──────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """Async-compatible circuit breaker (CLOSED → OPEN → HALF-OPEN)."""

    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60) -> None:
        self._threshold    = failure_threshold
        self._timeout      = reset_timeout
        self._failures     = 0
        self._last_failure = 0.0
        self._state        = self.CLOSED
        self._lock         = asyncio.Lock()

    async def execute(self, fn: Callable, *args, **kwargs):
        async with self._lock:
            if self._state == self.OPEN:
                if time.monotonic() - self._last_failure > self._timeout:
                    self._state = self.HALF_OPEN
                else:
                    raise RuntimeError("Circuit breaker OPEN")

        try:
            result = await fn(*args, **kwargs)
            async with self._lock:
                self._failures = 0
                self._state    = self.CLOSED
            return result
        except Exception as exc:
            async with self._lock:
                self._failures    += 1
                self._last_failure = time.monotonic()
                if self._failures >= self._threshold:
                    self._state = self.OPEN
            raise


# ──────────────────────────────────────────────────────────────────────────────
# Decorator helpers
# ──────────────────────────────────────────────────────────────────────────────

def async_retry(max_attempts: int = 3, delay: float = RETRY_DELAY):
    """Decorator: retry an async function on exception with exponential backoff."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        wait = min(delay * (2 ** attempt), 30.0)
                        logger.warning(
                            "%s retry %d/%d after %.1fs: %s",
                            fn.__name__, attempt + 1, max_attempts, wait, exc,
                        )
                        await asyncio.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


class ThreadSafeErrorCollector:
    """Async-safe list for collecting ProcessingError instances from coroutines."""

    def __init__(self) -> None:
        self._errors: List[ProcessingError] = []
        self._lock = asyncio.Lock()

    async def append(self, error: ProcessingError) -> None:
        async with self._lock:
            self._errors.append(error)

    async def extend(self, errors: List[ProcessingError]) -> None:
        async with self._lock:
            self._errors.extend(errors)

    def get_all(self) -> List[ProcessingError]:
        return list(self._errors)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base processor
# ──────────────────────────────────────────────────────────────────────────────

class BaseProcessor(ABC):
    @abstractmethod
    async def process(self, *args, **kwargs) -> Any: ...

    async def cleanup(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# DownloadProcessor
# ──────────────────────────────────────────────────────────────────────────────

class DownloadProcessor(BaseProcessor):
    """Downloads a remote URL to a local file."""

    def __init__(self, download_dir: str, security_validator: SecurityValidator) -> None:
        self.download_dir       = download_dir
        self.security_validator = security_validator
        self.circuit_breaker    = CircuitBreaker()

    @async_retry(max_attempts=3)
    async def process(self, url: str) -> Tuple[Optional[str], List[ProcessingError]]:
        errors: List[ProcessingError] = []

        if not self.security_validator.validate_url(url):
            errors.append(ProcessingError(
                stage=ProcessingStage.VALIDATION,
                error_type="invalid_url",
                message="URL failed security validation",
                details={"url": url},
                retryable=False,
            ))
            return None, errors

        try:
            logger.info("Downloading: %s", url)
            dl_result = await asyncio.to_thread(
                download_file, url, dest_dir=self.download_dir
            )
            if not dl_result.get("ok"):
                err_msg = dl_result.get("error", "Unknown download error")
                errors.append(ProcessingError(
                    stage=ProcessingStage.DOWNLOAD,
                    error_type="download_failure",
                    message=f"Download failed: {err_msg}",
                    details={"url": url},
                ))
                return None, errors

            local_path = dl_result.get("path")
            logger.info("Downloaded → %s", local_path)
            return local_path, errors

        except Exception as exc:
            logger.error("DownloadProcessor error: %s\n%s", exc, traceback.format_exc())
            errors.append(ProcessingError(
                stage=ProcessingStage.DOWNLOAD,
                error_type="download_exception",
                message=str(exc),
                details={"url": url},
            ))
            return None, errors

    async def cleanup(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# TranscriptionProcessor
# ──────────────────────────────────────────────────────────────────────────────

class TranscriptionProcessor(BaseProcessor):
    """Transcribes audio/video OR extracts text from documents."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._executor  = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TranscribeProc")

    @async_retry(max_attempts=2)
    async def process(
        self, file_path: str
    ) -> Tuple[Optional[Dict[str, Any]], List[ProcessingError]]:
        errors: List[ProcessingError] = []
        try:
            path   = Path(file_path)
            suffix = path.suffix.lower()
            logger.info("Processing %s (%s)", path.name, suffix)

            if suffix in SUPPORTED_AUDIO_VIDEO_EXTS:
                raw = await asyncio.to_thread(
                    transcribe_audio_or_video, str(path), model_name=self.model_name
                )
                # TranscriptionResult → dict
                content = raw.to_dict() if hasattr(raw, "to_dict") else raw
            else:
                raw = await asyncio.to_thread(extract_text_from_file, str(path))
                content = raw.to_dict() if hasattr(raw, "to_dict") else raw

            if not content or not content.get("ok"):
                err_msg = content.get("error", "Unknown error") if content else "No result"
                errors.append(ProcessingError(
                    stage=ProcessingStage.TRANSCRIPTION,
                    error_type="processing_failure",
                    message=f"Failed to process {path.name}: {err_msg}",
                    details={"file_path": file_path, "suffix": suffix},
                ))
            return content, errors

        except Exception as exc:
            logger.error("TranscriptionProcessor error: %s\n%s", exc, traceback.format_exc())
            errors.append(ProcessingError(
                stage=ProcessingStage.TRANSCRIPTION,
                error_type="processing_exception",
                message=str(exc),
                details={"file_path": file_path},
            ))
            return None, errors

    async def cleanup(self) -> None:
        self._executor.shutdown(wait=False)


# ──────────────────────────────────────────────────────────────────────────────
# IndexingProcessor
# ──────────────────────────────────────────────────────────────────────────────

class IndexingProcessor(BaseProcessor):
    """Chunks text and upserts each chunk into the vector store."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
        self.chunk_size = chunk_size
        self.overlap    = overlap

    async def process(
        self,
        service: Any,
        text: str,
        source_id: str,
        source: str,
        source_type: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[ProcessingError]]:
        """Chunk *text* and embed+upsert each chunk. Returns (indexed_chunks, errors)."""
        indexed: List[Dict[str, Any]] = []
        error_collector = ThreadSafeErrorCollector()

        if not text or not text.strip():
            await error_collector.append(ProcessingError(
                stage=ProcessingStage.CHUNKING,
                error_type="empty_text",
                message="Text is empty",
                details={"source": source},
                retryable=False,
            ))
            return indexed, error_collector.get_all()

        try:
            chunks = await asyncio.to_thread(
                chunk_text, text, chunk_size=self.chunk_size, chunk_overlap=self.overlap
            )
            logger.info("Created %d chunks from %d chars", len(chunks), len(text))
            total_chunks = len(chunks)
            semaphore    = asyncio.Semaphore(5)   # ≤5 concurrent embed+upsert calls

            async def _index_one(i: int, chunk: str) -> Optional[Dict[str, Any]]:
                async with semaphore:
                    try:
                        meta = {
                            "source_type":  source_type,
                            "source":       source,
                            "source_id":    source_id,
                            "chunk_index":  i,
                            "total_chunks": total_chunks,
                            "chunk_size":   len(chunk),
                            "timestamp":    datetime.utcnow().isoformat(),
                            **(extra_metadata or {}),
                        }
                        upsert_result = await asyncio.to_thread(
                            _make_embedding_and_upsert, service, chunk, source_id,
                            metadata=meta, chunk_index=i, total_chunks=total_chunks,
                        )
                        if upsert_result.get("ok"):
                            return {
                                "chunk_index": i, "chunk_size": len(chunk),
                                "upsert_id":  upsert_result.get("id"),
                                "metadata":   meta,
                            }
                        await error_collector.append(ProcessingError(
                            stage=ProcessingStage.INDEXING,
                            error_type="upsert_failure",
                            message=f"Upsert chunk {i}: {upsert_result.get('error')}",
                            details={"chunk_index": i},
                        ))
                        return None
                    except Exception as exc:
                        logger.error("Chunk %d error: %s", i, exc)
                        await error_collector.append(ProcessingError(
                            stage=ProcessingStage.INDEXING,
                            error_type="chunk_exception",
                            message=str(exc),
                            details={"chunk_index": i},
                        ))
                        return None

            tasks   = [_index_one(i, c) for i, c in enumerate(chunks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, dict):
                    indexed.append(r)
                elif isinstance(r, Exception):
                    await error_collector.append(ProcessingError(
                        stage=ProcessingStage.INDEXING,
                        error_type="task_exception",
                        message=str(r),
                    ))

            return indexed, error_collector.get_all()

        except Exception as exc:
            logger.error("IndexingProcessor error: %s\n%s", exc, traceback.format_exc())
            await error_collector.append(ProcessingError(
                stage=ProcessingStage.CHUNKING,
                error_type="chunking_exception",
                message=str(exc),
                details={"text_length": len(text)},
            ))
            return indexed, error_collector.get_all()

    async def cleanup(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# PipelineManager
# ──────────────────────────────────────────────────────────────────────────────

class PipelineManager:
    """
    Orchestrates the full ingest → process → index pipeline.

    ``process()`` is an async coroutine; use ``transcribe_and_index()`` or
    ``transcribe_url_to_vectorstore()`` for synchronous entry points.
    """

    def __init__(
        self,
        service: Any,
        config: Optional[PipelineConfig] = None,
        metrics: Optional[PipelineMetrics] = None,
    ) -> None:
        self.service    = service
        self.config     = config or PipelineConfig()
        self.metrics    = metrics or PipelineMetrics()
        self.security   = SecurityValidator()
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        self.is_shutting_down = False

        self.download_processor     = DownloadProcessor(
            download_dir=self._get_download_dir(),
            security_validator=self.security,
        )
        self.transcription_processor = TranscriptionProcessor(
            model_name=self.config.model_name
        )
        self.indexing_processor = IndexingProcessor(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_download_dir(self) -> str:
        try:
            if hasattr(self.service, "tools_registrar"):
                reg = self.service.tools_registrar
                if hasattr(reg, "dest_download_dir"):
                    return str(reg.dest_download_dir)
        except Exception:
            pass
        return str(DEFAULT_DOWNLOAD_DIR)

    def _is_remote(self, path_or_url: str) -> bool:
        low = (path_or_url or "").lower()
        return (
            low.startswith("http://") or
            low.startswith("https://") or
            "youtube.com" in low or
            "youtu.be"    in low
        )

    def _detect_source_type(self, path_or_url: str, is_remote: bool) -> SourceType:
        if is_remote:
            return SourceType.URL
        try:
            suffix = Path(path_or_url).suffix.lower()
            if suffix in SUPPORTED_AUDIO_VIDEO_EXTS:
                audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus"}
                return SourceType.AUDIO if suffix in audio_exts else SourceType.VIDEO
            if suffix in SUPPORTED_DOCUMENT_EXTS:
                return SourceType.DOCUMENT
        except Exception:
            pass
        return SourceType.LOCAL

    def _make_source_id(self, source: str) -> str:
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    # ── Core async process ────────────────────────────────────────────────────

    async def process(
        self,
        path_or_url: str,
        source_type: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        persist_db: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: validate → download (if remote) → transcribe/extract
        → chunk → embed+index.

        Returns a dict compatible with ``ProcessingResult.to_dict()``.
        """
        t0 = time.monotonic()
        extra_metadata = extra_metadata or {}
        persist_db     = persist_db if persist_db is not None else self.config.persist_db

        result = ProcessingResult(
            success=False, source=path_or_url,
            source_type=SourceType.UNKNOWN, status=ProcessingStatus.PROCESSING,
            metadata={"config": self.config.dict(), "extra_metadata": extra_metadata},
        )

        if self.is_shutting_down:
            result.errors.append(ProcessingError(
                stage=ProcessingStage.INITIALIZATION, error_type="service_unavailable",
                message="Pipeline is shutting down", retryable=True,
            ))
            result.status = ProcessingStatus.CANCELLED
            result.processing_time = time.monotonic() - t0
            return result.to_dict()

        try:
            # ── Source-type detection ─────────────────────────────────────
            is_remote = self._is_remote(path_or_url)
            det_type  = self._detect_source_type(path_or_url, is_remote)
            result.source_type = SourceType(source_type) if source_type else det_type
            logger.info("Pipeline start: %s (type=%s)", path_or_url, result.source_type.value)

            current_path = path_or_url

            # ── Download if remote ────────────────────────────────────────
            if is_remote:
                await self.rate_limiter.acquire()
                local_path, dl_errors = await self.download_processor.process(path_or_url)
                result.errors.extend(dl_errors)

                if local_path:
                    current_path = local_path
                    is_remote    = False
                elif dl_errors:
                    # Fatal: cannot continue without a file
                    result.status = ProcessingStatus.FAILED
                    result.processing_time = time.monotonic() - t0
                    self.metrics.record_failure(ProcessingStage.DOWNLOAD, "download_failed")
                    return result.to_dict()

            # ── Validate local file ───────────────────────────────────────
            if not is_remote:
                valid, err_msg = self.security.validate_file_path(current_path)
                if not valid:
                    result.errors.append(ProcessingError(
                        stage=ProcessingStage.VALIDATION,
                        error_type="invalid_file",
                        message=err_msg or "File validation failed",
                        details={"path": current_path},
                        retryable=False,
                    ))
                    result.status = ProcessingStatus.FAILED
                    result.processing_time = time.monotonic() - t0
                    return result.to_dict()

            # ── Transcription / extraction ────────────────────────────────
            transcription, tr_errors = await self.transcription_processor.process(current_path)
            result.errors.extend(tr_errors)

            if not transcription or not transcription.get("ok"):
                result.status = ProcessingStatus.FAILED
                result.processing_time = time.monotonic() - t0
                self.metrics.record_failure(ProcessingStage.TRANSCRIPTION, "processing_failed")
                return result.to_dict()

            text = transcription.get("text", "").strip()
            result.text = text
            result.metadata.update({
                "transcription_backend": transcription.get("backend"),
                "transcription_language": transcription.get("language"),
                "transcription_duration": transcription.get("duration"),
                "extraction_method": transcription.get("extraction_method"),
            })

            if not text:
                result.errors.append(ProcessingError(
                    stage=ProcessingStage.TEXT_EXTRACTION,
                    error_type="empty_text",
                    message="Transcription/extraction returned empty text",
                    details={"path": current_path},
                ))
                result.status = ProcessingStatus.PARTIAL
                result.processing_time = time.monotonic() - t0
                return result.to_dict()

            # ── Chunk + index ─────────────────────────────────────────────
            source_id = self._make_source_id(path_or_url)
            indexed, idx_errors = await self.indexing_processor.process(
                service=self.service,
                text=text,
                source_id=source_id,
                source=path_or_url,
                source_type=result.source_type.value,
                extra_metadata=extra_metadata,
            )
            result.errors.extend(idx_errors)
            result.chunks_indexed = len(indexed)

            # ── Optional DB persistence ───────────────────────────────────
            if persist_db:
                await self._persist_to_database(
                    source=path_or_url, source_id=source_id,
                    text=text, indexed=indexed, result=result,
                )

            # ── Finalise ──────────────────────────────────────────────────
            has_failures = any(e.severity == "error" for e in result.errors)
            result.success = not has_failures
            result.status  = ProcessingStatus.PARTIAL if (result.errors and result.success) else (
                ProcessingStatus.SUCCESS if result.success else ProcessingStatus.FAILED
            )
            result.processing_time = time.monotonic() - t0

            if result.success:
                self.metrics.record_success(result.processing_time, result.chunks_indexed)
            else:
                self.metrics.record_failure(ProcessingStage.COMPLETION, "partial_failure")

            logger.info(
                "Pipeline done: %s | chunks=%d | time=%.2fs | status=%s",
                path_or_url, result.chunks_indexed,
                result.processing_time, result.status.value,
            )

        except Exception as exc:
            logger.error("Pipeline unexpected error: %s\n%s", exc, traceback.format_exc())
            result.errors.append(ProcessingError(
                stage=ProcessingStage.COMPLETION,
                error_type="unexpected_exception",
                message=str(exc),
            ))
            result.status = ProcessingStatus.FAILED
            result.processing_time = time.monotonic() - t0
            self.metrics.record_failure(ProcessingStage.COMPLETION, "unexpected_exception")

        return result.to_dict()

    # ── DB persistence (best-effort) ──────────────────────────────────────────

    async def _persist_to_database(
        self,
        source: str,
        source_id: str,
        text: str,
        indexed: List[Dict[str, Any]],
        result: ProcessingResult,
    ) -> None:
        """Attempt to persist processing summary to a database (best-effort)."""
        try:
            db = getattr(self.service, "db_manager", None) or getattr(self.service, "db", None)
            if db is None:
                return
            record = {
                "source_id":     source_id,
                "source":        source,
                "text":          text[:MAX_TEXT_LENGTH_FOR_DB],
                "chunks_count":  len(indexed),
                "status":        result.status.value,
                "processing_time": result.processing_time,
                "created_at":    datetime.now(timezone.utc).isoformat(),
            }
            if hasattr(db, "insert"):
                await asyncio.to_thread(db.insert, "pipeline_results", record)
        except Exception as exc:
            logger.warning("DB persistence failed (non-fatal): %s", exc)

    async def shutdown(self) -> None:
        """Graceful shutdown — waits for active processors to finish."""
        self.is_shutting_down = True
        await self.download_processor.cleanup()
        await self.transcription_processor.cleanup()
        await self.indexing_processor.cleanup()
        logger.info("PipelineManager shutdown complete")


# ──────────────────────────────────────────────────────────────────────────────
# Synchronous wrappers (safe for non-async callers)
# ──────────────────────────────────────────────────────────────────────────────

def _run_in_new_loop(coro, timeout: float = 3600.0) -> Any:
    """
    Execute *coro* in a brand-new event loop on a dedicated thread.

    This is the safest way to call async code from synchronous contexts,
    including when a loop is already running in the calling thread (e.g.
    Jupyter or threaded HTTP servers).

    Args:
        coro:    Awaitable to run.
        timeout: Maximum seconds to wait for completion.
                 Raises RuntimeError if the coroutine does not finish in time.
                 Default 3600 s (1 hour) — long enough for any pipeline stage
                 while still preventing infinite hangs on dead event loops.
    """
    result_box: List[Any] = [None]
    exc_box: List[Optional[Exception]] = [None]

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box[0] = loop.run_until_complete(coro)
        except Exception as exc:
            exc_box[0] = exc
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        # Thread did not finish — coroutine is stuck or timed out
        raise RuntimeError(
            f"_run_in_new_loop: coroutine did not complete within {timeout:.0f}s. "
            "The thread is still running as a daemon and will be abandoned."
        )

    if exc_box[0] is not None:
        raise exc_box[0]
    return result_box[0]


def transcribe_and_index(
    service: Any,
    path_or_url: str,
    source_type: str             = "url",
    model_name: str              = DEFAULT_MODEL_NAME,
    chunk_size: int              = DEFAULT_CHUNK_SIZE,
    overlap: int                 = DEFAULT_CHUNK_OVERLAP,
    persist_db: bool             = False,
    extra_metadata: Optional[Dict[str, Any]] = None,
    max_retries: int             = MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Synchronous entry point: transcribe/extract *path_or_url* and index into
    the vector store attached to *service*.

    Returns the pipeline result dict (same schema as ``ProcessingResult.to_dict()``).
    """
    config  = PipelineConfig(
        model_name=model_name, chunk_size=chunk_size,
        chunk_overlap=overlap, max_retries=max_retries, persist_db=persist_db,
    )
    manager = PipelineManager(service, config)

    async def _run():
        try:
            return await manager.process(
                path_or_url=path_or_url,
                source_type=source_type,
                extra_metadata=extra_metadata,
                persist_db=persist_db,
            )
        finally:
            await manager.shutdown()

    return _run_in_new_loop(_run())


def transcribe_url_to_vectorstore(
    service: Any,
    url: str,
    source_type: str  = "web",
    model_name: str   = DEFAULT_MODEL_NAME,
    persist_db: bool  = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience wrapper: download *url*, transcribe/extract, and index.

    Returns pipeline result dict.
    """
    return transcribe_and_index(
        service=service,
        path_or_url=url,
        source_type=source_type,
        model_name=model_name,
        persist_db=persist_db,
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tool registration
# ──────────────────────────────────────────────────────────────────────────────

def register_pipeline_tools(tool_registry: Any, service: Any) -> None:
    """
    Register pipeline tools into *tool_registry* with *service* bound.

    Raises:
        ValueError: if either argument is None.
        RuntimeError: if registration itself fails.
    """
    if tool_registry is None:
        raise ValueError("tool_registry must not be None")
    if service is None:
        raise ValueError("service must not be None")

    try:
        def transcribe_url_tool(
            url: str,
            source_type: str = "web",
            model_name: str  = DEFAULT_MODEL_NAME,
            persist_db: bool = False,
        ) -> Dict[str, Any]:
            """
            Transcribe and index content from a URL into the vector store.

            Args:
                url:         HTTP/HTTPS URL or YouTube link to process.
                source_type: Content category (default: "web").
                model_name:  Whisper model size for audio/video.
                persist_db:  Persist summary to database.

            Returns:
                dict: success, source, chunks_indexed, processing_time, errors
            """
            return transcribe_url_to_vectorstore(
                service=service, url=url, source_type=source_type,
                model_name=model_name, persist_db=persist_db,
            )

        def transcribe_local_tool(
            path: str,
            source_type: str = "local",
            model_name: str  = DEFAULT_MODEL_NAME,
            persist_db: bool = False,
        ) -> Dict[str, Any]:
            """
            Transcribe and index a local file into the vector store.

            Supports audio, video, PDF, DOCX, Excel, CSV, and plain text.

            Args:
                path:        Absolute or relative path to the file.
                source_type: Content category (default: "local").
                model_name:  Whisper model size for audio/video.
                persist_db:  Persist summary to database.

            Returns:
                dict: success, source, chunks_indexed, processing_time, errors
            """
            return transcribe_and_index(
                service=service, path_or_url=path, source_type=source_type,
                model_name=model_name, persist_db=persist_db,
            )

        # Register URL tool
        tool_registry.register_function(
            transcribe_url_tool,
            name="transcribe_url_to_vectorstore",
            description="Transcribe and index URL content into vector store",
            schema={
                "url":         {"type": "string", "description": "HTTP/HTTPS or YouTube URL"},
                "source_type": {"type": "string", "default": "web",
                                "enum": ["web", "url", "audio", "video"]},
                "model_name":  {"type": "string", "default": DEFAULT_MODEL_NAME,
                                "enum": ["tiny", "base", "small", "medium", "large"]},
                "persist_db":  {"type": "boolean", "default": False},
            },
            timeout=3600, retries=1,
        )

        # Register local tool
        tool_registry.register_function(
            transcribe_local_tool,
            name="transcribe_local_to_vectorstore",
            description="Transcribe and index a local file into vector store",
            schema={
                "path":        {"type": "string", "description": "Path to local file"},
                "source_type": {"type": "string", "default": "local",
                                "enum": ["local", "audio", "video", "document"]},
                "model_name":  {"type": "string", "default": DEFAULT_MODEL_NAME,
                                "enum": ["tiny", "base", "small", "medium", "large"]},
                "persist_db":  {"type": "boolean", "default": False},
            },
            timeout=3600, retries=1,
        )

        logger.info("Pipeline tools registered: transcribe_url_to_vectorstore, transcribe_local_to_vectorstore")

    except Exception as exc:
        msg = f"Failed to register pipeline tools: {exc}"
        logger.error("%s\n%s", msg, traceback.format_exc())
        raise RuntimeError(msg) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Memory helper
# ──────────────────────────────────────────────────────────────────────────────

def _get_memory_usage() -> float:
    """Current process RSS in MB (0.0 if psutil unavailable)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1_048_576
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION ADDITIONS v5.0 — Checkpoint/resume, progress bus, stage health
# ─────────────────────────────────────────────────────────────────────────────

import json as _json
import time as _time
import threading as _threading


class PipelineCheckpoint:
    """
    Persist and restore pipeline run state to disk for crash recovery.
    
    Usage::
    
        cp = PipelineCheckpoint("/tmp/pipeline_run_001.json")
        cp.save({"stage": "download", "done": ["url1", "url2"], "failed": []})
        state = cp.load()   # → dict on success, {} if no checkpoint
        cp.clear()          # remove after successful completion
    """

    def __init__(self, path: str):
        self._path = path
        self._lock = _threading.Lock()

    def save(self, state: dict) -> bool:
        """Atomically write state to disk."""
        import os
        import tempfile
        state_with_ts = {**state, "_checkpoint_ts": _time.time()}
        try:
            dirpath = os.path.dirname(self._path) or "."
            with tempfile.NamedTemporaryFile(
                mode='w', dir=dirpath, delete=False, suffix='.tmp'
            ) as tmp:
                _json.dump(state_with_ts, tmp, default=str)
                tmp_path = tmp.name
            os.replace(tmp_path, self._path)
            return True
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")
            return False

    def load(self) -> dict:
        """Return saved state dict, or {} if not found/corrupt.

        Returns an empty dict (never None) so callers can safely iterate:
            for key, val in checkpoint.load().items(): ...
        """
        try:
            with open(self._path) as f:
                return _json.load(f)
        except (FileNotFoundError, _json.JSONDecodeError):
            return {}

    def clear(self) -> bool:
        try:
            os.remove(self._path)
            return True
        except FileNotFoundError:
            return True
        except Exception:
            return False

    def exists(self) -> bool:

        return os.path.exists(self._path)


class PipelineProgressBus:
    """
    Thread-safe event bus for pipeline progress reporting.
    
    Multiple consumers can subscribe to events. Events are delivered
    synchronously in the caller's thread but subscription/unsubscription
    is thread-safe.
    
    Usage::
    
        bus = PipelineProgressBus()
        bus.subscribe("my_ui", lambda evt: print(evt))
        bus.emit("download_started", {"url": "..."})
        bus.unsubscribe("my_ui")
    """

    def __init__(self):
        self._subs: Dict[str, Callable] = {}
        self._lock = _threading.RLock()
        self._history: list = []
        self._max_history = 200

    def subscribe(self, name: str, callback: Callable) -> None:
        with self._lock:
            self._subs[name] = callback

    def unsubscribe(self, name: str) -> None:
        with self._lock:
            self._subs.pop(name, None)

    def emit(self, event: str, data: dict = None) -> None:
        payload = {"event": event, "ts": _time.time(), "data": data or {}}
        with self._lock:
            self._history.append(payload)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            callbacks = list(self._subs.values())
        for cb in callbacks:
            try:
                cb(payload)
            except Exception as e:
                logger.warning(f"ProgressBus callback error: {e}")

    def recent_events(self, n: int = 20) -> list:
        with self._lock:
            return list(self._history[-n:])


class StageHealthTracker:
    """
    Track success/failure rates per pipeline stage for circuit-breaker logic.
    
    If a stage's error rate exceeds threshold over the last N executions,
    is_healthy() returns False and the pipeline can skip/retry that stage.
    """

    def __init__(self, window: int = 20, threshold: float = 0.5):
        self._window = window
        self._threshold = threshold
        self._records: Dict[str, list] = {}
        self._lock = _threading.Lock()

    def record(self, stage: str, success: bool) -> None:
        with self._lock:
            if stage not in self._records:
                self._records[stage] = []
            self._records[stage].append(1 if success else 0)
            if len(self._records[stage]) > self._window:
                self._records[stage].pop(0)

    def error_rate(self, stage: str) -> float:
        with self._lock:
            recs = self._records.get(stage, [])
            if not recs:
                return 0.0
            return 1.0 - (sum(recs) / len(recs))

    def is_healthy(self, stage: str) -> bool:
        return self.error_rate(stage) < self._threshold

    def all_health(self) -> Dict[str, dict]:
        with self._lock:
            return {
                stage: {
                    "error_rate": round(1.0 - (sum(recs) / len(recs)), 4) if recs else 0.0,
                    "healthy": (1.0 - sum(recs) / len(recs)) < self._threshold if recs else True,
                    "samples": len(recs),
                }
                for stage, recs in self._records.items()
            }


# Module-level singletons
_progress_bus = PipelineProgressBus()
_stage_health = StageHealthTracker(window=20, threshold=0.5)


def get_pipeline_progress_bus() -> "PipelineProgressBus":
    """Return the global pipeline progress event bus."""
    return _progress_bus


def get_stage_health() -> dict:
    """Return per-stage health metrics for all pipeline stages."""
    return _stage_health.all_health()
