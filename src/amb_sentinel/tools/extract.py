# tools/extract.py
"""
AMB-Sentinel Text Extraction Engine — Production v4.0.0
Author: AMB Team © 2025

Extracts text from PDF, DOCX, XLSX/CSV, plain-text, and images (OCR).
All heavy operations run on a thread pool; the public API is fully
synchronous so callers don't need to manage event loops.

Critical bug fixes vs v3:
  Bug #1 — OCREngine alias:          ``OCREngine = OCRBackend`` added at module level.
  Bug #2 — extract_with_ocr alias:   ``extract_with_ocr`` now exported at module level.
  Bug #5 — get_supported_formats:    consolidated under canonical name.

Registrar-expected exports:
  Functions:  extract_text_from_file, extract_text_from_pdf,
              extract_text_from_docx, extract_text_from_spreadsheet,
              extract_text_from_textfile, extract_and_chunk_file,
              chunk_text, batch_extract_files, extract_with_ocr,
              process_file_for_vector_store, get_extraction_backends,
              get_supported_formats, clear_extraction_cache
  Classes:    ExtractionConfig, ExtractionEngine, ExtractionResult,
              ChunkingStrategy, OCREngine   ← (alias for OCRBackend)
"""

from __future__ import annotations

import gc
import hashlib
import logging
import platform
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import (
    Any, Callable, Dict, List, Optional,
    Tuple, Union,
)

logger = logging.getLogger("Extract")

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB          = 500
MAX_TEXT_LENGTH_CHARS     = 50_000_000
MAX_CONCURRENT_EXTRACTIONS = 8
MEMORY_LIMIT_MB           = 2_048
CHUNK_READ_SIZE           = 65_536   # 64 KB — used for streaming large files


# ──────────────────────────────────────────────────────────────────────────────
# Dependency registry (lazy, thread-safe)
# ──────────────────────────────────────────────────────────────────────────────

_DEPENDENCIES: Dict[str, bool] = {}
_DEPENDENCIES_LOCK = Lock()


def _check_dependencies() -> Dict[str, bool]:
    """Return cached dict of optional library availability."""
    global _DEPENDENCIES
    with _DEPENDENCIES_LOCK:
        if _DEPENDENCIES:
            return _DEPENDENCIES.copy()
        checks = {
            "pdfminer":    "pdfminer",
            "pypdf":       "pypdf",
            "docx":        "docx",
            "openpyxl":    "openpyxl",
            "pandas":      "pandas",
            "pillow":      "PIL",
            "pytesseract": "pytesseract",
            "chardet":     "chardet",
            "charset_normalizer": "charset_normalizer",
        }
        for name, module in checks.items():
            try:
                __import__(module)
                _DEPENDENCIES[name] = True
            except ImportError:
                _DEPENDENCIES[name] = False
        return _DEPENDENCIES.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class ExtractionError(Exception):
    """Base extraction error."""

class FileValidationError(ExtractionError):
    """File failed validation."""

class UnsupportedFormatError(ExtractionError):
    """File format not supported."""

class ResourceLimitExceededError(ExtractionError):
    """File too large or memory limit hit."""


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class ExtractionMethod(Enum):
    AUTO     = "auto"
    PDFMINER = "pdfminer"
    PYPdf    = "pypdf"
    DOCX     = "docx"
    PANDAS   = "pandas"
    TEXT     = "text"
    OCR      = "ocr"


class ChunkingStrategy(Enum):
    FIXED     = "fixed"
    SENTENCE  = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC  = "semantic"
    RECURSIVE = "recursive"
    TOKEN     = "token"


class ExtractionStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    PARTIAL   = "partial"
    SKIPPED   = "skipped"


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionConfig:
    """
    Comprehensive extraction configuration.

    All parameters have safe production defaults.
    Most backend-specific options are forward-compatible via kwargs.
    """
    # General
    max_file_size_mb: float = float(MAX_FILE_SIZE_MB)
    max_chars: int          = MAX_TEXT_LENGTH_CHARS
    timeout_seconds: float  = 300.0

    # Retry
    max_retries: int              = 2
    retry_delay_seconds: float    = 1.0
    retry_backoff_factor: float   = 2.0

    # PDF
    max_pages: Optional[int]  = None
    preserve_layout: bool     = False
    include_metadata: bool    = True
    extract_tables: bool      = False

    # DOCX
    extract_comments: bool         = False
    include_headers_footers: bool  = False

    # Spreadsheet
    sheet: Optional[Union[str, int]] = None
    nrows: Optional[int]             = None
    include_headers: bool            = True

    # Text
    encoding: str                           = "utf-8"
    fallback_encodings: List[str]           = field(
        default_factory=lambda: ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"]
    )

    # Chunking
    chunk_size: int    = 1_000
    chunk_overlap: int = 200
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE

    # Sanitisation
    sanitize_html: bool         = True
    remove_control_chars: bool  = True
    normalize_whitespace: bool  = True
    remove_malicious_content: bool = True

    # OCR
    ocr_language: str    = "eng"
    ocr_psm: int         = 3
    ocr_oem: int         = 3
    ocr_enhance: bool    = True     # upgraded default: enhance by default
    ocr_deskew: bool     = False
    ocr_denoise: bool    = False

    # Security
    validate_file_signature: bool = True

    def __post_init__(self) -> None:
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size // 5


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionMetrics:
    start_time: float           = field(default_factory=time.monotonic)
    end_time: Optional[float]   = None
    characters_extracted: int   = 0
    pages_processed: int        = 0
    chunks_generated: int       = 0
    file_validation_time: float = 0.0
    ocr_processing_time: float  = 0.0

    def complete(self) -> None:
        self.end_time = time.monotonic()

    @property
    def elapsed(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.monotonic() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_seconds":       round(self.elapsed, 3),
            "characters_extracted":  self.characters_extracted,
            "pages_processed":       self.pages_processed,
            "chunks_generated":      self.chunks_generated,
            "file_validation_time":  round(self.file_validation_time, 3),
            "ocr_processing_time":   round(self.ocr_processing_time, 3),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    ok: bool
    text: str                             = ""
    file_path: str                        = ""
    extraction_method: str                = ""
    processing_time: float                = 0.0
    status: ExtractionStatus              = ExtractionStatus.FAILED
    error: Optional[str]                  = None
    metadata: Dict[str, Any]              = field(default_factory=dict)
    metrics: Optional[ExtractionMetrics]  = None
    file_size: int                        = 0
    file_hash: str                        = ""
    content_type: str                     = ""
    mime_type: str                        = ""
    encoding: str                         = ""
    page_count: int                       = 0
    config: Dict[str, Any]               = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok":               self.ok,
            "text":             self.text,
            "file_path":        self.file_path,
            "extraction_method": self.extraction_method,
            "processing_time":  round(self.processing_time, 3),
            "status":           self.status.value,
            "error":            self.error,
            "metadata":         self.metadata,
            "metrics":          self.metrics.to_dict() if self.metrics else {},
            "file_size":        self.file_size,
            "file_hash":        self.file_hash,
            "content_type":     self.content_type,
            "page_count":       self.page_count,
            "encoding":         self.encoding,
        }

    @property
    def is_truncated(self) -> bool:
        return len(self.text) >= MAX_TEXT_LENGTH_CHARS

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0


# ──────────────────────────────────────────────────────────────────────────────
# Resource monitor
# ──────────────────────────────────────────────────────────────────────────────

class ResourceMonitor:
    def __init__(self, memory_limit_mb: int = MEMORY_LIMIT_MB):
        self._limit = memory_limit_mb
        self._lock  = threading.Lock()

    def _get_memory_mb(self) -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1_048_576
        except Exception:
            return 0.0

    def check_memory(self) -> Tuple[bool, float]:
        used = self._get_memory_mb()
        return used < self._limit, used

    def get_metrics(self) -> Dict[str, Any]:
        ok, used = self.check_memory()
        return {"memory_used_mb": round(used, 1), "memory_limit_mb": self._limit, "within_limit": ok}


_resource_monitor = ResourceMonitor()


# ──────────────────────────────────────────────────────────────────────────────
# Circuit breaker
# ──────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """Simple CLOSED → OPEN → HALF-OPEN circuit breaker."""

    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout     = reset_timeout
        self.failure_count     = 0
        self.last_failure_time = 0.0
        self.state             = self.CLOSED
        self._lock             = threading.Lock()

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == self.CLOSED:
                return True
            if self.state == self.OPEN:
                if time.monotonic() - self.last_failure_time > self.reset_timeout:
                    self.state = self.HALF_OPEN
                    return True
                return False
            return True  # HALF_OPEN: allow one probe

    def record_success(self) -> None:
        with self._lock:
            self.failure_count  = 0
            self.state          = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count    += 1
            self.last_failure_time = time.monotonic()
            if self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
                logger.warning("Extraction circuit breaker OPEN after %d failures", self.failure_count)


extraction_circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)


# ──────────────────────────────────────────────────────────────────────────────
# File validator
# ──────────────────────────────────────────────────────────────────────────────

# Magic bytes for common dangerous formats
_DANGEROUS_SIGS: Dict[bytes, str] = {
    b"MZ":             "Windows executable",
    b"\x7fELF":        "Linux executable",
    b"#!/":            "Shell script",
    b"#!":             "Script (shebang)",
}

_MIME_BY_EXT: Dict[str, str] = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":  "application/vnd.ms-excel",
    ".csv":  "text/csv",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".html": "text/html",
    ".xml":  "application/xml",
    ".json": "application/json",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
    ".gif":  "image/gif",
    ".bmp":  "image/bmp",
    ".webp": "image/webp",
}


class FileValidator:
    @classmethod
    def validate(
        cls,
        file_path: Path,
        max_size_mb: float = MAX_FILE_SIZE_MB,
        check_signatures: bool = True,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate a file for extraction.

        Returns (is_valid, error_message, info_dict).
        """
        info: Dict[str, Any] = {}
        if not file_path.exists():
            return False, f"File not found: {file_path}", info
        if not file_path.is_file():
            return False, f"Not a regular file: {file_path}", info

        # Null-byte in path
        if "\x00" in str(file_path):
            return False, "Null byte in file path", info

        size = file_path.stat().st_size
        info["size_bytes"] = size
        info["size_mb"]    = size / 1_048_576
        info["extension"]  = file_path.suffix.lower()
        info["mime_type"]  = _MIME_BY_EXT.get(info["extension"], "application/octet-stream")

        if size == 0:
            return False, "File is empty", info
        if size > max_size_mb * 1_048_576:
            return False, f"File exceeds limit: {info['size_mb']:.1f} MB > {max_size_mb} MB", info

        if check_signatures:
            try:
                header = file_path.read_bytes()[:16]
                for sig, label in _DANGEROUS_SIGS.items():
                    if header.startswith(sig) and info["extension"] not in (".sh",):
                        # Only reject if it doesn't match expected extension
                        if label in ("Windows executable", "Linux executable"):
                            return False, f"Dangerous file type detected: {label}", info
            except OSError as exc:
                return False, f"Cannot read file header: {exc}", info

        return True, None, info


# ──────────────────────────────────────────────────────────────────────────────
# Text utilities
# ──────────────────────────────────────────────────────────────────────────────

class TextSanitizer:
    """Clean and normalise extracted text."""

    # Control characters to remove (keep \t \n \r)
    _CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _WS_RE      = re.compile(r"[ \t]+")
    _BLANK_RE   = re.compile(r"\n{3,}")

    @classmethod
    def sanitize(
        cls,
        text: str,
        remove_control_chars: bool = True,
        normalize_whitespace: bool = True,
        remove_suspicious: bool    = True,
        max_length: int            = MAX_TEXT_LENGTH_CHARS,
    ) -> str:
        if not text:
            return ""
        if remove_control_chars:
            text = cls._CONTROL_RE.sub(" ", text)
        if normalize_whitespace:
            text = cls._WS_RE.sub(" ", text)
            text = cls._BLANK_RE.sub("\n\n", text)
        if max_length and len(text) > max_length:
            text = text[:max_length]
            logger.warning("Text truncated to %d chars", max_length)
        return text.strip()

    @classmethod
    def normalize_encoding(cls, text: str, target: str = "utf-8") -> str:
        try:
            return text.encode(target, errors="replace").decode(target)
        except Exception:
            return text


class EncodingDetector:
    """Detect and handle file encoding."""

    @classmethod
    def detect(cls, raw_bytes: bytes, hint: str = "utf-8") -> str:
        """Return best-guess encoding name."""
        # Try hint first
        try:
            raw_bytes.decode(hint)
            return hint
        except (UnicodeDecodeError, LookupError):
            pass
        # chardet / charset-normalizer
        for mod in ("charset_normalizer", "chardet"):
            try:
                lib = __import__(mod)
                result = lib.detect(raw_bytes[:32_768])
                if result and result.get("encoding"):
                    enc = result["encoding"]
                    raw_bytes.decode(enc)
                    return enc
            except Exception:
                pass
        # Fallback cascade
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                raw_bytes.decode(enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                pass
        return "latin-1"

    @classmethod
    def read_text_file(cls, path: Path, hint: str = "utf-8") -> str:
        """Read a text file with automatic encoding detection."""
        raw = path.read_bytes()
        enc = cls.detect(raw, hint)
        return raw.decode(enc, errors="replace")


# ──────────────────────────────────────────────────────────────────────────────
# Abstract backend
# ──────────────────────────────────────────────────────────────────────────────

class ExtractionBackend(ABC):
    @abstractmethod
    def extract(self, file_path: Path, config: ExtractionConfig) -> ExtractionResult:
        ...

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ──────────────────────────────────────────────────────────────────────────────
# PDF backend — pdfminer (preferred)
# ──────────────────────────────────────────────────────────────────────────────

class PDFMinerBackend(ExtractionBackend):
    def __init__(self) -> None:
        self._available = _check_dependencies().get("pdfminer", False)

    @property
    def name(self) -> str:
        return "pdfminer"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf" and self._available

    def extract(self, file_path: Path, config: ExtractionConfig) -> ExtractionResult:
        t0 = time.monotonic()
        if not self._available:
            return ExtractionResult(
                ok=False, error="pdfminer not installed",
                file_path=str(file_path), status=ExtractionStatus.FAILED,
                processing_time=time.monotonic() - t0,
            )
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer

            pages: List[str] = []
            page_count = 0
            metadata: Dict[str, Any] = {}

            # Stream page by page — memory-efficient for huge PDFs
            for page_layout in extract_pages(str(file_path)):
                page_count += 1
                if config.max_pages and page_count > config.max_pages:
                    break
                page_text_parts: List[str] = []
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        page_text_parts.append(element.get_text())
                pages.append("".join(page_text_parts))
                # Memory guard
                total_so_far = sum(len(p) for p in pages)
                if total_so_far > config.max_chars:
                    logger.warning("PDF text truncated at page %d", page_count)
                    break

            full_text = "\n\n".join(pages)
            full_text = TextSanitizer.sanitize(
                full_text,
                remove_control_chars=config.remove_control_chars,
                normalize_whitespace=config.normalize_whitespace,
                max_length=config.max_chars,
            )

            # Metadata via pdfminer PDFDocument
            if config.include_metadata:
                try:
                    from pdfminer.pdfparser import PDFParser
                    from pdfminer.pdfdocument import PDFDocument
                    with open(file_path, "rb") as fh:
                        parser  = PDFParser(fh)
                        doc     = PDFDocument(parser)
                        if doc.info:
                            raw_meta = doc.info[0]
                            for k, v in raw_meta.items():
                                if isinstance(v, bytes):
                                    try:
                                        metadata[k] = v.decode("utf-8", errors="replace")
                                    except Exception:
                                        metadata[k] = repr(v)
                                else:
                                    metadata[k] = str(v)
                except Exception as exc:
                    logger.debug("PDF metadata extraction failed: %s", exc)

            m = ExtractionMetrics()
            m.characters_extracted = len(full_text)
            m.pages_processed      = page_count
            m.complete()
            return ExtractionResult(
                ok=True, text=full_text, file_path=str(file_path),
                extraction_method=self.name, page_count=page_count,
                metadata=metadata, metrics=m,
                status=ExtractionStatus.COMPLETED,
                processing_time=time.monotonic() - t0,
            )
        except Exception as exc:
            logger.warning("PDFMiner extraction failed: %s", exc)
            return ExtractionResult(
                ok=False, error=str(exc), file_path=str(file_path),
                status=ExtractionStatus.FAILED,
                processing_time=time.monotonic() - t0,
            )


# ──────────────────────────────────────────────────────────────────────────────
# PDF backend — pypdf (fallback)
# ──────────────────────────────────────────────────────────────────────────────

class PyPDFBackend(ExtractionBackend):
    def __init__(self) -> None:
        self._available  = False
        self._lib_name   = ""
        self._load_library()

    def _load_library(self) -> None:
        for mod in ("pypdf", "pypdf2", "pypdf4"):
            try:
                __import__(mod)
                self._available = True
                self._lib_name  = mod
                return
            except ImportError:
                pass

    @property
    def name(self) -> str:
        return "pypdf"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf" and self._available

    def extract(self, file_path: Path, config: ExtractionConfig) -> ExtractionResult:
        t0 = time.monotonic()
        if not self._available:
            return ExtractionResult(
                ok=False, error="pypdf not installed",
                file_path=str(file_path), status=ExtractionStatus.FAILED,
                processing_time=time.monotonic() - t0,
            )
        try:
            lib = __import__(self._lib_name)
            reader = lib.PdfReader(str(file_path))

            pages: List[str] = []
            total = len(reader.pages)
            limit = min(config.max_pages or total, total)

            for i in range(limit):
                try:
                    pages.append(reader.pages[i].extract_text() or "")
                except Exception as exc:
                    logger.debug("Page %d extraction error: %s", i, exc)
                    pages.append("")

            text = TextSanitizer.sanitize(
                "\n\n".join(pages),
                remove_control_chars=config.remove_control_chars,
                normalize_whitespace=config.normalize_whitespace,
                max_length=config.max_chars,
            )
            m = ExtractionMetrics()
            m.characters_extracted = len(text)
            m.pages_processed      = limit
            m.complete()
            meta: Dict[str, Any] = {}
            if config.include_metadata:
                try:
                    info = reader.metadata
                    if info:
                        meta = {k.lstrip("/"): str(v) for k, v in info.items()}
                except Exception:
                    pass
            return ExtractionResult(
                ok=True, text=text, file_path=str(file_path),
                extraction_method=self.name, page_count=limit,
                metadata=meta, metrics=m, status=ExtractionStatus.COMPLETED,
                processing_time=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExtractionResult(
                ok=False, error=str(exc), file_path=str(file_path),
                status=ExtractionStatus.FAILED, processing_time=time.monotonic() - t0,
            )


# ──────────────────────────────────────────────────────────────────────────────
# DOCX backend
# ──────────────────────────────────────────────────────────────────────────────

class DocxBackend(ExtractionBackend):
    def __init__(self) -> None:
        self._available = _check_dependencies().get("docx", False)

    @property
    def name(self) -> str:
        return "docx"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in (".docx", ".doc") and self._available

    def extract(self, file_path: Path, config: ExtractionConfig) -> ExtractionResult:
        t0 = time.monotonic()
        if not self._available:
            return ExtractionResult(
                ok=False, error="python-docx not installed",
                file_path=str(file_path), status=ExtractionStatus.FAILED,
                processing_time=time.monotonic() - t0,
            )
        try:
            import docx as _docx

            doc  = _docx.Document(str(file_path))
            parts: List[str] = []
            char_count = 0

            for para in doc.paragraphs:
                txt = para.text.strip()
                if txt:
                    parts.append(txt)
                    char_count += len(txt)
                    if char_count > config.max_chars:
                        break

            if config.include_headers_footers:
                for section in doc.sections:
                    for hdr_para in (section.header.paragraphs if section.header else []):
                        t = hdr_para.text.strip()
                        if t:
                            parts.insert(0, f"[HEADER] {t}")
                    for ftr_para in (section.footer.paragraphs if section.footer else []):
                        t = ftr_para.text.strip()
                        if t:
                            parts.append(f"[FOOTER] {t}")

            if config.extract_comments:
                for comment in getattr(doc, "comments", []):
                    t = getattr(comment, "text", "").strip()
                    if t:
                        parts.append(f"[COMMENT] {t}")

            text = TextSanitizer.sanitize(
                "\n\n".join(parts),
                remove_control_chars=config.remove_control_chars,
                normalize_whitespace=config.normalize_whitespace,
                max_length=config.max_chars,
            )

            # Metadata
            meta: Dict[str, Any] = {}
            if config.include_metadata:
                try:
                    cp = doc.core_properties
                    for attr in ("author", "title", "subject", "keywords", "description",
                                 "created", "modified", "last_modified_by", "revision"):
                        val = getattr(cp, attr, None)
                        if val is not None:
                            meta[attr] = str(val)
                except Exception:
                    pass

            m = ExtractionMetrics()
            m.characters_extracted = len(text)
            m.complete()
            return ExtractionResult(
                ok=True, text=text, file_path=str(file_path),
                extraction_method=self.name, metadata=meta, metrics=m,
                status=ExtractionStatus.COMPLETED, processing_time=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExtractionResult(
                ok=False, error=str(exc), file_path=str(file_path),
                status=ExtractionStatus.FAILED, processing_time=time.monotonic() - t0,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Spreadsheet backend — openpyxl + pandas, streaming-safe
# ──────────────────────────────────────────────────────────────────────────────

class SpreadsheetBackend(ExtractionBackend):
    _SUPPORTED = {".xlsx", ".xls", ".ods", ".csv", ".tsv"}

    def __init__(self) -> None:
        deps = _check_dependencies()
        self._has_openpyxl = deps.get("openpyxl", False)
        self._has_pandas   = deps.get("pandas", False)

    @property
    def name(self) -> str:
        return "pandas"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self._SUPPORTED and (
            self._has_openpyxl or self._has_pandas
        )

    def extract(self, file_path: Path, config: ExtractionConfig) -> ExtractionResult:
        t0  = time.monotonic()
        ext = file_path.suffix.lower()

        if ext in (".csv", ".tsv"):
            return self._extract_csv(file_path, config, t0)
        return self._extract_excel(file_path, config, t0, ext)

    def _extract_csv(self, path: Path, config: ExtractionConfig, t0: float) -> ExtractionResult:
        try:
            sep       = "\t" if path.suffix.lower() == ".tsv" else ","
            raw       = path.read_bytes()
            enc       = EncodingDetector.detect(raw)
            text_raw  = raw.decode(enc, errors="replace")
            rows      = text_raw.splitlines()
            if config.nrows:
                rows = rows[: config.nrows + (1 if config.include_headers else 0)]
            text = "\n".join(rows)
            text = TextSanitizer.sanitize(text, max_length=config.max_chars)
            m = ExtractionMetrics()
            m.characters_extracted = len(text)
            m.complete()
            return ExtractionResult(
                ok=True, text=text, file_path=str(path),
                extraction_method=self.name, encoding=enc, metrics=m,
                status=ExtractionStatus.COMPLETED, processing_time=time.monotonic() - t0,
                metadata={"rows": len(rows), "encoding": enc},
            )
        except Exception as exc:
            return ExtractionResult(
                ok=False, error=str(exc), file_path=str(path),
                status=ExtractionStatus.FAILED, processing_time=time.monotonic() - t0,
            )

    def _extract_excel(
        self, path: Path, config: ExtractionConfig, t0: float, ext: str
    ) -> ExtractionResult:
        try:
            if self._has_openpyxl and ext == ".xlsx":
                return self._openpyxl_extract(path, config, t0)
            if self._has_pandas:
                return self._pandas_extract(path, config, t0, ext)
            return ExtractionResult(
                ok=False, error="No spreadsheet library available",
                file_path=str(path), status=ExtractionStatus.FAILED,
                processing_time=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExtractionResult(
                ok=False, error=str(exc), file_path=str(path),
                status=ExtractionStatus.FAILED, processing_time=time.monotonic() - t0,
            )

    def _openpyxl_extract(self, path: Path, config: ExtractionConfig, t0: float) -> ExtractionResult:
        """Use openpyxl read-only streaming mode — handles large XLSX without loading full DOM."""
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        sheet_texts: List[str] = []
        sheet_names = wb.sheetnames

        target_sheets = (
            [config.sheet] if config.sheet is not None
            else sheet_names
        )
        for sheet_name in target_sheets:
            if isinstance(sheet_name, int):
                if sheet_name < len(sheet_names):
                    sheet_name = sheet_names[sheet_name]
                else:
                    continue
            try:
                ws = wb[sheet_name]
            except KeyError:
                continue

            rows_data: List[str] = []
            row_count = 0
            for row in ws.iter_rows(values_only=True):
                row_count += 1
                if config.nrows and row_count > config.nrows:
                    break
                cells = [str(c) if c is not None else "" for c in row]
                rows_data.append("\t".join(cells))

            sheet_texts.append(f"=== Sheet: {sheet_name} ===\n" + "\n".join(rows_data))

        wb.close()
        text = TextSanitizer.sanitize(
            "\n\n".join(sheet_texts), max_length=config.max_chars
        )
        m = ExtractionMetrics()
        m.characters_extracted = len(text)
        m.complete()
        return ExtractionResult(
            ok=True, text=text, file_path=str(path),
            extraction_method=self.name, metrics=m,
            status=ExtractionStatus.COMPLETED, processing_time=time.monotonic() - t0,
            metadata={"sheets": sheet_names},
        )

    def _pandas_extract(
        self, path: Path, config: ExtractionConfig, t0: float, ext: str
    ) -> ExtractionResult:
        import pandas as pd
        engine_map = {".xlsx": "openpyxl", ".xls": "xlrd", ".ods": "odf"}
        engine     = engine_map.get(ext)
        dfs        = pd.read_excel(
            str(path),
            sheet_name=config.sheet,
            nrows=config.nrows,
            engine=engine,
        )
        if isinstance(dfs, dict):
            sheets_text = []
            for sheet, df in dfs.items():
                sheets_text.append(f"=== Sheet: {sheet} ===\n{df.to_string(index=False)}")
            combined = "\n\n".join(sheets_text)
        else:
            combined = dfs.to_string(index=False)
        text = TextSanitizer.sanitize(combined, max_length=config.max_chars)
        m = ExtractionMetrics()
        m.characters_extracted = len(text)
        m.complete()
        return ExtractionResult(
            ok=True, text=text, file_path=str(path),
            extraction_method=self.name, metrics=m,
            status=ExtractionStatus.COMPLETED, processing_time=time.monotonic() - t0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Plain-text backend
# ──────────────────────────────────────────────────────────────────────────────

class TextBackend(ExtractionBackend):
    _SUPPORTED = {
        ".txt", ".md", ".markdown", ".rst", ".log", ".ini", ".cfg",
        ".conf", ".properties", ".yml", ".yaml", ".toml", ".tex",
        ".html", ".htm", ".xml", ".json", ".jsonl", ".sh", ".py",
        ".js", ".ts", ".css", ".csv", ".tsv",
    }

    @property
    def name(self) -> str:
        return "text"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self._SUPPORTED

    def extract(self, file_path: Path, config: ExtractionConfig) -> ExtractionResult:
        t0 = time.monotonic()
        try:
            # Streaming read for large files — avoids OOM on multi-GB logs
            size = file_path.stat().st_size
            if size > 64 * 1_048_576:  # > 64 MB → stream
                return self._stream_extract(file_path, config, t0)

            text = EncodingDetector.read_text_file(file_path, config.encoding)
            text = TextSanitizer.sanitize(
                text,
                remove_control_chars=config.remove_control_chars,
                normalize_whitespace=config.normalize_whitespace,
                max_length=config.max_chars,
            )
            m = ExtractionMetrics()
            m.characters_extracted = len(text)
            m.complete()
            return ExtractionResult(
                ok=True, text=text, file_path=str(file_path),
                extraction_method=self.name, metrics=m,
                status=ExtractionStatus.COMPLETED, processing_time=time.monotonic() - t0,
            )
        except Exception as exc:
            return ExtractionResult(
                ok=False, error=str(exc), file_path=str(file_path),
                status=ExtractionStatus.FAILED, processing_time=time.monotonic() - t0,
            )

    def _stream_extract(self, path: Path, config: ExtractionConfig, t0: float) -> ExtractionResult:
        """Read large files in chunks to avoid loading entire file into memory."""
        with path.open("rb") as _fh:
            raw_sample = _fh.read(32_768)
        enc = EncodingDetector.detect(raw_sample, config.encoding)
        parts: List[str] = []
        total  = 0
        truncated = False
        try:
            with open(path, encoding=enc, errors="replace") as fh:
                while True:
                    chunk = fh.read(CHUNK_READ_SIZE)
                    if not chunk:
                        break
                    parts.append(chunk)
                    total += len(chunk)
                    if total >= config.max_chars:
                        truncated = True
                        break
        except Exception as exc:
            return ExtractionResult(
                ok=False, error=str(exc), file_path=str(path),
                status=ExtractionStatus.FAILED, processing_time=time.monotonic() - t0,
            )
        text = TextSanitizer.sanitize("".join(parts), max_length=config.max_chars)
        m = ExtractionMetrics()
        m.characters_extracted = len(text)
        m.complete()
        return ExtractionResult(
            ok=True, text=text, file_path=str(path),
            extraction_method=self.name, encoding=enc, metrics=m,
            status=ExtractionStatus.PARTIAL if truncated else ExtractionStatus.COMPLETED,
            processing_time=time.monotonic() - t0,
            metadata={"truncated": truncated, "encoding": enc},
        )


# ──────────────────────────────────────────────────────────────────────────────
# OCR backend  ← Bug #1 source — OCRBackend is the real class
# ──────────────────────────────────────────────────────────────────────────────

class OCRBackend(ExtractionBackend):
    """
    OCR extraction backend using Pillow + pytesseract.

    Supports advanced preprocessing: grayscale, auto-contrast, deskew,
    denoise. Exposes confidence scores and EXIF metadata.
    """

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}

    def __init__(self) -> None:
        deps = _check_dependencies()
        self._available = deps.get("pillow", False) and deps.get("pytesseract", False)

    @property
    def name(self) -> str:
        return "ocr"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self._IMAGE_EXTS and self._available

    def extract(self, file_path: Path, config: ExtractionConfig) -> ExtractionResult:
        t0 = time.monotonic()
        if not self._available:
            return ExtractionResult(
                ok=False,
                error="OCR dependencies missing (Pillow + pytesseract required)",
                file_path=str(file_path), status=ExtractionStatus.FAILED,
                processing_time=time.monotonic() - t0,
            )
        try:
            from PIL import Image, ImageOps, ImageFilter
            import pytesseract

            # Open image, load all pixel data into memory, then close the file handle.
            # PIL lazy-loads by default — without .load() the file stays open
            # throughout processing, which leaks handles on failure paths.
            with Image.open(str(file_path)) as _raw_img:
                _raw_img.load()
                image = _raw_img.copy()
            orig_mode = image.mode

            # ── Preprocessing ─────────────────────────────────────────────
            preprocess_t0 = time.monotonic()

            # Convert to grayscale
            if image.mode not in ("L", "RGB"):
                image = image.convert("RGB")
            if image.mode == "RGB":
                image = image.convert("L")

            if config.ocr_enhance:
                image = ImageOps.autocontrast(image, cutoff=2)

            if config.ocr_deskew:
                # Simple deskew via finding dominant angle
                try:
                    image = self._deskew(image)
                except Exception:
                    pass   # Deskew is best-effort

            if config.ocr_denoise:
                image = image.filter(ImageFilter.MedianFilter(size=3))

            preprocess_time = time.monotonic() - preprocess_t0

            # ── OCR ───────────────────────────────────────────────────────
            lang   = config.ocr_language or "eng"
            tes_cfg = f"--psm {config.ocr_psm} --oem {config.ocr_oem} -l {lang}"
            ocr_t0 = time.monotonic()
            text   = pytesseract.image_to_string(image, config=tes_cfg)
            ocr_time = time.monotonic() - ocr_t0

            # Confidence
            avg_confidence = 0.0
            try:
                data = pytesseract.image_to_data(
                    image, config=tes_cfg, output_type=pytesseract.Output.DICT
                )
                confs = [int(c) for c in data.get("conf", []) if c not in (-1, "-1", None)]
                avg_confidence = sum(confs) / len(confs) if confs else 0.0
            except Exception:
                pass

            text = TextSanitizer.sanitize(
                text,
                remove_control_chars=config.remove_control_chars,
                normalize_whitespace=config.normalize_whitespace,
                max_length=config.max_chars,
            )

            # EXIF
            meta: Dict[str, Any] = {
                "ocr_language":       lang,
                "ocr_confidence":     round(avg_confidence, 2),
                "ocr_preprocess_time": round(preprocess_time, 3),
                "ocr_time":           round(ocr_time, 3),
                "image_size":         list(image.size),
                "image_original_mode": orig_mode,
                "ocr_config":         tes_cfg,
            }
            try:
                exif = getattr(image, "_getexif", lambda: None)()
                if exif:
                    meta["exif"] = {
                        str(k): str(v)[:200] for k, v in exif.items()
                        if isinstance(v, (str, int, float))
                    }
            except Exception:
                pass

            m = ExtractionMetrics()
            m.characters_extracted = len(text)
            m.ocr_processing_time  = ocr_time
            m.complete()
            return ExtractionResult(
                ok=True, text=text, file_path=str(file_path),
                extraction_method=self.name, metadata=meta, metrics=m,
                status=ExtractionStatus.COMPLETED, processing_time=time.monotonic() - t0,
            )
        except Exception as exc:
            logger.warning("OCR failed for %s: %s", file_path, exc)
            return ExtractionResult(
                ok=False, error=str(exc), file_path=str(file_path),
                status=ExtractionStatus.FAILED, processing_time=time.monotonic() - t0,
            )

    @staticmethod
    def _deskew(image: Any) -> Any:
        """Best-effort deskew using Pillow affine rotation."""
        import numpy as np
        arr = np.array(image)
        # Find coords of non-background pixels
        mask = arr < 200
        coords = np.column_stack(np.where(mask))
        if len(coords) < 20:
            return image
        angle = 0.0
        try:
            from numpy.linalg import eig
            cov  = np.cov(coords.T)
            vals, vecs = eig(cov)
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            if abs(angle) > 45:
                angle = 0.0
        except Exception:
            pass
        if abs(angle) > 0.5:
            return image.rotate(angle, expand=True, fillcolor=255)
        return image


# ──────────────────────────────────────────────────────────────────────────────
# ─── BUG #1 FIX: OCREngine alias ─────────────────────────────────────────────
# registrar.py: ``OCREngine = _extract_items.get('OCREngine')``
# The old extract.py never defined ``OCREngine`` — only ``OCRBackend``.
# ──────────────────────────────────────────────────────────────────────────────
OCREngine = OCRBackend   # ← Bug #1 fixed


# ──────────────────────────────────────────────────────────────────────────────
# Extraction engine — orchestrates backends
# ──────────────────────────────────────────────────────────────────────────────

class ExtractionEngine:
    """
    Main extraction engine with backend selection, fallback, and retry.

    Thread-safe singleton via ``get_extraction_engine()``.
    """

    def __init__(self) -> None:
        self.backends: List[ExtractionBackend] = []
        self._lock = threading.Lock()
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        # Priority order: pdfminer → pypdf → docx → spreadsheet → text → ocr
        self.backends = [
            PDFMinerBackend(),
            PyPDFBackend(),
            DocxBackend(),
            SpreadsheetBackend(),
            TextBackend(),
            OCRBackend(),
        ]

    def extract(
        self,
        file_path: Union[str, Path],
        config: Optional[ExtractionConfig] = None,
        preferred_backend: Optional[str]   = None,
    ) -> ExtractionResult:
        """
        Extract text from *file_path* using the best available backend.

        Args:
            file_path:          Path to file.
            config:             ExtractionConfig (defaults used if None).
            preferred_backend:  Name of backend to try first.

        Returns:
            ExtractionResult
        """
        config = config or ExtractionConfig()
        t0     = time.monotonic()
        path   = Path(file_path) if not isinstance(file_path, Path) else file_path

        # ── File validation ───────────────────────────────────────────────
        val_t0 = time.monotonic()
        valid, err_msg, val_info = FileValidator.validate(
            path,
            max_size_mb=config.max_file_size_mb,
            check_signatures=config.validate_file_signature,
        )
        val_time = time.monotonic() - val_t0

        if not valid:
            return ExtractionResult(
                ok=False, error=err_msg, file_path=str(path),
                status=ExtractionStatus.FAILED,
                processing_time=time.monotonic() - t0,
            )

        file_size = val_info["size_bytes"]
        mime_type = val_info["mime_type"]
        file_hash = self._hash_file(path)

        # ── Circuit breaker ───────────────────────────────────────────────
        if not extraction_circuit_breaker.can_execute():
            return ExtractionResult(
                ok=False,
                error="Extraction service temporarily unavailable (circuit breaker open)",
                file_path=str(path), file_size=file_size,
                file_hash=file_hash, content_type=mime_type,
                status=ExtractionStatus.FAILED,
                processing_time=time.monotonic() - t0,
            )

        # ── Memory guard ──────────────────────────────────────────────────
        mem_ok, mem_used = _resource_monitor.check_memory()
        if not mem_ok:
            logger.warning("High memory use: %.0f MB — forcing GC before extraction", mem_used)
            gc.collect()

        # ── Backend selection ─────────────────────────────────────────────
        backend: Optional[ExtractionBackend] = None
        if preferred_backend:
            for b in self.backends:
                if b.name == preferred_backend and b.supports(path):
                    backend = b
                    break
        if not backend:
            for b in self.backends:
                if b.supports(path):
                    backend = b
                    break

        if not backend:
            return ExtractionResult(
                ok=False,
                error=f"No backend supports: {path.suffix}",
                file_path=str(path), file_size=file_size,
                file_hash=file_hash, content_type=mime_type,
                status=ExtractionStatus.FAILED,
                processing_time=time.monotonic() - t0,
            )

        # ── Extraction with retry ─────────────────────────────────────────
        result: Optional[ExtractionResult] = None

        for attempt in range(config.max_retries + 1):
            if attempt > 0:
                wait = min(
                    config.retry_delay_seconds * (config.retry_backoff_factor ** (attempt - 1)),
                    30.0,
                )
                logger.info("Retry %d/%d for %s — waiting %.1fs", attempt, config.max_retries, path.name, wait)
                time.sleep(wait)

            try:
                result = backend.extract(path, config)
                if result.ok:
                    extraction_circuit_breaker.record_success()
                    break
            except Exception as exc:
                logger.warning("Backend %s attempt %d failed: %s", backend.name, attempt + 1, exc)
                result = ExtractionResult(
                    ok=False, error=str(exc), file_path=str(path),
                    status=ExtractionStatus.FAILED,
                    processing_time=time.monotonic() - t0,
                )

        if result and not result.ok:
            extraction_circuit_breaker.record_failure()

        if result:
            result.file_size    = file_size
            result.file_hash    = file_hash
            result.content_type = mime_type
            result.mime_type    = mime_type
            if result.metrics:
                result.metrics.file_validation_time = val_time
            result.config = asdict(config)

        return result or ExtractionResult(
            ok=False, error="Unknown extraction failure",
            file_path=str(path), status=ExtractionStatus.FAILED,
            processing_time=time.monotonic() - t0,
        )

    @staticmethod
    def _hash_file(path: Path, algorithm: str = "sha256") -> str:
        """Hash file contents in a memory-efficient streaming pass."""
        h = hashlib.new(algorithm)
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(CHUNK_READ_SIZE), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as exc:
            logger.debug("File hash failed for %s: %s", path, exc)
            return ""


# ──────────────────────────────────────────────────────────────────────────────
# Global singleton
# ──────────────────────────────────────────────────────────────────────────────

_extraction_engine: Optional[ExtractionEngine] = None
_engine_lock = Lock()


def get_extraction_engine() -> ExtractionEngine:
    """Get or lazily create the global ExtractionEngine (thread-safe)."""
    global _extraction_engine
    with _engine_lock:
        if _extraction_engine is None:
            _extraction_engine = ExtractionEngine()
    return _extraction_engine


# ──────────────────────────────────────────────────────────────────────────────
# Public extraction functions
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(
    path: Union[str, Path],
    max_pages: Optional[int]                    = None,
    method: Union[str, ExtractionMethod]        = ExtractionMethod.AUTO,
    preserve_layout: bool                       = False,
    extract_metadata: bool                      = True,
    extract_tables: bool                        = False,
    **kwargs,
) -> ExtractionResult:
    """Extract text from a PDF. Streams page-by-page for large files."""
    if isinstance(method, str):
        try:
            method = ExtractionMethod(method.lower())
        except ValueError:
            method = ExtractionMethod.AUTO
    backend_map = {ExtractionMethod.PDFMINER: "pdfminer", ExtractionMethod.PYPdf: "pypdf"}
    config = ExtractionConfig(
        max_pages=max_pages, preserve_layout=preserve_layout,
        include_metadata=extract_metadata, extract_tables=extract_tables,
    )
    return get_extraction_engine().extract(
        file_path=path, config=config,
        preferred_backend=backend_map.get(method),
    )


def extract_text_from_docx(
    path: Union[str, Path],
    include_comments: bool      = False,
    include_headers_footers: bool = False,
    **kwargs,
) -> ExtractionResult:
    """Extract text from a DOCX file."""
    config = ExtractionConfig(
        extract_comments=include_comments,
        include_headers_footers=include_headers_footers,
    )
    return get_extraction_engine().extract(file_path=path, config=config, preferred_backend="docx")


def extract_text_from_spreadsheet(
    path: Union[str, Path],
    sheet: Optional[Union[str, int]] = None,
    nrows: Optional[int]             = None,
    include_headers: bool            = True,
    engine: Optional[str]            = None,
    **kwargs,
) -> ExtractionResult:
    """Extract text from XLSX / XLS / CSV / ODS. Uses streaming mode for large Excel files."""
    config = ExtractionConfig(
        sheet=sheet, nrows=nrows, include_headers=include_headers,
    )
    return get_extraction_engine().extract(file_path=path, config=config, preferred_backend="pandas")


def extract_text_from_textfile(
    path: Union[str, Path],
    encoding: str                       = "utf-8",
    fallback_encodings: Optional[List[str]] = None,
    max_chars: int                      = MAX_TEXT_LENGTH_CHARS,
    detect_encoding: bool               = True,
    **kwargs,
) -> ExtractionResult:
    """Extract text from a plain-text file. Auto-detects encoding and streams large files."""
    config = ExtractionConfig(
        encoding=encoding,
        fallback_encodings=fallback_encodings or ["utf-8", "latin-1", "cp1252", "iso-8859-1"],
        max_chars=max_chars,
    )
    return get_extraction_engine().extract(file_path=path, config=config, preferred_backend="text")


def extract_text_from_file(
    path: Union[str, Path],
    method: Union[str, ExtractionMethod] = ExtractionMethod.AUTO,
    config: Optional[ExtractionConfig]   = None,
    **kwargs,
) -> ExtractionResult:
    """
    Auto-detect file type and extract text.

    This is the universal entry point — delegates to the correct backend.
    """
    if config is None:
        # Only pass valid ExtractionConfig fields from kwargs
        valid = {f for f in ExtractionConfig.__dataclass_fields__}  # type: ignore
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        config = ExtractionConfig(**filtered)
    elif kwargs:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

    if isinstance(method, str):
        try:
            method = ExtractionMethod(method.lower())
        except ValueError:
            method = ExtractionMethod.AUTO

    _backend_map = {
        ExtractionMethod.PDFMINER: "pdfminer",
        ExtractionMethod.PYPdf:    "pypdf",
        ExtractionMethod.DOCX:     "docx",
        ExtractionMethod.PANDAS:   "pandas",
        ExtractionMethod.TEXT:     "text",
        ExtractionMethod.OCR:      "ocr",
        ExtractionMethod.AUTO:     None,
    }
    return get_extraction_engine().extract(
        file_path=path, config=config,
        preferred_backend=_backend_map.get(method),
    )


def extract_text_with_ocr(
    path: Union[str, Path],
    language: str                       = "eng",
    psm: int                            = 3,
    oem: int                            = 3,
    enhance: bool                       = True,
    config: Optional[ExtractionConfig]  = None,
    **kwargs,
) -> ExtractionResult:
    """
    Extract text from an image file using OCR (Tesseract).

    Args:
        path:     Path to image file (.png, .jpg, .tiff, .bmp, …).
        language: Tesseract language code, e.g. ``"eng"``, ``"fra+eng"``.
        psm:      Page segmentation mode (1–13, default 3 = auto).
        oem:      OCR engine mode (0–3, default 3 = LSTM).
        enhance:  Apply auto-contrast image enhancement before OCR.
        config:   Optional ExtractionConfig override.
    """
    if config is None:
        valid = {f for f in ExtractionConfig.__dataclass_fields__}  # type: ignore
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        config = ExtractionConfig(**filtered)
    config.ocr_language = language
    config.ocr_psm      = psm
    config.ocr_oem      = oem
    config.ocr_enhance  = enhance
    return get_extraction_engine().extract(
        file_path=Path(path) if not isinstance(path, Path) else path,
        config=config, preferred_backend="ocr",
    )


# ── BUG #2 FIX: export ``extract_with_ocr`` at module level ──────────────────
# registrar.py: ``extract_with_ocr = _extract_items.get('extract_with_ocr')``
# Previously this was only defined as ``extract_text_with_ocr``, making the
# registrar silently receive None.
extract_with_ocr = extract_text_with_ocr   # ← Bug #2 fixed


# ──────────────────────────────────────────────────────────────────────────────
# Text chunking
# ──────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int                             = 1_000,
    chunk_overlap: int                          = 200,
    strategy: Union[str, ChunkingStrategy]     = ChunkingStrategy.SENTENCE,
    separators: Optional[List[str]]            = None,
    **kwargs,
) -> List[str]:
    """
    Split *text* into overlapping chunks using the selected strategy.

    Strategies:
        ``fixed``     — Hard character-count splits with partial-sentence repair.
        ``sentence``  — Respect sentence boundaries (.!?).
        ``paragraph`` — Split at blank lines.
        ``semantic``  — Try Markdown/numbered headings, fall back to paragraph.
        ``recursive`` — Iteratively split on a hierarchy of separators.
        ``token``     — Synonym for ``fixed`` (token-based split not implemented).

    Returns list of non-empty stripped chunks, none exceeding *chunk_size*.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    if isinstance(strategy, str):
        try:
            strategy = ChunkingStrategy(strategy.lower())
        except ValueError:
            strategy = ChunkingStrategy.SENTENCE

    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 5

    # ── Strategy implementations ───────────────────────────────────────────

    def _fixed_chunks(txt: str, size: int, overlap: int) -> List[str]:
        chunks: List[str] = []
        start = 0
        while start < len(txt):
            end   = min(start + size, len(txt))
            chunk = txt[start:end]
            # Try to snap end to a sentence boundary in the overlap zone
            if end < len(txt):
                zone = txt[max(start, end - overlap): end]
                best = max(
                    (zone.rfind(c) for c in (". ", "! ", "? ", "\n")),
                    default=-1,
                )
                if best > 0:
                    end   = start + (end - start - overlap) + best + 1
                    chunk = txt[start:end]
            stripped = chunk.strip()
            if stripped:
                chunks.append(stripped)
            next_start = end - overlap
            if next_start <= start:
                next_start = start + 1   # prevent infinite loop
            start = next_start
        return chunks

    def _sentence_chunks(txt: str, size: int, overlap: int) -> List[str]:
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$", txt)
        current = ""
        result: List[str] = []
        for sent in sents:
            sent = sent.strip()
            if not sent:
                continue
            candidate = (current + " " + sent).strip() if current else sent
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # Sentence itself longer than chunk_size → force split
                if len(sent) > size:
                    result.extend(_fixed_chunks(sent, size, overlap))
                    current = ""
                else:
                    current = sent
        if current:
            result.append(current)
        return result

    def _paragraph_chunks(txt: str, size: int, overlap: int) -> List[str]:
        paras   = re.split(r"\n\s*\n", txt)
        current = ""
        result: List[str] = []
        for para in paras:
            para = para.strip()
            if not para:
                continue
            candidate = (current + "\n\n" + para).strip() if current else para
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(para) > size:
                    result.extend(_fixed_chunks(para, size, overlap))
                    current = ""
                else:
                    current = para
        if current:
            result.append(current)
        return result

    def _recursive_chunks(txt: str, seps: List[str], size: int, overlap: int) -> List[str]:
        if len(txt) <= size:
            return [txt]
        if not seps:
            return _fixed_chunks(txt, size, overlap)
        sep, rest_seps = seps[0], seps[1:]
        if sep:
            parts = txt.split(sep)
        else:
            parts = list(txt)
        assembled: List[str] = []
        current = ""
        for part in parts:
            glue      = sep if current else ""
            candidate = current + glue + part
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    assembled.append(current)
                current = part
        if current:
            assembled.append(current)
        final: List[str] = []
        for seg in assembled:
            if len(seg) <= size:
                final.append(seg)
            else:
                final.extend(_recursive_chunks(seg, rest_seps, size, overlap))
        return final

    # ── Dispatch ─────────────────────────────────────────────────────────────

    if strategy == ChunkingStrategy.FIXED or strategy == ChunkingStrategy.TOKEN:
        raw_chunks = _fixed_chunks(text, chunk_size, chunk_overlap)

    elif strategy == ChunkingStrategy.SENTENCE:
        raw_chunks = _sentence_chunks(text, chunk_size, chunk_overlap)

    elif strategy == ChunkingStrategy.PARAGRAPH:
        raw_chunks = _paragraph_chunks(text, chunk_size, chunk_overlap)

    elif strategy == ChunkingStrategy.SEMANTIC:
        # Try section-level splits first
        section_patterns = [
            r"\n#+\s+",          # Markdown headers
            r"\n\d+\.\s+",       # Numbered sections
            r"\n[A-Z][A-Z ]+\n", # ALL-CAPS titles
            r"\n\*{3,}\n",       # Horizontal rules
        ]
        raw_chunks = []
        for pattern in section_patterns:
            secs = [s.strip() for s in re.split(pattern, text) if s.strip()]
            if len(secs) > 1 and all(len(s) <= chunk_size * 2 for s in secs):
                raw_chunks = secs
                break
        if not raw_chunks:
            raw_chunks = _paragraph_chunks(text, chunk_size, chunk_overlap)

    elif strategy == ChunkingStrategy.RECURSIVE:
        seps = separators or ["\n\n", "\n", ". ", " ", ""]
        raw_chunks = _recursive_chunks(text, seps, chunk_size, chunk_overlap)

    else:
        raw_chunks = _sentence_chunks(text, chunk_size, chunk_overlap)

    # ── Final pass: ensure no chunk exceeds chunk_size ─────────────────────
    final: List[str] = []
    for c in raw_chunks:
        c = c.strip()
        if not c:
            continue
        if len(c) <= chunk_size:
            final.append(c)
        else:
            final.extend(_fixed_chunks(c, chunk_size, chunk_overlap))

    logger.debug("chunk_text: %d chunks from %d chars", len(final), len(text))
    return final


# ──────────────────────────────────────────────────────────────────────────────
# extract_and_chunk_file
# ──────────────────────────────────────────────────────────────────────────────

def extract_and_chunk_file(
    path: Union[str, Path],
    chunk_size: int                             = 1_000,
    chunk_overlap: int                          = 200,
    chunking_strategy: Union[str, ChunkingStrategy] = ChunkingStrategy.SENTENCE,
    config: Optional[ExtractionConfig]         = None,
    **extraction_kwargs,
) -> Dict[str, Any]:
    """
    Extract text from *path* and immediately chunk it.

    Returns a dict with keys:
        ok, extraction_result, chunks, chunk_count,
        chunk_size, chunk_overlap, chunking_strategy, processing_time
    """
    t0 = time.monotonic()
    if config is None:
        valid = {f for f in ExtractionConfig.__dataclass_fields__}  # type: ignore
        filtered = {k: v for k, v in extraction_kwargs.items() if k in valid}
        config = ExtractionConfig(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            chunking_strategy=(
                chunking_strategy if isinstance(chunking_strategy, ChunkingStrategy)
                else ChunkingStrategy(str(chunking_strategy).lower())
            ),
            **filtered,
        )

    ext_result = extract_text_from_file(path, config=config)
    if not ext_result.ok:
        return {
            "ok": False, "error": ext_result.error,
            "extraction_result": ext_result.to_dict(),
            "chunks": [], "chunk_count": 0,
            "processing_time": time.monotonic() - t0,
        }

    chunks = chunk_text(
        ext_result.text, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, strategy=chunking_strategy,
    )
    if ext_result.metrics:
        ext_result.metrics.chunks_generated = len(chunks)

    strat_val = (
        chunking_strategy.value if isinstance(chunking_strategy, ChunkingStrategy)
        else str(chunking_strategy)
    )
    return {
        "ok": True,
        "extraction_result": ext_result.to_dict(),
        "chunks": chunks,
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunking_strategy": strat_val,
        "processing_time": round(time.monotonic() - t0, 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Parallel batch extraction
# ──────────────────────────────────────────────────────────────────────────────

def batch_extract_files(
    file_paths: List[Union[str, Path]],
    max_concurrent: int                                              = MAX_CONCURRENT_EXTRACTIONS,
    progress_callback: Optional[Callable[[int, int, str, Dict[str, Any]], None]] = None,
    config: Optional[ExtractionConfig]                              = None,
    **extraction_kwargs,
) -> List[ExtractionResult]:
    """
    Extract text from many files **in parallel** (order-preserved).

    Args:
        file_paths:         Files to process.
        max_concurrent:     Thread-pool size.
        progress_callback:  ``callback(done, total, file_str, stats_dict)``.
        config:             Shared ExtractionConfig.
        **extraction_kwargs: Passed to ExtractionConfig if config is None.

    Returns:
        List of ExtractionResult in the same order as *file_paths*.
    """
    if not file_paths:
        return []

    if config is None:
        valid = {f for f in ExtractionConfig.__dataclass_fields__}  # type: ignore
        filtered = {k: v for k, v in extraction_kwargs.items() if k in valid}
        config = ExtractionConfig(**filtered)

    total   = len(file_paths)
    workers = min(max(1, max_concurrent), MAX_CONCURRENT_EXTRACTIONS, total)
    logger.info("batch_extract_files: %d files, %d workers", total, workers)

    results: List[Optional[ExtractionResult]] = [None] * total
    completed_count = 0
    failed_count    = 0
    counter_lock    = threading.Lock()

    def _do_extract(idx: int, fp: Union[str, Path]) -> Tuple[int, ExtractionResult]:
        r = extract_text_from_file(fp, config=config)
        return idx, r

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="ExtractWorker"
    ) as pool:
        future_map: Dict[Future, int] = {
            pool.submit(_do_extract, i, fp): i
            for i, fp in enumerate(file_paths)
        }
        for fut in as_completed(future_map):
            orig_idx = future_map[fut]
            fp_str   = str(file_paths[orig_idx])
            try:
                idx, r = fut.result()
                results[idx] = r
                with counter_lock:
                    if r.ok:
                        completed_count += 1
                    else:
                        failed_count += 1
                    done = completed_count + failed_count
            except Exception as exc:
                logger.error("batch_extract_files error for %s: %s", fp_str, exc)
                r = ExtractionResult(
                    ok=False, error=str(exc), file_path=fp_str,
                    status=ExtractionStatus.FAILED, processing_time=0.0,
                )
                results[orig_idx] = r
                with counter_lock:
                    failed_count += 1
                    done = completed_count + failed_count

            if progress_callback:
                try:
                    progress_callback(
                        done, total, fp_str,
                        {"completed": completed_count, "failed": failed_count,
                         "current_result": r.to_dict()},  # type: ignore[union-attr]
                    )
                except Exception:
                    pass

    # Fill any slots that somehow weren't populated
    for i in range(total):
        if results[i] is None:
            results[i] = ExtractionResult(
                ok=False, error="Not executed",
                file_path=str(file_paths[i]), status=ExtractionStatus.FAILED,
            )

    ok_count   = sum(1 for r in results if r and r.ok)  # type: ignore
    total_time = sum(r.processing_time for r in results if r)  # type: ignore
    logger.info(
        "batch_extract_files done: %d/%d OK, %.2fs total processing time",
        ok_count, total, total_time,
    )
    return results  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# Vector store integration
# ──────────────────────────────────────────────────────────────────────────────

def _make_embedding_and_upsert(
    service: Any,
    text: str,
    source_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_index: Optional[int]          = None,
    total_chunks: Optional[int]         = None,
    embedding_model: Optional[str]      = None,
) -> Dict[str, Any]:
    """Create an embedding for *text* and upsert it into a vector store via *service*."""
    if not text or not text.strip():
        return {"ok": False, "error": "Empty text"}
    try:
        adapter = getattr(service, "embedding_adapter", None)
        store   = getattr(service, "vectorstore", None)
        if not adapter or not store:
            return {"ok": False, "error": "Service missing embedding_adapter or vectorstore"}
        embedding = adapter.embed(text, model=embedding_model)
        meta      = metadata or {}
        if chunk_index is not None:
            meta["chunk_index"]  = chunk_index
            meta["total_chunks"] = total_chunks
        meta["source_id"] = source_id
        meta["text_len"]  = len(text)
        store.upsert(
            ids=[f"{source_id}_chunk_{chunk_index}" if chunk_index is not None else source_id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[text],
        )
        return {"ok": True, "source_id": source_id, "chunk_index": chunk_index}
    except Exception as exc:
        logger.error("Embedding/upsert failed for %s: %s", source_id, exc)
        return {"ok": False, "error": str(exc), "source_id": source_id}


def process_file_for_vector_store(
    path: Union[str, Path],
    service: Any,
    chunk_size: int                             = 1_000,
    chunk_overlap: int                          = 200,
    chunking_strategy: Union[str, ChunkingStrategy] = ChunkingStrategy.SENTENCE,
    source_id: Optional[str]                   = None,
    metadata: Optional[Dict[str, Any]]         = None,
    config: Optional[ExtractionConfig]         = None,
    max_workers: int                            = 4,
    embedding_model: Optional[str]             = None,
) -> Dict[str, Any]:
    """
    Extract, chunk, embed, and upsert a file into a vector store.

    Chunks are processed in parallel (``max_workers`` threads).

    Returns dict with: ok, chunks_processed, chunks_failed, source_id,
    processing_time, error.
    """
    t0 = time.monotonic()
    path_obj  = Path(path)
    source_id = source_id or path_obj.stem

    result = extract_and_chunk_file(
        path_obj, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=chunking_strategy,
        config=config,
    )
    if not result["ok"]:
        return {"ok": False, "error": result.get("error"), "source_id": source_id,
                "chunks_processed": 0, "processing_time": time.monotonic() - t0}

    chunks       = result["chunks"]
    total_chunks = len(chunks)
    meta_base    = metadata or {}
    meta_base.update({"file_path": str(path_obj), "file_name": path_obj.name})

    processed = 0
    failed    = 0

    def _upsert(idx_chunk: Tuple[int, str]) -> Dict[str, Any]:
        idx, chunk = idx_chunk
        m = dict(meta_base)
        return _make_embedding_and_upsert(
            service=service, text=chunk, source_id=source_id,
            metadata=m, chunk_index=idx, total_chunks=total_chunks,
            embedding_model=embedding_model,
        )

    workers = min(max_workers, total_chunks) if total_chunks else 1
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="VecUpsert") as pool:
        for res in pool.map(_upsert, enumerate(chunks)):
            if res.get("ok"):
                processed += 1
            else:
                failed += 1
                logger.warning("Upsert failed for chunk: %s", res.get("error"))

    return {
        "ok":               failed == 0,
        "source_id":        source_id,
        "chunks_processed": processed,
        "chunks_failed":    failed,
        "total_chunks":     total_chunks,
        "processing_time":  round(time.monotonic() - t0, 3),
    }


def process_chunks_concurrently(
    chunks: List[str],
    processor: Callable[[int, str], Dict[str, Any]],
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Apply *processor(idx, chunk_text)* to all chunks in parallel.

    Returns results list in original order.
    """
    if not chunks:
        return []
    workers  = min(max_workers, len(chunks))
    results2: List[Optional[Dict[str, Any]]] = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_map: Dict[Future, int] = {
            pool.submit(processor, i, c): i for i, c in enumerate(chunks)
        }
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            try:
                results2[i] = fut.result()
            except Exception as exc:
                results2[i] = {"ok": False, "error": str(exc), "chunk_index": i}

    return [r or {"ok": False, "error": "Not executed"} for r in results2]


# ──────────────────────────────────────────────────────────────────────────────
# Backend / format introspection
# ──────────────────────────────────────────────────────────────────────────────

def get_extraction_backends() -> Dict[str, Any]:
    """
    Return a status report of all available extraction backends.

    Useful for health checks and debugging.
    """
    deps    = _check_dependencies()
    engine  = get_extraction_engine()

    # Tesseract CLI check
    tesseract_version: Optional[str] = None
    try:
        res = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            tesseract_version = res.stdout.splitlines()[0]
    except Exception:
        pass

    backends_info = [
        {
            "name":                b.name,
            "supports_pdf":        b.supports(Path("test.pdf")),
            "supports_docx":       b.supports(Path("test.docx")),
            "supports_text":       b.supports(Path("test.txt")),
            "supports_spreadsheet": b.supports(Path("test.xlsx")),
            "supports_images":     b.supports(Path("test.png")),
        }
        for b in engine.backends
    ]

    return {
        "dependencies": deps,
        "backends":     backends_info,
        "ocr": {
            "available": deps.get("pillow", False) and deps.get("pytesseract", False),
            "version":   tesseract_version,
        },
        "system": {
            "platform":        platform.system(),
            "platform_release": platform.release(),
            "python_version":  platform.python_version(),
        },
        "limits": {
            "max_file_size_mb":          MAX_FILE_SIZE_MB,
            "max_text_length_chars":     MAX_TEXT_LENGTH_CHARS,
            "max_concurrent_extractions": MAX_CONCURRENT_EXTRACTIONS,
            "memory_limit_mb":           MEMORY_LIMIT_MB,
        },
        "supported_formats": get_supported_formats(),
        "circuit_breaker": {
            "state":         extraction_circuit_breaker.state,
            "failure_count": extraction_circuit_breaker.failure_count,
        },
        "memory": _resource_monitor.get_metrics(),
    }


def get_supported_formats() -> Dict[str, List[str]]:
    """
    Return dict of supported file format categories → extension lists.

    Bug #5 note: previously there was naming drift; this is the canonical name.
    """
    return {
        "text": [
            ".txt", ".md", ".markdown", ".rst", ".log",
            ".ini", ".cfg", ".conf", ".properties",
            ".yml", ".yaml", ".toml", ".rtf", ".tex",
            ".sh", ".py", ".js", ".ts", ".css",
        ],
        "documents":     [".pdf", ".docx", ".doc", ".odt", ".epub", ".mobi"],
        "spreadsheets":  [".xlsx", ".xls", ".ods", ".csv", ".tsv"],
        "presentations": [".pptx", ".ppt", ".odp"],
        "html_xml":      [".html", ".htm", ".xhtml", ".xml", ".svg"],
        "data":          [".json", ".jsonl", ".ndjson"],
        "images":        [".png", ".jpg", ".jpeg", ".gif", ".bmp",
                          ".tiff", ".tif", ".webp"],
        "archives":      [".zip", ".tar", ".gz", ".bz2", ".xz", ".7z"],
    }


def clear_extraction_cache() -> None:
    """
    Reset the global ExtractionEngine singleton and clear the dependency cache.

    Useful after installing new OCR/PDF libraries at runtime.
    """
    global _extraction_engine, _DEPENDENCIES
    with _engine_lock:
        _extraction_engine = None
    with _DEPENDENCIES_LOCK:
        _DEPENDENCIES.clear()
    gc.collect()
    logger.info("Extraction cache cleared — engine will reinitialise on next call")


# ──────────────────────────────────────────────────────────────────────────────
# VectorStoreProcessor convenience class
# ──────────────────────────────────────────────────────────────────────────────

class VectorStoreProcessor:
    """
    High-level processor that extracts, chunks, embeds, and indexes files.

    Usage:
        proc = VectorStoreProcessor(service=my_service, chunk_size=800)
        stats = proc.process_files(["doc1.pdf", "report.docx"])
    """

    def __init__(
        self,
        service: Any,
        chunk_size: int    = 1_000,
        chunk_overlap: int = 200,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
        max_workers: int   = 4,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.service         = service
        self.chunk_size      = chunk_size
        self.chunk_overlap   = chunk_overlap
        self.strategy        = strategy
        self.max_workers     = max_workers
        self.embedding_model = embedding_model
        self._stats: Dict[str, int] = {
            "files_processed": 0, "files_failed": 0,
            "chunks_indexed": 0, "chunks_failed": 0,
        }

    def process_files(
        self,
        file_paths: List[Union[str, Path]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process and index a list of files.

        Returns aggregate statistics.
        """
        t0 = time.monotonic()
        for fp in file_paths:
            res = process_file_for_vector_store(
                path=fp, service=self.service,
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                chunking_strategy=self.strategy,
                metadata=metadata, max_workers=self.max_workers,
                embedding_model=self.embedding_model,
            )
            if res.get("ok"):
                self._stats["files_processed"]  += 1
                self._stats["chunks_indexed"]   += res.get("chunks_processed", 0)
                self._stats["chunks_failed"]    += res.get("chunks_failed", 0)
            else:
                self._stats["files_failed"] += 1
                logger.warning("VectorStoreProcessor: failed %s — %s", fp, res.get("error"))

        return {
            **self._stats,
            "total_files":      len(file_paths),
            "processing_time":  round(time.monotonic() - t0, 3),
        }

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self._stats)


# Production Additions v5.0
import re as _ext_re
from collections import Counter as _Counter


class ExtractionQualityScorer:
    """Score extracted text quality to filter boilerplate/garbage.
    Returns 0.0 (garbage) to 1.0 (clean article text).
    """

    def __init__(self, min_length: int = 100, min_alpha_ratio: float = 0.4):
        self._min_len = min_length
        self._min_alpha = min_alpha_ratio

    def score(self, text: str) -> float:
        """Return quality score 0.0-1.0."""
        if not text or len(text) < self._min_len:
            return 0.0
        score = 0.5
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio >= self._min_alpha:
            score += 0.2
        else:
            score -= 0.3
        sentences = len(_ext_re.findall(r'[.!?][\s\n]', text))
        if sentences >= 3:
            score += 0.15
        elif sentences >= 1:
            score += 0.05
        words = _ext_re.findall(r'\b\w{4,}\b', text.lower())
        if words:
            most_common_count = _Counter(words).most_common(1)[0][1]
            if most_common_count / len(words) > 0.3:
                score -= 0.3
            if len(set(words)) / len(words) > 0.6:
                score += 0.1
        return max(0.0, min(1.0, round(score, 4)))

    def is_quality(self, text: str, threshold: float = 0.3) -> bool:
        """Return True if text passes quality threshold."""
        return self.score(text) >= threshold


class SmartSentenceChunker:
    """Sentence-boundary-aware text chunker with overlap.
    Produces chunks of ~target_tokens with overlap_sentences context overlap.
    """

    def __init__(self, target_tokens: int = 256, overlap_sentences: int = 2,
                 chars_per_token: float = 4.0):
        self._target_chars = int(target_tokens * chars_per_token)
        self._overlap = overlap_sentences
        self._chars_per_tok = chars_per_token

    def _split_sentences(self, text: str) -> list:
        raw = _ext_re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if s.strip()]

    def chunk(self, text: str) -> list:
        """Return list of text chunks with sentence-level overlap."""
        if not text:
            return []
        if len(text) <= self._target_chars:
            return [text.strip()]
        sentences = self._split_sentences(text)
        if not sentences:
            return [text.strip()]
        chunks = []
        current_sents: list = []
        current_len = 0
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self._target_chars and current_sents:
                chunks.append(" ".join(current_sents))
                overlap_sents = current_sents[-self._overlap:] if self._overlap > 0 else []
                current_sents = overlap_sents + [sent]
                current_len = sum(len(s) for s in current_sents)
            else:
                current_sents.append(sent)
                current_len += sent_len
        if current_sents:
            chunks.append(" ".join(current_sents))
        return [c for c in chunks if c.strip()]

    def chunk_with_metadata(self, text: str, doc_id: str = "") -> list:
        """Return chunks as dicts with index/token metadata."""
        chunks = self.chunk(text)
        return [
            {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk,
                "char_count": len(chunk),
                "estimated_tokens": int(len(chunk) / self._chars_per_tok),
                "doc_id": doc_id,
            }
            for i, chunk in enumerate(chunks)
        ]


# Module-level singletons
_quality_scorer = ExtractionQualityScorer()
_smart_chunker = SmartSentenceChunker(target_tokens=256, overlap_sentences=2)


def score_extraction_quality(text: str) -> float:
    """Score extracted text quality (0.0=garbage, 1.0=clean)."""
    return _quality_scorer.score(text)


def smart_chunk_text(text: str, target_tokens: int = 256, overlap: int = 2) -> list:
    """Sentence-aware chunking with overlap. Returns list of text strings."""
    chunker = SmartSentenceChunker(target_tokens=target_tokens, overlap_sentences=overlap)
    return chunker.chunk(text)
