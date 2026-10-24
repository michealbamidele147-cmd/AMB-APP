# tools/__init__.py
"""
AMB-Sentinel Tools Package - Production-Ready Exports
Complete facade for all tool modules with comprehensive error handling.

Version: 1.0.0 (Production - Full Coverage)
Security Level: HIGH

Author: AMB Team


"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List

# Configure module logger
logger = logging.getLogger("Initials")

# ============================================================================
# CORE UTILITIES & CONSTANTS
# ============================================================================
from .utils import (
    DEFAULT_DOWNLOAD_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_LOG_DIR,
    cleanup_downloads,
    calculate_file_hash,
    validate_url,
    sanitize_filename,
    generate_safe_filename,
    SessionConfig,
    SecurityLevel,
    RetryStrategy,
    DownloadProgress,
    DownloadConfig,
    DownloadResult as UtilsDownloadResult,   # Bug #6: renamed to avoid collision
    CleanupResult,
)

# ============================================================================
# HTTP & NETWORK TOOLS
# ============================================================================
from .http import (
    http_fetch,
    http_fetch_batch,
    HttpConfig,
    HttpMethod,
    RetryStrategy as HttpRetryStrategy,
    RateLimiter,
    CircuitBreaker,
    CircuitState,
    ContentType,
    HttpError,
    CircuitBreakerOpenError,
    RateLimitExceededError,
    SecurityViolationError,
    ResponseTooLargeError,
    HttpResponse,
)

# ============================================================================
# DOWNLOAD TOOLS
# ============================================================================
from .download import (
    download_file,
    download_multiple,
    get_download_stats,
    validate_download_result,
    health_check as download_health_check,
    DownloadManager,
    DownloadOptions,
    DownloadResult,       # Bug #6 FIX: this is the richer download.py version
)

# ============================================================================
# LINK SEARCH TOOLS - WITH ENHANCED ERROR HANDLING
# ============================================================================
def _import_link_search_tools():
    """Safely import link search tools with fallback."""
    try:
        from .find_links import (
            search_links,
            search_with_dorks,
            find_file_links,
            find_hidden_resources,
            extract_links_from_page,
            expand_links,
            LinkSearchConfig,
            _ENGINE_HANDLERS
        )
        return {
            "search_links": search_links,
            "search_with_dorks": search_with_dorks,
            "find_file_links": find_file_links,
            "find_hidden_resources": find_hidden_resources,
            "extract_links_from_page": extract_links_from_page,
            "expand_links": expand_links,
            "LinkSearchConfig": LinkSearchConfig,
            "_ENGINE_HANDLERS": _ENGINE_HANDLERS,
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to import link search tools: {e}")
        return {}


_link_search_tools = _import_link_search_tools()
search_links = _link_search_tools.get("search_links")
search_with_dorks = _link_search_tools.get("search_with_dorks")
find_file_links = _link_search_tools.get("find_file_links")
find_hidden_resources = _link_search_tools.get("find_hidden_resources")
extract_links_from_page = _link_search_tools.get("extract_links_from_page")
expand_links = _link_search_tools.get("expand_links")
LinkSearchConfig = _link_search_tools.get("LinkSearchConfig")
_ENGINE_HANDLERS = _link_search_tools.get("_ENGINE_HANDLERS")


def get_available_search_engines() -> List[str]:
    """Get list of available search engines supported by the link search module."""
    try:
        if _ENGINE_HANDLERS:
            return list(_ENGINE_HANDLERS.keys())
        return []
    except Exception as e:
        logger.warning(f"⚠️ Failed to get search engines: {e}")
        return []


def create_link_search_config(**kwargs) -> Optional[LinkSearchConfig]:
    """Create link search configuration with custom settings."""
    try:
        if LinkSearchConfig:
            return LinkSearchConfig(**kwargs)
        return None
    except Exception as e:
        logger.error(f"❌ Failed to create LinkSearchConfig: {e}")
        return None

# ============================================================================
# SCRAPING TOOLS - WITH ENHANCED ERROR HANDLING
# ============================================================================
def _import_scraping_tools():
    """Safely import scraping tools with fallback."""
    try:
        from .scrape import (
            scrape_page,
            scrape_batch,
            scrape_async,
            scrape_batch_async,
            parse_feed,
            browser_scrape,
            get_scraping_stats,
            clear_scraping_cache,
            ScrapingConfig,
            ScrapingEngine,
            AntiDetectionLevel,
            RobotsPolicy,
            ContentCategory,
            DependencyManager,
            ContentExtractor,
            ScrapingResult,
            RobotsTxtChecker,
        )
        # ScrapingCache: Bug #7 FIX — import defensively; old scrape.py
        # had a dead import here. New scrape.py exports it correctly.
        try:
            from .scrape import ScrapingCache
        except ImportError:
            ScrapingCache = None
        return {
            "scrape_page": scrape_page,
            "scrape_batch": scrape_batch,
            "scrape_async": scrape_async,
            "scrape_batch_async": scrape_batch_async,
            "parse_feed": parse_feed,
            "browser_scrape": browser_scrape,
            "get_scraping_stats": get_scraping_stats,
            "clear_scraping_cache": clear_scraping_cache,
            "ScrapingConfig": ScrapingConfig,
            "ScrapingEngine": ScrapingEngine,
            "AntiDetectionLevel": AntiDetectionLevel,
            "RobotsPolicy": RobotsPolicy,
            "ContentCategory": ContentCategory,
            "DependencyManager": DependencyManager,
            "ContentExtractor": ContentExtractor,
            "ScrapingResult": ScrapingResult,
            "ScrapingCache": ScrapingCache,     # Bug #7 fixed — no longer crashes
            "RobotsTxtChecker": RobotsTxtChecker,
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to import scraping tools: {e}")
        return {}


_scraping_tools = _import_scraping_tools()
scrape_page = _scraping_tools.get("scrape_page")
scrape_batch = _scraping_tools.get("scrape_batch")
scrape_async = _scraping_tools.get("scrape_async")
scrape_batch_async = _scraping_tools.get("scrape_batch_async")
parse_feed = _scraping_tools.get("parse_feed")
browser_scrape = _scraping_tools.get("browser_scrape")
get_scraping_stats = _scraping_tools.get("get_scraping_stats")
clear_scraping_cache = _scraping_tools.get("clear_scraping_cache")
ScrapingConfig = _scraping_tools.get("ScrapingConfig")
ScrapingEngine = _scraping_tools.get("ScrapingEngine")
DependencyManager = _scraping_tools.get("DependencyManager")
ContentExtractor = _scraping_tools.get("ContentExtractor")


# Convenience functions
def rss_fetch(url: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
    """Fetch and parse RSS/Atom feed."""
    if not parse_feed:
        return {"ok": False, "error": "RSS parsing not available"}
    return parse_feed(url, **kwargs)


def google_news_search(query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
    """Search Google News via RSS feed."""
    if not parse_feed:
        return {"ok": False, "error": "RSS parsing not available"}

    try:
        from urllib.parse import quote
        feed_url = f"https://news.google.com/rss/search?q={quote(query)}"
        result = parse_feed(feed_url, parse_entries=False, **kwargs)
        if result.get("ok") and "entries" in result:
            result["entries"] = result["entries"][:limit]
        return result
    except Exception as e:
        logger.error(f"❌ Google News search failed: {e}")
        return {"ok": False, "error": str(e), "query": query}


def scrape_multiple(urls: List[str], **kwargs) -> List[Dict[str, Any]]:
    """Scrape multiple URLs (convenience wrapper)."""
    if not scrape_batch:
        return [{"ok": False, "error": "Scraping tools not available"} for _ in urls]
    try:
        return scrape_batch(urls, **kwargs)
    except Exception as e:
        logger.error(f"❌ Multiple scrape failed: {e}")
        return [{"ok": False, "error": str(e), "url": url} for url in urls]


def get_scraping_backends() -> Dict[str, bool]:
    """Get available scraping backends."""
    if DependencyManager:
        try:
            return DependencyManager.get_available()
        except Exception as e:
            logger.warning(f"⚠️ Failed to get scraping backends: {e}")
            return {}
    return {}

# ============================================================================
# EXTRACTION TOOLS - WITH ENHANCED ERROR HANDLING (Robust import wrapper)
# ============================================================================
def _import_extraction_tools():
    """Safely import extraction tools with fallback (robust to missing symbols).

    This loader:
    - Imports the `tools.extract` module via importlib to avoid relative import pitfalls.
    - Iterates over an explicit list of expected symbols and uses getattr(..., None)
      so a missing symbol does not cause an ImportError for the whole import.
    - Returns a dict mapping expected symbol names to either the real object or None.
    - Logs a warning on any exception and returns None mappings (caller handles missing).
    """
    items = [
        'extract_text_from_file',
        'extract_text_from_pdf',
        'extract_text_from_docx',
        'extract_text_from_spreadsheet',
        'extract_text_from_textfile',
        'extract_and_chunk_file',
        'chunk_text',
        'batch_extract_files',
        'extract_with_ocr',
        'process_file_for_vector_store',
        'get_extraction_backends',
        'ExtractionConfig',
        'ExtractionEngine',
        'ExtractionResult',
        'ChunkingStrategy',
        'OCREngine',
    ]

    try:
        import importlib
        module = importlib.import_module(f'{__package__}.extract')
        imported = {}
        for item in items:
            # Use getattr with default None to tolerate missing attributes
            try:
                imported[item] = getattr(module, item, None)
            except Exception as e:
                # Defensive: if getattr itself raises for some reason, catch and set None
                logger.warning(f"Unable to load symbol '{item}' from tools.extract: {e}")
                imported[item] = None
        return imported

    except Exception as e:
        # Log a warning but do not break initialization of the package
        logger.warning(f"⚠️ Failed to import extraction tools: {e}")
        # Return mapping with None for every expected symbol so callers can handle gracefully
        return {item: None for item in items}


_extraction_tools = _import_extraction_tools()

# Expose variables for backward-compatible imports in the package
extract_text_from_file = _extraction_tools.get("extract_text_from_file")
extract_text_from_pdf = _extraction_tools.get("extract_text_from_pdf")
extract_text_from_docx = _extraction_tools.get("extract_text_from_docx")
extract_text_from_spreadsheet = _extraction_tools.get("extract_text_from_spreadsheet")
extract_text_from_textfile = _extraction_tools.get("extract_text_from_textfile")
extract_and_chunk_file = _extraction_tools.get("extract_and_chunk_file")
chunk_text = _extraction_tools.get("chunk_text")
batch_extract_files = _extraction_tools.get("batch_extract_files")
extract_with_ocr = _extraction_tools.get("extract_with_ocr")
process_file_for_vector_store = _extraction_tools.get("process_file_for_vector_store")
get_extraction_backends_fn = _extraction_tools.get("get_extraction_backends")

# Classes / Config objects (may be None)
ExtractionConfig = _extraction_tools.get("ExtractionConfig")
ExtractionEngine = _extraction_tools.get("ExtractionEngine")
ExtractionResult = _extraction_tools.get("ExtractionResult")
ChunkingStrategy = _extraction_tools.get("ChunkingStrategy")
OCREngine = _extraction_tools.get("OCREngine")


def get_extraction_backends() -> Dict[str, bool]:
    """
    Get available extraction backends.

    This wrapper safely calls the underlying backend-discovery function if present.
    If the underlying function is missing or raises, we return an empty dict.
    """
    try:
        if callable(get_extraction_backends_fn):
            return get_extraction_backends_fn()
    except Exception as e:
        logger.warning(f"⚠️ Failed to get extraction backends: {e}")
    return {}

# ============================================================================
# VIDEO TOOLS - WITH ENHANCED ERROR HANDLING
# ============================================================================
def _import_video_tools():
    """Safely import video tools with fallback."""
    try:
        from .video import (
            download_video,
            download_multiple_videos,
            extract_video_thumbnail,
            video_info,
            get_video_backends,
            VideoProcessor,
            VideoConfig,
            VideoPlatform,
            VideoQuality,
            ThumbnailStrategy,
            VideoFormat,
            DownloadStatus,
            VideoMetadata,
            VideoResult,
        )
        return {
            "download_video": download_video,
            "download_multiple_videos": download_multiple_videos,
            "extract_video_thumbnail": extract_video_thumbnail,
            "video_info": video_info,
            "get_video_backends": get_video_backends,
            "VideoProcessor": VideoProcessor,
            "VideoConfig": VideoConfig,
            "VideoPlatform": VideoPlatform,
            "VideoQuality": VideoQuality,
            "ThumbnailStrategy": ThumbnailStrategy,
            "VideoFormat": VideoFormat,
            "DownloadStatus": DownloadStatus,
            "VideoMetadata": VideoMetadata,
            "VideoResult": VideoResult,
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to import video tools: {e}")
        return {}


_video_tools = _import_video_tools()
download_video = _video_tools.get("download_video")
download_multiple_videos = _video_tools.get("download_multiple_videos")
extract_video_thumbnail = _video_tools.get("extract_video_thumbnail")
video_info = _video_tools.get("video_info")
VideoProcessor = _video_tools.get("VideoProcessor")
VideoConfig = _video_tools.get("VideoConfig")
VideoPlatform = _video_tools.get("VideoPlatform")
VideoQuality = _video_tools.get("VideoQuality")


def get_video_backends() -> Dict[str, bool]:
    """Get available video processing backends."""
    if _video_tools.get("VideoProcessor"):
        try:
            processor = _video_tools["VideoProcessor"]()
            # v4.0.0 attribute names: _ytdlp, _ffmpeg, _http
            return {
                "yt_dlp":  processor._ytdlp.is_available(),
                "ffmpeg":  processor._ffmpeg.is_available(),
                "http":    processor._http.is_available(),
            }
        except Exception as e:
            logger.warning(f"⚠️ Failed to get video backends: {e}")
            return {}
    return {}


# ============================================================================
# TRANSCRIPTION TOOLS - WITH ENHANCED ERROR HANDLING
# ============================================================================
def _import_transcription_tools():
    """Safely import transcription tools with fallback."""
    try:
        from .transcribe import (
            transcribe_audio_or_video,
            transcribe_multiple,
            get_transcription_backends,
            clear_model_cache,
            get_supported_formats,
            TranscriptionConfig,
            TranscriptionResult,
            TranscriptionBackend,
            ComputeType,
            TranscriptionStatus,
            TranscriptionMetrics,
            TranscriptionBatchProcessor,
        )
        return {
            "transcribe_audio_or_video": transcribe_audio_or_video,
            "transcribe_multiple": transcribe_multiple,
            "get_transcription_backends": get_transcription_backends,
            "clear_model_cache": clear_model_cache,
            "get_supported_formats": get_supported_formats,
            "TranscriptionConfig": TranscriptionConfig,
            "TranscriptionResult": TranscriptionResult,
            "TranscriptionBackend": TranscriptionBackend,
            "ComputeType": ComputeType,
            "TranscriptionStatus": TranscriptionStatus,
            "TranscriptionMetrics": TranscriptionMetrics,
            "TranscriptionBatchProcessor": TranscriptionBatchProcessor,
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to import transcription tools: {e}")
        return {}


_transcription_tools = _import_transcription_tools()
transcribe_audio_or_video = _transcription_tools.get("transcribe_audio_or_video")
transcribe_multiple = _transcription_tools.get("transcribe_multiple")
clear_model_cache = _transcription_tools.get("clear_model_cache")
TranscriptionConfig = _transcription_tools.get("TranscriptionConfig")


def get_transcription_backends() -> Dict[str, bool]:
    """Get available transcription backends."""
    if _transcription_tools.get("get_transcription_backends"):
        try:
            return _transcription_tools["get_transcription_backends"]()
        except Exception as e:
            logger.warning(f"⚠️ Failed to get transcription backends: {e}")
            return {}
    return {}


def get_supported_transcription_formats() -> Dict[str, List[str]]:
    """Get supported audio/video formats for transcription."""
    if _transcription_tools.get("get_supported_formats"):
        try:
            return _transcription_tools["get_supported_formats"]()
        except Exception as e:
            logger.warning(f"⚠️ Failed to get supported formats: {e}")
            return {"audio": [], "video": []}
    return {"audio": [], "video": []}


# ============================================================================
# PIPELINE & INTEGRATION TOOLS
# ============================================================================
def _import_pipeline_tools():
    """Safely import pipeline tools with fallback."""
    try:
        from .pipeline import (
            register_pipeline_tools,
            transcribe_url_to_vectorstore,
            transcribe_and_index,
            PipelineManager,
            PipelineConfig,
            ProcessingStage,
            SourceType,
            ProcessingStatus,
            ProcessingError,
            ProcessingResult,
            PipelineMetrics,
            SecurityValidator,
            DownloadProcessor,
            TranscriptionProcessor,
            IndexingProcessor,
        )
        return {
            "register_pipeline_tools": register_pipeline_tools,
            "transcribe_url_to_vectorstore": transcribe_url_to_vectorstore,
            "transcribe_and_index": transcribe_and_index,
            "PipelineManager": PipelineManager,
            "PipelineConfig": PipelineConfig,
            "ProcessingStage": ProcessingStage,
            "SourceType": SourceType,
            "ProcessingStatus": ProcessingStatus,
            "ProcessingError": ProcessingError,
            "ProcessingResult": ProcessingResult,
            "PipelineMetrics": PipelineMetrics,
            "SecurityValidator": SecurityValidator,
            "DownloadProcessor": DownloadProcessor,
            "TranscriptionProcessor": TranscriptionProcessor,
            "IndexingProcessor": IndexingProcessor,
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to import pipeline tools: {e}")
        return {}


_pipeline_tools = _import_pipeline_tools()
register_pipeline_tools = _pipeline_tools.get("register_pipeline_tools")
transcribe_url_to_vectorstore = _pipeline_tools.get("transcribe_url_to_vectorstore")
transcribe_and_index = _pipeline_tools.get("transcribe_and_index")
PipelineManager = _pipeline_tools.get("PipelineManager")
PipelineConfig = _pipeline_tools.get("PipelineConfig")
SecurityValidator = _pipeline_tools.get("SecurityValidator")

# ============================================================================
# REGISTRAR & DATABASE TOOLS
# ============================================================================
from .registrar import ToolsRegistrar


# Optional database tools (lazy import with error handling)
def _import_db_tools():
    """Safely import optional database tools."""
    try:
        from .dbtools import (
            register_db_tools,
            DatabaseManager,
            DBTOOLS_AVAILABLE,
        )
        return {
            "register_db_tools": register_db_tools,
            "DatabaseManager": DatabaseManager,
            "DBTOOLS_AVAILABLE": DBTOOLS_AVAILABLE,
        }
    except Exception as e:
        logger.warning(f"ℹ️ Database tools not available: {e}")
        return {
            "register_db_tools": None,
            "DatabaseManager": None,
            "DBTOOLS_AVAILABLE": False,
        }


_db_tools = _import_db_tools()
register_db_tools = _db_tools.get("register_db_tools")
DatabaseManager = _db_tools.get("DatabaseManager")
DBTOOLS_AVAILABLE = _db_tools.get("DBTOOLS_AVAILABLE", False)

# ============================================================================
# PUBLIC API - COMPLETE EXPORT LIST
# ============================================================================
__all__ = [
    # Core utilities
    "DEFAULT_DOWNLOAD_DIR",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_LOG_DIR",
    "cleanup_downloads",
    "calculate_file_hash",
    "validate_url",
    "sanitize_filename",
    "generate_safe_filename",
    "SessionConfig",
    "SecurityLevel",
    "RetryStrategy",
    "DownloadProgress",
    "DownloadConfig",
    "DownloadResult",          # Rich version from download.py (Bug #6 fix)
    "UtilsDownloadResult",     # Original utils.py version (still accessible)
    "CleanupResult",
    # HTTP tools
    "http_fetch",
    "http_fetch_batch",
    "HttpConfig",
    "HttpMethod",
    "HttpResponse",
    "RateLimiter",
    "CircuitBreaker",
    "CircuitState",
    "ContentType",
    "HttpError",
    "CircuitBreakerOpenError",
    "RateLimitExceededError",
    "SecurityViolationError",
    "ResponseTooLargeError",
    # Link search tools
    "search_links",
    "search_with_dorks",
    "find_file_links",
    "find_hidden_resources",
    "extract_links_from_page",
    "expand_links",
    "LinkSearchConfig",
    "get_available_search_engines",
    "create_link_search_config",
    # Download tools
    "download_file",
    "download_multiple",
    "get_download_stats",
    "validate_download_result",
    "download_health_check",
    "DownloadManager",
    "DownloadOptions",
    # Scraping tools
    "scrape_page",
    "scrape_batch",
    "scrape_async",
    "scrape_batch_async",
    "parse_feed",
    "browser_scrape",
    "get_scraping_stats",
    "clear_scraping_cache",
    "get_scraping_backends",
    "rss_fetch",
    "google_news_search",
    "scrape_multiple",
    "ScrapingConfig",
    "ScrapingEngine",
    "DependencyManager",
    "ContentExtractor",
    # Extraction tools
    "extract_text_from_file",
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "extract_text_from_spreadsheet",
    "extract_text_from_textfile",
    "extract_and_chunk_file",
    "chunk_text",
    "batch_extract_files",
    "get_extraction_backends",
    "ExtractionConfig",
    # Video tools
    "download_video",
    "download_multiple_videos",
    "extract_video_thumbnail",
    "video_info",
    "get_video_backends",
    "VideoProcessor",
    "VideoConfig",
    "VideoPlatform",
    "VideoQuality",
    # Transcription tools
    "transcribe_audio_or_video",
    "transcribe_multiple",
    "get_transcription_backends",
    "clear_model_cache",
    "get_supported_transcription_formats",
    "TranscriptionConfig",
    # Pipeline tools
    "PipelineManager",
    "PipelineConfig",
    "SecurityValidator",
    "transcribe_url_to_vectorstore",
    "transcribe_and_index",
    "register_pipeline_tools",
    # Registrar
    "ToolsRegistrar",
    # Database (optional)
    "DatabaseManager",
    "register_db_tools",
    "DBTOOLS_AVAILABLE",
]


# ============================================================================
# INITIALIZATION & HEALTH CHECK
# ============================================================================
def get_tools_status() -> Dict[str, Any]:
    """Get status of all available tools."""
    return {
        "scraping": bool(scrape_page),
        "extraction": bool(extract_text_from_file),
        "video": bool(download_video),
        "transcription": bool(transcribe_audio_or_video),
        "pipeline": bool(PipelineManager),
        "database": DBTOOLS_AVAILABLE,
        "http": True,
        "download": True,
        "link_search": bool(search_links),
        "timestamp": datetime.now().isoformat() if hasattr(
            sys.modules.get("datetime"), "datetime"
        ) else None,
    }


def initialize_tools(config_path: Optional[str] = None) -> bool:
    """Initialize all tools with optional configuration."""
    logger.info("🔧 Initializing AMB-Sentinel Tools Package...")

    try:
        status = get_tools_status()

        # Check critical tools
        critical_available = status["http"] and status["download"]

        if critical_available:
            logger.info(f"✅ Tools initialized successfully: {status}")
            return True
        else:
            logger.error(f"❌ Critical tools missing: {status}")
            return False

    except Exception as e:
        logger.error(f"❌ Tools initialization failed: {e}")
        return False


# Auto-initialize on import
if __name__ != "__main__":
    try:
        initialize_tools()
    except Exception as e:
        logger.warning(f"⚠️ Auto-initialization warning: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION ADDITIONS v5.0 — Aggregate health + lazy new exports
# ─────────────────────────────────────────────────────────────────────────────

def get_full_system_health() -> dict:
    """Aggregate health report from all tool modules.
    
    Returns a single dict with health status, metrics, and error counts
    from every module. Safe to call at any time — each sub-call is
    independently try/except guarded.
    """
    health: dict = {"ok": True, "modules": {}, "ts": __import__("time").time()}

    def _safe(name, fn):
        try:
            health["modules"][name] = fn()
        except Exception as e:
            health["modules"][name] = {"error": str(e)}
            health["ok"] = False

    try:
        from .utils import get_response_cache, get_domain_delay_stats
        _safe("utils_response_cache", get_response_cache().stats)
        _safe("utils_domain_delays", get_domain_delay_stats)
    except Exception:
        pass

    try:
        from .http import get_http_metrics
        _safe("http_metrics", get_http_metrics)
    except Exception:
        pass

    try:
        from .dbtools import DatabaseManager
        dm = DatabaseManager()
        _safe("db_cache", dm._query_cache.stats)
    except Exception:
        pass

    try:
        from .scrape import get_scrape_health, get_content_deduplicator
        _safe("scrape_health", get_scrape_health)
        _safe("scrape_dedup", get_content_deduplicator().stats)
    except Exception:
        pass

    try:
        from .pipeline import get_stage_health
        _safe("pipeline_stage_health", get_stage_health)
    except Exception:
        pass

    try:
        from .find_links import get_engine_scores
        _safe("engine_scores", get_engine_scores)
    except Exception:
        pass

    try:
        from .download import get_download_metrics
        _safe("download_metrics", get_download_metrics)
    except Exception:
        pass

    try:
        # model_manager is at root, not under tools/
        from amb_sentinel.llm.model_manager import get_telemetry_summary
        _safe("llm_telemetry", get_telemetry_summary)
    except Exception:
        pass

    try:
        from amb_sentinel.llm.model_manager import get_backend_orchestrator
        orch = get_backend_orchestrator()
        _safe("backend_status", orch.status_report)
    except Exception:
        pass

    return health


# Lazy-load new production exports (won't break existing imports)
try:
    from .utils import _ResponseCache, AdaptiveDelayMixin, rotate_session_headers, get_domain_delay_stats
except ImportError:
    pass

try:
    from .http import _InFlightDeduplicator, HttpMetrics, get_http_metrics
except ImportError:
    pass

try:
    from .find_links import normalize_url, rank_links, get_engine_scores
except ImportError:
    pass

try:
    from .pipeline import PipelineCheckpoint, PipelineProgressBus, StageHealthTracker, get_pipeline_progress_bus, get_stage_health
except ImportError:
    pass

try:
    from .scrape import ContentDeduplicator, ScrapeHealthMonitor, get_content_deduplicator, get_scrape_health
except ImportError:
    pass

try:
    from .download import DownloadPriorityQueue, DownloadMetrics, get_download_queue, get_download_metrics
except ImportError:
    pass

try:
    from .extract import ExtractionQualityScorer, SmartSentenceChunker, score_extraction_quality, smart_chunk_text
except ImportError:
    pass

# model_manager lives at root (alongside ai_agent.py), NOT under tools/
# Import from root package to avoid duplicate singleton issues
try:
    import sys as _sys
    import os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from amb_sentinel.llm.model_manager import (
        AdaptiveTokenBudget, GenerationTelemetry,
        get_generation_telemetry, get_telemetry_summary,
        BackendOrchestrator, BackendType, get_backend_orchestrator,
        AdaptiveModelManager, get_model_manager,
    )
except ImportError:
    pass

try:
    from .dbtools import _mask_uri, _QueryCache, _QueryMetrics
except ImportError:
    pass
