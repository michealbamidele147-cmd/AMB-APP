# tools/registrar.py
"""
Production-Ready Tool Registration System for AMB-Sentinel AI Agent

This module provides comprehensive tool registration with:
- FULL function coverage from all tools modules
- Deep LLM integration capabilities
- Compatibility with ai_agent.py ToolRegistry
- Proper tool counting and metrics
- Rich parameter metadata for LLM understanding
- Security and validation
- Production-grade error handling

Security Level: HIGH

Author: AMB Team
copyright 2025

"""

import inspect
import difflib
import logging
import time
import concurrent.futures
from functools import wraps
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import importlib
import threading

# ============================================================================
# LINK SEARCH TOOLS - COMPLETE IMPORT AND REGISTRATION
# ============================================================================
from .find_links import (
    search_links,
    search_with_dorks,
    expand_links,
    extract_links_from_page,
    find_file_links,
    find_hidden_resources,
    LinkSearchConfig,
    _ENGINE_HANDLERS
)

# ============================================================================
# CORE IMPORTS
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

from .download import (
    download_file,
    download_multiple,
    get_download_stats,
    validate_download_result,
    health_check as download_health_check,
    DownloadManager,
    DownloadOptions,
    DownloadResult,
)

from .utils import (
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
    CleanupResult,
    DEFAULT_DOWNLOAD_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_LOG_DIR,
)

logger = logging.getLogger("Registrar")


# ============================================================================
# SAFE IMPORT UTILITIES - Enhanced
# ============================================================================
def _safe_import(module_path: str, items: list, module_name: str = None):
    """Safely import module items with comprehensive error handling."""
    try:
        if module_path.startswith('.'):
            package_name = __package__ or 'tools'
            abs_module = package_name + '.' + module_path.lstrip('.')
            module = importlib.import_module(abs_module)
        else:
            module = importlib.import_module(module_path)

        imported = {}
        for item in items:
            try:
                imported[item] = getattr(module, item, None)
            except Exception:
                imported[item] = None
                logger.warning(f"Item '{item}' not found in {module_name or module_path}")
        return imported

    except ImportError as e:
        logger.error(f"ℹ️ Could not import {module_name or module_path}: {e}")
        return {item: None for item in items}
    except Exception as e:
        logger.error(f"❌ Unexpected error importing {module_name or module_path}: {e}")
        return {item: None for item in items}


# ============================================================================
# SCRAPING TOOLS - COMPLETE IMPORT AND REGISTRATION
# ============================================================================
_scrape_items = _safe_import(
    '.scrape',
    [
        'scrape_page',
        'fetch_article_text',
        'scrape_batch',
        'scrape_async',
        'scrape_batch_async',
        'parse_feed',
        'parse_feeds_batch',
        'browser_scrape',
        'clear_scraping_cache',
        'get_scraping_stats',
        'ScrapingConfig',
        'ScrapingEngine',
        'ContentCategory',
        'RobotsPolicy',
        'AntiDetectionLevel',
        'DependencyManager',
        'ContentExtractor',
        'ScrapingResult',
        'RobotsTxtChecker',
        'URLValidator',
    ],
    'scrape module'
)

scrape_page = _scrape_items.get('scrape_page')
fetch_article_text = _scrape_items.get('fetch_article_text')
scrape_batch = _scrape_items.get('scrape_batch')
scrape_async = _scrape_items.get('scrape_async')
scrape_batch_async = _scrape_items.get('scrape_batch_async')
parse_feed        = _scrape_items.get('parse_feed')
parse_feeds_batch = _scrape_items.get('parse_feeds_batch')
browser_scrape = _scrape_items.get('browser_scrape')
clear_scraping_cache = _scrape_items.get('clear_scraping_cache')
get_scraping_stats = _scrape_items.get('get_scraping_stats')
ScrapingConfig = _scrape_items.get('ScrapingConfig')
ScrapingEngine = _scrape_items.get('ScrapingEngine')
ContentCategory = _scrape_items.get('ContentCategory')
RobotsPolicy = _scrape_items.get('RobotsPolicy')
AntiDetectionLevel = _scrape_items.get('AntiDetectionLevel')
DependencyManager = _scrape_items.get('DependencyManager')
ContentExtractor = _scrape_items.get('ContentExtractor')
ScrapingResult = _scrape_items.get('ScrapingResult')
RobotsTxtChecker = _scrape_items.get('RobotsTxtChecker')
URLValidator = _scrape_items.get('URLValidator')


# ============================================================================
# EXTRACTION TOOLS - COMPLETE IMPORT AND REGISTRATION
# ============================================================================
_extract_items = _safe_import(
    '.extract',
    [
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
    ],
    'extract module'
)

extract_text_from_file = _extract_items.get('extract_text_from_file')
extract_text_from_pdf = _extract_items.get('extract_text_from_pdf')
extract_text_from_docx = _extract_items.get('extract_text_from_docx')
extract_text_from_spreadsheet = _extract_items.get('extract_text_from_spreadsheet')
extract_text_from_textfile = _extract_items.get('extract_text_from_textfile')
extract_and_chunk_file = _extract_items.get('extract_and_chunk_file')
chunk_text = _extract_items.get('chunk_text')
batch_extract_files = _extract_items.get('batch_extract_files')
extract_with_ocr = _extract_items.get('extract_with_ocr')
process_file_for_vector_store = _extract_items.get('process_file_for_vector_store')
get_extraction_backends = _extract_items.get('get_extraction_backends')
ExtractionConfig = _extract_items.get('ExtractionConfig')
ExtractionEngine = _extract_items.get('ExtractionEngine')
ExtractionResult = _extract_items.get('ExtractionResult')
ChunkingStrategy = _extract_items.get('ChunkingStrategy')
OCREngine = _extract_items.get('OCREngine')

# ============================================================================
# VIDEO TOOLS - COMPLETE IMPORT AND REGISTRATION
# ============================================================================
_video_items = _safe_import(
    '.video',
    [
        'video_info',
        'download_video',
        'download_multiple_videos',
        'extract_video_thumbnail',
        'get_video_backends',
        'VideoProcessor',
        'VideoConfig',
        'VideoPlatform',
        'VideoQuality',
        'ThumbnailStrategy',
        'VideoFormat',
        'DownloadStatus',
        'VideoMetadata',
        'VideoResult',
        'VideoBackend',
        'YtDlpBackend',
        'GenericHttpBackend',
        'FfmpegBackend',
    ],
    'video module'
)

video_info = _video_items.get('video_info')
download_video = _video_items.get('download_video')
download_multiple_videos = _video_items.get('download_multiple_videos')
extract_video_thumbnail = _video_items.get('extract_video_thumbnail')
get_video_backends = _video_items.get('get_video_backends')
VideoProcessor = _video_items.get('VideoProcessor')
VideoConfig = _video_items.get('VideoConfig')
VideoPlatform = _video_items.get('VideoPlatform')
VideoQuality = _video_items.get('VideoQuality')
ThumbnailStrategy = _video_items.get('ThumbnailStrategy')
VideoFormat = _video_items.get('VideoFormat')
DownloadStatus = _video_items.get('DownloadStatus')
VideoMetadata = _video_items.get('VideoMetadata')
VideoResult = _video_items.get('VideoResult')
VideoBackend = _video_items.get('VideoBackend')
YtDlpBackend = _video_items.get('YtDlpBackend')
GenericHttpBackend = _video_items.get('GenericHttpBackend')
FfmpegBackend = _video_items.get('FfmpegBackend')

# ============================================================================
# TRANSCRIPTION TOOLS - COMPLETE IMPORT AND REGISTRATION
# ============================================================================
_transcribe_items = _safe_import(
    '.transcribe',
    [
        'transcribe_audio_or_video',
        'transcribe_multiple',
        'get_transcription_backends',
        'clear_model_cache',
        'get_supported_formats',
        'TranscriptionConfig',
        'TranscriptionResult',
        'TranscriptionBackend',
        'ComputeType',
        'TranscriptionStatus',
        'TranscriptionMetrics',
        'TranscriptionBatchProcessor',
        'ResourceMonitor',
        'CircuitBreaker',
    ],
    'transcribe module'
)

transcribe_audio_or_video = _transcribe_items.get('transcribe_audio_or_video')
transcribe_multiple = _transcribe_items.get('transcribe_multiple')
get_transcription_backends = _transcribe_items.get('get_transcription_backends')
clear_model_cache = _transcribe_items.get('clear_model_cache')
get_supported_formats = _transcribe_items.get('get_supported_formats')
TranscriptionConfig = _transcribe_items.get('TranscriptionConfig')
TranscriptionResult = _transcribe_items.get('TranscriptionResult')
TranscriptionBackend = _transcribe_items.get('TranscriptionBackend')
ComputeType = _transcribe_items.get('ComputeType')
TranscriptionStatus = _transcribe_items.get('TranscriptionStatus')
TranscriptionMetrics = _transcribe_items.get('TranscriptionMetrics')
TranscriptionBatchProcessor = _transcribe_items.get('TranscriptionBatchProcessor')
ResourceMonitor = _transcribe_items.get('ResourceMonitor')
TranscriptionCircuitBreaker = _transcribe_items.get('CircuitBreaker')

# ============================================================================
# PIPELINE TOOLS - COMPLETE IMPORT AND REGISTRATION
# ============================================================================
_pipeline_items = _safe_import(
    '.pipeline',
    [
        'register_pipeline_tools',
        'transcribe_url_to_vectorstore',
        'transcribe_and_index',
        'PipelineManager',
        'PipelineConfig',
        'ProcessingStage',
        'SourceType',
        'ProcessingStatus',
        'ProcessingError',
        'ProcessingResult',
        'PipelineMetrics',
        'SecurityValidator',
        'DownloadProcessor',
        'TranscriptionProcessor',
        'IndexingProcessor',
    ],
    'pipeline module'
)

register_pipeline_tools = _pipeline_items.get('register_pipeline_tools')
transcribe_url_to_vectorstore = _pipeline_items.get('transcribe_url_to_vectorstore')
transcribe_and_index = _pipeline_items.get('transcribe_and_index')
PipelineManager = _pipeline_items.get('PipelineManager')
PipelineConfig = _pipeline_items.get('PipelineConfig')
ProcessingStage = _pipeline_items.get('ProcessingStage')
SourceType = _pipeline_items.get('SourceType')
ProcessingStatus = _pipeline_items.get('ProcessingStatus')
ProcessingError = _pipeline_items.get('ProcessingError')
ProcessingResult = _pipeline_items.get('ProcessingResult')
PipelineMetrics = _pipeline_items.get('PipelineMetrics')
SecurityValidator = _pipeline_items.get('SecurityValidator')
DownloadProcessor = _pipeline_items.get('DownloadProcessor')
TranscriptionProcessor = _pipeline_items.get('TranscriptionProcessor')
IndexingProcessor = _pipeline_items.get('IndexingProcessor')

# ============================================================================
# DATABASE TOOLS - OPTIONAL COMPLETE IMPORT
# ============================================================================
_db_items = _safe_import(
    '.dbtools',
    [
        'register_db_tools',
        'DatabaseManager',
        'DatabaseConnectionError',
        'PSYCOPG2_AVAILABLE',
        'SQLALCHEMY_AVAILABLE',
        'MYSQL_AVAILABLE',
        'ORACLE_AVAILABLE',
        'DB2_AVAILABLE',
    ],
    'dbtools module'
)

register_db_tools = _db_items.get('register_db_tools')
DatabaseManager = _db_items.get('DatabaseManager')
DatabaseConnectionError = _db_items.get('DatabaseConnectionError')
PSYCOPG2_AVAILABLE = _db_items.get('PSYCOPG2_AVAILABLE', False)
SQLALCHEMY_AVAILABLE = _db_items.get('SQLALCHEMY_AVAILABLE', False)
MYSQL_AVAILABLE = _db_items.get('MYSQL_AVAILABLE', False)
ORACLE_AVAILABLE = _db_items.get('ORACLE_AVAILABLE', False)
DB2_AVAILABLE = _db_items.get('DB2_AVAILABLE', False)
DBTOOLS_AVAILABLE = any([PSYCOPG2_AVAILABLE, SQLALCHEMY_AVAILABLE, MYSQL_AVAILABLE,
                         ORACLE_AVAILABLE, DB2_AVAILABLE])


# ============================================================================
# SIGNATURE-AWARE ARGUMENT ALIAS RESOLUTION
# ============================================================================
# Curated per-tool alias table: maps the kwarg name the LLM may send to the
# real parameter name the underlying function expects.
#
# Format:  "tool_name": {"alias_sent_by_llm": "real_param_in_function", ...}
#
# These entries take PRIORITY over the automatic fuzzy-match resolver below.
# Add entries here whenever a new tool has a known LLM-hallucination pattern.
TOOL_PARAM_ALIASES: Dict[str, Dict[str, str]] = {
    # ── FEED PARSING ─────────────────────────────────────────────────────────
    # [BUG-FIX] LLM sends {"feeds": ["url1","url2"]} but parse_feed needs url=str
    "parse_feed": {
        "feeds":          "url",  # <-- THE bug from production log
        "feed_url":       "url",
        "feed":           "url",
        "rss_url":        "url",
        "rss":            "url",
        "atom_url":       "url",
        "source":         "url",
        "source_url":     "url",
        "link":           "url",
        "uri":            "url",
        "limit":          "max_entries",
        "count":          "max_entries",
        "max":            "max_entries",
        "num_entries":    "max_entries",
        "n":              "max_entries",
        "num":            "max_entries",
    },

    # ── WEB SEARCH ───────────────────────────────────────────────────────────
    # [FIX] LLM confuses search_links (search engine query tool) with URL
    # processing tools. It sometimes sends {"urls": [...]} or {"url": "..."}
    # thinking it should pass article URLs to be "searched". These are remapped
    # to "query" with URL→keyword coercion in the TYPE-COERCION pass below.
    "search_links": {
        "keywords":       "query",
        "search_query":   "query",
        "q":              "query",
        "term":           "query",
        "search_term":    "query",
        "search":         "query",
        "terms":          "query",
        "topic":          "query",
        "subject":        "query",
        # [FIX] LLM sends URLs thinking search_links fetches them - extract slug
        "url":            "query",
        "urls":           "query",
        "links":          "query",
        "href":           "query",
        "address":        "query",
        "source":         "query",
        "endpoint":       "query",
        "limit":          "max_results",
        "count":          "max_results",
        "num":            "max_results",
        "n":              "max_results",
        "max":            "max_results",
        "results":        "max_results",
    },

    # ── DORK SEARCH ──────────────────────────────────────────────────────────
    "search_with_dorks": {
        "query":          "dork_query",
        "search_query":   "dork_query",
        "q":              "dork_query",
        "queries":        "dork_query",
        "search":         "dork_query",
        "keywords":       "dork_query",
        "term":           "dork_query",
        "topic":          "dork_query",
        "subject":        "dork_query",
        "limit":          "max_results",
        "count":          "max_results",
        "num":            "max_results",
        "max":            "max_results",
    },

    # ── HTTP FETCH (single) ──────────────────────────────────────────────────
    "http_fetch": {
        "uri":            "url",
        "link":           "url",
        "endpoint":       "url",
        "address":        "url",
        "source":         "url",
        "target":         "url",
        "resource":       "url",
        "page":           "url",
        "webpage":        "url",
        "href":           "url",
    },

    # ── HTTP FETCH BATCH (multiple) ──────────────────────────────────────────
    "http_fetch_batch": {
        "url":            "urls",
        "uri":            "urls",
        "link":           "urls",
        "links":          "urls",
        "addresses":      "urls",
        "targets":        "urls",
        "resources":      "urls",
        "endpoints":      "urls",
        "pages":          "urls",
    },

    # ── SCRAPING ─────────────────────────────────────────────────────────────
    "scrape_page": {
        "uri":            "url",
        "link":           "url",
        "page":           "url",
        "page_url":       "url",
        "source":         "url",
        "address":        "url",
        "target":         "url",
        "webpage":        "url",
        "href":           "url",
        "endpoint":       "url",
    },
    "scrape_batch": {
        "url":            "urls",
        "uri":            "urls",
        "link":           "urls",
        "links":          "urls",
        "pages":          "urls",
        "addresses":      "urls",
        "targets":        "urls",
        "webpages":       "urls",
    },
    "scrape_async": {
        "uri":            "url",
        "link":           "url",
        "page":           "url",
        "source":         "url",
        "target":         "url",
        "href":           "url",
    },
    "scrape_batch_async": {
        "url":            "urls",
        "link":           "urls",
        "links":          "urls",
        "pages":          "urls",
    },
    "browser_scrape": {
        "uri":            "url",
        "link":           "url",
        "page":           "url",
        "source":         "url",
        "href":           "url",
    },

    # ── LINK FINDING ─────────────────────────────────────────────────────────
    "find_file_links": {
        "query":          "site",
        "url":            "site",
        "domain":         "site",
        "website":        "site",
        "base_url":       "site",
        "source":         "site",
        "host":           "site",
        "extensions":     "filetypes",
        "types":          "filetypes",
        "file_types":     "filetypes",
        "formats":        "filetypes",
        "ext":            "filetypes",
        "limit":          "max_results",
        "count":          "max_results",
        "num":            "max_results",
        "max":            "max_results",
    },
    "find_hidden_resources": {
        "url":            "base_url",
        "site":           "base_url",
        "domain":         "base_url",
        "source":         "base_url",
        "target":         "base_url",
        "host":           "base_url",
        "limit":          "max_results",
        "count":          "max_results",
        "num":            "max_results",
    },
    "expand_links": {
        "url":            "seed_urls",
        "urls":           "seed_urls",
        "seed_url":       "seed_urls",
        "seeds":          "seed_urls",
        "starting_urls":  "seed_urls",
        "start_urls":     "seed_urls",
        "links":          "seed_urls",
        "sources":        "seed_urls",
        "max":            "max_total",
        "limit":          "max_total",
        "count":          "max_total",
        "num":            "max_total",
    },
    "extract_links_from_page": {
        "uri":            "url",
        "link":           "url",
        "page_url":       "url",
        "source":         "url",
        "webpage":        "url",
        "href":           "url",
        "limit":          "max_links",
        "count":          "max_links",
        "max":            "max_links",
    },

    # ── URL VALIDATION ───────────────────────────────────────────────────────
    "validate_url": {
        "uri":            "url",
        "link":           "url",
        "address":        "url",
        "endpoint":       "url",
        "source":         "url",
        "resource":       "url",
        "href":           "url",
        "webpage":        "url",
    },

    # ── DOWNLOAD ─────────────────────────────────────────────────────────────
    "download_file": {
        "uri":            "url",
        "link":           "url",
        "source":         "url",
        "source_url":     "url",
        "target":         "url",
        "href":           "url",
        "resource":       "url",
        "dest":           "dest_dir",
        "directory":      "dest_dir",
        "folder":         "dest_dir",
        "output_dir":     "dest_dir",
        "output_path":    "dest_dir",
        "save_to":        "dest_dir",
        "download_dir":   "dest_dir",
        "output":         "dest_dir",
        "name":           "filename",
        "file_name":      "filename",
        "output_file":    "filename",
        "fname":          "filename",
    },
    "download_multiple": {
        "url":            "urls",
        "link":           "urls",
        "links":          "urls",
        "sources":        "urls",
        "targets":        "urls",
        "files":          "urls",
        "resources":      "urls",
    },

    # ── TEXT EXTRACTION ──────────────────────────────────────────────────────
    "extract_text_from_file": {
        "file_path":      "path",
        "filepath":       "path",
        "file":           "path",
        "filename":       "path",
        "input":          "path",
        "input_path":     "path",
        "document":       "path",
        "source":         "path",
        "pdf_path":       "path",
        "docx_path":      "path",
    },
    "extract_text_from_pdf": {
        "pdf_path":       "path",
        "file_path":      "path",
        "filepath":       "path",
        "file":           "path",
        "document":       "path",
        "pdf":            "path",
        "input":          "path",
        "input_path":     "path",
        "source":         "path",
    },
    "extract_text_from_docx": {
        "docx_path":      "path",
        "file_path":      "path",
        "filepath":       "path",
        "file":           "path",
        "document":       "path",
        "word_doc":       "path",
        "input":          "path",
        "input_path":     "path",
        "source":         "path",
    },
    "extract_text_from_spreadsheet": {
        "spreadsheet_path": "path",
        "file_path":      "path",
        "filepath":       "path",
        "file":           "path",
        "excel":          "path",
        "csv":            "path",
        "input":          "path",
        "input_path":     "path",
        "source":         "path",
    },
    "extract_text_from_textfile": {
        "file_path":      "path",
        "filepath":       "path",
        "file":           "path",
        "txt":            "path",
        "input":          "path",
        "input_path":     "path",
        "source":         "path",
    },
    "extract_and_chunk_file": {
        "file_path":      "path",
        "filepath":       "path",
        "file":           "path",
        "filename":       "path",
        "input":          "path",
        "input_path":     "path",
        "document":       "path",
        "source":         "path",
        "chunk_method":   "chunking_strategy",
        "strategy":       "chunking_strategy",
        "method":         "chunking_strategy",
        "size":           "chunk_size",
        "overlap":        "chunk_overlap",
    },
    "batch_extract_files": {
        "files":          "file_paths",
        "paths":          "file_paths",
        "documents":      "file_paths",
        "inputs":         "file_paths",
        "sources":        "file_paths",
    },

    # ── TEXT CHUNKING ────────────────────────────────────────────────────────
    "chunk_text": {
        "content":        "text",
        "body":           "text",
        "document":       "text",
        "input":          "text",
        "raw_text":       "text",
        "string":         "text",
        "passage":        "text",
        "size":           "chunk_size",
        "overlap":        "chunk_overlap",
        "method":         "strategy",
        "approach":       "strategy",
        "mode":           "strategy",
    },

    # ── VIDEO ────────────────────────────────────────────────────────────────
    "download_video": {
        "uri":            "url",
        "link":           "url",
        "video_url":      "url",
        "source":         "url",
        "href":           "url",
    },
    "download_multiple_videos": {
        "url":            "urls",
        "link":           "urls",
        "links":          "urls",
        "video_urls":     "urls",
        "sources":        "urls",
        "videos":         "urls",
        "dest":           "dest_dir",
        "directory":      "dest_dir",
        "folder":         "dest_dir",
        "output":         "dest_dir",
        "output_dir":     "dest_dir",
    },
    "video_info": {
        "uri":            "url",
        "link":           "url",
        "video_url":      "url",
        "source":         "url",
        "href":           "url",
    },
    "extract_video_thumbnail": {
        "video":          "video_path",
        "input":          "video_path",
        "file_path":      "video_path",
        "filepath":       "video_path",
        "path":           "video_path",
        "source":         "video_path",
        "output":         "output_path",
        "dest":           "output_path",
        "thumbnail":      "output_path",
        "output_file":    "output_path",
    },

    # ── TRANSCRIPTION ────────────────────────────────────────────────────────
    "transcribe_audio_or_video": {
        "file_path":      "path",
        "filepath":       "path",
        "file":           "path",
        "audio":          "path",
        "video":          "path",
        "input":          "path",
        "input_path":     "path",
        "source":         "path",
        "lang":           "language",
        "language_code":  "language",
        "locale":         "language",
        "model":          "model_name",
        "model_size":     "model_name",
        "backend_name":   "backend",
        "engine":         "backend",
    },
    "transcribe_multiple": {
        "files":          "paths",
        "inputs":         "paths",
        "file_paths":     "paths",
        "audios":         "paths",
        "videos":         "paths",
        "sources":        "paths",
    },

    # ── FEED BATCH ──────────────────────────────────────────────────────────
    "parse_feeds_batch": {
        "feed_urls":       "feeds",
        "urls":            "feeds",
        "url_list":        "feeds",
        "rss_feeds":       "feeds",
        "feed_list":       "feeds",
        "sources":         "feeds",
        "links":           "feeds",
        "limit":           "max_entries",
        "count":           "max_entries",
        "max":             "max_entries",
        "num":             "max_entries",
    },

    # ── OCR ─────────────────────────────────────────────────────────────────
    "extract_with_ocr": {
        "file":            "path",
        "file_path":       "path",
        "image":           "path",
        "image_path":      "path",
        "input":           "path",
        "input_path":      "path",
        "lang":            "language",
        "locale":          "language",
    },

    # ── VECTOR STORE PROCESSING ──────────────────────────────────────────────
    "process_file_for_vector_store": {
        "file":            "path",
        "file_path":       "path",
        "input":           "path",
        "input_path":      "path",
        "vectorstore":     "service",  # service = AIModelService ref
        "chunk":           "chunk_size",
        "size":            "chunk_size",
    },

    # ── DOWNLOAD CLEANUP ─────────────────────────────────────────────────────
    "cleanup_downloads": {
        "directory":       "max_age_hours",  # graceful: if string passed, drop
        "folder":          "max_age_hours",
        "dir":             "max_age_hours",
        "age":             "max_age_hours",
        "max_age":         "max_age_hours",
        "hours":           "max_age_hours",
        "max_size":        "max_total_size_gb",
        "size":            "max_total_size_gb",
        "size_gb":         "max_total_size_gb",
    },

    # ── SCRAPING CACHE ───────────────────────────────────────────────────────
    "clear_scraping_cache": {
        # Takes no parameters - any args just ignored via _signature_guard
    },

    # ── MODEL CACHE ──────────────────────────────────────────────────────────
    "clear_model_cache": {
        # Takes no parameters - any args just ignored via _signature_guard
    },

    # ── DOWNLOAD HEALTH ──────────────────────────────────────────────────────
    "download_health_check": {
        # Takes no parameters - any args just ignored via _signature_guard
    },

    # ── FILENAME / HASH UTILITIES ────────────────────────────────────────────
    "sanitize_filename": {
        # Real param is 'filename' — map all other terms to it
        "name":           "filename",
        "file_name":      "filename",
        "title":          "filename",
        "original":       "filename",
        "raw":            "filename",
        "input":          "filename",
        "value":          "filename",
        "text":           "filename",
        "string":         "filename",
    },
    "generate_safe_filename": {
        "uri":            "url",
        "link":           "url",
        "source":         "url",
        "source_url":     "url",
        "href":           "url",
        "mime":           "content_type",
        "type":           "content_type",
        "media_type":     "content_type",
        "mime_type":      "content_type",
    },
    "calculate_file_hash": {
        "file":           "file_path",
        "path":           "file_path",
        "filepath":       "file_path",
        "input":          "file_path",
        "source":         "file_path",
        "hash_algo":      "algorithm",
        "hash_type":      "algorithm",
        "algo":           "algorithm",
        "type":           "algorithm",
    },
    "validate_download_result": {
        "download":       "result",
        "response":       "result",
        "download_result": "result",
        "output":         "result",
    },
}


def _build_param_alias_resolver(
    func: Callable,
    tool_name: str,
    declared_schema: Optional[Dict[str, Any]]
) -> Dict[str, str]:
    """
    Build a {kwarg_name_from_llm: real_param_name} alias map for a tool.

    Resolution order (highest -> lowest priority):
      1. Curated TOOL_PARAM_ALIASES table  - precision, zero false positives
      2. Schema-vs-signature cross-map     - schema param not in sig -> find sig match
      3. difflib fuzzy match               - last resort, cutoff=0.70

    The returned dict is stored on the wrapper closure so alias resolution
    happens at near-zero cost every call (single dict lookup).
    """
    alias_map: Dict[str, str] = {}

    # Layer 1: curated table
    curated = TOOL_PARAM_ALIASES.get(tool_name, {})
    alias_map.update(curated)

    # Layers 2 & 3: automatic inspection
    try:
        sig = inspect.signature(func)
        real_params: set = set(sig.parameters.keys())

        if declared_schema and isinstance(declared_schema, dict):
            for schema_param in declared_schema.keys():
                # Already have a curated alias or param exists in real sig -> skip
                if schema_param in alias_map or schema_param in real_params:
                    continue

                # Layer 2: substring containment (e.g. "query" <-> "dork_query")
                candidates = [
                    p for p in real_params
                    if schema_param in p or p in schema_param
                ]
                if len(candidates) == 1:
                    alias_map[schema_param] = candidates[0]
                    logger.debug(
                        f"[AliasResolver] {tool_name}: '{schema_param}' -> "
                        f"'{candidates[0]}' (substring match)"
                    )
                    continue

                # Layer 3: difflib fuzzy match
                matches = difflib.get_close_matches(
                    schema_param, real_params, n=1, cutoff=0.70
                )
                if matches:
                    alias_map[schema_param] = matches[0]
                    logger.debug(
                        f"[AliasResolver] {tool_name}: '{schema_param}' -> "
                        f"'{matches[0]}' (fuzzy match)"
                    )

    except (ValueError, TypeError):
        pass
    except Exception as e:
        logger.debug(f"[AliasResolver] {tool_name}: introspection error: {e}")

    if alias_map:
        logger.debug(f"[AliasResolver] {tool_name} alias map built: {alias_map}")

    return alias_map


def _remap_kwargs(
    kwargs: Dict[str, Any],
    alias_map: Dict[str, str],
    tool_name: str
) -> Dict[str, Any]:
    """
    Apply alias_map to kwargs, translating LLM-sent names to real param names.

    Rules:
    - Only remap a kwarg if its name is in alias_map AND the real param name
      is NOT already present in kwargs (avoid clobbering an explicit correct arg).
    - Log a WARNING whenever a remap fires so LLM drift is trackable.

    [FIX-5] Added type-coercion pass after remapping:
    - If the real parameter is ``dork_query`` (or any string param) but the
      LLM sent a list (e.g. ``queries=["insecurity ...", "kidnapping ..."]``),
      join the list with \" OR \" to produce a valid query string.
    """
    if not alias_map:
        return kwargs

    remapped: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in alias_map:
            real_key = alias_map[k]
            if real_key not in kwargs:
                logger.warning(
                    f"[AliasResolver] Tool '{tool_name}': remapping kwarg "
                    f"'{k}' -> '{real_key}' (LLM used wrong parameter name)"
                )
                remapped[real_key] = v
            else:
                # Both alias and real key present; drop alias, keep real
                logger.debug(
                    f"[AliasResolver] Tool '{tool_name}': dropping alias "
                    f"'{k}' because '{real_key}' already present"
                )
        else:
            remapped[k] = v

    # ═══════════════════════════════════════════════════════════════════════
    # TYPE-COERCION PASS
    # The LLM sometimes sends the wrong type for a parameter.
    # Fix silently (with WARNING log) so the underlying function always
    # receives the correct Python type.
    # ═══════════════════════════════════════════════════════════════════════

    # URL params: list → take FIRST item
    # (e.g. url=["https://a.com","https://b.com"] → url="https://a.com")
    _URL_PARAMS = {"url", "uri", "link", "endpoint", "address", "source_url",
                   "feed_url", "feed", "page_url", "base_url", "video_url", "href"}

    # Query / text params: list → join with " OR "
    # (e.g. queries=["insecurity","kidnapping"] → dork_query="insecurity OR kidnapping")
    _QUERY_PARAMS = {"query", "dork_query", "search_query", "keywords", "text",
                     "content", "topic", "term", "search", "subject"}

    # File path params: list → take FIRST item
    _PATH_PARAMS = {"path", "file_path", "filepath", "pdf_path", "docx_path",
                    "spreadsheet_path", "video_path", "audio_path", "input_path"}

    # Name/string params: list → take FIRST item
    _STRING_PARAMS = {"name", "filename", "site", "domain", "host",
                      "language", "lang", "model_name", "model",
                      "algorithm", "strategy", "method", "backend"}

    # List params: single string → wrap in list
    _LIST_PARAMS = {"urls", "links", "paths", "file_paths", "seed_urls",
                    "sources", "targets", "files", "documents", "inputs",
                    "feeds", "video_urls", "audio_paths"}

    for param, val in list(remapped.items()):
        # ── list → single (URL / path / name params) ─────────────────────
        if isinstance(val, list) and val:
            if param in _URL_PARAMS or param in _PATH_PARAMS or param in _STRING_PARAMS:
                chosen = str(val[0]) if val[0] is not None else ""
                logger.warning(
                    f"[TypeCoerce] Tool '{tool_name}': param '{param}' received list "
                    f"({len(val)} items) — using first item: {chosen!r}. "
                    f"Use a batch tool to process all items."
                )
                remapped[param] = chosen
                continue

        # ── list → query string (query params — with smart URL-slug extraction) ─
        if isinstance(val, list) and param in _QUERY_PARAMS:
            import re as _re_coerce
            url_slugs, plain_terms = [], []
            for item in val:
                s = str(item).strip()
                if s.startswith(("http://", "https://")):
                    # Extract meaningful keywords from URL path
                    try:
                        from urllib.parse import urlparse as _up
                        path = _up(s).path.strip("/")
                        # Remove common boring path segments
                        path = _re_coerce.sub(r"\b(\d{4}|\d{2})\b", "", path)
                        slug = path.replace("-", " ").replace("/", " ").replace("_", " ")
                        slug = _re_coerce.sub(r"\.\w{2,4}$", "", slug)
                        slug = _re_coerce.sub(r"\s+", " ", slug).strip()
                        if slug and len(slug) > 4:
                            url_slugs.append(slug)
                    except Exception:
                        pass
                elif s:
                    plain_terms.append(s)
            parts = plain_terms + url_slugs
            joined = " OR ".join(p for p in parts if p) if parts else ""
            if not joined:
                joined = " OR ".join(str(x) for x in val if x)
            logger.warning(
                f"[TypeCoerce] Tool '{tool_name}': param '{param}' received list "
                f"— extracted query: {joined!r}"
            )
            remapped[param] = joined
            continue

        # ── single string → list (batch params) ─────────────────────────
        if isinstance(val, str) and param in _LIST_PARAMS:
            logger.warning(
                f"[TypeCoerce] Tool '{tool_name}': param '{param}' expects list "
                f"but received string — wrapping: [{val!r}]"
            )
            remapped[param] = [val]
            continue

        # ── string → int ──────────────────────────────────────────────────
        if isinstance(val, str) and param in {"max_results", "max_entries", "max_links",
                                               "max_total", "max_concurrent", "chunk_size",
                                               "chunk_overlap", "timeout", "depth",
                                               "max_pages", "limit", "count", "num", "n"}:
            try:
                remapped[param] = int(val)
                logger.debug(f"[TypeCoerce] Tool '{tool_name}': coerced '{param}' str→int: {val!r}")
            except (ValueError, TypeError):
                pass
            continue

        # ── string → bool ─────────────────────────────────────────────────
        if isinstance(val, str) and param in {"parse_entries", "respect_robots", "use_cache",
                                               "audio_only", "include_comments",
                                               "include_headers_footers", "include_headers",
                                               "preserve_layout", "extract_metadata",
                                               "detect_encoding", "detailed"}:
            remapped[param] = val.lower() not in ("false", "0", "no", "none", "")
            logger.debug(f"[TypeCoerce] Tool '{tool_name}': coerced '{param}' str→bool")
            continue

    return remapped


# ============================================================================
# MODULE-LEVEL EXECUTOR FOR RECOVERY TASKS (lazy initialized)
# ============================================================================
_recovery_executor = None
_recovery_lock = threading.Lock()

def _get_recovery_executor():
    global _recovery_executor
    with _recovery_lock:
        if _recovery_executor is None:
            _recovery_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="recovery"
            )
    return _recovery_executor

def _signature_guard(
    func: Callable,
    kwargs: Dict[str, Any],
    tool_name: str,
) -> Dict[str, Any]:
    """
    Drop any kwargs that are not in the function's signature.

    This is the universal last-resort defence against TypeErrors caused by
    the LLM sending unknown parameter names that were NOT caught by the
    alias resolver.  Instead of propagating a confusing TypeError that marks
    the goal as permanently stuck, we silently drop the unknown kwarg, log a
    WARNING, and let the call proceed with whatever valid kwargs remain.

    Skipped for functions that accept **kwargs (VAR_KEYWORD) since they
    would accept any parameter name anyway.

    [FIX] This is what prevents errors like:
        search_links() got an unexpected keyword argument 'urls'
    after alias remapping has already had its chance to handle it.
    """
    try:
        sig = inspect.signature(func)
        params = sig.parameters

        # If the function accepts **kwargs, nothing to filter
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs

        valid_params = set(params.keys())
        # Also exclude VAR_POSITIONAL (*args) from consideration
        valid_params -= {
            k for k, p in params.items()
            if p.kind == inspect.Parameter.VAR_POSITIONAL
        }

        unknown = {k for k in kwargs if k not in valid_params}
        if not unknown:
            return kwargs

        cleaned = {k: v for k, v in kwargs.items() if k in valid_params}
        logger.warning(
            f"[SignatureGuard] Tool '{tool_name}': dropped {len(unknown)} unknown "
            f"kwarg(s) {sorted(unknown)} — not in signature {sorted(valid_params)}. "
            f"Proceeding with valid kwargs: {sorted(cleaned.keys())}."
        )
        return cleaned
    except (ValueError, TypeError):
        # inspect.signature() can fail for some builtins — safe to skip
        return kwargs
    except Exception as e:
        logger.debug(f"[SignatureGuard] {tool_name}: introspection failed ({e}), skipping guard")
        return kwargs


def create_enhanced_tool_wrapper(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    param_info: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Create a production-grade wrapper for tool functions.

    Enhancements over the original:
    1. Signature-aware argument alias resolution - kwarg names sent by the LLM
       are transparently remapped to the real parameter names the underlying
       function expects.  This eliminates the class of errors:
         TypeError: <tool>() got an unexpected keyword argument '<alias>'
    2. Background context-recovery trigger on success (unchanged).
    3. Structured error logging with full exc_info (unchanged).

    The alias_map is built ONCE at wrapper creation time (O(1) per call)
    using inspect.signature() + TOOL_PARAM_ALIASES + difflib fuzzy matching.
    """
    tool_name = name or getattr(func, "__name__", str(func))

    # Build alias map once at registration time
    alias_map = _build_param_alias_resolver(func, tool_name, param_info)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Step 1: Remap aliased kwargs before forwarding to the real function
        resolved_kwargs = _remap_kwargs(kwargs, alias_map, tool_name)

        # Step 2: Signature guard — drop any kwargs still unknown after remapping.
        # This prevents TypeError: unexpected keyword argument from ever reaching
        # the underlying function (universal defence).
        resolved_kwargs = _signature_guard(func, resolved_kwargs, tool_name)

        try:
            # Pre-flight: verify required params present before calling
            try:
                _sig = inspect.signature(func)
                _missing = [
                    p for p, v in _sig.parameters.items()
                    if v.kind not in (inspect.Parameter.VAR_KEYWORD,
                                      inspect.Parameter.VAR_POSITIONAL)
                    and v.default is inspect.Parameter.empty
                    and p not in ("self", "cls")
                    and p not in resolved_kwargs
                ]
                if _missing:
                    logger.warning(
                        f"[Wrapper] '{tool_name}' still missing {_missing} after remapping. "
                        f"kwargs_in={list(kwargs.keys())}, resolved={list(resolved_kwargs.keys())}"
                    )
                    return {
                        "ok": False,
                        "error": f"Missing required argument(s): {_missing}",
                        "tool": tool_name,
                        "hint": f"{tool_name}({', '.join(_missing)}=<value>)",
                    }
            except Exception:
                pass  # Introspection failed – proceed
            result = func(*args, **resolved_kwargs)

            # Step 2: Background context-recovery trigger (non-blocking)
            try:
                from amb_sentinel.llm.model_manager import get_model_manager
                ctx_mgr = get_model_manager()

                def recovery_attempt():
                    try:
                        if hasattr(ctx_mgr, '_attempt_context_recovery_sync'):
                            ctx_mgr._attempt_context_recovery_sync()
                        else:
                            logger.debug("Context recovery skipped (sync method not available)")
                    except Exception as e:
                        logger.debug(f"Background recovery attempt failed: {e}")

                executor = _get_recovery_executor()
                executor.submit(recovery_attempt)
            except Exception as e:
                logger.debug(f"Recovery submission failed: {e}")

            return result

        except TypeError as e:
            # Escalation: if we still get TypeError after alias remapping,
            # log the actual signature so the developer knows exactly what went wrong.
            try:
                sig = inspect.signature(func)
                real_params = list(sig.parameters.keys())
            except Exception:
                real_params = ["<unable to inspect>"]

            logger.error(
                f"\u274c Tool {tool_name} failed with TypeError after alias remapping.\n"
                f"   Called with args={args} kwargs={kwargs}\n"
                f"   Resolved kwargs={resolved_kwargs}\n"
                f"   Function real params={real_params}\n"
                f"   Alias map used={alias_map}\n"
                f"   Error: {e}",
                exc_info=True
            )
            raise

        except Exception as e:
            logger.error(f"\u274c Tool {tool_name} failed: {e}", exc_info=True)
            raise

    wrapper._tool_metadata = {
        "name": tool_name,
        "description": description or func.__doc__ or "No description available",
        "parameters": param_info or {},
        "return_type": "Any",
        "category": "general",
        "version": "1.0.0",
        "alias_map": alias_map,  # expose for debugging/introspection
    }

    return wrapper

# ============================================================================
# PRODUCTION REGISTRAR CLASS - COMPLETE WITH ALL TOOLS REGISTERED
# ============================================================================
class ToolsRegistrar:
    """
    🛠️ Enterprise-Grade Tool Registration System - COMPLETE COVERAGE

    Features:
    - COMPLETE coverage of ALL tool functions
    - Deep LLM integration with rich metadata
    - Compatibility with ai_agent.py ToolRegistry
    - Tool counting and metrics
    - Security and validation
    - Async support
    - Production-grade error handling
    """

    def __init__(self, dest_download_dir: Optional[str] = None):
        self.dest_download_dir = str(dest_download_dir or DEFAULT_DOWNLOAD_DIR)
        Path(self.dest_download_dir).mkdir(parents=True, exist_ok=True)

        self.registered_tools = []
        self.tool_count = 0
        self.registration_errors = []

        logger.info(f"✅ ToolsRegistrar initialized with download dir: {self.dest_download_dir}")

    def _register_tool(
            self,
            tool_registry,
            func: Optional[Callable],
            name: Optional[str] = None,
            description: str = "",
            timeout: int = 60,
            retries: int = 3,
            parameters: Optional[Dict[str, Any]] = None,
            rate_limit: Optional[float] = None,
            requires_service: bool = False,
            enhanced_wrapper: bool = True,
            operation_type: Optional[str] = None,
            cost_per_unit: Optional[float] = None,
            # NEW: dependency metadata
            provides: Optional[List[str]] = None,
            consumes: Optional[List[str]] = None
    ) -> bool:
        if not func:
            return False

        if not callable(func):
            logger.info(f"⚠️ Skipping {name}: not callable")
            return False

        tool_name = name or getattr(func, "__name__", str(func))

        try:
            if hasattr(tool_registry, 'tools') and tool_name in tool_registry.tools:
                logger.info(f"ℹ️ Tool '{tool_name}' already registered, skipping")
                return False

            if enhanced_wrapper:
                wrapper_func = create_enhanced_tool_wrapper(
                    func,
                    name=tool_name,
                    description=description,
                    param_info=parameters
                )
                func_to_register = wrapper_func
            else:
                func_to_register = func

            # Register with all new fields
            tool_registry.register_function(
                func_to_register,
                name=tool_name,
                description=description,
                timeout=timeout,
                retries=retries,
                schema=parameters,
                rate_limit=rate_limit,
                operation_type=operation_type,
                cost_per_unit=cost_per_unit,
                provides=provides,      # NEW
                consumes=consumes        # NEW
            )

            self.registered_tools.append({
                'name': tool_name,
                'description': description[:100] + "..." if len(description) > 100 else description,
                'timeout': timeout,
                'retries': retries,
                'requires_service': requires_service,
                'enhanced_wrapper': enhanced_wrapper,
                'provides': provides,
                'consumes': consumes
            })
            self.tool_count += 1

            logger.info(f"✅ Registered tool: {tool_name} (Tool #{self.tool_count})")
            return True

        except Exception as e:
            error_msg = f"Failed to register '{tool_name}': {e}"
            logger.warning(f"⚠️ {error_msg}")
            self.registration_errors.append({
                'tool': tool_name,
                'error': str(e),
                'timestamp': time.time()
            })
            return False

    def register_all(self, tool_registry, service=None) -> None:
        logger.info("🔧 Starting COMPLETE tool registration (Production Mode)...")
        initial_count = self.tool_count

        # ====================================================================
        # LINK SEARCH TOOLS - COMPLETE
        # ====================================================================
        self._register_tool(
            tool_registry,
            search_links,
            name="search_links",
            description=(
                "🔍 WEB SEARCH ENGINE — Searches the internet for links matching a text query.\n"
                "REQUIRED: 'query' (string) – the search terms, e.g., 'kidnapping Nigeria 2026'.\n"
                "OPTIONAL: 'max_results' (int, default 50), 'engines' (list of engine names, e.g., ['google','bing']).\n"
                "RETURNS: dict with keys 'links' (list of URLs), 'count' (int), 'per_engine' (details).\n"
                "USE when you need to discover new URLs from a topic.\n"
                "DO NOT use if you already have a URL – use scrape_page or http_fetch instead."
            ),
            timeout=300,
            retries=2,
            parameters={
                "query": {"type": "string", "description": "Search keywords (required)", "required": True},
                "engines": {"type": "array", "items": {"type": "string"},
                            "description": "Search engines to use, e.g., ['duckduckgo', 'bing']", "default": ["duckduckgo", "bing", "google"]},
                "max_results": {"type": "integer", "description": "Maximum number of links to return", "default": 50},
                "timeout": {"type": "integer", "description": "Timeout per engine in seconds", "default": 30},
                "respect_robots": {"type": "boolean", "description": "Respect robots.txt", "default": True},
                "include_patterns": {"type": "array", "items": {"type": "string"},
                                     "description": "Regex patterns to include (e.g., [r'\\.pdf$'])"},
                "exclude_patterns": {"type": "array", "items": {"type": "string"},
                                     "description": "Regex patterns to exclude"},
                "use_proxy": {"type": "boolean", "description": "Use proxy rotation if configured", "default": False}
            },
            operation_type="network",
            provides=["search_results"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            search_with_dorks,
            name="search_with_dorks",
            description=(
                "🔍 ADVANCED SEARCH using Google Dorks syntax.\n"
                "REQUIRED: 'dork_query' (string) – the dork string, e.g., 'site:example.com filetype:pdf'.\n"
                "OPTIONAL: 'max_results' (int, default 100), 'engines' (list), 'filetypes' (list).\n"
                "RETURNS: dict with keys 'links' (list of URLs), 'count', 'per_dork_query'.\n"
                "USE for targeted searches like finding specific file types or hidden resources.\n"
                "Alias 'query' is automatically remapped to 'dork_query'."
            ),
            timeout=600,
            retries=2,
            parameters={
                "dork_query": {"type": "string", "required": True,
                               "description": "Google Dork search string, e.g., 'intitle:threat intelligence filetype:pdf'"},
                "query": {"type": "string", "required": False,
                          "description": "Alias for dork_query (remapped automatically)"},
                "filetypes": {"type": "array", "items": {"type": "string"},
                              "description": "Restrict to specific file types, e.g., ['pdf', 'doc']"},
                "engines": {"type": "array", "items": {"type": "string"}, "default": ["google", "bing"]},
                "max_results": {"type": "integer", "default": 100,
                                "description": "Maximum links per dork query"},
                "timeout": {"type": "integer", "default": 30},
                "respect_robots": {"type": "boolean", "default": True},
                "site": {"type": "string", "description": "Restrict search to this domain (e.g., 'example.com')"},
                "use_proxy": {"type": "boolean", "default": False}
            },
            operation_type="network",
            provides=["dork_results"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            find_file_links,
            name="find_file_links",
            description=(
                "📁 Find downloadable file links on a specific website.\n"
                "REQUIRED: 'site' (string) – domain to search, e.g., 'example.com'.\n"
                "OPTIONAL: 'filetypes' (list, default common types), 'max_results' (int).\n"
                "RETURNS: dict with keys 'links' (list of file URLs), 'count'.\n"
                "USE to discover PDFs, documents, etc. on a target site."
            ),
            timeout=300,
            retries=2,
            parameters={
                "site": {"type": "string", "required": True,
                         "description": "Domain to search, e.g., 'securityhumanrightshub.org'"},
                "filetypes": {"type": "array", "items": {"type": "string"},
                              "default": ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "csv", "zip"],
                              "description": "File extensions to look for"},
                "engines": {"type": "array", "items": {"type": "string"}, "default": ["google", "bing"]},
                "max_results": {"type": "integer", "default": 100},
                "timeout": {"type": "integer", "default": 30},
                "respect_robots": {"type": "boolean", "default": True},
                "use_proxy": {"type": "boolean", "default": False}
            },
            operation_type="network",
            provides=["file_links"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            find_hidden_resources,
            name="find_hidden_resources",
            description=(
                "🕵️ Discover hidden resources like admin panels, config files, backups on a website.\n"
                "REQUIRED: 'base_url' (string) – the website URL.\n"
                "OPTIONAL: 'patterns' (list) – custom dork patterns.\n"
                "RETURNS: dict with keys 'links' (list of discovered URLs), 'count'.\n"
                "USE for security reconnaissance on your own or authorized sites."
            ),
            timeout=300,
            retries=2,
            parameters={
                "base_url": {"type": "string", "required": True,
                             "description": "Base URL of the site to scan, e.g., 'https://example.com'"},
                "patterns": {"type": "array", "items": {"type": "string"},
                             "description": "Custom dork patterns to use"},
                "engines": {"type": "array", "items": {"type": "string"}, "default": ["google", "bing"]},
                "max_results": {"type": "integer", "default": 50},
                "timeout": {"type": "integer", "default": 30},
                "respect_robots": {"type": "boolean", "default": True},
                "use_proxy": {"type": "boolean", "default": False}
            },
            operation_type="network",
            provides=["hidden_resources"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            extract_links_from_page,
            name="extract_links_from_page",
            description=(
                "🔗 Extract all hyperlinks from a single web page.\n"
                "REQUIRED: 'url' (string) – the page URL.\n"
                "OPTIONAL: 'max_links' (int, default 200), 'include_patterns' (list), 'exclude_patterns'.\n"
                "RETURNS: dict with keys 'links' (list of URLs), 'count'.\n"
                "USE when you have a page and want to collect all links from it."
            ),
            timeout=120,
            retries=2,
            parameters={
                "url": {"type": "string", "required": True,
                        "description": "URL of the page to parse"},
                "max_links": {"type": "integer", "default": 200,
                              "description": "Maximum number of links to return"},
                "timeout": {"type": "integer", "default": 30},
                "respect_robots": {"type": "boolean", "default": True},
                "include_patterns": {"type": "array", "items": {"type": "string"},
                                     "description": "Only keep links matching these regex patterns"},
                "exclude_patterns": {"type": "array", "items": {"type": "string"},
                                     "description": "Exclude links matching these regex patterns"},
                "extract_file_links": {"type": "boolean", "default": False,
                                       "description": "If True, only return links to files (PDF, DOC, etc.)"}
            },
            operation_type="network",
            provides=["extracted_links"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            expand_links,
            name="expand_links",
            description=(
                "🕸️ Crawl from seed URLs and expand to collect more links (up to depth).\n"
                "REQUIRED: 'seed_urls' (list of strings) – starting URLs.\n"
                "OPTIONAL: 'depth' (int, default 1), 'max_total' (int, default 1000).\n"
                "RETURNS: dict with keys 'links' (all discovered URLs), 'total_count', 'depth_reached'.\n"
                "USE to perform a limited crawl from a set of starting points."
            ),
            timeout=900,
            retries=1,
            parameters={
                "seed_urls": {"type": "array", "items": {"type": "string"}, "required": True,
                              "description": "List of URLs to start crawling from"},
                "depth": {"type": "integer", "default": 1,
                          "description": "How many levels deep to crawl"},
                "max_per_page": {"type": "integer", "default": 50,
                                 "description": "Max links to extract per page"},
                "max_total": {"type": "integer", "default": 1000,
                              "description": "Stop when this many links are collected"},
                "timeout": {"type": "integer", "default": 30},
                "respect_robots": {"type": "boolean", "default": True},
                "follow_external": {"type": "boolean", "default": False,
                                    "description": "If False, stay within the same domain as seeds"},
                "include_patterns": {"type": "array", "items": {"type": "string"}},
                "exclude_patterns": {"type": "array", "items": {"type": "string"}}
            },
            operation_type="network",
            provides=["expanded_links"],
            consumes=[]
        )

        def get_available_search_engines():
            return list(_ENGINE_HANDLERS.keys())

        def create_link_search_config(**kwargs):
            try:
                return LinkSearchConfig(**kwargs)
            except Exception as e:
                logger.error(f"❌ Failed to create LinkSearchConfig: {e}")
                return None

        self._register_tool(
            tool_registry,
            get_available_search_engines,
            name="get_available_search_engines",
            description=(
                "List all search engines currently supported by the search_links tool.\n"
                "RETURNS: list of engine names, e.g., ['google','bing','duckduckgo']."
            ),
            timeout=5,
            retries=0,
            enhanced_wrapper=False,
            provides=["engine_list"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            create_link_search_config,
            name="create_link_search_config",
            description=(
                "Create a configuration object for link search tools.\n"
                "This is a factory – use it to build configs that you can pass to other tools if needed."
            ),
            timeout=10,
            retries=0,
            enhanced_wrapper=False,
            provides=["search_config"],
            consumes=[]
        )

        # ====================================================================
        # HTTP & NETWORK TOOLS - COMPLETE
        # ====================================================================
        self._register_tool(
            tool_registry,
            http_fetch,
            name="http_fetch",
            description=(
                "🌐 HTTP FETCH – fetches raw HTTP response from a URL.\n"
                "REQUIRED: 'url' (string).\n"
                "OPTIONAL: 'method' (default 'GET'), 'timeout' (int, default 30), 'headers' (dict).\n"
                "RETURNS: dict with keys 'ok', 'status_code', 'headers', 'text' (response body), 'content' (bytes).\n"
                "USE when you need the raw response, headers, or status code.\n"
                "For extracting content, consider scrape_page instead."
            ),
            timeout=60,
            retries=3,
            parameters={
                "url": {"type": "string", "required": True,
                        "description": "The URL to fetch"},
                "method": {"type": "string", "default": "GET",
                           "description": "HTTP method (GET, POST, PUT, DELETE, etc.)"},
                "timeout": {"type": "integer", "default": 30,
                            "description": "Request timeout in seconds"},
                "max_retries": {"type": "integer", "default": 3,
                                "description": "Number of retry attempts on failure"},
                "waf_bypass": {"type": "boolean", "default": True,
                               "description": "Enable WAF bypass headers"},
                "verify_ssl": {"type": "boolean", "default": True,
                               "description": "Verify SSL certificate"},
            },
            operation_type="network",
            provides=["http_response"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            http_fetch_batch,
            name="http_fetch_batch",
            description=(
                "🌐 Fetch multiple URLs concurrently and return results in order.\n"
                "REQUIRED: 'urls' (list of strings).\n"
                "OPTIONAL: 'max_concurrent' (int, default 5).\n"
                "RETURNS: list of result dicts (same structure as http_fetch) in the same order as input.\n"
                "USE when you need to fetch several pages at once."
            ),
            timeout=300,
            retries=2,
            parameters={
                "urls": {"type": "array", "items": {"type": "string"}, "required": True,
                         "description": "List of URLs to fetch"},
                "max_concurrent": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50,
                                   "description": "Number of parallel requests"},
            },
            operation_type="batch",
            provides=["batch_responses"],
            consumes=[]
        )

        # ====================================================================
        # DOWNLOAD TOOLS - COMPLETE
        # ====================================================================
        self._register_tool(
            tool_registry,
            download_file,
            name="download_file",
            description=(
                "⬇️ Download a single file from a URL and save it locally.\n"
                "REQUIRED: 'url' (string).\n"
                "OPTIONAL: 'dest_dir' (string, default configured download dir), 'filename' (string).\n"
                "RETURNS: dict with keys 'ok', 'status', 'path' (local file path), 'size', 'downloaded', 'error'.\n"
                "USE to download files (PDFs, images, documents) for later processing."
            ),
            timeout=900,
            retries=3,
            parameters={
                "url": {"type": "string", "required": True,
                        "description": "URL of the file to download"},
                "dest_dir": {"type": "string", "description": "Directory to save the file (default: system downloads dir)"},
                "filename": {"type": "string", "description": "Custom filename (overrides auto-generated)"},
                "timeout": {"type": "integer", "default": 120,
                            "description": "Download timeout in seconds"},
                "resume": {"type": "boolean", "default": True,
                           "description": "Resume partial downloads if possible"},
                "verify_hash": {"type": "boolean", "default": False,
                                "description": "Check file hash after download"},
            },
            operation_type="network",
            cost_per_unit=2.0,
            provides=["file_path"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            download_multiple,
            name="download_multiple",
            description=(
                "⬇️ Download multiple files concurrently.\n"
                "REQUIRED: 'urls' (list of strings).\n"
                "OPTIONAL: 'max_concurrent' (int, default 3).\n"
                "RETURNS: list of result dicts (same as download_file) in input order.\n"
                "USE to batch-download many files."
            ),
            timeout=3600,
            retries=2,
            parameters={
                "urls": {"type": "array", "items": {"type": "string"}, "required": True,
                         "description": "List of URLs to download"},
                "max_concurrent": {"type": "integer", "default": 3,
                                   "description": "Number of parallel downloads"},
            },
            operation_type="batch",
            cost_per_unit=2.0,
            provides=["file_paths"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            get_download_stats,
            name="get_download_stats",
            description=(
                "📊 Get statistics about downloaded files in a directory.\n"
                "OPTIONAL: 'directory' (string) – path to scan.\n"
                "RETURNS: dict with file counts, sizes, age distribution."
            ),
            timeout=30,
            retries=0,
            provides=["download_stats"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            validate_download_result,
            name="validate_download_result",
            description=(
                "✅ Validate a download result dict for completeness.\n"
                "REQUIRED: 'result' (dict) – the result from download_file.\n"
                "RETURNS: (bool, list of issues)."
            ),
            timeout=30,
            retries=0,
            provides=["validation_result"],
            consumes=["file_path"]
        )

        self._register_tool(
            tool_registry,
            download_health_check,
            name="download_health_check",
            description=(
                "🩺 Perform a health check on the download subsystem.\n"
                "RETURNS: dict with status, warnings, errors, and component availability."
            ),
            timeout=30,
            retries=0,
            provides=["health_status"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            cleanup_downloads,
            name="cleanup_downloads",
            description=(
                "🧹 Clean up old or large downloads from the download directory.\n"
                "OPTIONAL: 'max_age_hours' (float, default 168) – delete files older than this.\n"
                "          'max_total_size_gb' (float) – delete oldest files until under this limit.\n"
                "RETURNS: dict with cleaned file count and freed space."
            ),
            timeout=300,
            retries=0,
            parameters={
                "directory": {"type": "string", "description": "Directory to clean (default: download dir)"},
                "max_age_hours": {"type": "integer", "default": 168,
                                  "description": "Delete files older than this many hours"},
                "max_total_size_gb": {"type": "number", "default": 50.0,
                                      "description": "Keep total size under this GB limit"},
            },
            provides=["cleanup_report"],
            consumes=[]
        )

        # ====================================================================
        # SCRAPING TOOLS - COMPLETE
        # ====================================================================
        self._register_tool(
            tool_registry,
            scrape_page,
            name="scrape_page",
            description=(
                "📄 Extract text content, links, images, and metadata from a single web page.\n"
                "REQUIRED: 'url' (string).\n"
                "OPTIONAL: 'extract_links' (bool, default True), 'extract_text' (bool, default True).\n"
                "RETURNS: dict with keys 'ok', 'url', 'title', 'description', 'text_content', 'links', 'images', 'structured_data', etc.\n"
                "USE when you have a URL and want its human-readable content.\n"
                "For raw HTTP response, use http_fetch."
            ),
            timeout=60,
            retries=2,
            parameters={
                "url": {"type": "string", "required": True,
                        "description": "URL of the page to scrape"},
                "extract_text": {"type": "boolean", "default": True,
                                 "description": "Extract main article/text content"},
                "extract_links": {"type": "boolean", "default": True,
                                  "description": "Extract hyperlinks from the page"},
                "extract_images": {"type": "boolean", "default": False,
                                   "description": "Extract image URLs and metadata"},
                "extract_structured_data": {"type": "boolean", "default": True,
                                            "description": "Extract JSON-LD, Open Graph, etc."},
            },
            operation_type="network",
            provides=["scraped_data"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            fetch_article_text,
            name="fetch_article_text",
            description=(
                "📰 Validate → scrape → length-check a URL in one call.\n"
                "REQUIRED: 'url' (string).\n"
                "OPTIONAL: 'min_chars' (int, default 200).\n"
                "RETURNS: dict with 'ok', 'url', 'title', 'text_content', 'text_length'.\n"
                "USE this instead of validate_url + scrape_page + extract_text_from_file.\n"
                "The result handle $rN.text_content can be passed directly to store_raw_item."
            ),
            timeout=90,
            retries=1,
            parameters={
                "url": {"type": "string", "required": True},
                "min_chars": {"type": "integer", "default": 200},
            },
            operation_type="network",
            provides=["article"],
            consumes=[],
        )

        self._register_tool(
            tool_registry,
            scrape_batch,
            name="scrape_batch",
            description=(
                "📄 Scrape multiple web pages concurrently.\n"
                "REQUIRED: 'urls' (list of strings).\n"
                "OPTIONAL: 'max_concurrent' (int, default 3).\n"
                "RETURNS: list of result dicts (same as scrape_page) in input order."
            ),
            timeout=300,
            retries=1,
            parameters={
                "urls": {"type": "array", "items": {"type": "string"}, "required": True},
                "max_concurrent": {"type": "integer", "default": 3},
            },
            operation_type="batch",
            provides=["batch_scraped"],
            consumes=[]
        )

        if scrape_async:
            self._register_tool(
                tool_registry,
                scrape_async,
                name="scrape_async",
                description=(
                    "📄 Asynchronous version of scrape_page.\n"
                    "Same parameters and returns as scrape_page, but can be awaited in async code."
                ),
                timeout=60,
                retries=2,
                parameters={
                    "url": {"type": "string", "required": True},
                    "session_config": {"type": "object"},
                },
                operation_type="network",
                provides=["scraped_data_async"],
                consumes=[]
            )

        if scrape_batch_async:
            self._register_tool(
                tool_registry,
                scrape_batch_async,
                name="scrape_batch_async",
                description=(
                    "📄 Async batch scraping.\n"
                    "Same as scrape_batch but async-friendly."
                ),
                timeout=300,
                retries=1,
                parameters={
                    "urls": {"type": "array", "items": {"type": "string"}, "required": True},
                    "max_concurrent": {"type": "integer", "default": 5},
                },
                operation_type="batch",
                provides=["batch_scraped_async"],
                consumes=[]
            )

        self._register_tool(
            tool_registry,
            parse_feed,
            name="parse_feed",
            description=(
                "📡 Parse a single RSS or Atom feed.\n"
                "REQUIRED: 'url' (string) – the feed URL.\n"
                "OPTIONAL: 'max_entries' (int, default 100).\n"
                "RETURNS: dict with keys 'ok', 'url', 'title', 'entries' (list of {title, link, summary, published}), 'count'.\n"
                "USE to get article links and summaries from news feeds."
            ),
            timeout=30,
            retries=2,
            parameters={
                "url": {"type": "string", "required": True,
                        "description": "RSS/Atom feed URL, e.g., 'https://news.un.org/feed/subscribe/en/news/all/rss.xml'"},
                "max_entries": {"type": "integer", "description": "Maximum feed entries to return", "default": 100},
                "parse_entries": {"type": "boolean", "default": False,
                                  "description": "If True, also parse entry content (may be slow)"},
            },
            operation_type="network",
            provides=["feed_entries"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            parse_feeds_batch,
            name="parse_feeds_batch",
            description=(
                "📡 Parse multiple RSS/Atom feeds in one call.\n"
                "REQUIRED: 'feeds' (list of strings) – feed URLs.\n"
                "OPTIONAL: 'max_entries' (int, default 100) per feed.\n"
                "RETURNS: dict with aggregated entries from all feeds, plus per-feed details."
            ),
            timeout=120,
            retries=2,
            parameters={
                "feeds": {"type": "array", "items": {"type": "string"},
                          "description": "List of RSS/Atom feed URLs to parse", "required": True},
                "max_entries": {"type": "integer", "description": "Max entries per feed", "default": 100},
                "parse_entries": {"type": "boolean", "description": "Parse full entry details", "default": True},
            },
            operation_type="network",
            provides=["feed_entries"],
            consumes=[],
        )

        if browser_scrape:
            self._register_tool(
                tool_registry,
                browser_scrape,
                name="browser_scrape",
                description=(
                    "🌐 Scrape JavaScript-heavy pages using a headless browser.\n"
                    "REQUIRED: 'url' (string).\n"
                    "OPTIONAL: 'wait_time' (int) – seconds to wait after page load.\n"
                    "RETURNS: same as scrape_page.\n"
                    "USE for sites that require JavaScript rendering."
                ),
                timeout=120,
                retries=1,
                parameters={
                    "url": {"type": "string", "required": True},
                    "wait_time": {"type": "integer", "default": 5,
                                  "description": "Seconds to wait for dynamic content"},
                },
                operation_type="network",
                provides=["browser_scraped"],
                consumes=[]
            )

        self._register_tool(
            tool_registry,
            clear_scraping_cache,
            name="clear_scraping_cache",
            description=(
                "🧹 Clear the in-memory cache used by scrape_page.\n"
                "No parameters.\n"
                "RETURNS: dict with 'cleared' count."
            ),
            timeout=10,
            retries=0,
            provides=["cache_cleared"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            get_scraping_stats,
            name="get_scraping_stats",
            description=(
                "📊 Get statistics about scraping operations and cache.\n"
                "RETURNS: dict with cache hits/misses, available backends, etc."
            ),
            timeout=10,
            retries=0,
            provides=["scraping_stats"],
            consumes=[]
        )

        # ====================================================================
        # EXTRACTION TOOLS - COMPLETE
        # ====================================================================
        self._register_tool(
            tool_registry,
            extract_text_from_file,
            name="extract_text_from_file",
            description=(
                "📄 Extract plain text from various file formats (PDF, DOCX, XLSX, TXT, images with OCR).\n"
                "REQUIRED: 'path' (string) – file path.\n"
                "OPTIONAL: 'method' (string) – force a specific extraction method.\n"
                "RETURNS: dict with keys 'ok', 'text', 'file_path', 'extraction_method', 'error'.\n"
                "USE when you have a local file and need its text content."
            ),
            timeout=300,
            retries=2,
            parameters={
                "path": {"type": "string", "required": True,
                         "description": "Path to the file (absolute or relative)"},
                "method": {"type": "string", "default": "auto",
                           "enum": ["auto", "pdfminer", "pypdf", "docx", "pandas", "text", "ocr"],
                           "description": "Force a specific extraction backend"},
                "max_pages": {"type": "integer", "description": "For PDFs, limit pages extracted"},
                "ocr_enabled": {"type": "boolean", "default": False,
                                "description": "Enable OCR for images (requires tesseract)"},
            },
            operation_type="disk",
            provides=["extracted_text"],
            consumes=["file_path"]
        )

        self._register_tool(
            tool_registry,
            extract_text_from_pdf,
            name="extract_text_from_pdf",
            description=(
                "📄 Extract text from PDF files.\n"
                "REQUIRED: 'pdf_path' (string).\n"
                "RETURNS: same as extract_text_from_file."
            ),
            timeout=300,
            retries=2,
            parameters={
                "pdf_path": {"type": "string", "required": True},
                "page_range": {"type": "string", "description": "e.g., '1-5' to limit pages"},
            },
            operation_type="disk",
            provides=["pdf_text"],
            consumes=["file_path"]
        )

        self._register_tool(
            tool_registry,
            extract_text_from_docx,
            name="extract_text_from_docx",
            description=(
                "📄 Extract text from DOCX files.\n"
                "REQUIRED: 'docx_path' (string).\n"
                "RETURNS: same as extract_text_from_file."
            ),
            timeout=120,
            retries=2,
            parameters={
                "docx_path": {"type": "string", "required": True},
                "include_formatting": {"type": "boolean", "default": False},
            },
            operation_type="disk",
            provides=["docx_text"],
            consumes=["file_path"]
        )

        self._register_tool(
            tool_registry,
            extract_text_from_spreadsheet,
            name="extract_text_from_spreadsheet",
            description=(
                "📄 Extract text from Excel or CSV files.\n"
                "REQUIRED: 'spreadsheet_path' (string).\n"
                "OPTIONAL: 'sheet' (string or int) – sheet name or index.\n"
                "RETURNS: same as extract_text_from_file."
            ),
            timeout=120,
            retries=2,
            parameters={
                "spreadsheet_path": {"type": "string", "required": True},
                "sheet_name": {"type": "string", "description": "Specific sheet to extract"},
                "include_headers": {"type": "boolean", "default": True},
            },
            operation_type="disk",
            provides=["spreadsheet_data"],
            consumes=["file_path"]
        )

        self._register_tool(
            tool_registry,
            extract_text_from_textfile,
            name="extract_text_from_textfile",
            description=(
                "📄 Extract text from plain text files (TXT, MD, etc.).\n"
                "REQUIRED: 'file_path' (string).\n"
                "RETURNS: same as extract_text_from_file."
            ),
            timeout=60,
            retries=1,
            parameters={
                "file_path": {"type": "string", "required": True},
                "encoding": {"type": "string", "default": "utf-8",
                             "description": "File encoding (auto-detected if not specified)"},
            },
            operation_type="disk",
            provides=["text_content"],
            consumes=["file_path"]
        )

        self._register_tool(
            tool_registry,
            extract_and_chunk_file,
            name="extract_and_chunk_file",
            description=(
                "📄 Extract text from a file and split it into overlapping chunks.\n"
                "REQUIRED: 'path' (string).\n"
                "OPTIONAL: 'chunk_size' (int, default 1000), 'chunk_overlap' (int, default 200).\n"
                "RETURNS: dict with 'extraction_result' and 'chunks' list."
            ),
            timeout=600,
            retries=1,
            parameters={
                "path": {"type": "string", "required": True},
                "chunk_size": {"type": "integer", "default": 1000},
                "chunk_overlap": {"type": "integer", "default": 200},
                "chunking_strategy": {"type": "string", "default": "sentence",
                                       "enum": ["fixed", "sentence", "paragraph", "semantic", "recursive"]},
            },
            operation_type="mixed",
            provides=["chunks"],
            consumes=["file_path"]
        )

        if chunk_text:
            self._register_tool(
                tool_registry,
                chunk_text,
                name="chunk_text",
                description=(
                    "✂️ Split plain text into overlapping chunks.\n"
                    "REQUIRED: 'text' (string).\n"
                    "OPTIONAL: 'chunk_size' (int, default 1000), 'chunk_overlap' (int, default 200), 'strategy'.\n"
                    "RETURNS: list of strings (chunks)."
                ),
                timeout=60,
                retries=0,
                parameters={
                    "text": {"type": "string", "required": True},
                    "chunk_size": {"type": "integer", "default": 1000},
                    "chunk_overlap": {"type": "integer", "default": 200},
                    "strategy": {"type": "string", "default": "sentence",
                                 "enum": ["fixed", "sentence", "paragraph", "semantic", "recursive"]},
                },
                operation_type="compute",
                provides=["chunks"],
                consumes=["text"]
            )

        self._register_tool(
            tool_registry,
            batch_extract_files,
            name="batch_extract_files",
            description=(
                "📄 Extract text from multiple files concurrently.\n"
                "REQUIRED: 'paths' (list of strings).\n"
                "RETURNS: list of extraction results in input order."
            ),
            timeout=3600,
            retries=1,
            parameters={
                "paths": {"type": "array", "items": {"type": "string"}, "required": True},
                "max_concurrent": {"type": "integer", "default": 3},
            },
            operation_type="batch",
            provides=["batch_extractions"],
            consumes=["file_paths"]
        )

        if extract_with_ocr:
            self._register_tool(
                tool_registry,
                extract_with_ocr,
                name="extract_with_ocr",
                description=(
                    "🖼️ Extract text from images using OCR (Tesseract).\n"
                    "REQUIRED: 'image_path' (string).\n"
                    "OPTIONAL: 'language' (string, default 'eng').\n"
                    "RETURNS: same as extract_text_from_file."
                ),
                timeout=600,
                retries=1,
                parameters={
                    "image_path": {"type": "string", "required": True},
                    "language": {"type": "string", "default": "eng"},
                },
                operation_type="compute",
                provides=["ocr_text"],
                consumes=["file_path"]
            )

        if process_file_for_vector_store:
            self._register_tool(
                tool_registry,
                process_file_for_vector_store,
                name="process_file_for_vector_store",
                description=(
                    "📤 Extract, chunk, embed, and upsert a file into the vector store.\n"
                    "REQUIRES: 'file_path' (string) and access to service (automatically provided).\n"
                    "OPTIONAL: 'chunk_size' (int, default 1000).\n"
                    "RETURNS: dict with 'ok', 'chunks_processed', 'source_id'."
                ),
                timeout=600,
                retries=1,
                requires_service=True,
                parameters={
                    "file_path": {"type": "string", "required": True},
                    "chunk_size": {"type": "integer", "default": 1000},
                },
                operation_type="mixed",
                provides=["vector_ready"],
                consumes=["file_path"]
            )

        self._register_tool(
            tool_registry,
            get_extraction_backends,
            name="get_extraction_backends",
            description=(
                "🔍 List available text extraction backends.\n"
                "RETURNS: dict with backend names and availability booleans."
            ),
            timeout=10,
            retries=0,
            provides=["backend_list"],
            consumes=[]
        )

        # ====================================================================
        # VIDEO TOOLS - COMPLETE with dependency metadata
        # ====================================================================
        self._register_tool(
            tool_registry,
            video_info,
            name="video_info",
            description=(
                "🎥 Get metadata (title, duration, uploader, etc.) for a video URL.\n"
                "REQUIRED: 'url' (string).\n"
                "RETURNS: dict with title, duration, uploader, thumbnail_url, etc."
            ),
            timeout=60,
            retries=2,
            parameters={
                "url": {"type": "string", "required": True,
                        "description": "Video URL (YouTube, Vimeo, etc.)"},
                "extract_subtitles": {"type": "boolean", "default": False},
            },
            operation_type="network",
            provides=["video_metadata"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            download_video,
            name="download_video",
            description=(
                "⬇️ Download a video from a supported platform.\n"
                "REQUIRED: 'url' (string).\n"
                "OPTIONAL: 'dest_dir' (string), 'quality' (string, default 'best'), 'audio_only' (bool).\n"
                "RETURNS: dict with 'ok', 'path', 'metadata', etc."
            ),
            timeout=1800,
            retries=2,
            parameters={
                "url": {"type": "string", "required": True},
                "dest_dir": {"type": "string", "description": "Download directory"},
                "quality": {"type": "string", "default": "best",
                            "enum": ["best", "worst", "1080p", "720p", "480p", "audio_only"]},
                "download_subtitles": {"type": "boolean", "default": True},
            },
            operation_type="network",
            cost_per_unit=2.0,
            provides=["file_path"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            download_multiple_videos,
            name="download_multiple_videos",
            description=(
                "⬇️ Download multiple videos concurrently.\n"
                "REQUIRED: 'urls' (list of strings).\n"
                "OPTIONAL: 'max_concurrent' (int, default 2).\n"
                "RETURNS: list of result dicts."
            ),
            timeout=3600,
            retries=1,
            parameters={
                "urls": {"type": "array", "items": {"type": "string"}, "required": True},
                "max_concurrent": {"type": "integer", "default": 2},
            },
            operation_type="batch",
            provides=["file_paths"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            extract_video_thumbnail,
            name="extract_video_thumbnail",
            description=(
                "🖼️ Extract a thumbnail from a local video file.\n"
                "REQUIRED: 'video_path' (string).\n"
                "OPTIONAL: 'output_path' (string), 'strategy' (string).\n"
                "RETURNS: dict with 'ok', 'path'."
            ),
            timeout=120,
            retries=1,
            parameters={
                "video_path": {"type": "string", "required": True},
                "time_seconds": {"type": "number", "default": 5,
                                 "description": "Frame time to capture"},
                "width": {"type": "integer", "description": "Thumbnail width (preserves aspect)"},
            },
            operation_type="disk",
            provides=["thumbnail_path"],
            consumes=["file_path"]
        )

        self._register_tool(
            tool_registry,
            get_video_backends,
            name="get_video_backends",
            description=(
                "🔍 List available video processing backends (yt-dlp, http, ffmpeg).\n"
                "RETURNS: dict with availability booleans."
            ),
            timeout=10,
            retries=0,
            provides=["video_backends"],
            consumes=[]
        )

        # ====================================================================
        # TRANSCRIPTION TOOLS - COMPLETE with dependency metadata
        # ====================================================================
        self._register_tool(
            tool_registry,
            transcribe_audio_or_video,
            name="transcribe_audio_or_video",
            description=(
                "🎤 Transcribe an audio or video file to text using Whisper.\n"
                "REQUIRED: 'path' (string) – file path.\n"
                "OPTIONAL: 'model_name' (string, default 'small'), 'language' (string, auto-detect).\n"
                "RETURNS: dict with 'text', 'language', 'segments', 'duration'."
            ),
            timeout=3600,
            retries=0,
            parameters={
                "path": {"type": "string", "required": True,
                         "description": "Path to audio/video file"},
                "model_name": {"type": "string", "default": "small",
                               "enum": ["tiny", "base", "small", "medium", "large"]},
                "language": {"type": "string", "description": "ISO-639-1 language code, e.g., 'en' (auto if omitted)"},
                "backend": {"type": "string", "default": "auto",
                            "enum": ["auto", "faster_whisper", "openai_whisper"]},
                "word_timestamps": {"type": "boolean", "default": False},
            },
            operation_type="compute",
            provides=["transcript"],
            consumes=["file_path"]
        )

        self._register_tool(
            tool_registry,
            transcribe_multiple,
            name="transcribe_multiple",
            description=(
                "🎤 Transcribe multiple audio/video files concurrently.\n"
                "REQUIRED: 'paths' (list of strings).\n"
                "OPTIONAL: 'max_concurrent' (int, default 2).\n"
                "RETURNS: list of transcription results in input order."
            ),
            timeout=7200,
            retries=0,
            parameters={
                "paths": {"type": "array", "items": {"type": "string"}, "required": True},
                "model_name": {"type": "string", "default": "small"},
                "max_concurrent": {"type": "integer", "default": 2},
            },
            operation_type="batch",
            provides=["transcripts"],
            consumes=["file_paths"]
        )

        self._register_tool(
            tool_registry,
            get_transcription_backends,
            name="get_transcription_backends",
            description=(
                "🔍 List available transcription backends.\n"
                "RETURNS: dict with availability booleans."
            ),
            timeout=10,
            retries=0,
            provides=["transcription_backends"],
            consumes=[]
        )

        if clear_model_cache:
            self._register_tool(
                tool_registry,
                clear_model_cache,
                name="clear_model_cache",
                description=(
                    "🧹 Clear Whisper model cache to free memory.\n"
                    "No parameters.\n"
                    "RETURNS: dict with 'ok'."
                ),
                timeout=10,
                retries=0,
                provides=["cache_cleared"],
                consumes=[]
            )

        if get_supported_formats:
            self._register_tool(
                tool_registry,
                get_supported_formats,
                name="get_supported_formats",
                description=(
                    "📋 List supported audio/video formats for transcription.\n"
                    "RETURNS: dict with 'audio' and 'video' lists."
                ),
                timeout=5,
                retries=0,
                provides=["format_list"],
                consumes=[]
            )

        # NOTE: transcribe_url_to_vectorstore and transcribe_and_index are
        # registered exclusively by register_pipeline_tools() called in
        # register_with_service(). Registering them here too caused the
        # "Tool already registered, overwriting" warning on every startup.

        # ====================================================================
        # UTILITY TOOLS - COMPLETE
        # ====================================================================
        self._register_tool(
            tool_registry,
            calculate_file_hash,
            name="calculate_file_hash",
            description=(
                "🔐 Calculate cryptographic hash of a file (SHA256 by default).\n"
                "REQUIRED: 'file_path' (string).\n"
                "OPTIONAL: 'algorithm' (string, default 'sha256').\n"
                "RETURNS: string hash."
            ),
            timeout=300,
            retries=0,
            parameters={
                "file_path": {"type": "string", "required": True},
                "algorithm": {"type": "string", "default": "sha256",
                              "enum": ["md5", "sha1", "sha256", "sha512", "blake2b"]},
            },
            operation_type="disk",
            provides=["file_hash"],
            consumes=["file_path"]
        )

        self._register_tool(
            tool_registry,
            sanitize_filename,
            name="sanitize_filename",
            description=(
                "🧼 Remove unsafe characters from a filename.\n"
                "REQUIRED: 'filename' (string).\n"
                "RETURNS: safe filename string."
            ),
            timeout=5,
            retries=0,
            parameters={
                "filename": {"type": "string", "required": True},
                "max_length": {"type": "integer", "default": 255},
            },
            provides=["safe_filename"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            validate_url,
            name="validate_url",
            description=(
                "✅ Check if a URL is safe and properly formatted.\n"
                "REQUIRED: 'url' (string).\n"
                "RETURNS: bool (True if valid).\n"
                "USE before any fetch or download to avoid errors."
            ),
            timeout=5,
            retries=0,
            parameters={
                "url": {"type": "string", "required": True},
            },
            provides=["url_valid"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            generate_safe_filename,
            name="generate_safe_filename",
            description=(
                "🔐 Generate a safe filename from a URL, optionally with content type.\n"
                "REQUIRED: 'url' (string).\n"
                "RETURNS: string filename."
            ),
            timeout=5,
            retries=0,
            parameters={
                "url": {"type": "string", "required": True},
                "content_type": {"type": "string", "description": "MIME type to guess extension"},
            },
            provides=["safe_filename"],
            consumes=[]
        )

        # ====================================================================
        # DATABASE TOOLS (Optional) - COMPLETE
        # ====================================================================
        if DBTOOLS_AVAILABLE and register_db_tools:
            try:
                register_db_tools(
                    tool_registry,
                    config=getattr(service, "config", None),
                )
                logger.info("✅ Database tools registered (19 tools incl. store_raw_item)")
                self.tool_count += 19
            except Exception as e:
                logger.warning(f"⚠️ Failed to register database tools: {e}")
                self.registration_errors.append({
                    "tool": "database_tools",
                    "error": str(e),
                    "timestamp": time.time(),
                })

        # ====================================================================
        # HTTP COMPONENT FACTORIES
        # ====================================================================
        http_factories = [
            (HttpMethod, "create_http_method", "Create HTTP method enum for request configuration."),
            (HttpRetryStrategy, "create_http_retry_strategy", "Create HTTP retry strategy configuration."),
            (HttpConfig, "create_http_config", "Create HTTP configuration for request settings."),
            (RateLimiter, "create_rate_limiter", "Create rate limiter for controlling request rates."),
            (CircuitBreaker, "create_http_circuit_breaker", "Create HTTP circuit breaker for fault tolerance."),
            (CircuitState, "create_circuit_state", "Create circuit state for monitoring."),
            (ContentType, "create_content_type", "Create content type for HTTP headers."),
            (HttpResponse, "create_http_response", "Create HTTP response wrapper."),
        ]

        for factory_class, tool_name, description in http_factories:
            if factory_class:
                def create_factory(cls=factory_class, desc=description):
                    def factory(**kwargs):
                        try:
                            return cls(**kwargs)
                        except Exception as e:
                            logger.error(f"❌ Failed to create {cls.__name__}: {e}")
                            return None
                    factory.__doc__ = desc
                    return factory

                factory_func = create_factory()
                self._register_tool(
                    tool_registry,
                    factory_func,
                    name=tool_name,
                    description=description,
                    timeout=10,
                    retries=0,
                    enhanced_wrapper=False,
                    provides=["factory_instance"],
                    consumes=[]
                )

        # ====================================================================
        # DOWNLOAD COMPONENT FACTORIES
        # ====================================================================
        download_factories = [
            (DownloadOptions, "create_download_options", "Create download options configuration."),
            (DownloadResult, "create_download_result", "Create download result wrapper."),
            (DownloadManager, "create_download_manager", "Create download manager instance."),
        ]

        for factory_class, tool_name, description in download_factories:
            if factory_class:
                def create_factory(cls=factory_class, desc=description):
                    def factory(**kwargs):
                        try:
                            return cls(**kwargs)
                        except Exception as e:
                            logger.error(f"❌ Failed to create {cls.__name__}: {e}")
                            return None
                    factory.__doc__ = desc
                    return factory

                factory_func = create_factory()
                self._register_tool(
                    tool_registry,
                    factory_func,
                    name=tool_name,
                    description=description,
                    timeout=10,
                    retries=0,
                    enhanced_wrapper=False,
                    provides=["factory_instance"],
                    consumes=[]
                )

        # ====================================================================
        # UTILITY COMPONENT FACTORIES
        # ====================================================================
        utility_factories = [
            (SessionConfig, "create_session_config", "Create HTTP session configuration."),
            (SecurityLevel, "create_security_level", "Create security level enum."),
            (RetryStrategy, "create_retry_strategy", "Create retry strategy for operations."),
            (DownloadProgress, "create_download_progress", "Create download progress tracker."),
            (CleanupResult, "create_cleanup_result", "Create cleanup operation result."),
            (DownloadConfig, "create_download_config", "Create download configuration."),
        ]

        for factory_class, tool_name, description in utility_factories:
            if factory_class:
                def create_factory(cls=factory_class, desc=description):
                    def factory(**kwargs):
                        try:
                            return cls(**kwargs)
                        except Exception as e:
                            logger.error(f"❌ Failed to create {cls.__name__}: {e}")
                            return None
                    factory.__doc__ = desc
                    return factory

                factory_func = create_factory()
                self._register_tool(
                    tool_registry,
                    factory_func,
                    name=tool_name,
                    description=description,
                    timeout=10,
                    retries=0,
                    enhanced_wrapper=False,
                    provides=["factory_instance"],
                    consumes=[]
                )

        # ====================================================================
        # CONSTANT ACCESSORS
        # ====================================================================
        def get_default_cache_dir():
            return DEFAULT_CACHE_DIR

        def get_default_log_dir():
            return DEFAULT_LOG_DIR

        self._register_tool(
            tool_registry,
            get_default_cache_dir,
            name="get_default_cache_dir",
            description=(
                "📁 Get the default cache directory path.\n"
                "RETURNS: string path."
            ),
            timeout=5,
            retries=0,
            enhanced_wrapper=False,
            provides=["cache_dir"],
            consumes=[]
        )

        self._register_tool(
            tool_registry,
            get_default_log_dir,
            name="get_default_log_dir",
            description=(
                "📁 Get the default log directory path.\n"
                "RETURNS: string path."
            ),
            timeout=5,
            retries=0,
            enhanced_wrapper=False,
            provides=["log_dir"],
            consumes=[]
        )

        # ====================================================================
        # SCRAPING COMPONENT FACTORIES
        # ====================================================================
        scraping_factories = [
            (ContentCategory, "create_content_category", "Create content category enum."),
            (RobotsPolicy, "create_robots_policy", "Create robots.txt policy configuration."),
            (AntiDetectionLevel, "create_anti_detection_level", "Create anti-detection level for scraping."),
            (DependencyManager, "create_dependency_manager", "Create dependency manager for scraping."),
            (ContentExtractor, "create_content_extractor", "Create content extractor instance."),
            (ScrapingResult, "create_scraping_result", "Create scraping result wrapper."),
            (RobotsTxtChecker, "create_robots_txt_checker", "Create robots.txt checker instance."),
            (URLValidator, "create_url_validator", "Create URL validator instance."),
            (ScrapingConfig, "create_scraping_config", "Create scraping configuration."),
            (ScrapingEngine, "create_scraping_engine", "Create scraping engine instance."),
        ]

        for factory_class, tool_name, description in scraping_factories:
            if factory_class:
                def create_factory(cls=factory_class, desc=description):
                    def factory(**kwargs):
                        try:
                            return cls(**kwargs)
                        except Exception as e:
                            logger.error(f"❌ Failed to create {cls.__name__}: {e}")
                            return None
                    factory.__doc__ = desc
                    return factory

                factory_func = create_factory()
                self._register_tool(
                    tool_registry,
                    factory_func,
                    name=tool_name,
                    description=description,
                    timeout=15,
                    retries=1,
                    enhanced_wrapper=False,
                    provides=["factory_instance"],
                    consumes=[]
                )

        # ====================================================================
        # EXTRACTION COMPONENT FACTORIES
        # ====================================================================
        extraction_factories = [
            (ExtractionResult, "create_extraction_result", "Create extraction result wrapper."),
            (ChunkingStrategy, "create_chunking_strategy", "Create chunking strategy configuration."),
            (OCREngine, "create_ocr_engine", "Create OCR engine instance."),
            (ExtractionConfig, "create_extraction_config", "Create extraction configuration."),
            (ExtractionEngine, "create_extraction_engine", "Create extraction engine instance."),
        ]

        for factory_class, tool_name, description in extraction_factories:
            if factory_class:
                def create_factory(cls=factory_class, desc=description):
                    def factory(**kwargs):
                        try:
                            return cls(**kwargs)
                        except Exception as e:
                            logger.error(f"❌ Failed to create {cls.__name__}: {e}")
                            return None
                    factory.__doc__ = desc
                    return factory

                factory_func = create_factory()
                self._register_tool(
                    tool_registry,
                    factory_func,
                    name=tool_name,
                    description=description,
                    timeout=15,
                    retries=1,
                    enhanced_wrapper=False,
                    provides=["factory_instance"],
                    consumes=[]
                )

        # ====================================================================
        # VIDEO COMPONENT FACTORIES
        # ====================================================================
        video_factories = [
            (VideoPlatform, "create_video_platform", "Create video platform enum."),
            (VideoQuality, "create_video_quality", "Create video quality configuration."),
            (ThumbnailStrategy, "create_thumbnail_strategy", "Create thumbnail extraction strategy."),
            (VideoFormat, "create_video_format", "Create video format enum."),
            (DownloadStatus, "create_download_status", "Create download status tracker."),
            (VideoMetadata, "create_video_metadata", "Create video metadata wrapper."),
            (VideoResult, "create_video_result", "Create video processing result wrapper."),
            (VideoBackend, "create_video_backend", "Create video backend base instance."),
            (YtDlpBackend, "create_ytdlp_backend", "Create YouTube-DLP backend instance."),
            (GenericHttpBackend, "create_generic_http_backend", "Create generic HTTP backend instance."),
            (FfmpegBackend, "create_ffmpeg_backend", "Create FFmpeg backend instance."),
            (VideoProcessor, "create_video_processor", "Create video processor instance."),
            (VideoConfig, "create_video_config", "Create video configuration."),
        ]

        for factory_class, tool_name, description in video_factories:
            if factory_class:
                def create_factory(cls=factory_class, desc=description):
                    def factory(**kwargs):
                        try:
                            return cls(**kwargs)
                        except Exception as e:
                            logger.error(f"❌ Failed to create {cls.__name__}: {e}")
                            return None
                    factory.__doc__ = desc
                    return factory

                factory_func = create_factory()
                self._register_tool(
                    tool_registry,
                    factory_func,
                    name=tool_name,
                    description=description,
                    timeout=15,
                    retries=1,
                    enhanced_wrapper=False,
                    provides=["factory_instance"],
                    consumes=[]
                )

        # ====================================================================
        # TRANSCRIPTION COMPONENT FACTORIES
        # ====================================================================
        transcription_factories = [
            (TranscriptionResult, "create_transcription_result", "Create transcription result wrapper."),
            (TranscriptionBackend, "create_transcription_backend", "Create transcription backend instance."),
            (ComputeType, "create_compute_type", "Create compute type enum."),
            (TranscriptionStatus, "create_transcription_status", "Create transcription status tracker."),
            (TranscriptionMetrics, "create_transcription_metrics", "Create transcription metrics tracker."),
            (TranscriptionBatchProcessor, "create_transcription_batch_processor", "Create batch processor."),
            (ResourceMonitor, "create_resource_monitor", "Create resource monitor."),
            (TranscriptionCircuitBreaker, "create_transcription_circuit_breaker", "Create circuit breaker."),
            (TranscriptionConfig, "create_transcription_config", "Create transcription configuration."),
        ]

        for factory_class, tool_name, description in transcription_factories:
            if factory_class:
                def create_factory(cls=factory_class, desc=description):
                    def factory(**kwargs):
                        try:
                            return cls(**kwargs)
                        except Exception as e:
                            logger.error(f"❌ Failed to create {cls.__name__}: {e}")
                            return None
                    factory.__doc__ = desc
                    return factory

                factory_func = create_factory()
                self._register_tool(
                    tool_registry,
                    factory_func,
                    name=tool_name,
                    description=description,
                    timeout=15,
                    retries=1,
                    enhanced_wrapper=False,
                    provides=["factory_instance"],
                    consumes=[]
                )

        # ====================================================================
        # PIPELINE COMPONENT FACTORIES
        # ====================================================================
        pipeline_factories = [
            (ProcessingStage, "create_processing_stage", "Create processing stage enum."),
            (SourceType, "create_source_type", "Create source type enum."),
            (ProcessingStatus, "create_processing_status", "Create processing status tracker."),
            (ProcessingError, "create_processing_error", "Create processing error wrapper."),
            (ProcessingResult, "create_processing_result", "Create processing result wrapper."),
            (PipelineMetrics, "create_pipeline_metrics", "Create pipeline metrics tracker."),
            (SecurityValidator, "create_security_validator", "Create security validator instance."),
            (DownloadProcessor, "create_download_processor", "Create download processor instance."),
            (TranscriptionProcessor, "create_transcription_processor", "Create transcription processor instance."),
            (IndexingProcessor, "create_indexing_processor", "Create indexing processor instance."),
            (PipelineConfig, "create_pipeline_config", "Create pipeline configuration."),
            (PipelineManager, "create_pipeline_manager", "Create pipeline manager instance."),
        ]

        for factory_class, tool_name, description in pipeline_factories:
            if factory_class:
                def create_factory(cls=factory_class, desc=description):
                    def factory(**kwargs):
                        try:
                            return cls(**kwargs)
                        except Exception as e:
                            logger.error(f"❌ Failed to create {cls.__name__}: {e}")
                            return None
                    factory.__doc__ = desc
                    return factory

                factory_func = create_factory()
                self._register_tool(
                    tool_registry,
                    factory_func,
                    name=tool_name,
                    description=description,
                    timeout=15,
                    retries=1,
                    enhanced_wrapper=False,
                    provides=["factory_instance"],
                    consumes=[]
                )

        # ====================================================================
        # DATABASE COMPONENT FACTORIES (Optional)
        # ====================================================================
        if DBTOOLS_AVAILABLE:
            db_factories = [
                (DatabaseManager, "create_database_manager", "Create database manager instance."),
            ]

            for factory_class, tool_name, description in db_factories:
                if factory_class:
                    def create_factory(cls=factory_class, desc=description):
                        def factory(**kwargs):
                            try:
                                return cls(**kwargs)
                            except Exception as e:
                                logger.error(f"❌ Failed to create {cls.__name__}: {e}")
                                return None
                        factory.__doc__ = desc
                        return factory

                    factory_func = create_factory()
                    self._register_tool(
                        tool_registry,
                        factory_func,
                        name=tool_name,
                        description=description,
                        timeout=30,
                        retries=1,
                        enhanced_wrapper=False,
                        provides=["factory_instance"],
                        consumes=[]
                    )

            # Register database availability checkers
            def check_psycopg2_available():
                return PSYCOPG2_AVAILABLE

            def check_sqlalchemy_available():
                return SQLALCHEMY_AVAILABLE

            def check_mysql_available():
                return MYSQL_AVAILABLE

            def check_oracle_available():
                return ORACLE_AVAILABLE

            def check_db2_available():
                return DB2_AVAILABLE

            def check_dbtools_available():
                return DBTOOLS_AVAILABLE

            availability_checks = [
                (check_psycopg2_available, "check_psycopg2_available", "Check PostgreSQL support."),
                (check_sqlalchemy_available, "check_sqlalchemy_available", "Check SQLAlchemy support."),
                (check_mysql_available, "check_mysql_available", "Check MySQL support."),
                (check_oracle_available, "check_oracle_available", "Check Oracle support."),
                (check_db2_available, "check_db2_available", "Check DB2 support."),
                (check_dbtools_available, "check_dbtools_available", "Check any database tools."),
            ]

            for check_func, name, desc in availability_checks:
                self._register_tool(
                    tool_registry,
                    check_func,
                    name=name,
                    description=desc,
                    timeout=5,
                    retries=0,
                    enhanced_wrapper=False,
                    provides=["availability"],
                    consumes=[]
                )

        # ====================================================================
        # ERROR FACTORIES (for exception handling)
        # ====================================================================
        error_factories = [
            (HttpError, "create_http_error", "Create HTTP error exception."),
            (CircuitBreakerOpenError, "create_circuit_breaker_open_error", "Create circuit breaker open error."),
            (RateLimitExceededError, "create_rate_limit_exceeded_error", "Create rate limit exceeded error."),
            (SecurityViolationError, "create_security_violation_error", "Create security violation error."),
            (ResponseTooLargeError, "create_response_too_large_error", "Create response too large error."),
            (DatabaseConnectionError, "create_database_connection_error", "Create database connection error."),
        ]

        for factory_class, tool_name, description in error_factories:
            if factory_class:
                def create_factory(cls=factory_class, desc=description):
                    def factory(message="", **kwargs):
                        try:
                            return cls(message, **kwargs)
                        except Exception as e:
                            logger.error(f"❌ Failed to create {cls.__name__}: {e}")
                            return None
                    factory.__doc__ = desc
                    return factory

                factory_func = create_factory()
                self._register_tool(
                    tool_registry,
                    factory_func,
                    name=tool_name,
                    description=description,
                    timeout=10,
                    retries=0,
                    enhanced_wrapper=False,
                    provides=["error_instance"],
                    consumes=[]
                )

        # ====================================================================
        # ASYNC TOOLS REGISTRATION
        # ====================================================================
        async_tools = []

        if scrape_async:
            async_tools.append((scrape_async, "scrape_async", "Async web page scraping."))

        if scrape_batch_async:
            async_tools.append((scrape_batch_async, "scrape_batch_async", "Async batch web scraping."))

        for async_func, name, description in async_tools:
            self._register_tool(
                tool_registry,
                async_func,
                name=name,
                description=description,
                timeout=120,
                retries=2,
                enhanced_wrapper=True,
                provides=["async_result"],
                consumes=[]
            )

        # ====================================================================
        # PRINT REGISTRATION SUMMARY
        # ====================================================================
        new_registered = self.tool_count - initial_count
        logger.info(
            f"\n{'=' * 80}"
            f"\n✅ COMPLETE TOOL REGISTRATION FINISHED (PRODUCTION MODE)"
            f"\n{'=' * 80}"
            f"\nTotal Tools Registered: {self.tool_count}"
            f"\nTools in this batch: {new_registered}"
            f"\nRegistration Errors: {len(self.registration_errors)}"
            f"\nStatus: {'✅ READY FOR PRODUCTION' if len(self.registration_errors) == 0 else '⚠️ CHECK ERRORS'}"
            f"\n{'=' * 80}\n"
        )

        if self.registration_errors:
            logger.warning(f"⚠️ {len(self.registration_errors)} tools failed to register:")
            for error in self.registration_errors:
                logger.warning(f"  - {error['tool']}: {error['error']}")

        # Log categorized tool count
        tool_categories = {}
        for tool in self.registered_tools:
            name = tool['name']
            if 'http' in name or 'fetch' in name:
                tool_categories['HTTP'] = tool_categories.get('HTTP', 0) + 1
            elif 'download' in name:
                tool_categories['Download'] = tool_categories.get('Download', 0) + 1
            elif 'scrape' in name:
                tool_categories['Scrape'] = tool_categories.get('Scrape', 0) + 1
            elif 'extract' in name:
                tool_categories['Extract'] = tool_categories.get('Extract', 0) + 1
            elif 'video' in name:
                tool_categories['Video'] = tool_categories.get('Video', 0) + 1
            elif 'transcribe' in name:
                tool_categories['Transcribe'] = tool_categories.get('Transcribe', 0) + 1
            elif 'pipeline' in name:
                tool_categories['Pipeline'] = tool_categories.get('Pipeline', 0) + 1
            elif 'config' in name:
                tool_categories['Config'] = tool_categories.get('Config', 0) + 1
            elif 'db' in name or 'database' in name:
                tool_categories['Database'] = tool_categories.get('Database', 0) + 1
            elif 'init' in name or 'setup' in name:
                tool_categories['Initials'] = tool_categories.get('Initials', 0) + 1
            else:
                tool_categories['Utility'] = tool_categories.get('Utility', 0) + 1

        logger.info("📊 Tool Categories Summary:")
        for category, count in sorted(tool_categories.items()):
            logger.info(f"  - {category}: {count} tools")

    def register_with_service(
            self,
            service,
            cleanup_days: int = 7,
            cleanup_interval_hours: int = 24
    ) -> Dict[str, Any]:
        if not hasattr(service, 'tool_registry') or service.tool_registry is None:
            error_msg = "❌ Service has no tool_registry attribute"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        self.register_all(service.tool_registry, service)

        if register_pipeline_tools:
            try:
                register_pipeline_tools(service.tool_registry, service)
                logger.info("✅ Pipeline tools registered into service")
                self.tool_count += 2
            except Exception as e:
                logger.warning(f"⚠️ Failed to register pipeline tools: {e}")

        try:
            from apscheduler.triggers.interval import IntervalTrigger
            APSCHEDULER_AVAILABLE = True
        except ImportError:
            APSCHEDULER_AVAILABLE = False

        if (hasattr(service, "scheduler") and
                service.scheduler is not None and
                APSCHEDULER_AVAILABLE):
            try:
                job_id = "ai_tools_cleanup_downloads"

                try:
                    existing = service.scheduler.get_job(job_id)
                    if existing:
                        service.scheduler.remove_job(job_id)
                        logger.info(f"ℹ️ Removed existing cleanup job: {job_id}")
                except Exception:
                    pass

                trigger = IntervalTrigger(hours=cleanup_interval_hours)

                def cleanup_task():
                    try:
                        logger.info(f"🧹 Starting scheduled cleanup...")
                        cleanup_downloads(
                            directory=self.dest_download_dir,
                            max_age_hours=cleanup_days * 24
                        )
                        logger.info(f"✅ Scheduled cleanup completed")
                    except Exception as e:
                        logger.error(f"❌ Scheduled cleanup failed: {e}")

                service.scheduler.add_job(
                    func=cleanup_task,
                    trigger=trigger,
                    id=job_id,
                    replace_existing=True,
                    max_instances=1,
                    name="Download cleanup",
                )
                logger.info(
                    f"✅ Scheduled cleanup job: every {cleanup_interval_hours}h, "
                    f"removes files older than {cleanup_days} days"
                )
            except Exception as e:
                logger.error(f"❌ Failed to schedule cleanup job: {e}")

        return {
            "status": "success",
            "tools_registered": self.tool_count,
            "registration_errors": len(self.registration_errors),
            "registered_tools": self.registered_tools,
            "timestamp": time.time()
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_tools": self.tool_count,
            "registered_tools": self.registered_tools,
            "registration_errors": self.registration_errors,
            "timestamp": time.time()
        }

# ============================================================================
# GLOBAL FUNCTIONS (for convenience)
# ============================================================================
def get_available_search_engines():
    return list(_ENGINE_HANDLERS.keys())

def create_link_search_config(**kwargs):
    try:
        return LinkSearchConfig(**kwargs)
    except Exception as e:
        logger.error(f"❌ Failed to create LinkSearchConfig: {e}")
        return None
