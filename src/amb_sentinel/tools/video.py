# tools/video.py
"""
AMB-Sentinel Video Engine — Production v4.0.0
Author: AMB Team © 2025

Downloads and extracts metadata from online videos via a priority backend chain:
  1. yt-dlp     — supports 1 000+ platforms (YouTube, Vimeo, Twitter …)
  2. HTTP       — direct-download fallback for plain .mp4/.webm/… URLs
  3. ffmpeg     — thumbnail extraction and format transcoding

Key features
────────────
- VideoProcessor class with LRU metadata cache and parallel batch downloads
- YtDlpBackend wraps yt-dlp with full format / quality selection and progress hooks
- GenericHttpBackend handles direct media links with streaming download + resume
- FfmpegBackend provides thumbnail extraction, audio strip, and probe
- All backends degrade gracefully when optional binaries/libs are absent
- Anti-detection WAF headers (via utils) applied to HTTP downloads
- Parallel batch via ThreadPoolExecutor (order-preserved)

Registrar-expected exports
──────────────────────────
  video_info, download_video, download_multiple_videos,
  extract_video_thumbnail, get_video_backends,
  VideoProcessor, VideoConfig, VideoPlatform, VideoQuality,
  ThumbnailStrategy, VideoFormat, DownloadStatus,
  VideoMetadata, VideoResult, VideoBackend,
  YtDlpBackend, GenericHttpBackend, FfmpegBackend
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger("Video")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_CONCURRENT_DOWNLOADS = 4
DEFAULT_DOWNLOAD_DIR_NAME = "downloads/video"
CHUNK_SIZE               = 65_536      # 64 KB streaming chunks
MAX_FILE_SIZE_GB         = 20
INFO_CACHE_TTL_SECONDS   = 600        # 10 minutes

# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class VideoError(Exception):
    """Base video processing error."""

class VideoNotFoundError(VideoError):
    """URL or file does not exist."""

class VideoDownloadError(VideoError):
    """Download failed."""

class SecurityViolationError(VideoError):
    """URL or path failed security validation."""

class VideoProcessingError(VideoError):
    """Post-download processing failed."""


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class VideoPlatform(Enum):
    YOUTUBE    = "youtube"
    VIMEO      = "vimeo"
    TWITTER    = "twitter"
    FACEBOOK   = "facebook"
    INSTAGRAM  = "instagram"
    TIKTOK     = "tiktok"
    REDDIT     = "reddit"
    TWITCH     = "twitch"
    DAILYMOTION = "dailymotion"
    GENERIC    = "generic"
    LOCAL      = "local"
    UNKNOWN    = "unknown"


class VideoQuality(Enum):
    BEST        = "best"
    WORST       = "worst"
    HD_1080     = "1080p"
    HD_720      = "720p"
    SD_480      = "480p"
    SD_360      = "360p"
    AUDIO_ONLY  = "audio_only"
    CUSTOM      = "custom"


class ThumbnailStrategy(Enum):
    FIRST_FRAME  = "first_frame"
    MIDDLE_FRAME = "middle_frame"
    BEST_FRAME   = "best_frame"
    CUSTOM_TIME  = "custom_time"


class VideoFormat(Enum):
    MP4  = "mp4"
    WEBM = "webm"
    MKV  = "mkv"
    MP3  = "mp3"
    M4A  = "m4a"
    OGG  = "ogg"
    AAC  = "aac"
    BEST = "best"


class DownloadStatus(Enum):
    PENDING    = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoMetadata:
    """Rich metadata extracted before or during download."""
    title: str                          = ""
    description: str                    = ""
    url: str                            = ""
    platform: VideoPlatform             = VideoPlatform.UNKNOWN
    uploader: str                       = ""
    upload_date: Optional[str]          = None
    duration: float                     = 0.0
    view_count: int                     = 0
    like_count: int                     = 0
    thumbnail_url: Optional[str]        = None
    formats: List[Dict[str, Any]]       = field(default_factory=list)
    subtitles: Dict[str, Any]           = field(default_factory=dict)
    tags: List[str]                     = field(default_factory=list)
    categories: List[str]               = field(default_factory=list)
    chapters: List[Dict[str, Any]]      = field(default_factory=list)
    is_live: bool                       = False
    age_limit: int                      = 0
    extractor: str                      = ""
    webpage_url: str                    = ""
    raw_info: Dict[str, Any]            = field(default_factory=dict)
    error: Optional[str]                = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title":         self.title,
            "description":   self.description[:500] if self.description else "",
            "url":           self.url,
            "platform":      self.platform.value,
            "uploader":      self.uploader,
            "upload_date":   self.upload_date,
            "duration":      self.duration,
            "view_count":    self.view_count,
            "like_count":    self.like_count,
            "thumbnail_url": self.thumbnail_url,
            "formats_count": len(self.formats),
            "tags":          self.tags[:20],
            "is_live":       self.is_live,
            "extractor":     self.extractor,
            "error":         self.error,
        }


@dataclass
class VideoConfig:
    """Download and processing configuration."""
    quality: VideoQuality               = VideoQuality.BEST
    format: VideoFormat                 = VideoFormat.MP4
    audio_only: bool                    = False
    dest_dir: Union[str, Path]          = "downloads/video"
    filename: Optional[str]             = None
    max_file_size_gb: float             = float(MAX_FILE_SIZE_GB)
    timeout: int                        = 3_600   # 1 h
    max_retries: int                    = 3
    resume: bool                        = True
    verify_ssl: bool                    = True
    cookies_file: Optional[str]         = None
    proxy: Optional[str]                = None
    rate_limit_kbps: Optional[int]      = None   # yt-dlp bandwidth cap
    embed_subtitles: bool               = False
    embed_thumbnail: bool               = False
    extract_flat: bool                  = False
    custom_format: Optional[str]        = None   # raw yt-dlp format string
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    waf_bypass: bool                    = True

    def __post_init__(self) -> None:
        if isinstance(self.dest_dir, str):
            self.dest_dir = Path(self.dest_dir)
        if isinstance(self.quality, str):
            self.quality = VideoQuality(self.quality)
        if isinstance(self.format, VideoFormat):
            pass
        elif isinstance(self.format, str):
            self.format = VideoFormat(self.format)


@dataclass
class DownloadProgress:
    url: str                = ""
    status: DownloadStatus  = DownloadStatus.PENDING
    downloaded_bytes: int   = 0
    total_bytes: int        = 0
    speed_kbps: float       = 0.0
    eta_seconds: float      = 0.0
    percent: float          = 0.0
    filename: str           = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url, "status": self.status.value,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "speed_kbps": round(self.speed_kbps, 2),
            "eta_seconds": round(self.eta_seconds, 1),
            "percent": round(self.percent, 1),
            "filename": self.filename,
        }


@dataclass
class VideoResult:
    """Complete download result."""
    ok: bool
    status: DownloadStatus              = DownloadStatus.FAILED
    url: str                            = ""
    path: Optional[Path]                = None
    audio_path: Optional[Path]          = None
    thumbnail_path: Optional[Path]      = None
    subtitle_paths: List[Path]          = field(default_factory=list)
    metadata: Optional[VideoMetadata]   = None
    size: int                           = 0
    checksum: str                       = ""
    elapsed: float                      = 0.0
    backend: str                        = ""
    error: Optional[str]                = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok":             self.ok,
            "status":         self.status.value,
            "url":            self.url,
            "path":           str(self.path) if self.path else None,
            "audio_path":     str(self.audio_path) if self.audio_path else None,
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "subtitle_paths": [str(p) for p in self.subtitle_paths],
            "metadata":       self.metadata.to_dict() if self.metadata else {},
            "size":           self.size,
            "checksum":       self.checksum,
            "elapsed":        round(self.elapsed, 3),
            "backend":        self.backend,
            "error":          self.error,
        }


# ──────────────────────────────────────────────────────────────────────────────
# URL helpers
# ──────────────────────────────────────────────────────────────────────────────

def _detect_platform(url: str) -> VideoPlatform:
    l = url.lower()
    if "youtube.com" in l or "youtu.be" in l:    return VideoPlatform.YOUTUBE
    if "vimeo.com" in l:                          return VideoPlatform.VIMEO
    if "twitter.com" in l or "x.com" in l:        return VideoPlatform.TWITTER
    if "facebook.com" in l or "fb.com" in l:      return VideoPlatform.FACEBOOK
    if "instagram.com" in l:                      return VideoPlatform.INSTAGRAM
    if "tiktok.com" in l:                         return VideoPlatform.TIKTOK
    if "reddit.com" in l:                         return VideoPlatform.REDDIT
    if "twitch.tv" in l:                          return VideoPlatform.TWITCH
    if "dailymotion.com" in l:                    return VideoPlatform.DAILYMOTION
    return VideoPlatform.GENERIC


def _validate_url(url: str) -> Tuple[bool, Optional[str]]:
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, f"Unsupported scheme: {parsed.scheme}"
        if not parsed.netloc:
            return False, "Missing hostname"
        host = parsed.netloc.split(":")[0]
        if host.startswith("169.254.") or host in ("localhost", "127.0.0.1"):
            return False, "Private/loopback addresses blocked"
    except Exception as exc:
        return False, str(exc)
    return True, None


def _calculate_checksum(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Abstract backend
# ──────────────────────────────────────────────────────────────────────────────

class VideoBackend(ABC):
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def extract_info(self, url: str) -> VideoMetadata: ...

    @abstractmethod
    def download(self, url: str, config: VideoConfig) -> VideoResult: ...


# ──────────────────────────────────────────────────────────────────────────────
# yt-dlp backend
# ──────────────────────────────────────────────────────────────────────────────

class YtDlpBackend(VideoBackend):
    """
    Backend powered by yt-dlp (supports 1 000+ platforms).

    Falls back gracefully if yt-dlp is not installed.
    """

    def __init__(self) -> None:
        self._available = False
        try:
            import yt_dlp  # noqa
            self._available = True
        except ImportError:
            pass

    def is_available(self) -> bool:
        return self._available

    def extract_info(self, url: str) -> VideoMetadata:
        if not self._available:
            return VideoMetadata(url=url, error="yt-dlp not installed")
        try:
            import yt_dlp
            opts = {"quiet": True, "no_warnings": True, "extract_flat": False}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            return self._info_to_metadata(info, url)
        except Exception as exc:
            logger.warning("yt-dlp extract_info failed: %s", exc)
            return VideoMetadata(url=url, error=str(exc))

    def download(self, url: str, config: VideoConfig) -> VideoResult:
        if not self._available:
            return VideoResult(ok=False, url=url, backend="yt_dlp",
                               error="yt-dlp not installed")
        t0 = time.monotonic()
        tmp_dir = tempfile.mkdtemp(prefix="amb_video_")

        try:
            opts = self._build_opts(config, tmp_dir)

            progress_data: Dict[str, Any] = {}

            def _progress_hook(d: Dict[str, Any]) -> None:
                progress_data.update(d)
                if config.progress_callback:
                    status_str = d.get("status", "")
                    pct = 0.0
                    if d.get("total_bytes"):
                        pct = (d.get("downloaded_bytes", 0) / d["total_bytes"]) * 100
                    elif d.get("total_bytes_estimate"):
                        pct = (d.get("downloaded_bytes", 0) / d["total_bytes_estimate"]) * 100
                    config.progress_callback(status_str, {
                        "percent":     round(pct, 1),
                        "speed":       d.get("speed", 0),
                        "downloaded":  d.get("downloaded_bytes", 0),
                    })

            opts["progress_hooks"] = [_progress_hook]

            import yt_dlp
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                metadata = self._info_to_metadata(info, url)

            # Find the downloaded file(s)
            found_files = self._scan_temp_dir(Path(tmp_dir))
            if not found_files:
                return VideoResult(
                    ok=False, url=url, backend="yt_dlp",
                    error="yt-dlp reported success but no output file found",
                    elapsed=time.monotonic() - t0,
                )

            main_file = found_files[0]
            dest_dir  = Path(config.dest_dir).expanduser()
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / main_file.name
            shutil.move(str(main_file), dest_path)

            size     = dest_path.stat().st_size
            checksum = _calculate_checksum(dest_path)

            return VideoResult(
                ok=True, url=url, path=dest_path, status=DownloadStatus.COMPLETED,
                metadata=metadata, size=size, checksum=checksum,
                elapsed=time.monotonic() - t0, backend="yt_dlp",
            )

        except Exception as exc:
            logger.error("yt-dlp download failed: %s", exc)
            return VideoResult(
                ok=False, url=url, error=str(exc), backend="yt_dlp",
                status=DownloadStatus.FAILED, elapsed=time.monotonic() - t0,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_opts(self, config: VideoConfig, tmp_dir: str) -> Dict[str, Any]:
        opts: Dict[str, Any] = {
            "outtmpl":        os.path.join(tmp_dir, "%(title)s.%(ext)s"),
            "quiet":          not config.waf_bypass,
            "no_warnings":    True,
            "retries":        config.max_retries,
            "socket_timeout": 60,
            "http_chunk_size": CHUNK_SIZE,
            "continue":       config.resume,
            "nooverwrites":   False,
        }
        if config.audio_only or config.quality == VideoQuality.AUDIO_ONLY:
            opts["format"]     = "bestaudio/best"
            opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}]
        elif config.custom_format:
            opts["format"] = config.custom_format
        else:
            opts["format"] = self._get_format_string(config)

        if config.cookies_file:
            opts["cookiefile"] = config.cookies_file
        if config.proxy:
            opts["proxy"] = config.proxy
        if config.rate_limit_kbps:
            opts["ratelimit"] = config.rate_limit_kbps * 1024
        if not config.verify_ssl:
            opts["nocheckcertificate"] = True
        if config.embed_subtitles:
            opts["writesubtitles"]  = True
            opts["embedsubtitles"]  = True
            opts["subtitleslangs"]  = ["en"]
        if config.embed_thumbnail:
            opts["writethumbnail"] = True
            opts["embedthumbnail"] = True
        return opts

    def _get_format_string(self, config: VideoConfig) -> str:
        q_map = {
            VideoQuality.HD_1080: "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            VideoQuality.HD_720:  "bestvideo[height<=720]+bestaudio/best[height<=720]",
            VideoQuality.SD_480:  "bestvideo[height<=480]+bestaudio/best[height<=480]",
            VideoQuality.SD_360:  "bestvideo[height<=360]+bestaudio/best[height<=360]",
            VideoQuality.WORST:   "worstvideo+worstaudio/worst",
        }
        if config.quality in q_map:
            return q_map[config.quality]
        if config.format == VideoFormat.WEBM:
            return "bestvideo[ext=webm]+bestaudio[ext=webm]/best[ext=webm]/best"
        return "bestvideo+bestaudio/best"

    def _info_to_metadata(self, info: Dict[str, Any], url: str) -> VideoMetadata:
        return VideoMetadata(
            title        = info.get("title") or "",
            description  = info.get("description") or "",
            url          = url,
            platform     = _detect_platform(url),
            uploader     = info.get("uploader") or info.get("channel") or "",
            upload_date  = info.get("upload_date"),
            duration     = float(info.get("duration") or 0),
            view_count   = int(info.get("view_count") or 0),
            like_count   = int(info.get("like_count") or 0),
            thumbnail_url = info.get("thumbnail"),
            formats      = [
                {"id": f.get("format_id"), "ext": f.get("ext"),
                 "height": f.get("height"), "tbr": f.get("tbr")}
                for f in info.get("formats", [])[:20]
            ],
            subtitles    = {k: True for k in (info.get("subtitles") or {}).keys()},
            tags         = list(info.get("tags") or [])[:50],
            categories   = list(info.get("categories") or []),
            chapters     = info.get("chapters") or [],
            is_live      = bool(info.get("is_live")),
            age_limit    = int(info.get("age_limit") or 0),
            extractor    = info.get("extractor") or "",
            webpage_url  = info.get("webpage_url") or url,
        )

    def _scan_temp_dir(self, tmp: Path) -> List[Path]:
        """Return non-metadata files in download directory, largest first."""
        skip_exts = {".json", ".ytdl", ".part", ".description", ".annotations"}
        files = [
            f for f in tmp.rglob("*")
            if f.is_file() and f.suffix.lower() not in skip_exts
        ]
        return sorted(files, key=lambda f: f.stat().st_size, reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Generic HTTP backend
# ──────────────────────────────────────────────────────────────────────────────

class GenericHttpBackend(VideoBackend):
    """
    Direct HTTP streaming download for plain media URLs.

    Supports: resume (Range), WAF-bypass headers, progress callbacks.
    """

    def is_available(self) -> bool:
        try:
            import requests  # noqa
            return True
        except ImportError:
            return False

    def extract_info(self, url: str) -> VideoMetadata:
        import requests
        try:
            resp = requests.head(url, timeout=15, allow_redirects=True, headers=self._headers())
            ct   = resp.headers.get("Content-Type", "")
            size = int(resp.headers.get("Content-Length", 0))
            parsed = urlparse(url)
            filename = parsed.path.rsplit("/", 1)[-1]
            return VideoMetadata(
                url=url, title=filename or url,
                platform=_detect_platform(url),
                raw_info={"content_type": ct, "size_bytes": size,
                           "final_url": resp.url},
            )
        except Exception as exc:
            return VideoMetadata(url=url, error=str(exc))

    def download(self, url: str, config: VideoConfig) -> VideoResult:
        import requests
        t0 = time.monotonic()
        try:
            dest_dir = Path(config.dest_dir).expanduser()
            dest_dir.mkdir(parents=True, exist_ok=True)

            if config.filename:
                fname = config.filename
            else:
                parsed = urlparse(url)
                fname  = parsed.path.rsplit("/", 1)[-1] or "video_download"
                if not Path(fname).suffix:
                    fname += ".mp4"

            dest_path = dest_dir / fname
            part_path = dest_path.with_suffix(dest_path.suffix + ".part")

            headers = dict(self._headers())
            resume_pos = 0
            if config.resume and part_path.exists():
                resume_pos = part_path.stat().st_size
                headers["Range"] = f"bytes={resume_pos}-"

            resp = requests.get(
                url, headers=headers, stream=True,
                timeout=config.timeout, verify=config.verify_ssl,
                allow_redirects=True,
            )
            if resp.status_code == 416:
                # Range not satisfiable — already complete
                if part_path.exists():
                    part_path.rename(dest_path)
            elif resp.status_code not in (200, 206):
                return VideoResult(
                    ok=False, url=url, error=f"HTTP {resp.status_code}",
                    status=DownloadStatus.FAILED, elapsed=time.monotonic() - t0,
                    backend="http",
                )

            total = int(resp.headers.get("Content-Length", 0)) + resume_pos
            downloaded = resume_pos
            t_start = time.monotonic()

            mode = "ab" if resume_pos else "wb"
            with open(part_path, mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)
                        elapsed    = time.monotonic() - t_start + 0.001
                        speed_kbps = (downloaded - resume_pos) / elapsed / 1024
                        if config.progress_callback:
                            pct = (downloaded / total * 100) if total else 0
                            config.progress_callback("downloading", {
                                "percent": round(pct, 1), "speed_kbps": round(speed_kbps, 1),
                                "downloaded": downloaded, "total": total,
                            })

            part_path.rename(dest_path)
            size     = dest_path.stat().st_size
            checksum = _calculate_checksum(dest_path)
            meta     = VideoMetadata(
                url=url, title=fname, platform=_detect_platform(url),
            )
            return VideoResult(
                ok=True, url=url, path=dest_path, status=DownloadStatus.COMPLETED,
                metadata=meta, size=size, checksum=checksum,
                elapsed=time.monotonic() - t0, backend="http",
            )

        except Exception as exc:
            logger.error("HTTP download failed: %s", exc)
            return VideoResult(
                ok=False, url=url, error=str(exc), backend="http",
                status=DownloadStatus.FAILED, elapsed=time.monotonic() - t0,
            )

    @staticmethod
    def _headers() -> Dict[str, str]:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept":          "video/webm,video/mp4,*/*;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection":      "keep-alive",
        }


# ──────────────────────────────────────────────────────────────────────────────
# ffmpeg backend (thumbnails, probe, transcode)
# ──────────────────────────────────────────────────────────────────────────────

class FfmpegBackend:
    """
    ffmpeg/ffprobe utilities for thumbnail extraction, audio strip, and probe.

    Not a VideoBackend subclass — used as a helper tool.
    """

    def __init__(self) -> None:
        self._ffmpeg  = shutil.which("ffmpeg")
        self._ffprobe = shutil.which("ffprobe")

    def is_available(self) -> bool:
        return bool(self._ffmpeg)

    def extract_thumbnail(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        strategy: ThumbnailStrategy             = ThumbnailStrategy.MIDDLE_FRAME,
        timestamp: Optional[float]              = None,
        width: int                              = 640,
    ) -> Optional[Path]:
        """
        Extract a thumbnail from *video_path* and return the saved image path.

        Returns None if ffmpeg is unavailable or extraction fails.
        """
        if not self._ffmpeg:
            return None
        video_path = Path(video_path)
        if not video_path.exists():
            return None

        duration = self._get_duration(video_path)
        if timestamp is None:
            if strategy == ThumbnailStrategy.FIRST_FRAME:
                ts = "00:00:01"
            elif strategy == ThumbnailStrategy.BEST_FRAME:
                ts = str(duration * 0.25)
            else:
                ts = str(duration * 0.5) if duration > 0 else "00:00:01"
        else:
            ts = str(timestamp)

        if output_path is None:
            output_path = video_path.with_suffix(".thumb.jpg")
        output_path = Path(output_path)

        try:
            cmd = [
                self._ffmpeg, "-y", "-loglevel", "error",
                "-ss", str(ts), "-i", str(video_path),
                "-frames:v", "1",
                "-vf", f"scale={width}:-1",
                "-q:v", "2",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, timeout=120)
            return output_path if output_path.exists() else None
        except Exception as exc:
            logger.warning("Thumbnail extraction failed: %s", exc)
            return None

    def extract_audio(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        codec: str = "mp3",
        sample_rate: int = 16_000,
    ) -> Optional[Path]:
        """Strip audio track from *video_path*. Returns output path or None."""
        if not self._ffmpeg:
            return None
        video_path = Path(video_path)
        if output_path is None:
            output_path = video_path.with_suffix(f".{codec}")
        output_path = Path(output_path)
        try:
            cmd = [
                self._ffmpeg, "-y", "-loglevel", "error",
                "-i", str(video_path),
                "-vn", "-acodec", codec,
                "-ar", str(sample_rate), "-ac", "1",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, timeout=1800)
            return output_path if output_path.exists() else None
        except Exception as exc:
            logger.warning("Audio extraction failed: %s", exc)
            return None

    def transcode_video(
        self,
        src: Union[str, Path],
        dest: Union[str, Path],
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int         = 23,
    ) -> bool:
        """Transcode *src* to *dest*. Returns True on success."""
        if not self._ffmpeg:
            return False
        try:
            cmd = [
                self._ffmpeg, "-y", "-loglevel", "error",
                "-i", str(src),
                "-c:v", video_codec, "-crf", str(crf),
                "-c:a", audio_codec,
                str(dest),
            ]
            subprocess.run(cmd, check=True, timeout=7200)
            return True
        except Exception as exc:
            logger.warning("Transcode failed: %s", exc)
            return False

    def probe(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Run ffprobe and return JSON stream info."""
        if not self._ffprobe:
            return {}
        import json
        try:
            cmd = [
                self._ffprobe, "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return json.loads(res.stdout) if res.returncode == 0 else {}
        except Exception:
            return {}

    def _get_duration(self, path: Path) -> float:
        probe = self.probe(path)
        try:
            return float(probe.get("format", {}).get("duration", 0))
        except Exception:
            return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# VideoProcessor — orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class VideoProcessor:
    """
    Main video processor that selects the best backend and manages caching.

    Usage:
        proc = VideoProcessor()
        meta = proc.get_video_info("https://www.youtube.com/watch?v=...")
        result = proc.download_video("https://...", config)
    """

    def __init__(self, config: Optional[VideoConfig] = None) -> None:
        self.config   = config or VideoConfig()
        self._ytdlp   = YtDlpBackend()
        self._http    = GenericHttpBackend()
        self._ffmpeg  = FfmpegBackend()
        self._cache: Dict[str, Tuple[VideoMetadata, float]] = {}
        self._lock    = threading.Lock()

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_video_info(self, url: str, use_cache: bool = True) -> VideoMetadata:
        """Fetch metadata for *url*, using cache when available."""
        valid, err = _validate_url(url)
        if not valid:
            return VideoMetadata(url=url, error=err)

        if use_cache:
            with self._lock:
                cached = self._cache.get(url)
            if cached:
                meta, ts = cached
                if time.monotonic() - ts < INFO_CACHE_TTL_SECONDS:
                    return meta

        meta = None
        if self._ytdlp.is_available():
            meta = self._ytdlp.extract_info(url)
        if not meta or meta.error:
            meta = self._http.extract_info(url)

        if use_cache and meta and not meta.error:
            with self._lock:
                self._cache[url] = (meta, time.monotonic())

        return meta

    # ── Single download ───────────────────────────────────────────────────────

    def download_video(
        self,
        url: str,
        config: Optional[VideoConfig] = None,
    ) -> VideoResult:
        """
        Download a video from *url* using the best available backend.

        Falls back: yt-dlp → HTTP.
        """
        cfg   = config or self.config
        valid, err = _validate_url(url)
        if not valid:
            return VideoResult(ok=False, url=url, error=err)

        return self._download_with_retry(url, cfg)

    def _download_with_retry(self, url: str, config: VideoConfig) -> VideoResult:
        last_result: Optional[VideoResult] = None

        for attempt in range(config.max_retries + 1):
            if attempt > 0:
                wait = min(2.0 * (2 ** (attempt - 1)), 30.0)
                logger.info("Retry %d/%d for %s — waiting %.1fs", attempt, config.max_retries, url, wait)
                time.sleep(wait)

            # Prefer yt-dlp for known platforms
            platform = _detect_platform(url)
            use_ytdlp = (
                self._ytdlp.is_available() and
                platform not in (VideoPlatform.GENERIC, VideoPlatform.UNKNOWN)
            )
            result = (
                self._ytdlp.download(url, config) if use_ytdlp
                else self._http.download(url, config) if self._http.is_available()
                else VideoResult(ok=False, url=url, error="No download backend available")
            )
            last_result = result
            if result.ok:
                return result

            # Fallback: if yt-dlp failed, try HTTP
            if use_ytdlp and not result.ok and self._http.is_available():
                logger.info("yt-dlp failed — falling back to HTTP for %s", url)
                result = self._http.download(url, config)
                last_result = result
                if result.ok:
                    return result

        return last_result or VideoResult(ok=False, url=url, error="All download attempts failed")

    # ── Thumbnail ─────────────────────────────────────────────────────────────

    def extract_thumbnail(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        strategy: ThumbnailStrategy             = ThumbnailStrategy.MIDDLE_FRAME,
        timestamp: Optional[float]              = None,
        width: int                              = 640,
    ) -> Optional[Path]:
        """Extract a thumbnail from a local video file."""
        return self._ffmpeg.extract_thumbnail(
            video_path, output_path, strategy, timestamp, width
        )

    # ── Parallel batch ────────────────────────────────────────────────────────

    def download_multiple(
        self,
        urls: List[str],
        config: Optional[VideoConfig]                               = None,
        max_concurrent: int                                         = MAX_CONCURRENT_DOWNLOADS,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[VideoResult]:
        """
        Download multiple videos in parallel (order-preserved).

        Args:
            urls:             List of URLs to download.
            config:           Shared VideoConfig applied to all downloads.
            max_concurrent:   Thread-pool size.
            progress_callback: ``cb(done, total, url)``.
        """
        if not urls:
            return []

        cfg     = config or self.config
        total   = len(urls)
        workers = min(max(1, max_concurrent), MAX_CONCURRENT_DOWNLOADS, total)

        results: List[Optional[VideoResult]] = [None] * total
        done_count = 0
        counter_lock = threading.Lock()

        def _dl(idx: int, url: str) -> Tuple[int, VideoResult]:
            r = self.download_video(url, cfg)
            return idx, r

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="VideoDownload") as pool:
            fut_map: Dict[Future, int] = {
                pool.submit(_dl, i, u): i for i, u in enumerate(urls)
            }
            for fut in as_completed(fut_map):
                orig = fut_map[fut]
                try:
                    idx, r = fut.result(timeout=cfg.timeout)
                    results[idx] = r
                except Exception as exc:
                    results[orig] = VideoResult(
                        ok=False, url=urls[orig],
                        error=str(exc), status=DownloadStatus.FAILED,
                    )

                with counter_lock:
                    done_count += 1
                    done = done_count

                if progress_callback:
                    try:
                        progress_callback(done, total, urls[orig])
                    except Exception:
                        pass

        for i in range(total):
            if results[i] is None:
                results[i] = VideoResult(
                    ok=False, url=urls[i], error="Not executed",
                )

        ok_n = sum(1 for r in results if r and r.ok)
        logger.info("download_multiple: %d/%d OK", ok_n, total)
        return results  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# Public convenience functions
# ──────────────────────────────────────────────────────────────────────────────

_default_processor: Optional[VideoProcessor] = None
_proc_lock = threading.Lock()


def _get_processor() -> VideoProcessor:
    global _default_processor
    with _proc_lock:
        if _default_processor is None:
            _default_processor = VideoProcessor()
    return _default_processor


def video_info(
    url: str,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Return metadata dict for *url*.

    Always returns a dict (never raises). Check ``result["error"]`` for failures.
    """
    meta = _get_processor().get_video_info(url, use_cache=use_cache)
    return meta.to_dict()


def download_video(
    url: str,
    dest_dir: Union[str, Path]      = "downloads/video",
    quality: Union[str, VideoQuality] = VideoQuality.BEST,
    audio_only: bool                = False,
    format: Union[str, VideoFormat] = VideoFormat.MP4,
    filename: Optional[str]         = None,
    timeout: int                    = 3_600,
    max_retries: int                = 3,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Download a single video and return a result dict.

    Returns dict with keys: ok, status, url, path, size, metadata, elapsed, error.
    """
    if isinstance(quality, str):
        quality = VideoQuality(quality)
    if isinstance(format, str):
        format = VideoFormat(format)

    config = VideoConfig(
        dest_dir=dest_dir, quality=quality, audio_only=audio_only,
        format=format, filename=filename, timeout=timeout,
        max_retries=max_retries, progress_callback=progress_callback,
    )
    result = _get_processor().download_video(url, config)
    return result.to_dict()


def download_multiple_videos(
    urls: List[str],
    dest_dir: Union[str, Path] = "downloads/video",
    max_concurrent: int        = MAX_CONCURRENT_DOWNLOADS,
    quality: Union[str, VideoQuality] = VideoQuality.BEST,
    audio_only: bool           = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Download multiple videos in parallel. Returns list of result dicts."""
    if isinstance(quality, str):
        quality = VideoQuality(quality)
    config = VideoConfig(dest_dir=dest_dir, quality=quality, audio_only=audio_only)
    results = _get_processor().download_multiple(
        urls, config=config, max_concurrent=max_concurrent
    )
    return [r.to_dict() for r in results]


def extract_video_thumbnail(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    strategy: Union[str, ThumbnailStrategy] = ThumbnailStrategy.MIDDLE_FRAME,
    timestamp: Optional[float]              = None,
    width: int                              = 640,
) -> Dict[str, Any]:
    """
    Extract a thumbnail from a local video file.

    Returns: ``{"ok": bool, "path": str|None, "error": str|None}``.
    """
    if isinstance(strategy, str):
        strategy = ThumbnailStrategy(strategy)
    result_path = _get_processor().extract_thumbnail(
        video_path, output_path, strategy, timestamp, width
    )
    if result_path and result_path.exists():
        return {"ok": True, "path": str(result_path)}
    return {"ok": False, "path": None, "error": "Thumbnail extraction failed (ffmpeg required)"}


def get_video_backends() -> Dict[str, Any]:
    """Return availability and capability of each video backend."""
    proc  = _get_processor()
    ytdlp = proc._ytdlp
    http  = proc._http
    ffm   = proc._ffmpeg

    ytdlp_version: Optional[str] = None
    if ytdlp.is_available():
        try:
            import yt_dlp
            ytdlp_version = yt_dlp.version.__version__
        except Exception:
            pass

    return {
        "yt_dlp":   {
            "available": ytdlp.is_available(),
            "version":   ytdlp_version,
            "note":      "Supports 1000+ platforms",
        },
        "http":     {
            "available": http.is_available(),
            "note":      "Direct URL download fallback",
        },
        "ffmpeg":   {
            "available": ffm.is_available(),
            "ffmpeg":    ffm._ffmpeg,
            "ffprobe":   ffm._ffprobe,
            "note":      "Thumbnail / transcode / probe",
        },
        "max_concurrent": MAX_CONCURRENT_DOWNLOADS,
        "supported_platforms": [p.value for p in VideoPlatform],
        "supported_qualities":  [q.value for q in VideoQuality],
    }


# Production Additions v5.0
import threading as _vid_threading


class VideoMetrics:
    """Track video download success rates per platform."""

    def __init__(self):
        self._lock = _vid_threading.Lock()
        self._records: dict = {}

    def record(self, platform: str, success: bool, size_bytes: int = 0) -> None:
        with self._lock:
            r = self._records.setdefault(platform, {"ok": 0, "err": 0, "bytes": 0})
            if success:
                r["ok"] += 1
                r["bytes"] += size_bytes
            else:
                r["err"] += 1

    def summary(self) -> dict:
        with self._lock:
            return {
                p: {
                    "success": r["ok"],
                    "errors": r["err"],
                    "success_rate": round(r["ok"] / (r["ok"] + r["err"]), 4) if (r["ok"] + r["err"]) else 0.0,
                    "total_mb": round(r["bytes"] / 1_048_576, 2),
                }
                for p, r in self._records.items()
            }


_video_metrics = VideoMetrics()


def get_video_metrics() -> dict:
    """Return per-platform video download performance."""
    return _video_metrics.summary()
