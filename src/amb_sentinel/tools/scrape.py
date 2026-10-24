# tools/scrape.py
"""
AMB-Sentinel Scrape — Production-grade web scraping.

Version: 1.0.0
Author: AMB Team

Features:
- Multi-engine: HTTP+BS4 (default), Playwright (JS sites), RSS feeds
- Anti-detection: realistic headers, random delays, human-mimic timing
- Parallel batch scraping with ThreadPoolExecutor
- Caching with configurable TTL
- Robots.txt compliance (configurable policy)
- Link / image / structured-data extraction
- SERP result link extraction for search engines
- LRU in-memory cache + optional disk cache
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import random
import re
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from ipaddress import AddressValueError, ip_address
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse, ParseResult, urljoin, urlunparse
from urllib.robotparser import RobotFileParser

try:
    from bs4 import BeautifulSoup, Tag
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None  # type: ignore
    warnings.warn("beautifulsoup4 not installed — scraping severely limited")

try:
    import lxml  # noqa: F401
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False

try:
    import feedparser  # type: ignore
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    feedparser = None  # type: ignore

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout  # type: ignore
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    sync_playwright = None  # type: ignore

try:
    from .http import (
        HttpConfig, http_fetch, http_fetch_batch,
        RateLimiter, CircuitBreaker,
        SecurityViolationError, ResponseTooLargeError,
    )
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    warnings.warn("http module unavailable")

try:
    from .utils import (
        SessionConfig, validate_url, sanitize_filename,
        calculate_file_hash, DEFAULT_CACHE_DIR, SecurityLevel,
        create_waf_headers, USER_AGENTS,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ]

logger = logging.getLogger("Scrape")

def _attr_str(v: Any) -> str:
    """Coerce a BeautifulSoup attribute value to a hashable ``str``.

    ``Tag.get()`` returns ``str | AttributeValueList | None``, where
    ``AttributeValueList`` is a ``list`` subclass. A ``list`` is
    unhashable, so using the raw return value as a ``dict`` key raises
    ``TypeError: unhashable type: 'AttributeValueList'`` on pages where
    a ``<meta>`` attribute happens to be multi-valued. This helper
    guarantees a plain ``str`` regardless of shape.

    * ``None``      → ``""``
    * ``list/tuple`` → ``", ".join(str(x) for x in v)``
    * ``str``        → unchanged
    """
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)

# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class ScrapingEngine(Enum):
    HTTP       = "http"
    PLAYWRIGHT = "playwright"
    RSS        = "rss"
    AUTO       = "auto"


class ContentCategory(Enum):
    ARTICLE       = "article"
    NEWS          = "news"
    PRODUCT       = "product"
    FORUM         = "forum"
    BLOG          = "blog"
    DOCUMENTATION = "documentation"
    ECOMMERCE     = "ecommerce"
    SOCIAL        = "social"
    SEARCH_RESULT = "search_result"
    UNKNOWN       = "unknown"


class RobotsPolicy(Enum):
    STRICT   = "strict"
    MODERATE = "moderate"
    IGNORE   = "ignore"


class AntiDetectionLevel(Enum):
    NONE     = "none"
    BASIC    = "basic"
    ADVANCED = "advanced"
    STEALTH  = "stealth"


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class ScrapingConfig:
    """Full scraping configuration with safe production defaults."""
    engine: ScrapingEngine        = ScrapingEngine.AUTO
    timeout: int                  = 30
    max_retries: int              = 3
    retry_delay: float            = 1.0
    max_redirects: int            = 10
    max_response_size: int        = 20 * 1024 * 1024   # 20 MB
    max_text_length: int          = 500_000
    rate_limit_rpm: int           = 30
    concurrent_requests: int      = 8
    delay_min: float              = 0.3
    delay_max: float              = 1.5
    human_mimic: bool             = True
    robots_policy: RobotsPolicy   = RobotsPolicy.MODERATE
    user_agent: Optional[str]     = None
    proxy: Optional[Dict[str, str]] = None
    verify_ssl: bool              = True
    anti_detection: AntiDetectionLevel = AntiDetectionLevel.ADVANCED
    cache_enabled: bool           = True
    cache_ttl: int                = 3600
    extract_structured: bool      = True
    extract_images: bool          = True
    extract_links: bool           = True
    max_links: int                = 200
    javascript_enabled: bool      = False
    headless: bool                = True
    viewport_width: int           = 1920
    viewport_height: int          = 1080
    wait_for_selector: Optional[str] = None
    wait_timeout: int             = 10
    screenshot_enabled: bool      = False
    # Anti-detection extras
    rotate_user_agent: bool       = True
    add_referer: bool             = True
    accept_cookies: bool          = True


@dataclass
class ScrapingResult:
    """Structured scraping result matching registrar expectations."""
    ok: bool
    url: str
    engine: str
    final_url: Optional[str]          = None
    timestamp: str                     = field(default_factory=lambda: datetime.now().isoformat())
    content_category: str              = ContentCategory.UNKNOWN.value
    title: str                         = ""
    description: str                   = ""
    meta_tags: Dict[str, str]          = field(default_factory=dict)
    headers: Dict[str, str]            = field(default_factory=dict)
    text_content: str                  = ""
    html_content: str                  = ""
    content: str                       = ""   # alias for text_content (used by find_links)
    structured_data: Dict[str, Any]    = field(default_factory=dict)
    links: List[Dict[str, Any]]        = field(default_factory=list)
    images: List[Dict[str, Any]]       = field(default_factory=list)
    status_code: int                   = 0
    response_size: int                 = 0
    processing_time: float             = 0.0
    error: Optional[str]               = None
    error_type: Optional[str]          = None
    warnings: List[str]                = field(default_factory=list)
    redirected: bool                   = False
    redirect_chain: List[Dict[str, Any]] = field(default_factory=dict)
    cache_hit: bool                    = False
    cache_key: Optional[str]           = None
    fingerprint: Optional[str]         = None
    metadata: Dict[str, Any]           = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        return d


# ──────────────────────────────────────────────
# LRU in-memory cache
# ──────────────────────────────────────────────

class ScrapingCache:
    """Thread-safe in-memory TTL cache for scraping results."""

    def __init__(self, maxsize: int = 256, default_ttl: int = 3600):
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                value, expires = self._store[key]
                if time.monotonic() < expires:
                    self._hits += 1
                    return value
                del self._store[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            # Evict oldest if full
            if len(self._store) >= self._maxsize:
                oldest = min(self._store.items(), key=lambda kv: kv[1][1])
                del self._store[oldest[0]]
            self._store[key] = (value, time.monotonic() + (ttl or self._default_ttl))

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._hits = 0
            self._misses = 0
            return count

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(1, self._hits + self._misses),
            }


_scraping_cache = ScrapingCache()
_scraping_stats: Dict[str, int] = defaultdict(int)
_stats_lock = RLock()


# ──────────────────────────────────────────────
# Robots.txt checker
# ──────────────────────────────────────────────

class RobotsTxtChecker:
    """Robots.txt compliance with caching and configurable policy."""

    def __init__(self, policy: RobotsPolicy = RobotsPolicy.MODERATE,
                 cache_ttl: int = 3600, user_agent: str = "AMBSentinel/4.0"):
        self._policy = policy
        self._cache_ttl = cache_ttl
        self._user_agent = user_agent
        self._cache: Dict[str, Tuple[bool, float]] = {}
        self._parser_cache: Dict[str, Tuple[RobotFileParser, float]] = {}
        self._lock = RLock()

    def can_fetch(self, url: str, user_agent: Optional[str] = None) -> bool:
        if self._policy == RobotsPolicy.IGNORE:
            return True
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        except Exception:
            return True

        cache_key = f"{robots_url}:{user_agent or self._user_agent}"
        with self._lock:
            if cache_key in self._cache:
                allowed, ts = self._cache[cache_key]
                if time.monotonic() - ts < self._cache_ttl:
                    return allowed

        try:
            if HTTP_AVAILABLE:
                resp = http_fetch(robots_url, timeout=8, max_retries=1, waf_bypass=False)
                robots_text = resp.get("text", "") if resp.get("ok") else ""
            else:
                import urllib.request
                with urllib.request.urlopen(robots_url, timeout=8) as r:
                    robots_text = r.read().decode("utf-8", errors="ignore")
        except Exception:
            robots_text = ""

        if not robots_text:
            allowed = self._policy != RobotsPolicy.STRICT
        else:
            parser = RobotFileParser()
            parser.parse(robots_text.splitlines())
            ua = user_agent or self._user_agent
            allowed = parser.can_fetch(ua, url)
            if not allowed and self._policy == RobotsPolicy.MODERATE:
                allowed = parser.can_fetch("*", url)

        with self._lock:
            self._cache[cache_key] = (allowed, time.monotonic())
        return allowed


# ──────────────────────────────────────────────
# URL Validator
# ──────────────────────────────────────────────

class URLValidator:
    """Security-focused URL validator (SSRF prevention)."""

    MAX_LEN = 2048
    BLOCKED_SCHEMES = {"file", "data", "javascript", "gopher", "ftp"}

    @classmethod
    def validate(
        cls, url: str, allow_localhost: bool = False
    ) -> Tuple[bool, Optional[str], Optional[ParseResult]]:
        """Returns (is_valid, error_message, parsed_url)."""
        if not url or len(url) > cls.MAX_LEN:
            return False, "URL too long or empty", None
        try:
            parsed = urlparse(url)
        except Exception as exc:
            return False, f"Parse error: {exc}", None
        if parsed.scheme.lower() not in {"http", "https"}:
            return False, f"Unsupported scheme: {parsed.scheme}", None
        if not parsed.hostname:
            return False, "Missing hostname", None
        hostname = parsed.hostname.lower()
        if hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} and not allow_localhost:
            return False, f"Blocked host: {hostname}", None
        try:
            ip = ip_address(hostname)
            if (ip.is_private or ip.is_loopback or ip.is_link_local) and not allow_localhost:
                return False, f"Private IP blocked: {hostname}", None
        except (AddressValueError, ValueError):
            pass
        if "\x00" in url or "%00" in url.lower():
            return False, "Null byte in URL", None
        return True, None, parsed


# ──────────────────────────────────────────────
# Content extractor
# ──────────────────────────────────────────────

class ContentExtractor:
    """Extract structured content from a BeautifulSoup document."""

    @staticmethod
    def extract_title(soup: Any) -> str:
        for sel in ["title", "h1", '[property="og:title"]']:
            el = soup.select_one(sel)
            if el:
                return el.get("content", el.get_text(strip=True))[:500]
        return ""

    @staticmethod
    def extract_description(soup: Any) -> str:
        for sel in [
            'meta[name="description"]',
            'meta[property="og:description"]',
            'meta[name="twitter:description"]',
        ]:
            el = soup.select_one(sel)
            if el and el.get("content"):
                return el["content"][:1000]
        p = soup.find("p")
        if p:
            t = p.get_text(strip=True)
            if 40 < len(t) < 800:
                return t
        return ""

    @staticmethod
    def extract_main_content(soup: Any) -> str:
        # Remove noise
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "iframe", "form", "button", "noscript",
                          "svg", "figure"]):
            tag.decompose()

        # Priority selectors
        for sel in [
            "article", "main", '[role="main"]',
            ".article-body", ".post-content", ".entry-content",
            ".story-body", ".article-content", "#content", "#main",
        ]:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    return text

        # Heuristic: collect longest paragraphs
        body = soup.find("body") or soup
        paras = []
        for p in body.find_all(["p", "div"]):
            t = p.get_text(strip=True)
            if len(t) < 60:
                continue
            links = p.find_all("a")
            link_chars = sum(len(a.get_text(strip=True)) for a in links)
            if len(t) > 0 and link_chars / len(t) < 0.6:
                paras.append(t)
        if paras:
            return "\n\n".join(paras)

        return body.get_text(separator="\n", strip=True)

    @staticmethod
    def extract_links(soup: Any, base_url: str, max_links: int = 200) -> List[Dict[str, Any]]:
        """Extract all anchor links, normalized and deduplicated."""
        links: List[Dict[str, Any]] = []
        seen: set = set()
        base_parsed = urlparse(base_url)

        for a in soup.find_all("a", href=True):
            if len(links) >= max_links:
                break
            href = (a.get("href") or "").strip()
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            try:
                full = urljoin(base_url, href)
                p = urlparse(full)
                if p.scheme not in ("http", "https"):
                    continue
                norm = urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
                if norm in seen:
                    continue
                seen.add(norm)
                links.append({
                    "url": full,
                    "text": a.get_text(strip=True)[:200],
                    "title": a.get("title", ""),
                    "is_external": p.netloc != base_parsed.netloc,
                    "href": href,
                })
            except Exception:
                continue
        return links

    @staticmethod
    def extract_serp_links(soup: Any, base_url: str, engine: str = "generic") -> List[str]:
        """
        Extract actual result URLs from search engine results pages.

        Handles DuckDuckGo, Bing, Google, Brave, Startpage, Qwant, Yandex.
        Falls back to generic link extraction.
        """
        result_urls: List[str] = []
        seen: set = set()

        # Generic: look for result links with certain patterns
        # Most SERPs use <a> with href containing actual URLs embedded as redirect URLs
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href:
                continue

            # DuckDuckGo redirects: /l/?uddg=<encoded_url>
            if "/l/?" in href and "uddg=" in href:
                from urllib.parse import parse_qs, unquote
                qs = parse_qs(urlparse(href).query)
                if "uddg" in qs:
                    real = unquote(qs["uddg"][0])
                    if real.startswith("http") and real not in seen:
                        seen.add(real)
                        result_urls.append(real)
                continue

            # Bing redirects: /search?... or result wrapper
            if "bing.com" in href.lower():
                continue

            # Google redirects: /url?q=<url>
            if "/url?q=" in href or "/url?url=" in href:
                from urllib.parse import parse_qs, unquote
                qs = parse_qs(urlparse(href).query)
                real = qs.get("q", qs.get("url", [None]))[0]
                if real and real.startswith("http") and real not in seen:
                    seen.add(real)
                    result_urls.append(real)
                continue

            # Brave: result links are direct
            # Startpage: /proxy/...
            if "/proxy/" in href:
                continue

            # Direct http(s) links that are not the search engine itself
            try:
                full = urljoin(base_url, href)
                p = urlparse(full)
                if p.scheme not in ("http", "https"):
                    continue
                # Skip search engine own pages
                domain = p.netloc.lower().removeprefix("www.")
                if any(domain.startswith(se) for se in (
                    "duckduckgo.com", "bing.com", "google.com", "google.",
                    "yandex.com", "yandex.", "brave.com", "startpage.com",
                    "qwant.com",
                )):
                    continue
                if full not in seen:
                    seen.add(full)
                    result_urls.append(full)
            except Exception:
                continue

        return result_urls

    @staticmethod
    def extract_images(soup: Any, base_url: str, max_images: int = 30) -> List[Dict[str, Any]]:
        images: List[Dict[str, Any]] = []
        seen: set = set()
        for img in soup.find_all("img", src=True):
            if len(images) >= max_images:
                break
            src = (img.get("src") or "").strip()
            if not src or src.startswith("data:"):
                continue
            try:
                full = urljoin(base_url, src)
                if full in seen:
                    continue
                seen.add(full)
                images.append({
                    "src": full,
                    "alt": img.get("alt", ""),
                    "title": img.get("title", ""),
                    "width": img.get("width"),
                    "height": img.get("height"),
                })
            except Exception:
                continue
        return images

    @staticmethod
    def extract_structured_data(soup: Any) -> Dict[str, Any]:
        sd: Dict[str, Any] = {"json_ld": [], "open_graph": {}, "twitter_cards": {}, "meta_tags": {}}
        try:
            for s in soup.find_all("script", type="application/ld+json"):
                try:
                    sd["json_ld"].append(json.loads(s.string))
                except Exception:
                    pass
            for m in soup.find_all("meta", attrs={"property": re.compile(r"^og:")}):
                p = _attr_str(m.get("property"))
                c = _attr_str(m.get("content"))
                if p and c:
                    sd["open_graph"][p] = c
            for m in soup.find_all("meta", attrs={"name": re.compile(r"^twitter:")}):
                n = _attr_str(m.get("name"))
                c = _attr_str(m.get("content"))
                if n and c:
                    sd["twitter_cards"][n] = c
            for m in soup.find_all("meta"):
                name = _attr_str(
                    m.get("name") or m.get("property") or m.get("http-equiv")
                )
                content = _attr_str(m.get("content"))
                if name and content:
                    sd["meta_tags"][name] = content
        except Exception:
            pass
        return sd

    @staticmethod
    def clean_text(text: str, max_length: int = 0) -> str:
        if not text:
            return ""
        text = "".join(c for c in text if c.isprintable() or c in "\n\t")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        return text


# ──────────────────────────────────────────────
# Dependency manager
# ──────────────────────────────────────────────

class DependencyManager:
    _cache: Dict[str, bool] = {}
    _lock = RLock()

    @classmethod
    def check(cls, name: str) -> bool:
        with cls._lock:
            if name in cls._cache:
                return cls._cache[name]
            mapping = {
                "beautifulsoup": "bs4", "lxml": "lxml",
                "feedparser": "feedparser", "playwright": "playwright",
                "selenium": "selenium",
            }
            mod = mapping.get(name, name)
            try:
                __import__(mod)
                cls._cache[name] = True
            except ImportError:
                cls._cache[name] = False
            return cls._cache[name]

    @classmethod
    def get_available(cls) -> Dict[str, bool]:
        return {n: cls.check(n) for n in
                ("beautifulsoup", "lxml", "feedparser", "playwright", "selenium")}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _humanize_delay(min_s: float = 0.3, max_s: float = 1.5) -> None:
    time.sleep(random.triangular(min_s, max_s, (min_s + max_s) / 2))


def _detect_content_category(url: str, soup: Any) -> str:
    url_l = url.lower()
    if any(k in url_l for k in ("/news/", "/article/", "/blog/", "/post/")):
        return ContentCategory.NEWS.value
    if any(k in url_l for k in ("/product/", "/shop/", "/buy/", "/item/")):
        return ContentCategory.PRODUCT.value
    if soup:
        title = (soup.find("title") or soup.new_tag("x")).get_text().lower()
        if any(k in title for k in ("news", "article", "report")):
            return ContentCategory.NEWS.value
    return ContentCategory.UNKNOWN.value


def _build_http_config(cfg: ScrapingConfig) -> "HttpConfig":
    if not HTTP_AVAILABLE:
        return None  # type: ignore
    return HttpConfig(
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        waf_bypass=(cfg.anti_detection != AntiDetectionLevel.NONE),
        browser_fingerprint=(cfg.anti_detection in {AntiDetectionLevel.ADVANCED, AntiDetectionLevel.STEALTH}),
        verify_ssl=cfg.verify_ssl,
        follow_redirects=True,
        max_redirects=cfg.max_redirects,
        max_response_size=cfg.max_response_size,
        proxy=cfg.proxy,
        human_mimic=(cfg.anti_detection == AntiDetectionLevel.STEALTH),
    )

def fetch_article_text(url: str, min_chars: int = 200) -> Dict[str, Any]:
    """Validate a URL, scrape it, and return clean article text if above min_chars.

    Collapses the LLM's STEP 2/3/4 (validate → scrape → length check) into one
    deterministic tool call. Returns the same shape `store_raw_item` expects.
    """
    from .utils import validate_url as _validate
    from .scrape import scrape_page as _scrape

    if not url or not isinstance(url, str):
        return {"ok": False, "error": "url is required", "url": url}

    v = _validate(url)
    # validate_url returns bool in this codebase; tolerate future dict form.
    ok = v is True or (isinstance(v, dict) and v.get("valid") is True)
    if not ok:
        return {"ok": False, "error": f"invalid URL: {url}", "url": url}

    page = _scrape(url)
    if not isinstance(page, dict) or not page.get("ok", True):
        return {
            "ok": False,
            "error": (page.get("error") if isinstance(page, dict) else "scrape failed") or "scrape failed",
            "url": url,
        }

    text = (page.get("text_content") or page.get("text") or "").strip()
    if len(text) < int(min_chars):
        return {
            "ok": False,
            "error": f"text too short ({len(text)} < {min_chars} chars)",
            "url": url,
            "text_length": len(text),
        }

    return {
        "ok": True,
        "url": url,
        "title": page.get("title", "") or "",
        "text_content": text,
        "text_length": len(text),
        "source_type": "news",
    }

# ──────────────────────────────────────────────
# Core scrape_page
# ──────────────────────────────────────────────

def scrape_page(
    url: str,
    selectors: Optional[List[str]] = None,
    config: Optional[ScrapingConfig] = None,
    http_config: Optional["HttpConfig"] = None,
    use_cache: bool = True,
    robots_checker: Optional[RobotsTxtChecker] = None,
) -> Dict[str, Any]:
    """
    Scrape a web page with full extraction.

    Returns a dict compatible with ScrapingResult.to_dict():
      ok, url, engine, title, description, text_content, content,
      links, images, structured_data, headers, status_code, etc.
    """
    t0 = time.monotonic()
    cfg = config or ScrapingConfig()

    # Validate
    ok_v, err, _ = URLValidator.validate(url)
    if not ok_v:
        return ScrapingResult(
            ok=False, url=url, engine=ScrapingEngine.HTTP.value, error=err,
            processing_time=time.monotonic() - t0,
        ).to_dict()

    # Choose engine
    engine = cfg.engine
    if engine == ScrapingEngine.AUTO:
        engine = ScrapingEngine.PLAYWRIGHT if (cfg.javascript_enabled and PLAYWRIGHT_AVAILABLE) else ScrapingEngine.HTTP

    # RSS feeds
    if engine == ScrapingEngine.RSS or url.lower().endswith((".rss", ".atom", ".xml")):
        return parse_feed(url, config=cfg)

    # Playwright
    if engine == ScrapingEngine.PLAYWRIGHT and PLAYWRIGHT_AVAILABLE:
        return _playwright_scrape(url, cfg)

    # ── HTTP + BS4 ──────────────────────────────
    if not BS4_AVAILABLE:
        return ScrapingResult(
            ok=False, url=url, engine="http",
            error="beautifulsoup4 not installed",
            processing_time=time.monotonic() - t0,
        ).to_dict()

    # Robots
    if cfg.robots_policy != RobotsPolicy.IGNORE:
        checker = robots_checker or RobotsTxtChecker(policy=cfg.robots_policy)
        if not checker.can_fetch(url, cfg.user_agent):
            return ScrapingResult(
                ok=False, url=url, engine="http",
                error="Disallowed by robots.txt",
                processing_time=time.monotonic() - t0,
            ).to_dict()

    # Cache check
    cache_key: Optional[str] = None
    if use_cache and cfg.cache_enabled:
        cache_key = hashlib.sha256(f"{url}:{json.dumps(selectors or [])}".encode()).hexdigest()
        cached = _scraping_cache.get(cache_key)
        if cached:
            cached["cache_hit"] = True
            return cached

    # Human delay
    if cfg.human_mimic:
        _humanize_delay(cfg.delay_min, cfg.delay_max)

    # Fetch
    h_cfg = http_config or _build_http_config(cfg)
    if HTTP_AVAILABLE:
        response = http_fetch(
            url=url, method="GET",
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            waf_bypass=(cfg.anti_detection != AntiDetectionLevel.NONE),
            browser_fingerprint=(cfg.anti_detection in {AntiDetectionLevel.ADVANCED, AntiDetectionLevel.STEALTH}),
            verify_ssl=cfg.verify_ssl,
            follow_redirects=True,
            max_redirects=cfg.max_redirects,
            max_response_size=cfg.max_response_size,
        )
    else:
        import requests as _req
        ua = random.choice(USER_AGENTS) if cfg.rotate_user_agent else "AMB-Sentinel/4.0"
        r = _req.get(url, headers={"User-Agent": ua}, timeout=cfg.timeout,
                     allow_redirects=True, verify=cfg.verify_ssl)
        response = {
            "ok": r.ok, "url": str(r.url),
            "status_code": r.status_code,
            "headers": dict(r.headers),
            "text": r.text,
            "content": r.content,
            "error": None,
            "elapsed": 0,
        }

    if not response.get("ok"):
        return ScrapingResult(
            ok=False, url=url, engine="http",
            error=response.get("error", f"HTTP {response.get('status_code', 0)}"),
            status_code=response.get("status_code", 0),
            processing_time=time.monotonic() - t0,
        ).to_dict()

    html = response.get("text") or ""
    if not html and response.get("content"):
        try:
            html = response["content"].decode("utf-8", errors="replace")
        except Exception:
            html = ""

    # Parse
    parser = "lxml" if LXML_AVAILABLE else "html.parser"
    try:
        soup = BeautifulSoup(html, parser)
    except Exception as exc:
        return ScrapingResult(
            ok=False, url=url, engine="http", error=f"Parse error: {exc}",
            processing_time=time.monotonic() - t0,
        ).to_dict()

    extractor = ContentExtractor()
    title       = extractor.extract_title(soup)
    description = extractor.extract_description(soup)
    main_text   = extractor.extract_main_content(soup)
    cleaned     = extractor.clean_text(main_text, cfg.max_text_length)

    links: List[Dict[str, Any]] = []
    if cfg.extract_links:
        # Try SERP extraction first (for search engines)
        serp = extractor.extract_serp_links(soup, url)
        if len(serp) > 3:
            links = [{"url": u, "text": "", "is_external": True} for u in serp]
        else:
            links = extractor.extract_links(soup, url, cfg.max_links)

    images     = extractor.extract_images(soup, url) if cfg.extract_images else []
    structured = extractor.extract_structured_data(soup) if cfg.extract_structured else {}

    meta_tags: Dict[str, str] = {}
    for m in soup.find_all("meta"):
        n = _attr_str(m.get("name") or m.get("property") or m.get("http-equiv"))
        c = _attr_str(m.get("content"))
        if n and c:
            meta_tags[n] = c

    fp = hashlib.sha256(html.encode()).hexdigest()[:16]

    result = ScrapingResult(
        ok=True,
        url=url,
        final_url=response.get("url", url),
        engine=ScrapingEngine.HTTP.value,
        title=title,
        description=description,
        meta_tags=meta_tags,
        headers=response.get("headers", {}),
        text_content=cleaned,
        content=cleaned,
        html_content=html[:100_000],
        structured_data=structured,
        links=links,
        images=images,
        status_code=response.get("status_code", 200),
        response_size=len(html),
        processing_time=time.monotonic() - t0,
        redirected=response.get("url", url) != url,
        redirect_chain=response.get("redirect_history", []),
        fingerprint=fp,
        metadata={
            "parser": parser,
            "selectors": selectors or [],
            "response_elapsed": response.get("elapsed", 0),
            "content_type": response.get("content_type", ""),
        },
    )
    result.content_category = _detect_content_category(url, soup)

    if use_cache and cfg.cache_enabled and cache_key:
        _scraping_cache.set(cache_key, result.to_dict(), cfg.cache_ttl)

    with _stats_lock:
        _scraping_stats["pages_scraped"] += 1

    logger.info("Scraped %s → %d chars, %d links in %.2fs",
                url, len(cleaned), len(links), result.processing_time)
    return result.to_dict()


# ──────────────────────────────────────────────
# Playwright scrape
# ──────────────────────────────────────────────

def _playwright_scrape(url: str, cfg: ScrapingConfig) -> Dict[str, Any]:
    """Headless browser scrape via Playwright."""
    t0 = time.monotonic()
    if not PLAYWRIGHT_AVAILABLE:
        return scrape_page(url, config=cfg)  # fallback to HTTP

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=cfg.headless,
                args=[
                    "--no-sandbox", "--disable-setuid-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                ],
            )
            ctx = browser.new_context(
                viewport={"width": cfg.viewport_width, "height": cfg.viewport_height},
                user_agent=cfg.user_agent or random.choice(USER_AGENTS),
                ignore_https_errors=not cfg.verify_ssl,
                locale="en-US",
                timezone_id="America/New_York",
            )
            page = ctx.new_page()
            # Anti-detection JS
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.chrome = {runtime: {}};
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            """)
            if cfg.human_mimic:
                _humanize_delay(0.3, 1.0)

            resp = page.goto(url, wait_until="domcontentloaded", timeout=cfg.timeout * 1000)
            if cfg.wait_for_selector:
                try:
                    page.wait_for_selector(cfg.wait_for_selector, timeout=cfg.wait_timeout * 1000)
                except PWTimeout:
                    pass

            html = page.content()
            status = resp.status if resp else 200
            final_url = page.url

            screenshot_data = None
            if cfg.screenshot_enabled:
                import base64
                screenshot_data = base64.b64encode(page.screenshot()).decode()

            browser.close()

        parser = "lxml" if LXML_AVAILABLE else "html.parser"
        soup = BeautifulSoup(html, parser)
        extractor = ContentExtractor()
        cleaned = extractor.clean_text(
            extractor.extract_main_content(soup), cfg.max_text_length
        )
        links = extractor.extract_links(soup, url, cfg.max_links) if cfg.extract_links else []
        images = extractor.extract_images(soup, url) if cfg.extract_images else []

        result = ScrapingResult(
            ok=True, url=url, final_url=final_url, engine=ScrapingEngine.PLAYWRIGHT.value,
            title=extractor.extract_title(soup),
            description=extractor.extract_description(soup),
            text_content=cleaned, content=cleaned,
            html_content=html[:100_000],
            links=links, images=images,
            status_code=status,
            response_size=len(html),
            processing_time=time.monotonic() - t0,
            redirected=final_url != url,
            metadata={"screenshot": bool(screenshot_data)},
        )
        if screenshot_data:
            result.metadata["screenshot_data"] = screenshot_data
        return result.to_dict()

    except Exception as exc:
        logger.error("Playwright scrape failed for %s: %s", url, exc)
        # Fallback to HTTP
        fallback_cfg = ScrapingConfig(**{**cfg.__dict__, "engine": ScrapingEngine.HTTP})
        return scrape_page(url, config=fallback_cfg)


# ──────────────────────────────────────────────
# RSS / Atom feed parser
# ──────────────────────────────────────────────

def parse_feed(
    url: str,
    max_entries: int = 100,
    config: Optional[ScrapingConfig] = None,
    parse_entries: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Parse RSS / Atom feed.

    Returns: ok, url, title, entries, count, error
    """
    t0 = time.monotonic()
    cfg = config or ScrapingConfig()

    if FEEDPARSER_AVAILABLE:
        try:
            feed = feedparser.parse(url)
            entries = []
            for e in (feed.entries or [])[:max_entries]:
                entries.append({
                    "title": getattr(e, "title", ""),
                    "link": getattr(e, "link", ""),
                    "summary": getattr(e, "summary", ""),
                    "published": getattr(e, "published", ""),
                    "author": getattr(e, "author", ""),
                })
            return {
                "ok": True, "url": url,
                "title": getattr(feed.feed, "title", ""),
                "entries": entries,
                "count": len(entries),
                "elapsed": time.monotonic() - t0,
                "error": None,
            }
        except Exception as exc:
            return {"ok": False, "url": url, "error": str(exc), "entries": [], "count": 0}

    # Fallback: fetch XML and parse manually
    try:
        if HTTP_AVAILABLE:
            resp = http_fetch(url, timeout=cfg.timeout)
            xml_text = resp.get("text", "")
        else:
            import urllib.request
            with urllib.request.urlopen(url, timeout=20) as r:
                xml_text = r.read().decode("utf-8", errors="replace")

        if BS4_AVAILABLE and xml_text:
            soup = BeautifulSoup(xml_text, "xml" if LXML_AVAILABLE else "html.parser")
            entries = []
            for item in (soup.find_all("item") or soup.find_all("entry"))[:max_entries]:
                entries.append({
                    "title": (item.find("title") or soup.new_tag("x")).get_text(strip=True),
                    "link": (item.find("link") or soup.new_tag("x")).get("href")
                             or (item.find("link") or soup.new_tag("x")).get_text(strip=True),
                    "summary": (item.find("description") or item.find("summary")
                                or soup.new_tag("x")).get_text(strip=True)[:500],
                })
            title_tag = soup.find("title")
            return {
                "ok": True, "url": url,
                "title": title_tag.get_text(strip=True) if title_tag else "",
                "entries": entries,
                "count": len(entries),
                "elapsed": time.monotonic() - t0,
                "error": None,
            }
    except Exception as exc:
        pass

    return {"ok": False, "url": url, "error": "feedparser unavailable and manual parse failed",
            "entries": [], "count": 0}


# ──────────────────────────────────────────────
# Batch scraping (parallel)
# ──────────────────────────────────────────────

def scrape_batch(
    urls: List[str],
    config: Optional[ScrapingConfig] = None,
    max_concurrent: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Scrape multiple URLs in parallel (order-preserved).
    """
    cfg = config or ScrapingConfig()
    workers = max_concurrent or cfg.concurrent_requests
    n = len(urls)
    results: List[Optional[Dict[str, Any]]] = [None] * n

    def _one(idx: int, url: str) -> Tuple[int, Dict[str, Any]]:
        r = scrape_page(url, config=cfg, **kwargs)
        return idx, r

    if workers <= 1 or n <= 1:
        for i, url in enumerate(urls):
            _, r = _one(i, url)
            results[i] = r
        return results  # type: ignore

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, i, url): i for i, url in enumerate(urls)}
        for fut in concurrent.futures.as_completed(futs):
            try:
                idx, r = fut.result()
                results[idx] = r
            except Exception as exc:
                orig = futs[fut]
                results[orig] = {"ok": False, "url": urls[orig], "error": str(exc), "links": []}

    for i in range(n):
        if results[i] is None:
            results[i] = {"ok": False, "url": urls[i], "error": "Not executed", "links": []}
    return results  # type: ignore


# ──────────────────────────────────────────────
# Async wrappers
# ──────────────────────────────────────────────

async def scrape_async(url: str, **kwargs) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: scrape_page(url, **kwargs))


async def scrape_batch_async(urls: List[str], **kwargs) -> List[Dict[str, Any]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: scrape_batch(urls, **kwargs))


# ──────────────────────────────────────────────
# Browser scrape (public API wrapper)
# ──────────────────────────────────────────────

def browser_scrape(
    url: str,
    engine: Union[str, ScrapingEngine] = ScrapingEngine.PLAYWRIGHT,
    config: Optional[ScrapingConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Scrape using a headless browser."""
    cfg = config or ScrapingConfig()
    cfg.javascript_enabled = True
    if isinstance(engine, str):
        engine = ScrapingEngine(engine)
    cfg.engine = engine
    if engine == ScrapingEngine.PLAYWRIGHT and PLAYWRIGHT_AVAILABLE:
        return _playwright_scrape(url, cfg)
    return scrape_page(url, config=cfg)


# ──────────────────────────────────────────────
# Stats / cache management
# ──────────────────────────────────────────────

def get_scraping_stats() -> Dict[str, Any]:
    with _stats_lock:
        stats = dict(_scraping_stats)
    stats.update(_scraping_cache.stats())
    stats["bs4_available"] = BS4_AVAILABLE
    stats["playwright_available"] = PLAYWRIGHT_AVAILABLE
    stats["lxml_available"] = LXML_AVAILABLE
    stats["feedparser_available"] = FEEDPARSER_AVAILABLE
    return stats


def clear_scraping_cache() -> Dict[str, Any]:
    cleared = _scraping_cache.clear()
    return {"cleared": cleared, "ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION ADDITIONS v5.0 — Content dedup, scrape health, request variance
# ─────────────────────────────────────────────────────────────────────────────

import hashlib as _hashlib
import threading as _threading


class ContentDeduplicator:
    """
    Thread-safe bloom-filter-like content fingerprint registry.
    
    Uses SHA-256 of normalized text content to detect near-duplicates.
    Prevents the same article body from being stored multiple times
    even if fetched from different URLs (mirrors, reposts).
    
    Usage::
    
        dedup = ContentDeduplicator(max_entries=50_000)
        if not dedup.is_seen(text):
            dedup.mark_seen(text)
            store(text)
    """

    def __init__(self, max_entries: int = 50_000):
        self._seen: set = set()
        self._max = max_entries
        self._lock = _threading.Lock()
        self._total_checks = 0
        self._duplicates = 0

    @staticmethod
    def fingerprint(text: str) -> str:
        """Normalize + hash text for deduplication."""
        import re
        # Strip whitespace runs, lowercase, remove punctuation noise
        norm = re.sub(r'\s+', ' ', text.lower().strip())
        # Use first 2000 chars for speed (enough to identify duplicates)
        return _hashlib.sha256(norm[:2000].encode('utf-8', errors='replace')).hexdigest()

    def is_seen(self, text: str) -> bool:
        """Return True if this content was already processed."""
        fp = self.fingerprint(text)
        with self._lock:
            self._total_checks += 1
            if fp in self._seen:
                self._duplicates += 1
                return True
            return False

    def mark_seen(self, text: str) -> None:
        """Register content as processed."""
        fp = self.fingerprint(text)
        with self._lock:
            if len(self._seen) >= self._max:
                # Evict oldest ~10% (set doesn't maintain order — just clear oldest bucket)
                self._seen = set(list(self._seen)[int(self._max * 0.1):])
            self._seen.add(fp)

    def stats(self) -> dict:
        with self._lock:
            return {
                "registered": len(self._seen),
                "max_entries": self._max,
                "total_checks": self._total_checks,
                "duplicates_blocked": self._duplicates,
                "dupe_rate": round(self._duplicates / self._total_checks, 4)
                if self._total_checks else 0.0,
            }


class ScrapeHealthMonitor:
    """
    Per-domain scrape success tracking for automatic skip of broken sources.
    
    After consecutive_fail_threshold failures in a row with no successes,
    the domain enters a cooldown period during which scrape requests are
    skipped. This prevents wasting time on sites that consistently block us.
    """

    def __init__(self, fail_threshold: int = 5, cooldown_s: float = 3600.0):
        self._threshold = fail_threshold
        self._cooldown = cooldown_s
        self._counts: dict = {}   # domain → {"streak": int, "blocked_until": float}
        self._lock = _threading.Lock()

    @staticmethod
    def _domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except Exception:
            return url[:50]

    def record_success(self, url: str) -> None:
        d = self._domain(url)
        with self._lock:
            self._counts.setdefault(d, {"streak": 0, "blocked_until": 0.0})
            self._counts[d]["streak"] = 0

    def record_failure(self, url: str) -> None:
        import time
        d = self._domain(url)
        with self._lock:
            rec = self._counts.setdefault(d, {"streak": 0, "blocked_until": 0.0})
            rec["streak"] += 1
            if rec["streak"] >= self._threshold:
                rec["blocked_until"] = time.time() + self._cooldown
                logger.warning(f"ScrapeHealth: {d} blocked for {self._cooldown}s after {rec['streak']} failures")

    def is_blocked(self, url: str) -> bool:
        import time
        d = self._domain(url)
        with self._lock:
            rec = self._counts.get(d, {})
            blocked_until = rec.get("blocked_until", 0.0)
            if time.time() < blocked_until:
                return True
            if blocked_until and time.time() >= blocked_until:
                # Cooldown expired — reset streak
                rec["streak"] = 0
                rec["blocked_until"] = 0.0
            return False

    def all_health(self) -> dict:
        import time
        with self._lock:
            return {
                d: {
                    "consecutive_failures": rec["streak"],
                    "blocked": time.time() < rec.get("blocked_until", 0.0),
                    "blocked_until": rec.get("blocked_until", 0.0),
                }
                for d, rec in self._counts.items()
            }


# Module-level singletons
_content_deduplicator = ContentDeduplicator(max_entries=50_000)
_scrape_health = ScrapeHealthMonitor(fail_threshold=5, cooldown_s=3600.0)


def get_content_deduplicator() -> "ContentDeduplicator":
    """Return the global content deduplication instance."""
    return _content_deduplicator


def get_scrape_health() -> dict:
    """Return per-domain scrape health status."""
    return _scrape_health.all_health()
