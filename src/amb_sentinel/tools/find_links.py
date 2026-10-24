# tools/find_links.py
"""
AMB-Sentinel FindLinks — Production-grade link/URL discovery.

Version: 4.0.0
Author: AMB Team © 2025

Key fixes over v3:
1. All 7 search engines run SIMULTANEOUSLY (ThreadPoolExecutor) — not sequentially.
2. search_links() returns ok=False when ALL engines fail (not ok=True with 0 links).
3. find_hidden_resources() includes 'links' key in result (was 'hidden_resources').
4. True result extraction from SERP pages via scrape.py ContentExtractor.extract_serp_links.
5. Timeout enforced per-engine via futures.
6. Filter, deduplicate, and normalise applied consistently.

Functions:
  search_links(query, …)
  search_with_dorks(dork_query, …)   ← registrar alias: 'query' → 'dork_query'
  find_file_links(site, …)
  find_hidden_resources(base_url, …)
  extract_links_from_page(url, …)
  expand_links(seed_urls, …)
"""

from __future__ import annotations
import threading
import os
import concurrent.futures
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urlparse

from .scrape import (
    scrape_page, URLValidator, RobotsTxtChecker, ScrapingConfig,
    AntiDetectionLevel, RobotsPolicy,
)
from .http import http_fetch

logger = logging.getLogger("FindLink")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class LinkSearchConfig:
    """Configuration for link-search and expansion tools."""
    engines: List[str] = field(default_factory=lambda: [
        "duckduckgo", "bing", "google", "yandex", "brave", "startpage", "qwant"
    ])
    max_results_default: int      = 100
    per_request_timeout: int      = 30
    respect_robots: bool          = True
    rate_limit_wait_seconds: float = 0.3   # between per-engine calls (when sequential)
    enable_proxies: bool          = False
    proxy_list: List[str]         = field(default_factory=list)
    searxng_url: str = field(default_factory=lambda: os.environ.get("AMB_SEARXNG_URL", ""))
    rss_feeds: List[str] = field(
        default_factory=lambda: [
            u.strip() for u in os.environ.get("AMB_RSS_FEEDS", "").split(",") if u.strip()
        ]
    )
    max_concurrent_searches: int  = 7      # all engines simultaneously
    retry_attempts: int           = 2
    cache_results: bool           = True
    # Dork-specific
    max_dork_queries_per_engine: int = 5
    # Built-in dork patterns
    dork_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "common_files": [
            "site:{site} filetype:pdf",
            "site:{site} filetype:doc OR filetype:docx",
            "site:{site} filetype:xls OR filetype:xlsx",
            "site:{site} filetype:ppt OR filetype:pptx",
            "site:{site} filetype:txt OR filetype:csv",
            "site:{site} filetype:zip OR filetype:rar",
        ],
        "hidden_resources": [
            "site:{site} inurl:admin",
            "site:{site} inurl:login",
            "site:{site} inurl:config",
            "site:{site} inurl:backup",
            "site:{site} intitle:index of",
            "site:{site} inurl:.git",
            "site:{site} inurl:.env",
        ],
        "sensitive_files": [
            "site:{site} ext:sql",
            "site:{site} ext:log",
            "site:{site} ext:env",
            "site:{site} ext:yml OR ext:yaml",
            "site:{site} ext:bak OR ext:old",
        ],
    })


_DEFAULT_CONFIG = LinkSearchConfig()


# ──────────────────────────────────────────────
# URL helpers
# ──────────────────────────────────────────────

_TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_content",
             "utm_term", "fbclid", "gclid", "msclkid"}


def _normalize_url(url: str) -> str:
    """Remove fragments, strip tracking params, normalise scheme."""
    try:
        if "#" in url:
            url = url.split("#")[0]
        parsed = urlparse(url)
        if not parsed.scheme:
            url = "https://" + url
            parsed = urlparse(url)
        # Strip tracking query params
        if parsed.query:
            kept = "&".join(
                p for p in parsed.query.split("&")
                if not any(p.startswith(t) for t in _TRACKING)
            )
            from urllib.parse import urlunparse
            url = urlunparse((parsed.scheme, parsed.netloc, parsed.path,
                              parsed.params, kept, ""))
        return url
    except Exception:
        return url


def _dedupe(urls: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for u in urls:
        key = _normalize_url(u).rstrip("/").lower()
        if key not in seen:
            seen.add(key)
            out.append(_normalize_url(u))
    return out


def _filter(links: List[str],
            include: Optional[List[str]] = None,
            exclude: Optional[List[str]] = None) -> List[str]:
    result = []
    for lnk in links:
        if exclude and any(re.search(p, lnk, re.IGNORECASE) for p in exclude):
            continue
        if include and not any(re.search(p, lnk, re.IGNORECASE) for p in include):
            continue
        result.append(lnk)
    return result


def _collect_links_from_scrape(scrape_res: Dict[str, Any], max_links: int = 500) -> List[str]:
    """Extract link URLs from a scrape_page result."""
    urls: List[str] = []
    for l in scrape_res.get("links", []):
        href = l.get("url") or l.get("href") or l.get("link")
        if href:
            urls.append(href)
            if max_links and len(urls) >= max_links:
                break
    return urls


# ──────────────────────────────────────────────
# Per-engine search handlers
# Each returns (List[str] links, Optional[str] error)
# ──────────────────────────────────────────────

def _search_engine(
    engine: str,
    query: str,
    max_results: int,
    timeout: int,
    respect_robots: bool,
    proxy: Optional[str] = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Dispatch to the correct search-engine endpoint and extract result links.

    Uses scrape_page with STEALTH anti-detection so SERP pages are
    fetched with realistic browser headers, then ContentExtractor pulls
    the actual result URLs via extract_serp_links().
    """
    if engine == "searxng":
        return _searxng_search(query, max_results, timeout)
    if engine == "rss":
        return _rss_search(query, max_results, timeout)

    _ENDPOINTS: Dict[str, str] = {
        "duckduckgo": "https://html.duckduckgo.com/html/?q={q}",
        "bing":       "https://www.bing.com/search?q={q}&count={n}",
        "google":     "https://www.google.com/search?q={q}&num={n}&hl=en",
        "yandex":     "https://yandex.com/search/?text={q}&numdoc={n}",
        "brave":      "https://search.brave.com/search?q={q}&count={n}",
        "startpage":  "https://www.startpage.com/do/dsearch?query={q}&language=english",
        "qwant":      "https://www.qwant.com/?q={q}&t=web",
    }
    if engine not in _ENDPOINTS:
        return [], f"unsupported_engine:{engine}"

    q = quote_plus(query)
    url = _ENDPOINTS[engine].format(q=q, n=min(max_results, 50))

    cfg = ScrapingConfig(
        timeout=timeout,
        max_retries=_DEFAULT_CONFIG.retry_attempts,
        cache_enabled=_DEFAULT_CONFIG.cache_results,
        anti_detection=AntiDetectionLevel.ADVANCED,
        human_mimic=False,          # speed — no human delay inside parallel workers
        robots_policy=RobotsPolicy.MODERATE if respect_robots else RobotsPolicy.IGNORE,
        proxy={"http": proxy, "https": proxy} if proxy else None,
        extract_links=True,
        extract_images=False,
        extract_structured=False,
        max_links=max_results * 3,
    )

    try:
        res = scrape_page(url, config=cfg, use_cache=False)
        if not res.get("ok"):
            return [], res.get("error", "scrape_failed")
        # Prefer SERP-aware link extraction (already done by scrape_page)
        links = _collect_links_from_scrape(res, max_results * 2)
        # Also mine raw HTML for direct URLs
        html = res.get("html_content", "")
        if html:
            extra = re.findall(r'https?://[^\s"\'<>\\]{10,400}', html)
            links += extra
        return _dedupe(links)[:max_results], None
    except Exception as exc:
        logger.debug("Engine %s error for %r: %s", engine, query[:50], exc)
        return [], str(exc)


_ENGINE_ALIASES = {"yahoo": "duckduckgo", "ddg": "duckduckgo", "google news": "rss", "news": "rss"}
_SERP_BLOCKED: Dict[str, float] = {}
_SERP_BLOCK_TTL_S = 6 * 3600


def _select_engines(requested: Optional[List[str]]) -> List[str]:
    """Return a de-duplicated, valid, healthy engine list. Providers
    first; SERP scraping last."""
    valid = set(_ENGINE_HANDLERS) | {"searxng", "rss"}
    out: List[str] = []
    for e in (requested or _DEFAULT_CONFIG.engines):
        e = _ENGINE_ALIASES.get(str(e).strip().lower(), str(e).strip().lower())
        if e in valid and e not in out:
            out.append(e)
    now = time.monotonic()
    out = [e for e in out if now - _SERP_BLOCKED.get(e, -1e9) > _SERP_BLOCK_TTL_S]
    order = {"searxng": 0, "rss": 1}
    out.sort(key=lambda e: order.get(e, 2))
    if _DEFAULT_CONFIG.searxng_url and "searxng" not in out:
        out.insert(0, "searxng")
    if _DEFAULT_CONFIG.rss_feeds and "rss" not in out:
        out.insert(1 if out and out[0] == "searxng" else 0, "rss")
    return _engine_scorer.ranked_engines(out) if out else ["duckduckgo"]


def _searxng_search(query: str, max_results: int, timeout: int) -> Tuple[List[str], Optional[str]]:
    base = _DEFAULT_CONFIG.searxng_url.rstrip("/")
    if not base:
        return [], "searxng_not_configured"
    r = http_fetch(
        f"{base}/search",
        params={"q": query, "format": "json", "language": "en"},
        timeout=timeout, max_retries=1, waf_bypass=False,
    )
    if not r.get("ok"):
        return [], r.get("error") or f"http_{r.get('status_code')}"
    data = r.get("json_data") or {}
    links = [x.get("url") for x in (data.get("results") or []) if x.get("url")]
    return _dedupe(links)[:max_results], (None if links else "no_results")


def _rss_search(query: str, max_results: int, timeout: int) -> Tuple[List[str], Optional[str]]:
    import feedparser
    terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    hits: List[str] = []
    for feed_url in _DEFAULT_CONFIG.rss_feeds:
        r = http_fetch(feed_url, timeout=timeout, max_retries=1, waf_bypass=False)
        if not r.get("ok") or not r.get("text"):
            continue
        for e in feedparser.parse(r["text"]).entries:
            blob = f"{e.get('title', '')} {e.get('summary', '')}".lower()
            if any(t in blob for t in terms) and e.get("link"):
                hits.append(e["link"])
    return _dedupe(hits)[:max_results], (None if hits else "no_matching_entries")

# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def search_links(
    query: str,
    engines: Optional[List[str]] = None,
    max_results: int = 50,
    timeout: Optional[int] = None,
    respect_robots: Optional[bool] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    use_proxy: bool = False,
) -> Dict[str, Any]:
    """
    Search the web for links related to *query* using multiple engines IN PARALLEL.

    Returns:
        ok           — True if at least one engine returned results
        query        — echoed query
        links        — deduplicated, filtered link list
        count        — len(links)
        per_engine   — per-engine breakdown
        errors       — list of engine errors
        elapsed      — wall-clock seconds
        engines_used — engine list used
    """
    t0 = time.monotonic()
    eng_list = _select_engines(engines)
    to = int(timeout or _DEFAULT_CONFIG.per_request_timeout)
    robots = respect_robots if respect_robots is not None else _DEFAULT_CONFIG.respect_robots

    proxy: Optional[str] = None
    if use_proxy and _DEFAULT_CONFIG.enable_proxies and _DEFAULT_CONFIG.proxy_list:
        proxy = random.choice(_DEFAULT_CONFIG.proxy_list)

    aggregated: List[str] = []
    per_engine: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    any_ok = False

    # Run all engines concurrently
    workers = min(len(eng_list), _DEFAULT_CONFIG.max_concurrent_searches)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                _search_engine, eng, query, max_results, to, robots, proxy
            ): eng
            for eng in eng_list
        }
        try:
            for fut in concurrent.futures.as_completed(futs, timeout=to + 15):
                eng = futs[fut]
                try:
                    links, err = fut.result()
                    if err:
                        if "robots" in str(err).lower():
                            _SERP_BLOCKED[eng] = time.monotonic()
                        _engine_scorer.update(eng, results=0, errors=1, dupes=0)
                        per_engine[eng] = {"ok": False, "links": [], "count": 0, "error": err}
                        errors.append(f"{eng}:{err}")
                    else:
                        if include_patterns or exclude_patterns:
                            links = _filter(links, include_patterns, exclude_patterns)
                        per_engine[eng] = {"ok": True, "links": links, "count": len(links), "error": None}
                        _engine_scorer.update(eng, results=len(links), errors=0, dupes=0)
                        aggregated.extend(links)
                        if links:
                            any_ok = True
                except concurrent.futures.TimeoutError:
                    per_engine[eng] = {"ok": False, "links": [], "count": 0, "error": "timeout"}
                    errors.append(f"{eng}:timeout")
                except Exception as exc:
                    per_engine[eng] = {"ok": False, "links": [], "count": 0, "error": str(exc)}
                    errors.append(f"{eng}:{exc}")
        except concurrent.futures.TimeoutError:
            errors.append("search:overall_timeout")

    aggregated = _dedupe(aggregated)
    if include_patterns or exclude_patterns:
        aggregated = _filter(aggregated, include_patterns, exclude_patterns)
    if max_results and len(aggregated) > max_results:
        aggregated = aggregated[:max_results]

    logger.info("search_links %r → %d links from %d engines in %.2fs",
                query[:60], len(aggregated), len(eng_list), time.monotonic() - t0)

    return {
        "ok": any_ok,
        "query": query,
        "links": aggregated,
        "count": len(aggregated),
        "per_engine": per_engine,
        "errors": errors,
        "elapsed": round(time.monotonic() - t0, 3),
        "engines_used": eng_list,
    }


def search_with_dorks(
    dork_query: str,
    filetypes: Optional[List[str]] = None,
    engines: Optional[List[str]] = None,
    max_results: int = 100,
    timeout: Optional[int] = None,
    respect_robots: Optional[bool] = None,
    site: Optional[str] = None,
    use_proxy: bool = False,
) -> Dict[str, Any]:
    """
    Search using advanced Google-dork queries.

    Alias 'query' is auto-remapped to 'dork_query' by the registrar alias resolver.

    Args:
        dork_query  : Base dork string, e.g. "filetype:pdf", "inurl:admin"
        filetypes   : Optional list of file extensions to constrain results
        engines     : Search engines to use (default: all)
        max_results : Max links per dork query
        timeout     : Per-request timeout
        respect_robots: Respect robots.txt
        site        : Constrain search to this domain
        use_proxy   : Use proxy rotation if configured
    """
    t0 = time.monotonic()

    # Build dork query list
    dork_queries = _generate_dork_queries(dork_query, site, filetypes,
                                          _DEFAULT_CONFIG.dork_patterns.get("common_files", []))
    dork_queries = dork_queries[:_DEFAULT_CONFIG.max_dork_queries_per_engine]
    logger.info("search_with_dorks: %d dork queries", len(dork_queries))

    all_links: List[str] = []
    per_dork: Dict[str, Any] = {}
    all_errors: List[str] = []

    # Run dork queries sequentially (each spawns parallel engine calls internally)
    for i, q in enumerate(dork_queries):
        result = search_links(
            query=q,
            engines=engines,
            max_results=max_results,
            timeout=timeout,
            respect_robots=respect_robots,
            use_proxy=use_proxy,
        )
        per_dork[q] = {
            "links": result.get("links", []),
            "count": result.get("count", 0),
            "ok": result.get("ok", False),
            "errors": result.get("errors", []),
        }
        if result.get("ok"):
            all_links.extend(result.get("links", []))
        all_errors.extend(result.get("errors", []))

        if i < len(dork_queries) - 1:
            time.sleep(_DEFAULT_CONFIG.rate_limit_wait_seconds)

    all_links = _dedupe(all_links)
    if max_results and len(all_links) > max_results:
        all_links = all_links[:max_results]

    return {
        "ok": len(all_links) > 0,
        "dork_query": dork_query,
        "site": site,
        "filetypes": filetypes,
        "total_links": all_links,
        "links": all_links,         # alias for consistent API
        "per_dork_query": per_dork,
        "total_count": len(all_links),
        "count": len(all_links),
        "errors": all_errors,
        "elapsed": round(time.monotonic() - t0, 3),
    }


def find_file_links(
    site: str,
    filetypes: Optional[List[str]] = None,
    engines: Optional[List[str]] = None,
    max_results: int = 100,
    timeout: Optional[int] = None,
    respect_robots: Optional[bool] = None,
    use_proxy: bool = False,
) -> Dict[str, Any]:
    """
    Find downloadable files on *site* using filetype dorks.

    Returns: ok, site, filetypes, links, count, errors, queries_used
    """
    t0 = time.monotonic()
    if filetypes is None:
        filetypes = ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
                     "txt", "csv", "zip", "rar"]

    queries: List[str] = []
    for ft in filetypes:
        queries.append(f"site:{site} filetype:{ft}")
        queries.append(f"site:{site} ext:{ft}")
    for d in ("downloads", "files", "documents", "media", "uploads", "assets"):
        queries.append(f"site:{site} inurl:{d}")
    queries = queries[:12]

    all_links: List[str] = []
    errors: List[str] = []
    per_q = max(5, max_results // max(len(queries), 1))

    # Run queries in parallel
    def _q(query: str) -> List[str]:
        r = search_links(query=query, engines=engines, max_results=per_q,
                         timeout=timeout, respect_robots=respect_robots,
                         use_proxy=use_proxy)
        if r.get("ok"):
            return r.get("links", [])
        errors.extend(r.get("errors", []))
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
        futs = [pool.submit(_q, q) for q in queries]
        for fut in concurrent.futures.as_completed(futs):
            try:
                all_links.extend(fut.result())
            except Exception as exc:
                errors.append(str(exc))

    # Filter to only target extensions
    if filetypes:
        ext_set = {f".{ft.lower().lstrip('.')}" for ft in filetypes}
        all_links = [l for l in all_links
                     if any(l.lower().endswith(e) for e in ext_set)] or all_links

    all_links = _dedupe(all_links)
    if max_results and len(all_links) > max_results:
        all_links = all_links[:max_results]

    return {
        "ok": len(all_links) > 0 or len(errors) == 0,
        "site": site,
        "filetypes": filetypes,
        "links": all_links,
        "count": len(all_links),
        "errors": errors,
        "queries_used": queries,
        "elapsed": round(time.monotonic() - t0, 3),
    }


def find_hidden_resources(
    base_url: str,
    patterns: Optional[List[str]] = None,
    engines: Optional[List[str]] = None,
    max_results: int = 50,
    timeout: Optional[int] = None,
    respect_robots: Optional[bool] = None,
    use_proxy: bool = False,
) -> Dict[str, Any]:
    """
    Discover admin panels, config files, backups, and sensitive paths on a site.

    FIX: Now returns 'links' key (in addition to 'hidden_resources') for
    consistency with other tools that iterate result.get('links', []).

    Returns: ok, base_url, links, hidden_resources, count, patterns_used, errors
    """
    t0 = time.monotonic()
    parsed = urlparse(base_url)
    site = parsed.netloc or base_url

    if patterns is None:
        patterns = (
            _DEFAULT_CONFIG.dork_patterns.get("hidden_resources", []) +
            _DEFAULT_CONFIG.dork_patterns.get("sensitive_files", [])
        )

    # Substitute {site} placeholder
    resolved_patterns = [p.replace("{site}", site) for p in patterns][:12]

    all_links: List[str] = []
    errors: List[str] = []
    n_per = max(5, max_results // max(len(resolved_patterns), 1))

    def _pat(pattern: str) -> List[str]:
        r = search_links(
            query=pattern, engines=engines,
            max_results=n_per, timeout=timeout,
            respect_robots=respect_robots, use_proxy=use_proxy,
        )
        if r.get("ok"):
            # Keep only links on the same domain
            same_domain = []
            for lnk in r.get("links", []):
                try:
                    lp = urlparse(lnk)
                    if lp.netloc.endswith(site) or site.endswith(lp.netloc):
                        same_domain.append(lnk)
                except Exception:
                    pass
            return same_domain or r.get("links", [])  # fallback: all links
        errors.extend(r.get("errors", []))
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(resolved_patterns))) as pool:
        futs = [pool.submit(_pat, p) for p in resolved_patterns]
        for fut in concurrent.futures.as_completed(futs):
            try:
                all_links.extend(fut.result())
            except Exception as exc:
                errors.append(str(exc))

    all_links = _dedupe(all_links)
    if max_results and len(all_links) > max_results:
        all_links = all_links[:max_results]

    return {
        "ok": True,
        "base_url": base_url,
        "site": site,
        "links": all_links,             # ← primary key for downstream compatibility
        "hidden_resources": all_links,  # ← legacy key kept for backward compat
        "count": len(all_links),
        "patterns_used": resolved_patterns,
        "errors": errors,
        "elapsed": round(time.monotonic() - t0, 3),
    }


def extract_links_from_page(
    url: str,
    max_links: int = 200,
    timeout: Optional[int] = None,
    respect_robots: Optional[bool] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    extract_file_links: bool = False,
) -> Dict[str, Any]:
    """
    Fetch a page and extract all links from it.

    Returns: ok, url, links, count, error, elapsed
    """
    t0 = time.monotonic()
    to = int(timeout or _DEFAULT_CONFIG.per_request_timeout)
    robots = respect_robots if respect_robots is not None else _DEFAULT_CONFIG.respect_robots

    ok_v, err, _ = URLValidator.validate(url)
    if not ok_v:
        return {"ok": False, "url": url, "links": [], "count": 0,
                "error": err, "elapsed": time.monotonic() - t0}

    if robots:
        checker = RobotsTxtChecker()
        if not checker.can_fetch(url):
            return {"ok": False, "url": url, "links": [], "count": 0,
                    "error": "disallowed-by-robots", "elapsed": time.monotonic() - t0}

    if extract_file_links and not include_patterns:
        include_patterns = [
            r"\.(pdf|docx?|xlsx?|pptx?|txt|csv|zip|rar|tar\.gz|mp4|mp3)$"
        ]

    cfg = ScrapingConfig(
        timeout=to,
        extract_links=True,
        extract_images=False,
        extract_structured=False,
        max_links=max_links * 3,
        human_mimic=False,
        cache_enabled=_DEFAULT_CONFIG.cache_results,
    )
    try:
        res = scrape_page(url, config=cfg, use_cache=False)
        if not res.get("ok"):
            return {"ok": False, "url": url, "links": [], "count": 0,
                    "error": res.get("error"), "elapsed": time.monotonic() - t0}

        links = _collect_links_from_scrape(res, max_links * 2)
        # Also mine HTML
        html = res.get("html_content", "") or res.get("content", "")
        if html:
            extra = re.findall(r'https?://[^\s"\'<>\\]{10,300}', html)
            links += extra

        links = _dedupe(links)
        if include_patterns or exclude_patterns:
            links = _filter(links, include_patterns, exclude_patterns)
        if max_links and len(links) > max_links:
            links = links[:max_links]

        return {
            "ok": True, "url": url, "links": links, "count": len(links),
            "error": None, "elapsed": round(time.monotonic() - t0, 3),
        }
    except Exception as exc:
        logger.error("extract_links_from_page failed for %s: %s", url, exc)
        return {"ok": False, "url": url, "links": [], "count": 0,
                "error": str(exc), "elapsed": round(time.monotonic() - t0, 3)}


def expand_links(
    seed_urls: List[str],
    depth: int = 1,
    max_per_page: int = 50,
    max_total: int = 1000,
    timeout: Optional[int] = None,
    respect_robots: Optional[bool] = None,
    follow_external: bool = False,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Crawl seed URLs up to *depth* levels deep, collecting all links.

    Returns: ok, seeds, links, total_count, errors, elapsed, depth_reached
    """
    t0 = time.monotonic()
    to = int(timeout or _DEFAULT_CONFIG.per_request_timeout)
    robots = respect_robots if respect_robots is not None else _DEFAULT_CONFIG.respect_robots

    base_domains: Set[str] = set()
    for u in seed_urls:
        try:
            base_domains.add(urlparse(_normalize_url(u)).netloc)
        except Exception:
            pass

    collected: List[str] = list(_dedupe(seed_urls))
    visited: Set[str] = set()
    errors: List[str] = []
    to_visit = list(collected)
    level_reached = 0

    for level in range(depth):
        if not to_visit or len(collected) >= max_total:
            break
        level_reached = level + 1
        next_round: List[str] = []

        # Fetch current level in parallel
        def _fetch(u: str) -> Tuple[str, List[str]]:
            norm = _normalize_url(u).rstrip("/")
            res = extract_links_from_page(
                norm, max_links=max_per_page,
                timeout=to, respect_robots=robots,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            if res.get("ok"):
                return norm, res.get("links", [])
            return norm, []

        unfetched = [u for u in to_visit if _normalize_url(u).rstrip("/") not in visited]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(_fetch, u): u for u in unfetched}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    norm, page_links = fut.result()
                    visited.add(norm)
                    for lnk in page_links:
                        if len(collected) >= max_total:
                            break
                        if not follow_external:
                            try:
                                if urlparse(_normalize_url(lnk)).netloc not in base_domains:
                                    continue
                            except Exception:
                                continue
                        ln = _normalize_url(lnk)
                        if ln not in visited and ln not in collected:
                            collected.append(ln)
                            next_round.append(ln)
                except Exception as exc:
                    errors.append(str(exc))

        to_visit = next_round

    combined = _dedupe(seed_urls + collected)
    if include_patterns or exclude_patterns:
        combined = _filter(combined, include_patterns, exclude_patterns)
    if max_total and len(combined) > max_total:
        combined = combined[:max_total]

    return {
        "ok": True,
        "seeds": list(_dedupe(seed_urls)),
        "links": combined,
        "total_count": len(combined),
        "errors": errors,
        "elapsed": round(time.monotonic() - t0, 3),
        "depth_reached": level_reached,
        "unique_domains": len({urlparse(_normalize_url(u)).netloc for u in combined}),
    }


# ──────────────────────────────────────────────
# Internal: dork query generation
# ──────────────────────────────────────────────

def _generate_dork_queries(
    base_query: str,
    site: Optional[str] = None,
    filetypes: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
) -> List[str]:
    queries: List[str] = [base_query]

    if site:
        queries.append(f"site:{site}")
        queries.append(f"site:{site} {base_query}")

    if filetypes:
        ft_str = " OR ".join(f"filetype:{ft}" for ft in filetypes)
        queries.append(f"{base_query} {ft_str}")
        if site:
            queries.append(f"site:{site} {ft_str}")

    if patterns:
        for p in patterns[:8]:
            if site:
                queries.append(p.replace("{site}", site))
            else:
                queries.append(f"{base_query} {p}")

    queries.extend([
        f"inurl:{base_query}",
        f"intitle:{base_query}",
        f"intext:{base_query}",
    ])
    if site:
        queries += [
            f"inurl:{site} {base_query}",
            f"site:{site} inurl:upload",
            f"site:{site} inurl:download",
        ]

    return _dedupe(queries)[:20]


# ──────────────────────────────────────────────
# Engine handler table (for __init__.py export)
# ──────────────────────────────────────────────

_ENGINE_HANDLERS = {
    eng: (lambda q, max_results=50, timeout=30, respect_robots=True, proxy=None, _e=eng:
          _search_engine(_e, q, max_results, timeout, respect_robots, proxy))
    for eng in ("duckduckgo", "bing", "google", "yandex", "brave", "startpage", "qwant")
}


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION ADDITIONS v5.0 — Engine scoring, link ranking, URL normalizer
# ─────────────────────────────────────────────────────────────────────────────

from urllib.parse import urlparse as _urlparse, urlencode as _urlencode, parse_qs as _parse_qs, urlunparse as _urlunparse


class _EngineScorer:
    """
    Track per-engine reliability and result quality to bias future searches.
    
    Score model:
    - Base score = 1.0 per engine
    - +0.1 per successful result returned
    - -0.2 per timeout/error
    - -0.05 per duplicate result (already seen from other engines)
    Scores are exponentially weighted (alpha=0.1) so recent behavior counts more.
    """

    def __init__(self, alpha: float = 0.1):
        self._scores: Dict[str, float] = {}
        self._alpha = alpha
        self._lock = threading.Lock()

    def update(self, engine: str, results: int, errors: int, dupes: int) -> None:
        with self._lock:
            delta = results * 0.1 - errors * 0.2 - dupes * 0.05
            old = self._scores.get(engine, 1.0)
            self._scores[engine] = (1 - self._alpha) * old + self._alpha * (1.0 + delta)

    def get_score(self, engine: str) -> float:
        with self._lock:
            return self._scores.get(engine, 1.0)

    def ranked_engines(self, engines: list) -> list:
        """Return engines sorted by descending score."""
        with self._lock:
            return sorted(engines, key=lambda e: self._scores.get(e, 1.0), reverse=True)

    def all_scores(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._scores)


_engine_scorer = _EngineScorer()


def normalize_url(url: str, remove_tracking: bool = True) -> str:
    """
    Canonicalize a URL for deduplication and caching.
    
    Steps applied:
    1. Lowercase scheme + host
    2. Remove default ports (80/443)
    3. Sort query params alphabetically
    4. Remove known tracking params (utm_*, fbclid, gclid, etc.)
    5. Remove trailing slash from path (except root)
    
    Args:
        url: Raw URL string
        remove_tracking: If True, strip common tracking query params
        
    Returns:
        Normalized URL string, or original on parse failure
    """
    _TRACKING_PARAMS = frozenset({
        'utm_source','utm_medium','utm_campaign','utm_term','utm_content',
        'fbclid','gclid','msclkid','yclid','_ga','ref','source',
        'mc_eid','mc_cid','hsCtaTracking',
    })

    try:
        p = _urlparse(url)
        scheme = p.scheme.lower()
        host = p.netloc.lower()
        # Strip default ports
        if host.endswith(':80') and scheme == 'http':
            host = host[:-3]
        elif host.endswith(':443') and scheme == 'https':
            host = host[:-4]
        
        path = p.path.rstrip('/') or '/'
        
        # Handle query params
        qs = _parse_qs(p.query, keep_blank_values=False)
        if remove_tracking:
            qs = {k: v for k, v in qs.items() if k.lower() not in _TRACKING_PARAMS}
        # Sort for determinism
        sorted_qs = _urlencode(sorted(qs.items()), doseq=True) if qs else ''
        
        return _urlunparse((scheme, host, path, p.params, sorted_qs, ''))
    except Exception:
        return url


def rank_links(
    links: list,
    keywords: list = None,
    boost_domains: list = None,
    penalize_domains: list = None,
) -> list:
    """
    Rank a list of link dicts by relevance and quality signals.
    
    Scoring signals (higher = better):
    - keyword match in URL/title (+0.2 per match)
    - HTTPS scheme (+0.1)
    - Short path depth (+0.05 per fewer slashes)
    - Boosted domain (+0.3)
    - Penalized domain (-0.5)
    - Duplicate URL (de-ranked to bottom)
    
    Args:
        links: List of dicts with at least {"url": ..., "title": ...}
        keywords: Terms to look for in URL/title
        boost_domains: Domain substrings that get a positive boost
        penalize_domains: Domain substrings penalized (ads, spam, etc.)
        
    Returns:
        List of link dicts sorted by descending score, deduped by normalized URL
    """
    keywords = [k.lower() for k in (keywords or [])]
    boost_domains = [d.lower() for d in (boost_domains or [])]
    penalize_domains = [d.lower() for d in (penalize_domains or ['doubleclick', 'googlesyndication', 'adservice'])]

    seen_urls: set = set()
    scored: list = []

    for link in links:
        url = link.get('url') or link.get('href') or ''
        title = (link.get('title') or link.get('text') or '').lower()
        norm = normalize_url(url)

        if not url or norm in seen_urls:
            continue
        seen_urls.add(norm)

        score = 0.5  # baseline
        url_lower = url.lower()

        # HTTPS bonus
        if url_lower.startswith('https://'):
            score += 0.1

        # Keyword matches
        for kw in keywords:
            if kw in url_lower or kw in title:
                score += 0.2

        # Domain boosts/penalties
        for dom in boost_domains:
            if dom in url_lower:
                score += 0.3
        for dom in penalize_domains:
            if dom in url_lower:
                score -= 0.5

        # Depth penalty (fewer path segments is cleaner)
        depth = url_lower.count('/')
        score += max(0.0, 0.05 * (5 - depth))

        scored.append({**link, 'url': url, '_score': round(score, 4)})

    scored.sort(key=lambda x: x['_score'], reverse=True)
    return scored


def get_engine_scores() -> dict:
    """Return current engine reliability scores (useful for monitoring)."""
    return _engine_scorer.all_scores()

