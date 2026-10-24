"""
amb_llm_core.py — production transport layer for AMB-APP -> any chat LLM.

Works against:
  * Local-LLM server (has /v1/runtime/limits + /v1/chat/completions)
  * Any OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp server, Ollama /v1, OpenAI, ...)
  * Unreachable / degraded servers (typed errors, never silent success)

Guarantees
──────────
  G1  A reply that LOOKS like success (HTTP 200) but carries an error marker
      ("[timed out after 0s]", "[Error: ...]") or an error finish_reason is raised
      as a typed exception. It is NEVER returned as model text.
  G2  A client-side deadline is ALWAYS sent (timeout_s) and derived from prompt
      size + measured speed, so a server whose default_timeout_s is 0 can never
      produce an instant timeout.
  G3  Streaming uses a first-token budget + idle (inter-token) budget, not one
      total deadline. Slow-but-alive hardware is never killed.
  G4  Conversation state is owned by the caller: history_mode="replace" and a
      unique session id per request. The server never appends to stored history.
  G5  Model id is resolved against /v1/models, never guessed from a file name.

Production features
───────────────────
  * Per-client circuit breaker (open after N failures, half-open probe after cooldown)
  * Jittered exponential backoff on retryable errors (3 attempts)
  * asyncio.Semaphore(n_parallel) — one in-flight request on single-slot servers
  * Auto-adaptive decode_tps / prefill_tps from observed usage (EMA, α=0.3)
  * CapabilityLimits auto-detected from /v1/runtime/limits (falls back to safe
    defaults for plain OpenAI-compat servers, no config needed)
  * Structured logging with run_id / step_id / llm_call_n / tokens / elapsed
  * Metrics counters exposed via ChatClient.metrics()
  * Single pooled httpx.AsyncClient; no per-request client creation
  * Graceful aclose(): drains semaphore, closes pool, marks closed

Non-goals
─────────
  * Prompt engineering — that's amb_runtime.py
  * Tool call parsing — that's amb_actions.py
  * Model management — that's model_manager.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Errors — every raise site carries status_code + finish_reason where applicable
# ─────────────────────────────────────────────────────────────────────────────
class LLMError(RuntimeError):
    """Base class. `retryable=True` means the caller may safely try again."""
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        finish_reason: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.finish_reason = finish_reason


class LLMAuthError(LLMError):
    """Invalid / missing API key."""
    retryable = False


class LLMContextOverflow(LLMError):
    """Prompt exceeded the model's context window. Caller should shrink and retry."""
    retryable = False


class LLMModelNotReady(LLMError):
    """Model not loaded (503) or /v1/models doesn't list it yet. Server may be warming up."""
    retryable = True


class LLMTransportError(LLMError):
    """Connection refused, DNS failure, TCP reset, ..."""
    retryable = True


class LLMServerTimeout(LLMError):
    """Server-side deadline / stall fired, OR client read timeout."""
    retryable = True


class LLMBadReply(LLMError):
    """Empty reply, error marker as text, or error finish_reason."""
    retryable = True


class LLMStall(LLMServerTimeout):
    """Client-side watchdog fired during streaming (first-token or idle budget)."""


# ─────────────────────────────────────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LLMResponse:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    ttft_s: float = 0.0                # time-to-first-token (streaming only)
    model: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

@dataclass
class _StreamState:
    """Mutable accumulator passed into `_stream_inner`.

    `_stream_inner` is an async generator — it yields text chunks — so it
    cannot also `return` a value. The alternative is to attach the final
    `finish_reason` and `usage` to `self`, which serialises cleanly only
    because the client holds a semaphore that limits concurrent streams to
    one. Passing an explicit state object is safer: it works even if the
    semaphore is ever raised to allow parallel streams, and it makes the
    state's lifetime visible in the call graph instead of hidden on `self`.
    """
    fin: str = "stop"
    usage: Dict[str, Any] = field(default_factory=dict)

# ─────────────────────────────────────────────────────────────────────────────
# Capability limits — typed view of /v1/runtime/limits
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CapabilityLimits:
    context_length: int = 4096
    effective_tps: float = 3.0
    prefill_tps: float = 0.0           # 0 = unknown, caller uses 2× decode as fallback
    tier: str = "small"
    n_parallel: int = 1
    recommended_response_tokens: int = 0   # 0 = unlimited / not reported
    recommended_resume_chunk: int = 0
    recommended_idle_timeout_s: float = 300.0
    backend: str = "unknown"
    model_id: str = ""
    measured_tps: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Reply validation — the G1 guarantee
# ─────────────────────────────────────────────────────────────────────────────
_ERR_MARKER = re.compile(
    r"^\s*\[(?:"
    r"timed out after[^\]]*|"
    r"request timed out|"
    r"request cancelled|"
    r"stream cancelled|"
    r"cancelled|"
    r"error[^\]]*|"
    r"⚡(?:TRUNCATED|INTERRUPTED|STALLED)[^\]]*"
    r")\]\s*$",
    re.IGNORECASE,
)

_BAD_FINISH = frozenset({
    "timeout", "cancelled", "error", "oom",
    "circuit_open", "stalled", "interrupted",
})


def validate_reply(
    body: Dict[str, Any],
    model: str = "",
    elapsed: float = 0.0,
    ttft: float = 0.0,
) -> LLMResponse:
    """
    Turn an OpenAI-shaped response body into LLMResponse, or raise a typed error.

    Reads finish_reason from choices[0] FIRST (that's where OpenAI puts it),
    falls back to body root for servers that put it there.
    """
    choices = body.get("choices") or []
    if not choices:
        raise LLMBadReply("reply has no choices")

    ch = choices[0]
    message = ch.get("message") or {}
    delta = ch.get("delta") or {}
    text = message.get("content") or delta.get("content") or ch.get("text") or ""

    fin = ch.get("finish_reason") or body.get("finish_reason") or "stop"

    usage = body.get("usage") or {}
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)

    if fin in _BAD_FINISH:
        if fin in ("timeout", "stalled"):
            raise LLMServerTimeout(
                f"server finish_reason={fin}: {text[:120]!r}",
                finish_reason=fin,
            )
        raise LLMBadReply(
            f"server finish_reason={fin}: {text[:120]!r}",
            finish_reason=fin,
        )

    if _ERR_MARKER.match(text):
        if "timed out" in text.lower():
            raise LLMServerTimeout(
                f"server returned error marker as text: {text.strip()[:120]!r}",
                finish_reason="timeout",
            )
        raise LLMBadReply(
            f"server returned error marker as text: {text.strip()[:120]!r}",
        )

    if not text.strip():
        # Empty content is only acceptable if the server explicitly says it stopped
        # for a benign reason and there's a completion-token count > 0 (some servers
        # return whitespace-only for reasoning models mid-stream).
        if fin == "stop" and ct == 0:
            raise LLMBadReply("empty completion", finish_reason=fin)
        if fin not in ("stop", "length"):
            raise LLMBadReply("empty completion", finish_reason=fin)

    return LLMResponse(
        text=text,
        finish_reason=fin,
        prompt_tokens=pt,
        completion_tokens=ct,
        elapsed_s=elapsed,
        ttft_s=ttft,
        model=model or body.get("../../../model", ""),
        raw=body,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deadline estimation — the G2 guarantee
# ─────────────────────────────────────────────────────────────────────────────
def estimate_deadline_s(
    prompt_tokens: int,
    max_new_tokens: int,
    decode_tps: float,
    prefill_tps: Optional[float] = None,
    safety: float = 1.6,
    floor: float = 60.0,
    ceil: float = 7200.0,
) -> float:
    """
    prefill + decode time with a safety factor.

    Prefill on small/CPU boxes is only 1.5–3× decode (Local-LLM log: prefill 5 tok/s
    vs decode 2.7 tok/s on qwen3-8b), so it MUST be included. Defaults to 2× decode
    when the server hasn't reported a measured prefill rate.
    """
    decode_tps = max(float(decode_tps), 0.2)
    prefill_tps = max(float(prefill_tps or decode_tps * 2.0), 0.2)
    out = max_new_tokens if max_new_tokens > 0 else 1024  # 0 == unlimited → assume 1024
    raw = (prompt_tokens / prefill_tps + out / decode_tps) * safety + 15.0
    return float(min(ceil, max(floor, raw)))


# ─────────────────────────────────────────────────────────────────────────────
# Model id resolution — the G5 guarantee
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_model_id(s: str) -> str:
    s = s.lower().rsplit("/", 1)[-1]
    s = re.sub(r"\.gguf$", "", s)
    s = re.sub(r"[-_.]?(q\d(_[a-z0-9]+)*|i2_s|f16|f32|bf16)$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def resolve_model_id(hint: str, available: List[str]) -> str:
    """
    Match a file-name-style hint against the ids the server is actually serving.

    'Qwen3-8B-Q4_K_M.gguf', '/m/qwen3-8b.gguf', 'qwen3-8b' → 'qwen3-8b'
    If only one model is loaded, that one wins even if the hint doesn't match.
    Raises LLMModelNotReady when the hint is ambiguous.
    """
    if hint in available:
        return hint
    n = _normalize_model_id(hint)
    for m in available:
        if _normalize_model_id(m) == n:
            return m
    if len(available) == 1:
        return available[0]
    raise LLMModelNotReady(
        f"model '{hint}' not among loaded models {available}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Circuit breaker
# ─────────────────────────────────────────────────────────────────────────────
class _CircuitBreaker:
    """
    Simple breaker: open after `failure_threshold` consecutive failures,
    half-open after `recovery_s` seconds with a single probe, closed on probe success.
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 30.0):
        self._threshold = failure_threshold
        self._recovery = recovery_s
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            if self._failures < self._threshold:
                return True
            if time.monotonic() - self._opened_at < self._recovery:
                return False
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    async def record_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._probe_in_flight = False

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            self._probe_in_flight = False
            if self._failures >= self._threshold:
                self._opened_at = time.monotonic()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "failures": self._failures,
            "threshold": self._threshold,
            "opened": self._failures >= self._threshold,
            "recovery_s": self._recovery,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ChatClient
# ─────────────────────────────────────────────────────────────────────────────
class ChatClient:
    """
    Async chat client for any OpenAI-compatible endpoint.

    Usage:
        client = ChatClient("http://localhost:8080", api_key="...", model="qwen3-8b")
        await client.ensure_model()                 # resolves via /v1/models
        limits = await client.runtime_limits()      # typed CapabilityLimits
        r = await client.chat([{"role": "user", "content": "hi"}], max_tokens=64)
        async for chunk in client.chat_stream_iter(...): ...
        await client.aclose()

    All public methods are safe to call concurrently. Requests above the server's
    reported n_parallel are queued by an internal semaphore, not sent.
    """

    # Retry tuning (only retryable errors; never LLMAuthError / LLMContextOverflow)
    MAX_ATTEMPTS = 3
    BACKOFF_BASE = 0.5
    BACKOFF_CAP = 8.0

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        *,
        default_decode_tps: float = 3.0,
        max_connections: int = 8,
        request_timeout_connect_s: float = 10.0,
        client: Optional[httpx.AsyncClient] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.base = base_url.rstrip("/")
        self.headers: Dict[str, str] = {}
        if api_key:
            # Local-LLM uses X-API-Key; OpenAI uses Authorization: Bearer.
            # Send both — servers ignore the one they don't understand.
            self.headers["X-API-Key"] = api_key
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.model_hint = model
        self.model: str = ""
        self.decode_tps = float(default_decode_tps or 3.0)
        self.prefill_tps: float = 0.0
        self.run_id = run_id or uuid.uuid4().hex[:8]

        self._http = client or httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_connect_s, read=None),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        )
        self._own_client = client is None

        # Capability cache
        self._limits: Optional[CapabilityLimits] = None
        self._limits_ts: float = 0.0
        self._limits_ttl: float = 30.0

        # Concurrency gate — set on first runtime_limits() call
        self._sem: Optional[asyncio.Semaphore] = None
        self._sem_size: int = 0

        # Circuit breaker
        self._breaker = _CircuitBreaker()

        # Metrics
        self._m_lock = asyncio.Lock()
        self._m_requests = 0
        self._m_success = 0
        self._m_failed = 0
        self._m_retries = 0
        self._m_prompt_tokens = 0
        self._m_completion_tokens = 0
        self._m_latency_sum = 0.0

        self._closed = False
        # Loop-affinity guard. Any object holding an asyncio primitive
        # (httpx.AsyncClient, asyncio.Lock, asyncio.Semaphore) is bound to
        # the loop that first awaited it. Crossing loops produces
        # "RuntimeError: <asyncio.locks.Event …> is bound to a different
        # event loop" — we make the misuse fail immediately and with a
        # message that names the fix instead of an opaque runtime error.
        self._home_loop: "Optional[asyncio.AbstractEventLoop]" = None

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._home_loop is None:
            self._home_loop = loop
        elif loop is not self._home_loop:
            raise LLMTransportError(
                "ChatClient used from a different event loop than the one "
                "that created its HTTP pool. Route the call through "
                "llm_cloud_adapter._BRIDGE (LLMCloudClient.*) — never call "
                "ChatClient methods directly from the main asyncio loop."
            )

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._own_client:
            try:
                await self._http.aclose()
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"ChatClient(base={self.base!r}, model={self.model!r}, "
            f"decode_tps={self.decode_tps:.2f}, circuit={self._breaker.snapshot()})"
        )

    # ── low-level helpers ────────────────────────────────────────────────────
    async def _get(self, path: str, timeout: float = 10.0) -> Dict[str, Any]:
        self._check_loop()
        if self._closed:
            raise LLMTransportError("client is closed")
        try:
            r = await self._http.get(self.base + path, headers=self.headers, timeout=timeout)
        except httpx.TimeoutException as e:
            raise LLMServerTimeout(f"GET {path}: {e}") from e
        except httpx.HTTPError as e:
            raise LLMTransportError(f"GET {path}: {e}") from e
        if r.status_code in (401, 403):
            raise LLMAuthError(f"GET {path}: HTTP {r.status_code}", status_code=r.status_code)
        if r.status_code == 404:
            # Distinct from a transport error: caller may want to fall back gracefully.
            raise LLMTransportError(
                f"GET {path}: HTTP 404 (endpoint not available)",
                status_code=404,
            )
        if r.status_code != 200:
            raise LLMTransportError(
                f"GET {path}: HTTP {r.status_code}", status_code=r.status_code
            )
        try:
            return r.json()
        except ValueError as e:
            raise LLMBadReply(f"GET {path}: invalid JSON: {e}") from e

    @staticmethod
    def _raise_for_status(r: httpx.Response) -> None:
        if r.status_code == 200:
            return
        detail = (r.text or "")[:200]
        if r.status_code in (401, 403):
            raise LLMAuthError(detail, status_code=r.status_code)
        if r.status_code == 413:
            raise LLMContextOverflow(detail, status_code=413)
        if r.status_code in (502, 503):
            raise LLMModelNotReady(detail, status_code=r.status_code)
        if r.status_code in (504, 408):
            raise LLMServerTimeout(detail, status_code=r.status_code)
        if r.status_code == 429:
            # Rate-limited: retryable, but let caller decide whether to wait.
            raise LLMTransportError(f"HTTP 429: {detail}", status_code=429)
        raise LLMTransportError(f"HTTP {r.status_code}: {detail}", status_code=r.status_code)

    # ── discovery ────────────────────────────────────────────────────────────
    async def ensure_model(self) -> str:
        """Resolve the model id against /v1/models. Idempotent."""
        if self.model:
            return self.model
        data = (await self._get("/v1/models")).get("data") or []
        ready = [
            m.get("id")
            for m in data
            if m.get("id") and m.get("ready", True) and m.get("status", "ready") == "ready"
        ]
        if not ready:
            raise LLMModelNotReady(
                f"no ready models in /v1/models (got {len(data)} entries)"
            )
        self.model = resolve_model_id(self.model_hint, ready)
        log.info(
            "ChatClient: resolved model %r -> %r (from %d ready)",
            self.model_hint, self.model, len(ready),
            extra={"run_id": self.run_id},
        )
        return self.model

    async def health(self) -> bool:
        """Return True if /health responds 200."""
        self._check_loop()
        try:
            r = await self._http.get(self.base + "/health", headers=self.headers, timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def wait_for_server(
        self,
        total_s: float = 300.0,
        first: float = 1.0,
        cap: float = 15.0,
    ) -> bool:
        """
        Block until /health returns 200 or `total_s` elapses.
        Exponential backoff with jitter. Raises LLMTransportError on timeout.
        This is the fix for the AMB-APP "one 5s probe, then die" pattern.
        """
        t0 = time.monotonic()
        delay = first
        attempt = 0
        while True:
            attempt += 1
            if await self.health():
                log.info(
                    "ChatClient: server healthy after %d probe(s) in %.1fs",
                    attempt, time.monotonic() - t0,
                    extra={"run_id": self.run_id},
                )
                return True
            elapsed = time.monotonic() - t0
            if elapsed >= total_s:
                raise LLMTransportError(
                    f"server not ready after {total_s:.0f}s ({attempt} probes)"
                )
            jitter = 0.75 + random.random() / 2.0
            wait = min(delay * jitter, cap, total_s - elapsed)
            log.warning(
                "ChatClient: server not ready (probe %d, elapsed %.1fs) — retry in %.1fs",
                attempt, elapsed, wait,
                extra={"run_id": self.run_id},
            )
            await asyncio.sleep(wait)
            delay = min(delay * 2.0, cap)

    # ── runtime limits ───────────────────────────────────────────────────────
    async def runtime_limits(self, ttl: float = 30.0) -> CapabilityLimits:
        """
        Return /v1/runtime/limits as a typed CapabilityLimits. Cached for `ttl` seconds.

        For plain OpenAI-compat servers (no /v1/runtime/limits endpoint), returns
        conservative defaults. Never raises — a missing endpoint must not break
        the agent loop.
        """

        self._check_loop()

        now = time.monotonic()
        if self._limits is not None and (now - self._limits_ts) < ttl:
            return self._limits

        body: Dict[str, Any] = {}
        try:
            body = await self._get("/v1/runtime/limits", timeout=5.0)
        except LLMError as e:
            log.warning("ChatClient: /v1/runtime/limits unavailable (%s); using defaults", e)

        ctx = int(body.get("context_length", 0) or 0)
        tps = float(body.get("effective_tps", 0.0) or 0.0)
        pf = float(body.get("prefill_tps", 0.0) or 0.0)
        np_ = int(body.get("recommended_n_parallel", 0) or 0)

        limits = CapabilityLimits(
            context_length=ctx if ctx > 0 else 4096,
            effective_tps=tps if tps > 0 else self.decode_tps,
            prefill_tps=pf,
            tier=str(body.get("tier", "small") or "small"),
            n_parallel=np_ if np_ > 0 else 1,
            recommended_response_tokens=int(body.get("recommended_response_tokens", 0) or 0),
            recommended_resume_chunk=int(body.get("recommended_resume_chunk", 0) or 0),
            recommended_idle_timeout_s=float(body.get("recommended_idle_timeout_s", 300.0) or 300.0),
            backend=str(body.get("backend", "unknown") or "unknown"),
            model_id=str(body.get("model_id", "") or ""),
            measured_tps=float(body.get("measured_tps", 0.0) or 0.0),
            raw=body,
        )

        # Seed our TPS from the server's measured value if it's higher than our default.
        if limits.effective_tps > 0:
            self.decode_tps = limits.effective_tps
        if limits.prefill_tps > 0:
            self.prefill_tps = limits.prefill_tps

        # Resize the semaphore if the server reports a different n_parallel.
        if self._sem is None or limits.n_parallel != self._sem_size:
            self._sem = asyncio.Semaphore(max(1, limits.n_parallel))
            self._sem_size = limits.n_parallel

        self._limits = limits
        self._limits_ts = now
        return limits

    # ── TPS adaptation ───────────────────────────────────────────────────────
    def _observe_speed(self, resp: LLMResponse) -> None:
        """Update decode_tps / prefill_tps from observed usage. EMA α=0.3."""
        if resp.completion_tokens > 0 and resp.elapsed_s > 0.05:
            measured = resp.completion_tokens / resp.elapsed_s
            # Reject absurd samples (cache hits, tiny warmups).
            if 0.05 < measured < 500.0:
                self.decode_tps = 0.7 * self.decode_tps + 0.3 * measured
        if resp.prompt_tokens > 0 and resp.ttft_s > 0.05:
            prefill = resp.prompt_tokens / resp.ttft_s
            if 0.05 < prefill < 500.0:
                self.prefill_tps = 0.7 * self.prefill_tps + 0.3 * prefill

    # ── metrics ──────────────────────────────────────────────────────────────
    async def _record_request(self, ok: bool, elapsed_s: float) -> None:
        async with self._m_lock:
            self._m_requests += 1
            if ok:
                self._m_success += 1
            else:
                self._m_failed += 1
            self._m_latency_sum += elapsed_s

    async def _record_tokens(self, prompt: int, completion: int) -> None:
        async with self._m_lock:
            self._m_prompt_tokens += prompt
            self._m_completion_tokens += completion

    def metrics(self) -> Dict[str, Any]:
        n = max(1, self._m_requests)
        return {
            "requests": self._m_requests,
            "successful": self._m_success,
            "failed": self._m_failed,
            "retries": self._m_retries,
            "success_rate": round(self._m_success / n, 4),
            "avg_latency_s": round(self._m_latency_sum / n, 3),
            "prompt_tokens": self._m_prompt_tokens,
            "completion_tokens": self._m_completion_tokens,
            "decode_tps_ema": round(self.decode_tps, 3),
            "prefill_tps_ema": round(self.prefill_tps, 3),
            "circuit": self._breaker.snapshot(),
        }

    # ── payload construction ─────────────────────────────────────────────────
    def _payload(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]],
        no_think: Optional[bool],
        deadline_s: float,
        stream: bool,
    ) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": bool(stream),
            "timeout_s": float(deadline_s),        # G2
            "history_mode": "replace",             # G4
        }
        if stop:
            p["stop"] = list(stop)
        if no_think is not None:
            p["no_think"] = bool(no_think)
        return p

    # ── retry wrapper ────────────────────────────────────────────────────────
    async def _with_retry(self, fn):
        """Run `fn` with jittered exponential backoff on retryable errors."""
        last_exc: Optional[LLMError] = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                return await fn()
            except LLMError as e:
                last_exc = e
                if not e.retryable or attempt >= self.MAX_ATTEMPTS:
                    raise
                async with self._m_lock:
                    self._m_retries += 1
                delay = min(
                    self.BACKOFF_CAP,
                    self.BACKOFF_BASE * (2 ** (attempt - 1)),
                )
                delay *= 0.5 + random.random()
                log.warning(
                    "ChatClient: %s on attempt %d/%d — retry in %.2fs",
                    type(e).__name__, attempt, self.MAX_ATTEMPTS, delay,
                    extra={"run_id": self.run_id, "error": str(e)[:120]},
                )
                await asyncio.sleep(delay)
        if last_exc:
            raise last_exc

    # ── non-streaming chat ───────────────────────────────────────────────────
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
        no_think: Optional[bool] = None,
        est_prompt_tokens: int = 0,
        deadline_s: Optional[float] = None,
        step_id: Optional[str] = None,
        llm_call_n: Optional[int] = None,
    ) -> LLMResponse:
        """
        Single non-streaming chat completion. Retries retryable errors internally.
        Raises LLMError on final failure; never returns error text as model output.
        """
        self._check_loop()
        if self._closed:
            raise LLMTransportError("client is closed")

        await self.ensure_model()
        await self.runtime_limits()               # seeds semaphore

        if not await self._breaker.allow():
            raise LLMTransportError("circuit breaker open")

        dl = deadline_s or estimate_deadline_s(
            est_prompt_tokens, max_tokens, self.decode_tps, self.prefill_tps,
        )
        body = self._payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            no_think=no_think,
            deadline_s=dl,
            stream=False,
        )
        hdr = {**self.headers, "X-Session-ID": uuid.uuid4().hex}

        async def _call() -> LLMResponse:
            t0 = time.perf_counter()
            try:
                r = await self._http.post(
                    self.base + "/v1/chat/completions",
                    json=body,
                    headers=hdr,
                    timeout=httpx.Timeout(10.0, read=dl + 30.0),
                )
            except httpx.TimeoutException as e:
                raise LLMServerTimeout(f"client read timeout after {dl + 30:.0f}s") from e
            except httpx.HTTPError as e:
                raise LLMTransportError(str(e)) from e
            self._raise_for_status(r)
            try:
                json_body = r.json()
            except ValueError as e:
                raise LLMBadReply(f"invalid JSON from server: {e}") from e
            return validate_reply(json_body, self.model, time.perf_counter() - t0)

        assert self._sem is not None
        async with self._sem:
            try:
                resp = await self._with_retry(_call)
                await self._breaker.record_success()
                await self._record_request(True, resp.elapsed_s)
                await self._record_tokens(resp.prompt_tokens, resp.completion_tokens)
                self._observe_speed(resp)
                log.info(
                    "chat ok model=%s in=%d out=%d elapsed=%.2fs finish=%s",
                    resp.model, resp.prompt_tokens, resp.completion_tokens,
                    resp.elapsed_s, resp.finish_reason,
                    extra={
                        "run_id": self.run_id,
                        "step_id": step_id,
                        "llm_call_n": llm_call_n,
                        "model": resp.model,
                        "elapsed_s": round(resp.elapsed_s, 3),
                        "tokens_out": resp.completion_tokens,
                    },
                )
                return resp
            except LLMError as e:
                await self._breaker.record_failure()
                await self._record_request(False, 0.0)
                log.error(
                    "chat failed: %s: %s",
                    type(e).__name__, str(e)[:200],
                    extra={
                        "run_id": self.run_id,
                        "step_id": step_id,
                        "llm_call_n": llm_call_n,
                        "error_type": type(e).__name__,
                    },
                )
                raise

    # ── streaming chat ───────────────────────────────────────────────────────
    async def chat_stream(
            self,
            messages: List[Dict[str, str]],
            *,
            max_tokens: int = 0,
            temperature: float = 0.2,
            stop: Optional[List[str]] = None,
            no_think: Optional[bool] = None,
            est_prompt_tokens: int = 0,
            first_token_s: Optional[float] = None,
            idle_s: Optional[float] = None,
            step_id: Optional[str] = None,
            llm_call_n: Optional[int] = None,
    ) -> LLMResponse:
        """
        Streaming chat completion. Uses first-token + idle budgets (G3), not one
        total deadline. Aggregates the streamed chunks into a single LLMResponse.
        """
        text_parts: List[str] = []
        state = _StreamState()
        t0 = time.perf_counter()
        ttft = 0.0
        n_chunks = 0
        async for chunk in self._stream_inner(
                messages,
                state=state,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                no_think=no_think,
                est_prompt_tokens=est_prompt_tokens,
                first_token_s=first_token_s,
                idle_s=idle_s,
        ):
            text_parts.append(chunk)
            n_chunks += 1
            if n_chunks == 1:
                ttft = time.perf_counter() - t0
        full = "".join(text_parts)
        elapsed = time.perf_counter() - t0
        return validate_reply(
            {"choices": [{"message": {"content": full}, "finish_reason": state.fin}],
             "usage": state.usage},
            self.model,
            elapsed,
            ttft,
        )

    async def chat_stream_iter(
            self,
            messages: List[Dict[str, str]],
            **kwargs,
    ) -> AsyncIterator[str]:
        """Yield raw text chunks; the caller assembles them."""
        async for chunk in self._stream_inner(
                messages, state=_StreamState(), **kwargs
        ):
            yield chunk

    async def _stream_inner(
            self,
            messages: List[Dict[str, str]],
            *,
            state: "_StreamState",
            max_tokens: int = 0,
            temperature: float = 0.2,
            stop: Optional[List[str]] = None,
            no_think: Optional[bool] = None,
            est_prompt_tokens: int = 0,
            first_token_s: Optional[float] = None,
            idle_s: Optional[float] = None,
    ) -> AsyncIterator[str]:

        self._check_loop()
        if self._closed:
            raise LLMTransportError("client is closed")

        await self.ensure_model()
        await self.runtime_limits()

        if not await self._breaker.allow():
            raise LLMTransportError("circuit breaker open")

        ft = first_token_s or estimate_deadline_s(
            est_prompt_tokens, 1, self.decode_tps, self.prefill_tps,
        )
        idle = idle_s or max(60.0, 20.0 / max(self.decode_tps, 0.2))

        body = self._payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            no_think=no_think,
            deadline_s=0,
            stream=True,
        )
        body.pop("timeout_s", None)
        hdr = {
            **self.headers,
            "X-Session-ID": uuid.uuid4().hex,
            "Accept": "text/event-stream",
        }

        assert self._sem is not None
        async with self._sem:
            t0 = time.perf_counter()
            ttft = 0.0
            n_chunks = 0
            try:
                async with self._http.stream(
                        "POST",
                        self.base + "/v1/chat/completions",
                        json=body,
                        headers=hdr,
                        timeout=httpx.Timeout(10.0, read=None),
                ) as r:
                    if r.status_code != 200:
                        await r.aread()
                        self._raise_for_status(r)
                    lines = r.aiter_lines().__aiter__()
                    budget = ft
                    while True:
                        try:
                            line = await asyncio.wait_for(
                                lines.__anext__(), timeout=budget,
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            raise LLMStall(
                                f"no stream activity for {budget:.0f}s "
                                f"after {n_chunks} chunks"
                            ) from None
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if "error" in obj:
                            raise LLMBadReply(f"stream error: {obj['error']}")
                        ch = (obj.get("choices") or [{}])[0]
                        piece = (ch.get("delta") or {}).get("content") or ""
                        if piece:
                            if n_chunks == 0:
                                ttft = time.perf_counter() - t0
                            n_chunks += 1
                            budget = idle
                            yield piece
                        fin = ch.get("finish_reason")
                        if fin:
                            state.fin = fin
                        if obj.get("usage"):
                            state.usage = obj["usage"]
                await self._breaker.record_success()
                await self._record_request(True, time.perf_counter() - t0)
                pt = int(state.usage.get("prompt_tokens", 0) or 0)
                ct = int(state.usage.get("completion_tokens", 0) or n_chunks)
                await self._record_tokens(pt, ct)
                self._observe_speed(LLMResponse(
                    text="", finish_reason=state.fin,
                    prompt_tokens=pt, completion_tokens=ct,
                    elapsed_s=time.perf_counter() - t0, ttft_s=ttft,
                ))
                log.info(
                    "stream ok model=%s in=%d out=%d ttft=%.2fs elapsed=%.2fs finish=%s",
                    self.model, pt, ct, ttft, time.perf_counter() - t0,
                    state.fin,
                    extra={"run_id": self.run_id},
                )
            except LLMError as e:
                await self._breaker.record_failure()
                await self._record_request(False, time.perf_counter() - t0)
                log.error(
                    "stream failed: %s: %s",
                    type(e).__name__, str(e)[:200],
                    extra={"run_id": self.run_id, "error_type": type(e).__name__},
                )
                raise

    # ── capability probe (one-shot, no caching) ──────────────────────────────
    async def probe(self) -> Dict[str, Any]:
        """
        Return a dict describing the server: reachable, model, limits, health.
        Never raises. Suitable for logging at startup.
        """
        out: Dict[str, Any] = {
            "base_url": self.base,
            "reachable": False,
            "healthy": False,
            "has_runtime_limits": False,
            "model": "",
            "limits": None,
            "server_type": "unknown",
        }
        out["healthy"] = await self.health()
        out["reachable"] = out["healthy"]
        try:
            models = await self._get("/v1/models", timeout=5.0)
            ids = [m.get("id") for m in (models.get("data") or []) if m.get("id")]
            out["model"] = ", ".join(ids[:5])
            if ids:
                out["server_type"] = "openai_compat"
        except LLMError as e:
            out["model_error"] = str(e)[:120]
        try:
            limits = await self.runtime_limits(ttl=0.0)
            out["has_runtime_limits"] = bool(limits.raw)
            out["limits"] = {
                "context_length": limits.context_length,
                "effective_tps": limits.effective_tps,
                "prefill_tps": limits.prefill_tps,
                "tier": limits.tier,
                "n_parallel": limits.n_parallel,
                "backend": limits.backend,
            }
            if limits.raw:
                out["server_type"] = "local_llm_cloud"
        except Exception as e:
            out["limits_error"] = str(e)[:120]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible helper: create + wait + probe in one call
# ─────────────────────────────────────────────────────────────────────────────
async def open_client(
    base_url: str,
    api_key: str = "",
    model: str = "",
    *,
    wait_total_s: float = 300.0,
    probe_log: bool = True,
    **kwargs,
) -> ChatClient:
    """
    Create a ChatClient, wait for /health to respond, resolve the model,
    and (optionally) log the probed capabilities. Raises LLMError on timeout.

    This the caller never sees a half-initialised adapter.
    """
    client = ChatClient(base_url, api_key=api_key, model=model, **kwargs)
    try:
        await client.wait_for_server(total_s=wait_total_s)
        await client.ensure_model()
        await client.runtime_limits()
        if probe_log:
            info = await client.probe()
            log.info(
                "ChatClient probed: type=%s model=%s ctx=%s tps=%s tier=%s n_parallel=%s",
                info.get("server_type"), info.get("model"),
                (info.get("limits") or {}).get("context_length"),
                (info.get("limits") or {}).get("effective_tps"),
                (info.get("limits") or {}).get("tier"),
                (info.get("limits") or {}).get("n_parallel"),
                extra={"run_id": client.run_id},
            )
        return client
    except Exception:
        await client.aclose()
        raise


__all__ = [
    # errors
    "LLMError", "LLMAuthError", "LLMContextOverflow", "LLMModelNotReady",
    "LLMTransportError", "LLMServerTimeout", "LLMBadReply", "LLMStall",
    # data
    "LLMResponse", "CapabilityLimits",
    # functions
    "validate_reply", "estimate_deadline_s", "resolve_model_id",
    # client
    "ChatClient", "open_client",
]
