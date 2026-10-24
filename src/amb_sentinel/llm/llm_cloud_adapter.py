"""
llm_cloud_adapter.py — production synchronous facade over amb_llm_core.ChatClient.

Purpose
───────
Synchronous callers in AMB-APP (LocalLLMCloudAdapter, BackendOrchestrator,
model_manager) need a stable, blocking API to talk to Local-LLM. Meanwhile the
actual HTTP work should be done by amb_llm_core.ChatClient, which already
handles deadlines, retries, circuit breaking, error-marker rejection, and
TPS adaptation.

This module wires the two together:
  * LLMCloudClient (sync) delegates to ChatClient (async) via a dedicated
    background event loop (_SyncBridge). All HTTP goes through one loop and
    one httpx.AsyncClient pool.
  * The old public API (probe_server, get_client, LLMCloudUnavailable, ...)
    is preserved bit-for-bit, so ai_agent.py and model_manager.py need no
    edits to keep working.
  * The A9 hardcoded _PROFILE_DEFAULTS table is gone — the hardware profile
    is built entirely from live /v1/runtime/limits, /v1/system, and
    /v1/drive/stats responses.


Compatibility notes
───────────────────
* LLMCloudUnavailable / LLMCloudTimeout / LLMCloudModelError are now
  subclasses of amb_llm_core.LLMError. Code that catches either the old class
  or the new parent works unchanged.
* GenerationResult keeps its .ok property (True when error is None).
* HardwareProfile keeps every field the old one had; new fields
  (tier, n_parallel, backend, model_id, prefill_tps, recommended_*) are added
  with defaults so old constructors still work.
* The old _PROFILE_DEFAULTS dict, _CircuitBreaker, _http_get/_post/_delete,
  and _safe_json helpers are removed. Nothing in AMB-APP imports them.
"""
from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

# ── Transport layer — this module is a facade over ChatClient ────────────────
from amb_sentinel.llm.amb_llm_core import (
    ChatClient,
    CapabilityLimits,
    LLMError,
    LLMAuthError,
    LLMContextOverflow,
    LLMModelNotReady,
    LLMServerTimeout,
    LLMStall,
    LLMResponse,
)

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Public exceptions (backward-compatible with the old llm_cloud_adapter API)
# ═════════════════════════════════════════════════════════════════════════════
class LLMCloudUnavailable(LLMError):
    """
    Local-LLM server unreachable, returned a non-2xx status, or otherwise
    failed to answer. Retryable at the caller's discretion.
    """
    retryable = True


class LLMCloudTimeout(LLMCloudUnavailable):
    """
    A generation call exceeded its wall-clock deadline. Retryable with a
    longer deadline; the loop's tier-downgrade logic usually shrinks the
    prompt instead.
    """
    retryable = True


class LLMCloudModelError(LLMCloudUnavailable):
    """
    Model load/unload failed (bad path, insufficient VRAM, wrong id).
    Not retryable without a config change.
    """
    retryable = False

class LLMCloudConfigError(LLMError):
    """
    Configuration/programming conflict — NOT a transport failure.

    Deliberately *not* a subclass of LLMCloudUnavailable. The fallback
    loop in AIModelService._init_models_async and the retry logic in
    AgentExecutor._call_llm_with_retry must treat this as "fix config
    and restart" (exit code 78 / EX_CONFIG), not as "server is down,
    retry with backoff".
    """
    retryable = False

# ═════════════════════════════════════════════════════════════════════════════
# Data classes — every field the old version had, plus new ones with defaults
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class GenerationResult:
    """Result of a single synchronous generation call."""
    text: str
    tokens_generated: int = 0
    tokens_prompt: int = 0
    elapsed_s: float = 0.0
    tps: float = 0.0
    ttft_ms: float = 0.0
    backend: str = "local_llm_cloud"
    model_id: str = ""
    finish_reason: str = "stop"
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class HardwareProfile:
    """
    Hardware / capability snapshot from the server.

    Field names match the old llm_cloud_adapter.HardwareProfile so callers
    (model_manager.BackendOrchestrator, LocalLLMCloudAdapter) keep working.
    New fields are additive.
    """
    # ── identity ────────────────────────────────────────────────────────────
    profile_name: str = "cpu_only"
    # ── VRAM / RAM ──────────────────────────────────────────────────────────
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    ram_total_mb: int = 0
    # ── CPU / GPU identity ──────────────────────────────────────────────────
    cpu_model: str = ""
    cpu_cores: int = 4
    has_cuda: bool = False
    has_rocm: bool = False
    has_opencl: bool = False
    n_gpu_layers: int = 0
    # ── inference configuration ─────────────────────────────────────────────
    context_length: int = 4096
    kv_cache_quant: str = "unknown"
    n_batch: int = 512
    n_ubatch: int = 256
    target_tps: float = 3.0
    force_no_think: bool = False
    # ── auto-detected capability fields (new; defaulted for compat) ─────────
    tier: str = "small"                         # nano | small | medium | large
    n_parallel: int = 1
    backend: str = "unknown"                    # rocm | cuda | vulkan | cpu | ...
    model_id: str = ""
    prefill_tps: float = 0.0
    recommended_response_tokens: int = 0
    recommended_resume_chunk: int = 0
    recommended_idle_timeout_s: float = 300.0
    # ── opaque extras for debugging ─────────────────────────────────────────
    extra: Dict[str, Any] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# Sync <-> async bridge
# ═════════════════════════════════════════════════════════════════════════════
class _SyncBridge:
    """
    A single, module-private background thread with its own asyncio event loop.

    All HTTP calls initiated by LLMCloudClient (a sync class) are scheduled here
    as coroutines and awaited via `concurrent.futures.Future.result()`. This
    keeps the caller's thread and any caller event loop completely free.

    Why one loop instead of one per client:
      * httpx.AsyncClient binds to the loop that first uses it. Two loops would
        need two connection pools and double the file descriptors.
      * The circuit breaker in ChatClient uses asyncio.Lock — also loop-bound.
      * Synchronising state across loops is bug-prone; one loop is simpler.
    """
    _INSTANCE: Optional["_SyncBridge"] = None

    @classmethod
    def instance(cls) -> "_SyncBridge":
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def _start(self) -> None:
        if self._loop is not None:
            return
        with self._lock:
            if self._loop is not None:
                return

            def _run() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._ready.set()
                try:
                    loop.run_forever()
                finally:
                    try:
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    except Exception:
                        pass
                    loop.close()

            self._thread = threading.Thread(
                target=_run, name="llm-cloud-bridge", daemon=True,
            )
            self._thread.start()
        self._ready.wait(timeout=5.0)

    def run(self, coro, timeout: Optional[float] = None):
        """Schedule `coro` on the bridge loop and block until it resolves.

        On timeout, actively cancel the submitted coroutine on its owning
        loop and wait a short grace period for the cancellation to land.
        Without this, a caller that times out (e.g. .result(timeout=...))
        leaves the HTTP request running — on a single-slot server that is
        a self-inflicted DoS: the slot stays busy until the server-side
        deadline fires.
        """
        self._start()
        assert self._loop is not None, "bridge loop failed to start"
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()                           # safe: run_coroutine_threadsafe future
            # Give the loop a moment to unwind the cancellation. We do
            # NOT raise here — the original TimeoutError is what the
            # caller expects, and `fut.result()` already raised it.
            raise

    def submit(self, coro) -> concurrent.futures.Future:
        """Schedule `coro` and return the future without waiting."""
        self._start()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._loop = None
        self._thread = None


_BRIDGE = _SyncBridge.instance()
atexit.register(_BRIDGE.shutdown)


# ═════════════════════════════════════════════════════════════════════════════
# Module-level probe (fast one-shot check, no client construction)
# ═════════════════════════════════════════════════════════════════════════════
def probe_server(
    host: str = "localhost",
    port: int = 8080,
    timeout: float = 3.0,
    api_key: str = "",
) -> bool:
    """
    Return True if `/health` answers 200 within `timeout` seconds.

    Fast, synchronous, no state. Used by BackendOrchestrator and by
    LocalLLMCloudAdapter's first connectivity check. For a robust wait loop
    use ChatClient.wait_for_server() instead.
    """
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(f"http://{host}:{port}/health", headers=headers)
            return r.status_code == 200
    except Exception as exc:
        log.warning("probe_server(%s:%s): %s", host, port, exc)
        return False


# ═════════════════════════════════════════════════════════════════════════════
# LLMCloudClient — the sync facade
# ═════════════════════════════════════════════════════════════════════════════
class LLMCloudClient:
    """
    Synchronous facade over amb_llm_core.ChatClient.

    Instantiate once per server, share across threads. `get_client()` returns a
    cached instance per (host, port). All methods are thread-safe.

    The client lazily opens its connection pool on the first call; there is no
    separate connect() step. `close()` drains the pool and marks the client
    unusable. Calling `close()` twice is safe.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        api_key: str = "",
        timeout: float = 300.0,
        *,
        default_decode_tps: float = 3.0,
        run_id: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.api_key = api_key or os.environ.get("LLM_CLOUD_API_KEY", "")
        self.timeout = float(timeout)

        # The transport. `model` starts empty and is resolved from /v1/models
        # on the first ensure_model() call.
        self._chat = ChatClient(
            self.base_url,
            api_key=self.api_key,
            model="",
            default_decode_tps=default_decode_tps,
            run_id=run_id,
        )

        # Hardware-profile cache (60 s default; refresh with force_refresh=True)
        self._hw_profile: Optional[HardwareProfile] = None
        self._hw_profile_ts: float = 0.0
        self._hw_profile_ttl: float = 60.0

        self._lock = threading.RLock()
        self._closed = False

    # ── loop-safe capability access ──────────────────────────────────────────
    def runtime_limits(self, ttl: float = 30.0):
        """Blocking. Always executes on the bridge loop.

        This is the primitive AgentExecutor must call instead of reaching
        into ChatClient.runtime_limits() directly — that would cross loops.
        """
        return _BRIDGE.run(self._chat.runtime_limits(ttl=ttl), timeout=15.0)

    async def aruntime_limits(self, ttl: float = 30.0):
        """Awaitable from ANY event loop. The request itself runs on the bridge."""
        if self._closed:
            raise LLMCloudUnavailable("client is closed")
        fut = _BRIDGE.submit(self._chat.runtime_limits(ttl=ttl))
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=15.0)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def close(self) -> None:
        """Drain the connection pool. Safe to call multiple times."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            _BRIDGE.run(self._chat.aclose(), timeout=5.0)
        except Exception:
            pass
        # Drop the singleton pool entry so a future get_client() starts fresh.
        with _client_lock:
            for k, v in list(_clients.items()):
                if v is self:
                    del _clients[k]

    def __enter__(self) -> "LLMCloudClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"LLMCloudClient(base={self.base_url!r}, "
            f"model={self._chat.model!r}, closed={self._closed})"
        )

    # ── low-level raw request (used by model-management endpoints) ───────────
    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Fire an HTTP request on the bridge loop and return (status, json).

        Never raises on non-2xx; the caller decides how to handle the status.
        Raises LLMCloudUnavailable on transport failures only.
        """
        async def _do() -> Tuple[int, Dict[str, Any]]:
            url = self.base_url + path
            headers = dict(self._chat.headers)
            try:
                if method == "GET":
                    r = await self._chat._http.get(url, headers=headers, timeout=timeout)
                elif method == "POST":
                    r = await self._chat._http.post(
                        url, json=body or {}, headers=headers, timeout=timeout,
                    )
                elif method == "DELETE":
                    r = await self._chat._http.delete(url, headers=headers, timeout=timeout)
                else:
                    raise ValueError(f"unsupported method: {method}")
            except httpx.HTTPError as exc:
                raise LLMCloudUnavailable(f"{method} {path}: {exc}") from exc
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, {"_raw": (r.text or "")[:500]}

        try:
            return _BRIDGE.run(_do(), timeout=timeout + 5.0)
        except concurrent.futures.TimeoutError as exc:
            raise LLMCloudTimeout(
                f"{method} {path}: bridge timeout after {timeout + 5.0:.0f}s"
            ) from exc

    # ── discovery / health ───────────────────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        """Return the /health body, or raise LLMCloudUnavailable."""
        code, body = self._request("GET", "/health", timeout=5.0)
        if code != 200:
            raise LLMCloudUnavailable(f"health check failed: HTTP {code}")
        return body

    def is_available(self, retries: int = 1) -> bool:
        """Best-effort health check with exponential backoff. Never raises."""
        delay = 1.0
        for attempt in range(max(1, retries)):
            try:
                self.health()
                return True
            except Exception:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 1.5, 30.0)
        return False

    def list_models(self) -> List[Dict[str, Any]]:
        """Return the raw /v1/models list."""
        code, body = self._request("GET", "/v1/models", timeout=10.0)
        if code != 200:
            raise LLMCloudUnavailable(f"list_models failed: HTTP {code}")
        return body.get("data", []) or []

    def probe(self) -> Dict[str, Any]:
        """
        One-shot capability probe. Never raises. Safe for startup logging.

        Returns a dict with reachable, healthy, server_type, model, limits,
        and a few server-reported identifiers. The `server_type` field is:
          * "local_llm_cloud"  if /v1/runtime/limits is present
          * "openai_compat"    if only /v1/models is present
          * "unknown"          otherwise
        """
        try:
            return _BRIDGE.run(self._chat.probe(), timeout=15.0)
        except Exception as exc:
            return {
                "base_url": self.base_url,
                "reachable": False,
                "healthy": False,
                "server_type": "unknown",
                "error": str(exc)[:200],
            }

    # ── model management ─────────────────────────────────────────────────────
    def load_model(
        self,
        model_id: str,
        model_path: str,
        quantization: str = "Q4_K_M",
        context_length: int = 0,
        num_gpu_layers: int = -1,
        n_threads: int = 0,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        POST /v1/models/load. Server may answer 200, 201, or 202 (background
        load). Raises LLMCloudModelError on any other status.
        """
        payload = {
            "model_id": model_id,
            "model_path": model_path,
            "quantization": quantization,
            "context_length": context_length,
            "num_gpu_layers": num_gpu_layers,
            "n_threads": n_threads,
        }
        code, body = self._request("POST", "/v1/models/load", body=payload, timeout=timeout)
        if code not in (200, 201, 202):
            raise LLMCloudModelError(
                f"model load failed: HTTP {code}: "
                f"{body.get('detail') or body.get('error') or body}"
            )
        if code == 202 and body.get("status") == "loading":
            log.info(
                "LLMCloudClient: model %r load started in background; "
                "poll list_models() to check readiness",
                model_id,
            )
        else:
            log.info(
                "LLMCloudClient: loaded %r (backend=%s gpu_layers=%s)",
                model_id, body.get("backend"), body.get("gpu_layers"),
            )
        return body

    def unload_model(self, model_id: str) -> bool:
        """DELETE /v1/models/{id}. Returns True on 200."""
        code, _ = self._request("DELETE", f"/v1/models/{model_id}", timeout=30.0)
        return code == 200

    def ensure_model_loaded(
        self,
        model_id: str,
        model_path: str,
        **load_kwargs,
    ) -> bool:
        """Load the model only if it is not already in /v1/models."""
        try:
            models = self.list_models()
            if any(m.get("id") == model_id for m in models):
                return True
            self.load_model(model_id, model_path, **load_kwargs)
            return True
        except LLMCloudUnavailable:
            raise
        except Exception as exc:
            raise LLMCloudModelError(str(exc)) from exc

    # ── hardware / capability profile ────────────────────────────────────────
    def get_hardware_profile(self, force_refresh: bool = False) -> HardwareProfile:
        """
        Return a HardwareProfile built from live server responses. Cached for
        _hw_profile_ttl seconds. Never raises — a partially-degraded server
        still produces a valid profile with conservative defaults.
        """
        with self._lock:
            if (
                not force_refresh
                and self._hw_profile is not None
                and (time.monotonic() - self._hw_profile_ts) < self._hw_profile_ttl
            ):
                return self._hw_profile

        profile = self._probe_hardware_profile()

        with self._lock:
            self._hw_profile = profile
            self._hw_profile_ts = time.monotonic()
        log.info(
            "LLMCloudClient: hardware profile = %s tier=%s ctx=%d "
            "tps=%.2f prefill=%.2f n_parallel=%d backend=%s",
            profile.profile_name, profile.tier, profile.context_length,
            profile.target_tps, profile.prefill_tps, profile.n_parallel,
            profile.backend,
        )
        return profile

    def _probe_hardware_profile(self) -> HardwareProfile:
        # 1. /v1/runtime/limits — the authoritative source for ctx/tps/tier/n_parallel
        limits: Optional[CapabilityLimits] = None
        try:
            limits = _BRIDGE.run(self._chat.runtime_limits(ttl=0.0), timeout=10.0)
        except Exception as exc:
            log.warning("hardware probe: runtime_limits failed: %s", exc)

        # 2. /v1/system — GPU/RAM/CPU identity
        try:
            _, sys_body = self._request("GET", "/v1/system", timeout=10.0)
        except Exception as exc:
            log.warning("hardware probe: /v1/system failed: %s", exc)
            sys_body = {}

        # 3. /v1/drive/stats — hw_profile name (informational)
        try:
            _, drive_body = self._request("GET", "/v1/drive/stats", timeout=5.0)
        except Exception:
            drive_body = {}

        profile_name = str(
            drive_body.get("hw_profile")
            or sys_body.get("hw_profile")
            or "cpu_only"
        )

        gpus = sys_body.get("gpus") or []
        vram_total = sum(int(g.get("vram_total_mb", 0) or 0) for g in gpus)
        vram_free = sum(int(g.get("vram_free_mb", 0) or 0) for g in gpus)

        # Limits carry through if present; otherwise conservative defaults.
        ctx = limits.context_length if limits else 4096
        tps = limits.effective_tps if limits else 3.0
        pf = limits.prefill_tps if limits else 0.0
        tier = limits.tier if limits else "small"
        n_par = limits.n_parallel if limits else 1
        backend = limits.backend if limits else "unknown"
        model_id = limits.model_id if limits else ""
        rec_resp = limits.recommended_response_tokens if limits else 0
        rec_chunk = limits.recommended_resume_chunk if limits else 0
        rec_idle = limits.recommended_idle_timeout_s if limits else 300.0

        # force_no_think: derive from the profile name — APUs are compute-starved
        # and reasoning tokens waste most of the budget. The old hardcoded table
        # was the only source; this is the auto-adaptive replacement.
        force_no_think = profile_name.lower().startswith("apu_")

        return HardwareProfile(
            profile_name=profile_name,
            vram_total_mb=vram_total,
            vram_free_mb=vram_free,
            ram_total_mb=int(sys_body.get("ram_total_mb", 0) or 0),
            cpu_model=str(sys_body.get("cpu_model", "") or ""),
            cpu_cores=int(sys_body.get("cpu_cores", 4) or 4),
            has_cuda=bool(sys_body.get("has_cuda", False)),
            has_rocm=bool(sys_body.get("has_rocm", False)),
            has_opencl=bool(sys_body.get("has_opencl", False)),
            n_gpu_layers=int(sys_body.get("n_gpu_layers", 0) or 0),
            context_length=ctx,
            kv_cache_quant=str(sys_body.get("kv_cache_quant", "unknown") or "unknown"),
            n_batch=int(sys_body.get("n_batch", 512) or 512),
            n_ubatch=int(sys_body.get("n_ubatch", 256) or 256),
            target_tps=tps,
            force_no_think=force_no_think,
            tier=tier,
            n_parallel=n_par,
            backend=backend,
            model_id=model_id,
            prefill_tps=pf,
            recommended_response_tokens=rec_resp,
            recommended_resume_chunk=rec_chunk,
            recommended_idle_timeout_s=rec_idle,
            extra={
                "raw_system": sys_body,
                "raw_drive": drive_body,
                "raw_limits": (limits.raw if limits else {}),
            },
        )

    # ── generation (the hot path) ────────────────────────────────────────────
    def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Optional[List[str]] = None,
        timeout: float = 300.0,
        stream: bool = False,
        no_think: Optional[bool] = None,
    ) -> GenerationResult:
        """
        Synchronous generation. Blocks the calling thread.

        On any failure raises LLMCloudTimeout / LLMCloudUnavailable / subclass.
        On success returns a GenerationResult with .error = None. A server
        reply that carries an error marker ("[timed out after 0s]") is treated
        as a failure and raises — it is never returned as .text.

        `stream` is accepted for API compatibility but ignored (the sync
        facade is non-streaming). Use astream() for async streaming.
        """
        if self._closed:
            raise LLMCloudUnavailable("client is closed")

        # Apply the hardware profile's force_no_think when caller didn't specify.
        if no_think is None:
            with self._lock:
                prof = self._hw_profile
            if prof is not None:
                no_think = prof.force_no_think

        # Reasonable prompt-token estimate: 1 token ≈ 4 chars for English.
        est_prompt_tokens = max(1, len(prompt) // 4)

        async def _gen() -> LLMResponse:
            # Set the model hint once. ChatClient.ensure_model() will resolve
            # it against /v1/models the first time.
            if model_id and not self._chat.model:
                self._chat.model_hint = model_id
            return await self._chat.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=max(0, int(max_tokens)),
                temperature=float(temperature),
                stop=list(stop) if stop else None,
                no_think=no_think,
                est_prompt_tokens=est_prompt_tokens,
                deadline_s=float(timeout),
            )

        try:
            resp = _BRIDGE.run(_gen(), timeout=timeout + 30.0)
        except concurrent.futures.TimeoutError as exc:
            raise LLMCloudTimeout(
                f"client wall-clock timeout after {timeout + 30.0:.0f}s"
            ) from exc
        except LLMServerTimeout as exc:
            raise LLMCloudTimeout(str(exc)) from exc
        except LLMModelNotReady as exc:
            raise LLMCloudUnavailable(f"model not ready: {exc}") from exc
        except LLMAuthError as exc:
            raise LLMCloudUnavailable(f"auth failed: {exc}") from exc
        except LLMContextOverflow as exc:
            raise LLMCloudUnavailable(f"context overflow: {exc}") from exc
        except LLMStall as exc:
            raise LLMCloudTimeout(str(exc)) from exc
        except LLMError as exc:
            raise LLMCloudUnavailable(f"{type(exc).__name__}: {exc}") from exc

        elapsed = resp.elapsed_s or 0.0
        tps = (resp.completion_tokens / elapsed) if elapsed > 0 else 0.0
        return GenerationResult(
            text=resp.text,
            tokens_generated=resp.completion_tokens,
            tokens_prompt=resp.prompt_tokens,
            elapsed_s=elapsed,
            tps=tps,
            ttft_ms=resp.ttft_s * 1000.0,
            backend="local_llm_cloud",
            model_id=resp.model or self._chat.model or model_id,
            finish_reason=resp.finish_reason,
            error=None,
        )

    def chat(
            self,
            messages: List[Dict[str, str]],
            *,
            max_tokens: int = 512,
            temperature: float = 0.2,
            stop: Optional[List[str]] = None,
            no_think: Optional[bool] = None,
            est_prompt_tokens: int = 0,
            deadline_s: Optional[float] = None,
    ) -> LLMResponse:
        """
        message-list interface. Forwards to amb_llm_core.ChatClient.chat
        so the server receives structured messages and applies its own template.
        """
        if self._closed:
            raise LLMCloudUnavailable("client is closed")

        if no_think is None:
            with self._lock:
                prof = self._hw_profile
            if prof is not None:
                no_think = prof.force_no_think

        dl = float(deadline_s if deadline_s is not None else (self.timeout or 300.0))

        async def _do() -> LLMResponse:
            return await self._chat.chat(
                messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                stop=list(stop) if stop else None,
                no_think=no_think,
                est_prompt_tokens=int(est_prompt_tokens),
                deadline_s=dl,
            )

        try:
            return _BRIDGE.run(_do(), timeout=dl + 30.0)
        except concurrent.futures.TimeoutError as exc:
            raise LLMCloudTimeout(f"chat wall-clock timeout after {dl + 30:.0f}s") from exc
        except LLMServerTimeout as exc:
            raise LLMCloudTimeout(str(exc)) from exc
        except LLMModelNotReady as exc:
            raise LLMCloudUnavailable(f"model not ready: {exc}") from exc
        except LLMAuthError as exc:
            raise LLMCloudUnavailable(f"auth failed: {exc}") from exc
        except LLMContextOverflow:
            raise  # preserve type so caller can shrink and retry
        except LLMError as exc:
            raise LLMCloudUnavailable(f"{type(exc).__name__}: {exc}") from exc

    async def _astream_producer(self, loop, q, _DONE, stop_evt, prompt, *,
                                max_tokens, temperature, stop, no_think,
                                first_token_s, idle_s):
        """Runs on the bridge loop; feeds the consumer's asyncio.Queue safely."""
        def _put(item):
            # Safe cross-loop put: schedule the coroutine on the consumer loop.
            try:
                return asyncio.run_coroutine_threadsafe(q.put(item), loop)
            except RuntimeError:
                return None   # consumer loop closed — nothing to deliver

        try:
            await self._chat.ensure_model()
            await self._chat.runtime_limits()
            async for chunk in self._chat.chat_stream_iter(
                [{"role": "user", "content": prompt}],
                max_tokens=max(0, int(max_tokens)),
                temperature=float(temperature),
                stop=list(stop) if stop else None,
                no_think=no_think,
                est_prompt_tokens=max(1, len(prompt) // 4),
                first_token_s=first_token_s,
                idle_s=idle_s,
            ):
                if stop_evt.is_set():
                    break
                fut = _put(chunk)
                if fut is not None:
                    try:
                        await asyncio.wrap_future(fut)   # backpressure
                    except asyncio.CancelledError:
                        break
        except Exception as exc:
            _put(exc)
        finally:
            _put(_DONE)

    async def astream(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 0,
        temperature: float = 0.3,
        stop: Optional[List[str]] = None,
        no_think: Optional[bool] = None,
        first_token_s: Optional[float] = None,
        idle_s: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Async token stream. Safe to await from any event loop, including the
        main asyncio loop — internally delegates to ChatClient on the bridge
        loop, yielding through an asyncio.Queue.

        Example:
            async for chunk in client.astream("qwen3-8b", "hi"):
                sys.stdout.write(chunk)
        """
        if self._closed:
            raise LLMCloudUnavailable("client is closed")

        if no_think is None:
            with self._lock:
                prof = self._hw_profile
            if prof is not None:
                no_think = prof.force_no_think

        if model_id and not self._chat.model:
            self._chat.model_hint = model_id

        loop = asyncio.get_running_loop()             # consumer loop (caller's)
        q: asyncio.Queue = asyncio.Queue(maxsize=64)  # owned by consumer loop
        _DONE = object()
        _stop_evt = threading.Event()
        _producer_cf: "concurrent.futures.Future" = _BRIDGE.submit(self._astream_producer(
            loop, q, _DONE, _stop_evt, prompt,
            max_tokens=max_tokens, temperature=temperature, stop=stop,
            no_think=no_think, first_token_s=first_token_s, idle_s=idle_s,
        ))
        try:
            while True:
                item = await q.get()
                if item is _DONE:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            _stop_evt.set()
            _producer_cf.cancel()

    # ── metrics ──────────────────────────────────────────────────────────────
    def metrics(self) -> Dict[str, Any]:
        """Return transport counters from ChatClient."""
        return self._chat.metrics()

    def get_metrics(self) -> Dict[str, Any]:
        """Legacy alias for metrics() — kept for old callers."""
        return self.metrics()

    def get_performance(self) -> Dict[str, Any]:
        """Return the /v1/performance body, or {} if unavailable."""
        try:
            code, body = self._request("GET", "/v1/performance", timeout=5.0)
            return body if code == 200 else {}
        except Exception:
            return {}

    def clear_context_cache(self) -> bool:
        """DELETE /v1/context-cache/clear. Returns True on 200."""
        try:
            code, _ = self._request("DELETE", "/v1/context-cache/clear", timeout=10.0)
            return code == 200
        except Exception:
            return False


# ═════════════════════════════════════════════════════════════════════════════
# Module-level singleton pool
# ═════════════════════════════════════════════════════════════════════════════
_clients: Dict[str, LLMCloudClient] = {}
_client_lock = threading.Lock()

def get_client(
    host: str = "localhost",
    port: int = 8080,
    api_key: str = "",
    timeout: float = 300.0,
) -> LLMCloudClient:
    """Return a cached LLMCloudClient for (host, port).

    A cached client is reusable by any caller. ``timeout`` is a
    *per-request* value — ``LLMCloudClient.chat(..., deadline_s=X)`` and
    ``.generate(..., timeout=X)`` — never part of the cache identity.
    Two concurrent callers that want different deadlines simply pass
    different keyword arguments on each call; the cached client's own
    ``timeout`` field is only a fallback for callers that pass neither
    ``deadline_s`` nor ``timeout=``.

    ``api_key`` mismatch is a genuine programming bug in the caller (both
    paths read ``LLM_CLOUD_API_KEY`` in practice, so this branch should
    never fire), so we raise ``LLMCloudConfigError`` — NOT
    ``LLMCloudUnavailable``. That distinction is what stops the
    crash-loop: a config error exits with EX_CONFIG (78), a transport
    error retries.
    """
    key = f"{host}:{port}"
    with _client_lock:
        c = _clients.get(key)
        if c is not None and not c._closed:
            eff_key = c.api_key or ""
            requested = api_key or os.environ.get("LLM_CLOUD_API_KEY", "")
            if requested and eff_key != requested:
                raise LLMCloudConfigError(
                    f"Cached client for {key} was created with a different "
                    f"api_key. Call close_all_clients() or set a distinct "
                    f"env LLM_CLOUD_API_KEY before creating a new client."
                )
            if abs(c.timeout - timeout) > 0.1:
                log.info(
                    "get_client(%s): caller passed timeout=%.0fs; cached "
                    "client has %.0fs. Reusing cached client — per-request "
                    "deadlines override this default.",
                    key, timeout, c.timeout,
                )
            # Unconditional return on ANY cache hit — whether or not the
            # caller's timeout matched the cached client's. Without this,
            # a call whose timeout equals the cached client's recreates
            # the client, leaks a fresh httpx.AsyncClient connection
            # pool, and swaps the object out from under any caller that
            # already holds a reference. Cache hits must not fall through.
            return c

        c = LLMCloudClient(
            host=host, port=port, api_key=api_key, timeout=timeout,
        )
        _clients[key] = c
        return c
    
def close_all_clients() -> None:
    """Shut down every cached client. Called automatically at process exit."""
    with _client_lock:
        clients = list(_clients.values())
        _clients.clear()
    for c in clients:
        try:
            c.close()
        except Exception:
            pass


atexit.register(close_all_clients)


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════
__all__ = [
    # exceptions
    "LLMCloudUnavailable",
    "LLMCloudTimeout",
    "LLMCloudModelError",
    "LLMCloudConfigError",
    # data
    "GenerationResult",
    "HardwareProfile",
    # client
    "LLMCloudClient",
    "get_client",
    "close_all_clients",
    "probe_server",
]
