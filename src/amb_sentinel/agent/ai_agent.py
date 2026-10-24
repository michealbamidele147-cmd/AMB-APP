#!/usr/bin/env python3
from __future__ import annotations

"""
ai_agent.py - Production-ready local-first AI agent framework (no LangChain)

Enhanced for production with:
- Improved error handling and resilience
- Enhanced configuration validation
- Better resource management and cleanup
- Thread/process safety improvements
- Graceful degradation strategies
- Enhanced monitoring and health checks
- Instruction-driven agent execution from config
- Database integration for data collection and analysis

1. Enhanced logging with consistent format: [timestamp] LEVEL logger_name: message
2. Separate log files for AgentExecutor and DataAnalyzer
3. Concurrent autonomous execution of AgentExecutor and DataAnalyzer
4. Production-ready orchestration with proper state management
5. Complete DatabaseManager integration for all database operations

Author: AMB Team
"""

import asyncio
import concurrent.futures
import json
import inspect
import logging
import os
import signal
import sys
import time
import uuid
import aiohttp
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import functools
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Type, Set
import threading
import random
import psutil
from collections import defaultdict, deque

# Paths and lazy extras — added in phase 4a
from amb_sentinel import paths
from amb_sentinel._extras import lazy

# Load .env from config_dir first, then cwd. No-op if python-dotenv isn't installed.
try:
    from dotenv import load_dotenv
    _env_file = paths.config_dir() / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    load_dotenv()
except ImportError:
    pass

# Core system libs. torch is optional: install with `pip install 'amb-sentinel[ml]'`.
# Lazy import keeps `sentinel health` working on a core-only install.
torch = lazy("torch", "ml")
import yaml

# Scheduler for background jobs
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Enhanced embedding support
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Quick fixes for common issues
import warnings

warnings.filterwarnings("ignore", message="probability tensor contains")

# ---------------------------------------------------------------------
# Enhanced type definitions and constants
# ---------------------------------------------------------------------
T = TypeVar('T')
RETRY_DELAY = 0.5
DEFAULT_TIMEOUT = 30.0
MAX_LOG_SIZE = 50 * 1024 * 1024  # 50MB max log size
LOG_BACKUP_COUNT = 5
MAX_CONCURRENT_REQUESTS = 50
REQUEST_TIMEOUT = 300  # 5 minutes
HEALTH_CHECK_INTERVAL = 60  # seconds
AGENT_EXECUTION_INTERVAL = 600  # 10 minutes between agent runs
DATA_ANALYSIS_INTERVAL = 300  # 5 minutes between data analysis cycles

from amb_sentinel.llm.model_manager import (
    get_model_manager, TokenMetrics,
    get_backend_orchestrator,
)

from amb_sentinel import paths

# Tools and scraper imports
from amb_sentinel.tools import ToolsRegistrar, DEFAULT_DOWNLOAD_DIR

# ── amb_llm_core typed errors ────────────────────────────────────────────
# Adapters must raise these instead of returning "ERROR: …"
# strings. The retry logic in AgentExecutor and any future caller of
# amb_runtime.AgentRun can then distinguish retryable transport failures
# (LLMTransportError, LLMServerTimeout) from bad-reply failures
# (LLMBadReply, LLMContextOverflow) and handle each correctly.
try:
    from amb_sentinel.llm.amb_llm_core import (
        LLMError,
        LLMAuthError,
        LLMContextOverflow,
        LLMModelNotReady,
        LLMTransportError,
        LLMServerTimeout,
        LLMBadReply,
        LLMStall,
    )
    _AMB_LLM_CORE_AVAILABLE = True
except ImportError:
    _AMB_LLM_CORE_AVAILABLE = False

    # Fallback definitions so adapters still raise typed exceptions when
    # amb_llm_core is not importable (e.g. running standalone).
    class LLMError(RuntimeError):
        retryable: bool = False
    class LLMAuthError(LLMError):            retryable = False
    class LLMContextOverflow(LLMError):      retryable = False
    class LLMModelNotReady(LLMError):        retryable = True
    class LLMTransportError(LLMError):       retryable = True
    class LLMServerTimeout(LLMError):        retryable = True
    class LLMBadReply(LLMError):             retryable = True
    class LLMStall(LLMServerTimeout):        pass

class ConfigError(RuntimeError):
    """Fatal configuration error; main() exits with EX_CONFIG (78) so
    systemd's RestartPreventExitStatus=78 stops the crash loop instead
    of restarting every 30 s against the same broken config.
    """

# Database tools
try:
    from amb_sentinel.tools.dbtools import DatabaseManager, register_db_tools, _safe_ident as _db_safe_ident

    DBTOOLS_AVAILABLE = True
except ImportError:
    DBTOOLS_AVAILABLE = False
    DatabaseManager = None
    register_db_tools = None

    # Fallback validator if dbtools not installed
    import re as _re
    _DB_IDENT_RE = _re.compile(r'^[A-Za-z_][A-Za-z0-9_$]{0,62}$')

    def _db_safe_ident(name: str) -> str:
        """Safe SQL identifier quoting (fallback when dbtools unavailable)."""
        if not isinstance(name, str) or not _DB_IDENT_RE.match(name):
            raise ValueError(
                f"Unsafe SQL identifier rejected: {name!r}. "
                "Only letters, digits, underscores and $ are permitted."
            )
        return '"' + name.replace('"', '""') + '"'

from amb_sentinel.logging_setup import configure, get_logger
# sets up every log file in one shot
configure()
# Initialize logging
logger = get_logger("ai_agent")

# ---------------------------------------------------------------------
# Optional imports (keep as is)
# ---------------------------------------------------------------------
_OPTIONAL_IMPORTS = {
    'openai': False,
    'transformers': False,
    'llama_cpp': False,
    'gpt4all': False,
    'redis': False,
    'sentence_transformers': False,
}

try:
    import openai
    from openai import OpenAIError
    from packaging import version

    if version.parse(openai.__version__) >= version.parse("1.0.0"):
        from openai import OpenAI
    _OPTIONAL_IMPORTS['openai'] = True
except (ImportError, AttributeError) as e:
    logger.warning(f"⚠️ OpenAI not available: {e}")

try:
    import tiktoken

    _OPTIONAL_IMPORTS['tiktoken'] = True
except ImportError:
    _OPTIONAL_IMPORTS['tiktoken'] = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig

    _OPTIONAL_IMPORTS['transformers'] = True
except ImportError as e:
    logger.warning(f"⚠️ Transformers not available: {e}")

try:
    from llama_cpp import Llama

    _OPTIONAL_IMPORTS['llama_cpp'] = True
except ImportError as e:
    logger.warning(f"⚠️ llama_cpp not available: {e}")

try:
    from gpt4all import GPT4All

    _OPTIONAL_IMPORTS['gpt4all'] = True
except ImportError as e:
    logger.warning(f"⚠️ GPT4All not available: {e}")

try:
    import redis
    from redis.exceptions import RedisError

    _OPTIONAL_IMPORTS['redis'] = True
except ImportError as e:
    logger.warning(f"⚠️ Redis not available: {e}")

try:
    from sentence_transformers import SentenceTransformer

    _OPTIONAL_IMPORTS['sentence_transformers'] = True
except ImportError as e:
    logger.warning(f"⚠️ Sentence transformers not available: {e}")

# ---------------------------------------------------------------------
# Enhanced decorators and utilities (keep as is)
# ---------------------------------------------------------------------
def retry(times: int = 3, delay: float = RETRY_DELAY,
          exceptions: Tuple[Type[Exception], ...] = (Exception,),
          logger: Optional[logging.Logger] = None):
    """🔁 Enhanced retry decorator with configurable exceptions and logging."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            log = logger or logging.getLogger(func.__module__)

            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < times:
                        wait_time = delay * (2 ** (attempt - 1))
                        log.warning(
                            f"🔁 Retry {attempt}/{times} for {func.__name__}: {e}. "
                            f"Waiting {wait_time:.2f}s before retry."
                        )
                        time.sleep(wait_time)
                    else:
                        log.error(
                            f"❌ All {times} attempts failed for {func.__name__}: {e}",
                            exc_info=True
                        )
            raise last_exc if last_exc else Exception(f"Failed after {times} attempts")

        return wrapper

    return decorator


def async_retry(times: int = 3, delay: float = RETRY_DELAY,
                exceptions: Tuple[Type[Exception], ...] = (Exception,),
                logger: Optional[logging.Logger] = None):
    """🔁 Async retry decorator with configurable exceptions."""

    def decorator(coro_func):
        @functools.wraps(coro_func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            log = logger or logging.getLogger(coro_func.__module__)

            for attempt in range(1, times + 1):
                try:
                    return await coro_func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < times:
                        wait_time = delay * (2 ** (attempt - 1))
                        log.warning(
                            f"🔁 Async retry {attempt}/{times} for {coro_func.__name__}: {e}. "
                            f"Waiting {wait_time:.2f}s before retry."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        log.error(
                            f"❌ All {times} async attempts failed for {coro_func.__name__}: {e}",
                            exc_info=True
                        )
            raise last_exc if last_exc else Exception(f"Failed after {times} attempts")

        return wrapper

    return decorator


class Singleton(type):
    """🔒 Thread-safe Singleton metaclass for shared resources."""
    _instances = {}
    _lock = RLock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DaemonWorkerPool:
    """
    Minimal single-worker replacement for ThreadPoolExecutor whose
    worker is a raw daemon thread.

    Why this exists
    ---------------
    `concurrent.futures.ThreadPoolExecutor` registers every worker in
    `concurrent.futures.thread._threads_queues`. At interpreter exit,
    the module-level `atexit` handler `_python_exit` iterates that
    registry and calls `t.join()` on every entry. A worker stuck
    inside a C call that never returns — `llama_decode`, a CUDA sync,
    an mmap on a slow filesystem — therefore prevents the process
    from exiting, and the `daemon=True` flag does **not** help: it
    only affects the later daemon-thread kill phase, which runs
    *after* atexit.

    This pool has the same interface as ThreadPoolExecutor for the
    small subset the adapters use (`submit`, `shutdown`), returns
    ordinary `concurrent.futures.Future` objects so callers that
    rely on `asyncio.wrap_future` and `concurrent.futures.wait`
    continue to work unchanged, and runs its worker on a bare
    `threading.Thread(daemon=True)` that is **not** registered in
    `_threads_queues`. A stuck call therefore cannot block process
    exit; the OS reaps the daemon thread.

    The pool is not a general-purpose concurrency primitive: no
    dynamic resizing, no `max_workers > 1`, no shutdown-with-cancel-
    futures. Adapters that want a single serial worker (all current
    callers of `_executor` in this file) are exactly the intended use.

    Ordering guarantees
    -------------------
    `submit()` and `shutdown()` both enqueue under the same lock, so
    a task submitted before a shutdown is always queued *before* the
    stop sentinel. The worker drains in FIFO order and therefore
    either runs the task or observes the stop signal — never both
    out of order. A `submit()` after `shutdown()` raises `RuntimeError`
    like `ThreadPoolExecutor` does.

    Cancellation safety
    -------------------
    If the awaiting coroutine times out (`asyncio.wait_for`), it
    cancels the wrapping `asyncio.Future`, which forwards the cancel
    to the underlying `concurrent.futures.Future`. When the worker
    later reaches `set_result` / `set_exception`, the base class
    raises `concurrent.futures.InvalidStateError`. The worker loop
    swallows that specific exception — otherwise a single timeout
    would kill the worker thread and every subsequent request would
    hang forever waiting for a task that is never dequeued.
    """

    _STOP = object()

    def __init__(self, name: str = "daemon_worker") -> None:
        import queue as _queue  # local import; keep module imports unchanged
        self._queue: "_queue.Queue" = _queue.Queue()
        self._name = name
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._start_thread()

    # ── lifecycle ───────────────────────────────────────────────────────
    def _start_thread(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopped = False
            t = threading.Thread(
                target=self._run,
                name=self._name,
                daemon=True,
            )
            self._thread = t
            t.start()

    def _run(self) -> None:
        """
        Worker loop. Runs on the raw daemon thread.

        Catches BaseException around `fn()` so a KeyboardInterrupt or
        SystemExit raised inside a task is delivered to the awaiting
        future rather than killing the worker thread and leaving all
        subsequent submissions queued forever.
        """
        while True:
            item = self._queue.get()
            if item is self._STOP:
                return
            fn, fut = item
            # Required before set_result / set_exception on a
            # concurrent.futures.Future; also handles the case where
            # the caller already cancelled the future.
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                result = fn()
            except BaseException as exc:
                try:
                    fut.set_exception(exc)
                except concurrent.futures.InvalidStateError:
                    # The awaiting coroutine timed out and cancelled
                    # this future while fn() was still running. The
                    # caller already saw CancelledError; there is
                    # nothing more to deliver, and we must not let the
                    # exception escape the worker loop or the worker
                    # thread would die and all subsequent requests
                    # would hang.
                    pass
            else:
                try:
                    fut.set_result(result)
                except concurrent.futures.InvalidStateError:
                    # Same race on the success path.
                    pass

    # ── public API ──────────────────────────────────────────────────────
    def submit(self, fn, *args, **kwargs) -> concurrent.futures.Future:
        """Enqueue `fn(*args, **kwargs)` and return a Future for its result."""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        if args or kwargs:
            # functools is imported at module level.
            fn = functools.partial(fn, *args, **kwargs)
        with self._lock:
            if self._stopped:
                raise RuntimeError(
                    f"{self._name}: submit() after shutdown"
                )
            # Put under the lock so a concurrent shutdown() cannot slip
            # its sentinel in front of this item. FIFO ordering on the
            # queue then guarantees the task runs before the worker sees
            # the stop signal.
            self._queue.put((fn, fut))
        return fut

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Signal the worker to stop after draining the queue.

        Returns
        -------
        True
            The worker thread has exited (or had already exited).
        False
            `wait=True` was requested and the thread is still alive
            after `timeout` seconds — the worker is almost certainly
            blocked inside a C call. This is not an error: the thread
            is a daemon thread and will not block interpreter exit.

        `timeout=None` with `wait=True` means "wait forever" (matches
        ThreadPoolExecutor semantics); most callers should pass an
        explicit small timeout so adapter shutdown does not block on
        a stuck C call.
        """
        with self._lock:
            already_stopped = self._stopped
            self._stopped = True
            if not already_stopped:
                # Same lock as submit() → ordering is guaranteed even
                # under concurrent calls.
                self._queue.put(self._STOP)

        t = self._thread
        if t is None:
            return True
        if wait:
            t.join(timeout=timeout)
        return not t.is_alive()

    @property
    def thread(self) -> Optional[threading.Thread]:
        """Diagnostic accessor — the raw worker thread (never joined by us)."""
        return self._thread

# ---------------------------------------------------------------------
# Enhanced configuration classes (keep as is)
# ---------------------------------------------------------------------
@dataclass
class ModelConfig:
    """🤖 Configuration for a model provider with validation."""
    provider: str
    model_name: str
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = 256
    timeout: int = DEFAULT_TIMEOUT
    device: str = "auto"
    max_retries: int = 3
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    batch_size: int = 1
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    n_gpu_layers: Optional[int] = None
    n_threads: Optional[int] = None
    chat_template: Optional[str] = None  # e.g., "llama-3", "qwen", "gemma", "none"

    def __post_init__(self):
        """🔍 Validate configuration after initialization."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"Temperature must be between 0 and 2, got {self.temperature}")

        if self.max_tokens and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")


@dataclass
class PgVectorConfig:
    """🧬 pgvector‑specific configuration."""
    enabled: bool = True
    dimension: int = 384
    index_type: str = "ivfflat"
    lists: int = 100
    probes: int = 10
    operator: str = "vector_cosine_ops"

    def __post_init__(self):
        if self.dimension <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}")
        if self.lists <= 0:
            raise ValueError(f"lists must be positive, got {self.lists}")
        if self.probes <= 0:
            raise ValueError(f"probes must be positive, got {self.probes}")


@dataclass
class DatabaseConfig:
    """🗄️ Database configuration with validation."""
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    user: str = ""
    password: str = ""
    min_connections: int = 1
    max_connections: int = 10
    timeout: int = 10
    sslmode: str = "prefer"
    pgvector: PgVectorConfig = field(default_factory=PgVectorConfig)

    def __post_init__(self):
        """🔍 Validate database configuration."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"Invalid port number: {self.port}")

        if self.min_connections < 0:
            raise ValueError(f"min_connections must be non-negative, got {self.min_connections}")

        if self.max_connections <= 0:
            raise ValueError(f"max_connections must be positive, got {self.max_connections}")

        if self.min_connections > self.max_connections:
            raise ValueError(
                f"min_connections ({self.min_connections}) cannot exceed "
                f"max_connections ({self.max_connections})"
            )
        self.use_pgvector = self.pgvector.enabled


@dataclass
class RedisConfig:
    """🔴 Redis configuration with validation."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True

    def __post_init__(self):
        """🔍 Validate Redis configuration."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"Invalid port number: {self.port}")

        if self.db < 0 or self.db > 15:
            raise ValueError(f"Redis DB must be between 0 and 15, got {self.db}")


@dataclass
class ServiceConfig:
    """🚀 Main service configuration with comprehensive settings."""
    models: List[ModelConfig]
    primary_model: str
    agent_instruction: str = ""  # Instruction for agent execution
    database: Optional[DatabaseConfig] = None
    redis: Optional[RedisConfig] = None
    health_check_interval: int = HEALTH_CHECK_INTERVAL
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS
    request_timeout: int = REQUEST_TIMEOUT
    embedding_model: str = ""
    embedding_cache_dir: str = ""
    embedding_cache_size: int = 1000
    embedding_enable_download: bool = False
    embedding_timeout: int = 30
    embedding_device: str = ""
    embedding_similarity_threshold: float = 0.85
    download_dir: Optional[str] = None
    enable_monitoring: bool = True
    metrics_port: int = 9090
    auto_run_agent: bool = True  # Auto-run agent with instruction on startup
    max_agent_iterations: int = 10
    max_total_tokens: int = 2048  # Max iterations for agent execution
    agent_execution_interval: int = AGENT_EXECUTION_INTERVAL
    data_analysis_interval: int = DATA_ANALYSIS_INTERVAL  # Replaces threat_analysis_interval
    enable_concurrent_execution: bool = True
    require_real_backend: bool = True  # never silently swap to DummyAdapter

    def __post_init__(self):
        """🔍 Validate service configuration."""
        if not self.models:
            raise ValueError("At least one model must be configured")

        if self.health_check_interval <= 0:
            raise ValueError(f"health_check_interval must be positive, got {self.health_check_interval}")

        if self.max_concurrent_requests <= 0:
            raise ValueError(f"max_concurrent_requests must be positive, got {self.max_concurrent_requests}")

        if self.request_timeout <= 0:
            raise ValueError(f"request_timeout must be positive, got {self.request_timeout}")


# ---------------------------------------------------------------------
# Enhanced Base Adapter Interface (keep as is)
# ---------------------------------------------------------------------
class BaseLLMAdapter(ABC):
    """🤖 Abstract adapter interface for all LLM backends with enhanced error handling."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._initialized = False
        # Init guard must be an asyncio.Lock, NOT a threading.RLock.
        #
        # The old code held self._lock (an RLock) across
        # `await self._initialize_impl()`. That is unsafe: the RLock is
        # owned by the OS thread that runs the event loop, and while the
        # coroutine is suspended at the await the loop cannot run any
        # other coroutine that needs the same lock — the loop is
        # effectively deadlocked. asyncio.Lock() cooperates with the
        # event loop and is the correct primitive for a critical section
        # that contains an await.
        self._init_lock = asyncio.Lock()
        # Kept for backwards compatibility with subclasses / callers
        # that historically took `self._lock` for short, non-await
        # critical sections. Never hold this across an await.
        self._lock = RLock()
        self._model_loaded = False
        self._last_used = None
        self._shutting_down = False
        self._shutdown_flag = False

    async def initialize(self) -> bool:
        """⚡ Initialize the model with enhanced error handling and lazy loading."""
        if self._initialized:
            return True

        async with self._init_lock:
            if self._initialized:
                return True

            try:
                await self._initialize_impl()
                self._initialized = True
                self._model_loaded = True
                self._last_used = time.time()
                logger.info(f"✅ Model {self.config.provider} initialized successfully")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to initialize model {self.config.provider}: {e}", exc_info=True)
                self._initialized = False
                self._model_loaded = False
                raise

    @abstractmethod
    async def _initialize_impl(self) -> None:
        """⚙️ Implementation-specific initialization."""
        pass

    @abstractmethod
    async def generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            timeout: Optional[float] = None
    ) -> str:
        """📝 Generate text from prompt with enhanced error handling."""
        pass

    async def batch_generate(
            self,
            prompts: List[str],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            timeout: Optional[float] = None
    ) -> List[str]:
        """📦 Generate text for multiple prompts with parallel execution where supported."""
        results = []
        for prompt in prompts:
            try:
                result = await self.generate(prompt, max_tokens, temperature, timeout)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Batch generation failed for prompt: {e}")
                results.append(f"ERROR: {e}")
        return results

    async def chat(
            self,
            messages: List[Dict[str, str]],
            *,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            timeout: Optional[float] = None,
            no_think: Optional[bool] = None,
            est_prompt_tokens: int = 0,
            **kw,
    ):
        """
        Default: flatten to one string and forward to generate() so legacy
        adapters (OpenAI, HF, LlamaCpp, GPT4All, Dummy) keep working unchanged.

        LocalLLMCloudAdapter overrides this so messages reach the server
        untouched, letting the server apply its own chat template (Qwen ChatML
        for qwen3-8b) instead of receiving a Mistral [INST] block wrapped in
        Qwen ChatML.
        """
        parts = []
        for m in messages:
            role = (m.get("role") or "user").strip()
            content = m.get("content") or ""
            parts.append(f"### {role}\n{content}")
        prompt = "\n\n".join(parts)

        text = await self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        from amb_sentinel.llm.amb_llm_core import LLMResponse
        return LLMResponse(text=text, finish_reason="stop")

    def is_healthy(self) -> bool:
        """💚 Check if the model is healthy and ready for inference."""
        return self._initialized and self._model_loaded

    def get_metrics(self) -> Dict[str, Any]:
        """📊 Get model metrics for monitoring."""
        return {
            "initialized": self._initialized,
            "model_loaded": self._model_loaded,
            "last_used": self._last_used,
            "provider": self.config.provider,
            "model_name": self.config.model_name
        }

    async def attempt_context_recovery(self):
        """
        Wrapper that calls the model manager's recovery method.
        Called after successful tool executions.
        """
        ctx_mgr = get_model_manager()
        await ctx_mgr.attempt_context_recovery()

    def validate_tool_call(
            self,
            tool_name: str,
            tool_args: Dict[str, Any],
            tool_registry: 'ToolRegistry'
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a tool call before execution.
        Prevents hallucinated/invalid tool calls.

        Returns:
            (is_valid, error_message)
        """
        # Check if tool exists
        if not hasattr(tool_registry, 'tools'):
            return False, "Tool registry not available"

        if tool_name not in tool_registry.tools:
            return False, f"Tool '{tool_name}' not found in registry"

        tool = tool_registry.tools[tool_name]

        # Check required parameters
        if hasattr(tool, 'schema') and tool.schema:
            for param_name, param_spec in tool.schema.items():
                if param_spec.get('required', False) and param_name not in tool_args:
                    return False, f"Required parameter '{param_name}' missing"

        return True, None

    async def generate_with_adaptation(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            timeout: Optional[float] = None,
            allow_degradation: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate with real-time adaptation and fallback strategies.

        Returns:
            (generated_text, execution_metadata)
        """
        model_manager = get_model_manager()
        execution_meta = {
            "original_strategy": None,
            "strategy_degraded": False,
            "compression_applied": False,
            "attempt_number": 1,
            "max_attempts": 3
        }

        for attempt in range(1, execution_meta["max_attempts"] + 1):
            try:
                execution_meta["attempt_number"] = attempt

                # Check memory pressure before generation
                await model_manager.adapt_to_memory_pressure()

                # Calculate token budget
                token_budget = model_manager.calculate_token_budget(
                    prompt_text=prompt,
                    estimated_prompt_tokens=None,
                    max_generation_tokens=max_tokens if max_tokens is not None else (self.config.max_tokens if self.config.max_tokens is not None else 256)
                )

                # Handle token pressure - compress if needed
                prompt_to_use = prompt
                if not token_budget.is_feasible and allow_degradation:
                    logger.warning(f"⚠️ Attempt {attempt}: Token budget infeasible, compressing prompt")
                    physical = getattr(model_manager, '_actual_model_n_ctx', 0)
                    compress_base = physical if physical > 0 else token_budget.total_budget
                    target_tokens = int(compress_base * 0.6)
                    prompt_to_use = model_manager.adapt_prompt_to_context(prompt, target_tokens)
                    execution_meta["compression_applied"] = True
                    max_tokens = min(token_budget.recommended_max_gen, max_tokens or 256)
                    logger.info(
                        f"✂️ Compressed to {target_tokens} tokens "
                        f"(physical_base={compress_base}, soft_budget={token_budget.total_budget})"
                    )

                # Generate with adaptive timeout
                use_timeout = timeout or token_budget.recommended_timeout

                result = await self.generate(
                    prompt_to_use,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=use_timeout
                )

                if not result.startswith("ERROR"):
                    return result, execution_meta
                else:
                    # Error from adapter - try degradation
                    if attempt < execution_meta["max_attempts"] and allow_degradation:
                        logger.warning(f"⚠️ Generation error on attempt {attempt}, retrying with degradation")
                        await asyncio.sleep(1)
                        continue
                    else:
                        return result, execution_meta

            except asyncio.TimeoutError:
                if attempt < execution_meta["max_attempts"] and allow_degradation:
                    logger.warning(f"⚠️ Timeout on attempt {attempt}, reducing context and retrying")
                    model_manager.reduce_context_window()
                    await asyncio.sleep(2)
                    continue
                else:
                    return f"ERROR: Timeout after {attempt} attempts", execution_meta

            except Exception as e:
                if attempt < execution_meta["max_attempts"]:
                    logger.warning(f"⚠️ Exception on attempt {attempt}: {e}, retrying")
                    await asyncio.sleep(1)
                    continue
                else:
                    return f"ERROR: {str(e)[:200]}", execution_meta

        return "ERROR: Exhausted all adaptation attempts", execution_meta

    async def validate_and_execute_tool(
            self,
            tool_name: str,
            tool_args: Dict[str, Any],
            tool_registry: 'ToolRegistry',
            timeout: float = 30.0
    ) -> Tuple[str, bool]:
        """
        Safely validate and execute a tool call with error handling.

        Returns:
            (result, success)
        """
        # Validate tool exists
        if not hasattr(tool_registry, 'tools') or tool_name not in tool_registry.tools:
            logger.warning(f"⚠️ Tool '{tool_name}' not found in registry")
            return f"ERROR: Tool '{tool_name}' not available", False

        tool = tool_registry.tools[tool_name]

        # Validate parameters
        if hasattr(tool, 'schema') and tool.schema:
            for param_name, param_spec in tool.schema.items():
                if param_spec.get('required', False) and param_name not in tool_args:
                    logger.warning(f"⚠️ Required param '{param_name}' missing for tool '{tool_name}'")
                    return f"ERROR: Missing required parameter '{param_name}'", False

        # Execute with timeout
        try:
            tool_fn = getattr(tool, "func", tool)
            if inspect.iscoroutinefunction(tool_fn):
                result = await asyncio.wait_for(tool_fn(**tool_args), timeout=timeout)
            else:
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: tool_fn(**tool_args)),
                    timeout=timeout,
                )

            return str(result), True

        except asyncio.TimeoutError:
            logger.error(f"⏱️ Tool '{tool_name}' execution timed out after {timeout}s")
            return f"ERROR: Tool execution timed out", False
        except Exception as e:
            logger.error(f"❌ Tool '{tool_name}' failed: {e}")
            return f"ERROR: {str(e)[:200]}", False

    def set_shutdown_flag(self):
        """🚩 Set shutdown flag to interrupt ongoing operations."""
        self._shutting_down = True
        self._shutdown_flag = True

    def shutdown(self, timeout: float = 10.0) -> bool:
        """
        🛑 Gracefully shut down the adapter.
        - Sets shutdown flags to reject new requests.
        - Waits for pending operations (if any) up to `timeout` seconds.
        - Releases external resources.

        Returns:
            True if all pending operations completed, False if timeout occurred.
        """
        self._shutting_down = True
        self._shutdown_flag = True
        # Base adapter has no background threads – nothing to wait for.
        return True


# ---------------------------------------------------------------------
# Adapter Implementations (keep as is, but add logging fix)
# ---------------------------------------------------------------------
if _OPTIONAL_IMPORTS['openai']:
    class OpenAIAdapter(BaseLLMAdapter):
        """
        🤖 Production‑grade OpenAI adapter with adaptive context management,
        token budgeting, prompt compression, and graceful shutdown.

        Features:
        - Uses tiktoken for accurate token counting (fallback to word estimate)
        - Integrates with ModelManager for token budgets, timeouts, and memory adaptation
        - Thread‑safe execution with pending future tracking
        - Graceful shutdown with timeout for pending API calls
        - Comprehensive logging and error handling
        """

        def __init__(self, config: ModelConfig):
            super().__init__(config)
            self._client = None
            self._use_new_api = False
            self._encoding = None
            # Single-worker daemon-thread pool. Not a ThreadPoolExecutor:
            # OpenAI HTTP calls can hang on a slow or unreachable socket,
            # and a ThreadPoolExecutor worker stuck inside `requests` /
            # `httpx` would be registered in
            # `concurrent.futures.thread._threads_queues` and joined by the
            # atexit handler at interpreter exit. See DaemonWorkerPool for
            # the full rationale.
            self._executor = DaemonWorkerPool(
                name=f"openai_{config.model_name.replace('/', '_')}"
            )
            self._pending_futures = set()
            self._pending_lock = threading.RLock()
            self.context_manager = get_model_manager()  # singleton

            # Store initial values for potential recovery
            self._initial_context = None  # Not applicable for OpenAI (API fixed)
            self._initial_max_tokens = self.config.max_tokens

            # Optional: track tokens/sec from past calls for better timeout estimation
            self.measured_tokens_per_sec = None

        # ------------------------------------------------------------------
        # Initialization
        # ------------------------------------------------------------------
        async def _initialize_impl(self) -> None:
            """
            ⚡ Initialize OpenAI client and tokenizer.
            Logs all steps for transparency.
            """
            logger.info(f"🔧 Initializing OpenAI adapter: {self.config.provider}")

            try:
                # Determine API version and create client
                if hasattr(openai, 'OpenAI'):
                    self._client = openai.OpenAI(
                        api_key=self.config.api_key,
                        base_url=self.config.base_url,
                        timeout=self.config.timeout
                    )
                    self._use_new_api = True
                    logger.info("✅ Using OpenAI API v1.0+")
                else:
                    openai.api_key = self.config.api_key
                    if self.config.base_url:
                        openai.api_base = self.config.base_url
                    self._use_new_api = False
                    logger.info("✅ Using legacy OpenAI API")

                # Set up tokenizer for the model (if tiktoken available)
                if _OPTIONAL_IMPORTS.get('tiktoken'):
                    try:
                        # Map model to encoding; fallback to cl100k_base for recent models
                        encoding_name = tiktoken.encoding_for_model(self.config.model_name).name
                        self._encoding = tiktoken.get_encoding(encoding_name)
                        logger.info(f"✅ tiktoken encoder ready for {self.config.model_name}")
                    except Exception:
                        # Fallback to a safe default
                        self._encoding = tiktoken.get_encoding("cl100k_base")
                        logger.info("ℹ️ tiktoken fallback to cl100k_base")
                else:
                    logger.info("ℹ️ tiktoken not available, using word‑based token estimation")

                logger.info(f"✅ OpenAI adapter initialized (new_api={self._use_new_api})")

                OPENAI_CTX_MAP = {
                    "gpt-4o": 128_000,
                    "gpt-4o-mini": 128_000,
                    "gpt-4-turbo": 128_000,
                    "gpt-4-turbo-preview": 128_000,
                    "gpt-4-32k": 32_768,
                    "gpt-4": 8_192,
                    "gpt-3.5-turbo-16k": 16_384,
                    "gpt-3.5-turbo": 16_384,
                    "gpt-3.5-turbo-instruct": 4_096,
                }
                model_lower = self.config.model_name.lower()
                openai_ctx = None
                for key, ctx in OPENAI_CTX_MAP.items():
                    if key in model_lower:
                        openai_ctx = ctx
                        break

                if openai_ctx is None:
                    # Unknown model: fall back to config value or a safe 8192
                    openai_ctx = self.config.context_length or 8_192
                    logger.info(
                        f"ℹ️ OpenAI model '{self.config.model_name}' not in known ctx map, "
                        f"using {openai_ctx} tokens"
                    )
                else:
                    logger.info(
                        f"📌 OpenAI '{self.config.model_name}' context window: {openai_ctx} tokens"
                    )

                self.context_manager.set_actual_n_ctx(openai_ctx)

            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI adapter: {e}")
                raise

        # ------------------------------------------------------------------
        # Context recovery (delegates to ModelManager)
        # ------------------------------------------------------------------
        def can_recover_context(self, current_pressure: float) -> bool:
            """
            Check if context recovery is possible.
            For OpenAI, this simply checks memory pressure via ModelManager.
            """
            try:
                mem = psutil.virtual_memory()
                # Recover if memory pressure < 60% (recovery is just a hint for future calls)
                return mem.percent < 60
            except Exception:
                return False

        async def attempt_context_recovery(self):
            """
            Safely attempt to recover context window via ModelManager.
            This is called after successful tool executions to gradually increase
            the context window if memory permits.
            """
            try:
                await self.context_manager.attempt_context_recovery()
            except Exception as e:
                logger.warning(f"Context recovery attempt failed: {e}")

        # ------------------------------------------------------------------
        # Token counting
        # ------------------------------------------------------------------
        def _count_tokens(self, text: str) -> int:
            """
            Count tokens accurately using tiktoken if available,
            otherwise fall back to word‑based estimation.
            """
            if self._encoding:
                try:
                    return len(self._encoding.encode(text))
                except Exception:
                    pass
            # Fallback: ~1.3 tokens per word
            return max(1, int(len(text.split()) * 1.3))

        @property
        def supports_grammar(self):
            return False

        # ------------------------------------------------------------------
        # Generation (with full adaptation)
        # ------------------------------------------------------------------
        @async_retry(times=3, delay=1.0, exceptions=(openai.OpenAIError,))
        async def generate(
                self,
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                timeout: Optional[float] = None
        ) -> str:
            """
            📝 Generate text using OpenAI API with adaptive token management.
            Integrates with ModelManager for token budgets, prompt compression,
            and system‑aware timeouts.
            """
            # Shutdown checks
            if self._shutting_down or self._shutdown_flag:
                logger.warning("⚠️ Generation cancelled: adapter is shutting down")
                return "ERROR: Adapter shutting down"

            # Memory pressure check
            try:
                mem = psutil.virtual_memory()
                if mem.available < 500 * 1024 * 1024:  # less than 500 MB free
                    logger.error("🚨 Critical low memory – aborting generation")
                    return "ERROR: System memory too low to generate safely"
            except ImportError:
                pass  # psutil not installed, skip check

            await self.context_manager.adapt_to_memory_pressure()

            # Ensure initialized
            if not self._initialized:
                try:
                    await self.initialize()
                except Exception as e:
                    logger.error(f"❌ Model initialization failed: {e}")
                    return f"ERROR: Model initialization failed: {str(e)[:100]}"

            # ------------------------------------------------------------------
            # 1. TOKEN BUDGET CALCULATION & PROMPT ADAPTATION
            # ------------------------------------------------------------------
            max_tokens = max_tokens if max_tokens is not None else (self.config.max_tokens if self.config.max_tokens is not None else 256)
            temperature = temperature if temperature is not None else (self.config.temperature if self.config.temperature is not None else 0.7)
            temperature = max(1e-8, min(temperature, 2.0))

            # Get token budget from context manager
            token_budget = None
            prompt_to_use = prompt
            compressed_applied = False

            try:
                # Estimate prompt tokens
                estimated_tokens = self._count_tokens(prompt)

                token_budget = self.context_manager.calculate_token_budget(
                    prompt_text=prompt,
                    estimated_prompt_tokens=estimated_tokens,
                    max_generation_tokens=max_tokens,
                    tokens_per_sec=self.measured_tokens_per_sec  # may be None
                )

                # Log any warnings
                if token_budget.warnings:
                    for warning in token_budget.warnings:
                        logger.warning(f"⚠️ {warning}")

                # Use recommended generation tokens
                max_tokens = token_budget.recommended_max_gen

                # If the prompt itself doesn't fit, compress it
                if not token_budget.is_feasible:
                    logger.warning("⚠️ Prompt exceeds context – compressing...")
                    physical = getattr(self.context_manager, '_actual_model_n_ctx', 0)
                    compress_base = physical if physical > 0 else token_budget.total_budget
                    target_tokens = int(compress_base * 0.6)
                    logger.info(f"✂️ OpenAI compress target: {target_tokens} tokens (base={compress_base})")

                    prompt_to_use = self.context_manager.adapt_prompt_to_context(
                        prompt,
                        target_tokens=target_tokens
                    )
                    compressed_applied = True

                    # Re‑estimate tokens after compression
                    estimated_tokens = self._count_tokens(prompt_to_use)
                    token_budget = self.context_manager.calculate_token_budget(
                        prompt_text=prompt_to_use,
                        estimated_prompt_tokens=estimated_tokens,
                        max_generation_tokens=max_tokens,
                        tokens_per_sec=self.measured_tokens_per_sec
                    )
                    max_tokens = token_budget.recommended_max_gen

                logger.info(
                    f"Token budget: prompt={estimated_tokens}, "
                    f"max_gen={max_tokens}, compressed={compressed_applied}"
                )

            except Exception as e:
                logger.warning(f"⚠️ Token budget calculation failed: {e}, using conservative values")
                max_tokens = min(max_tokens, 256)
                estimated_tokens = self._count_tokens(prompt)

            # ------------------------------------------------------------------
            # 2. TIMEOUT MANAGEMENT
            # ------------------------------------------------------------------
            use_timeout = token_budget.recommended_timeout if token_budget else (
                        timeout or self.config.timeout or 120.0)

            # ------------------------------------------------------------------
            # 3. EXECUTE THE API CALL IN A THREAD
            # ------------------------------------------------------------------
            def _call():
                """Synchronous OpenAI call."""
                try:
                    if self._shutting_down or self._shutdown_flag:
                        logger.warning("⚠️ OpenAI call interrupted: adapter shutting down")
                        return "ERROR: Generation cancelled"

                    if self._use_new_api:
                        response = self._client.chat.completions.create(
                            model=self.config.model_name,
                            messages=[{"role": "user", "content": prompt_to_use}],
                            temperature=temperature,
                            max_tokens=max(1, int(max_tokens)),
                            timeout=use_timeout
                        )
                        return response.choices[0].message.content
                    else:
                        response = openai.ChatCompletion.create(
                            model=self.config.model_name,
                            messages=[{"role": "user", "content": prompt_to_use}],
                            temperature=temperature,
                            max_tokens=max(1, int(max_tokens)),
                            timeout=use_timeout
                        )
                        return response["choices"][0]["message"]["content"]
                except Exception as e:
                    # If shutdown is in progress, treat as cancellation
                    if self._shutting_down or self._shutdown_flag:
                        logger.warning(f"⚠️ OpenAI call interrupted: {str(e)[:100]}")
                        return "ERROR: Generation cancelled"
                    raise

            future = self._executor.submit(_call)
            with self._pending_lock:
                self._pending_futures.add(future)
                future.add_done_callback(lambda f: self._pending_futures.discard(f))

            try:
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=use_timeout + 5.0  # extra buffer for thread startup
                )
                self._last_used = time.time()
                return result
            except asyncio.TimeoutError:
                logger.error("⏱️ OpenAI request timed out")
                # Do NOT cancel the future – it's already running
                return "ERROR: Generation timed out"
            except asyncio.CancelledError:
                logger.warning("⚠️ OpenAI generation cancelled")
                return "ERROR: Generation cancelled"
            except Exception as e:
                logger.error(f"❌ OpenAI generation failed: {str(e)[:200]}")
                return f"ERROR: {str(e)[:200]}"
            finally:
                with self._pending_lock:
                    self._pending_futures.discard(future)

        # ------------------------------------------------------------------
        # Shutdown
        # ------------------------------------------------------------------
        def shutdown(self, timeout: float = 10.0) -> bool:
            """
            🛑 Wait for pending OpenAI calls and release executor.
            Returns True if all pending calls completed within timeout.
            """
            logger.info("🛑 OpenAIAdapter shutdown initiated")
            super().shutdown(timeout)

            # Wait for already‑submitted futures
            with self._pending_lock:
                futures = list(self._pending_futures)

            not_done = set()
            if futures:
                try:
                    done, not_done = concurrent.futures.wait(
                        futures,
                        timeout=timeout,
                        return_when=concurrent.futures.ALL_COMPLETED
                    )
                    if not_done:
                        logger.warning(f"⚠️ {len(not_done)} OpenAI call(s) still running after timeout")
                    else:
                        logger.info("✅ All pending OpenAI calls completed")
                except Exception as e:
                    logger.error(f"❌ Error waiting for futures: {e}")

            # Shutdown executor and wait
            # Signal the worker to stop. Do NOT wait indefinitely: an
            # OpenAI call stuck on a slow socket cannot be interrupted
            # from Python, and blocking here would just delay the caller.
            # The worker is a raw daemon thread, so interpreter exit will
            # not join it either.
            logger.info("🛑 Signalling OpenAIAdapter worker to stop ...")
            try:
                exited = self._executor.shutdown(wait=True, timeout=2.0)
                if exited:
                    logger.info("✅ OpenAIAdapter worker thread exited cleanly")
                else:
                    logger.warning(
                        "⚠️ OpenAIAdapter worker thread still inside an "
                        "HTTP call after 2s. It is a daemon thread and "
                        "will not block process exit; leaving it to the OS."
                    )
            except Exception as e:
                logger.error(f"❌ Error signalling OpenAIAdapter worker: {e}")

            logger.info("✅ OpenAIAdapter shutdown complete")
            return len(not_done) == 0


        # ------------------------------------------------------------------
        # Health and metrics
        # ------------------------------------------------------------------
        def is_healthy(self) -> bool:
            """💚 Check if the adapter is healthy."""
            return self._initialized and self._client is not None

        def get_metrics(self) -> Dict[str, Any]:
            """📊 Get adapter metrics."""
            base = super().get_metrics()
            base.update({
                "api_version": "new" if self._use_new_api else "legacy",
                "tokenizer_available": self._encoding is not None,
                "tokens_per_sec": self.measured_tokens_per_sec,  # may be None
            })
            return base

if _OPTIONAL_IMPORTS['transformers']:
    class HFAdapter(BaseLLMAdapter):
        """
        🤖 Production‑grade HuggingFace Transformers adapter with:
        - Automatic device selection (CUDA, MPS, CPU) with fallback
        - Adaptive context management via ModelManager
        - Token‑aware prompt compression and overflow handling
        - Real‑time memory monitoring and graceful degradation
        - Thread‑safe execution with pending future tracking
        - Graceful shutdown with timeout
        - Comprehensive logging and error handling
        """

        def __init__(self, config: ModelConfig):
            super().__init__(config)
            self._tokenizer = None
            self._model = None
            self._device = self._resolve_device()
            self._generation_config = None
            # Single-worker daemon-thread pool. Not a ThreadPoolExecutor:
            # that class registers its workers in
            # `concurrent.futures.thread._threads_queues`, whose atexit
            # handler will join them on interpreter exit — a worker stuck
            # inside HF's `from_pretrained` or a `.generate()` call cannot
            # be interrupted from Python and would hang shutdown.
            self._executor = DaemonWorkerPool(
                name=f"hf_{config.model_name.replace('/', '_')}"
            )
            self._pending_futures = set()
            self._pending_lock = threading.RLock()
            self.context_manager = get_model_manager()  # singleton

            # Store initial values for recovery
            self._initial_context = None  # will be set after loading
            self._initial_max_tokens = self.config.max_tokens

            # Optional: measure tokens/sec (can be estimated later)
            self.measured_tokens_per_sec = None

        # ------------------------------------------------------------------
        # Device and resource detection (mirroring LlamaCppAdapter)
        # ------------------------------------------------------------------
        def _resolve_device(self) -> str:
            """
            ⚙️ Safely resolve compute device with robust GPU detection and CPU fallback.
            Returns a string like 'cuda', 'mps', or 'cpu'.
            """
            device = self.config.device.lower()

            if device == "auto" or device == "cuda":
                try:
                    if not torch.cuda.is_available():
                        logger.info("ℹ️ CUDA not available, falling back to CPU")
                        return "cpu"

                    # Test if CUDA actually works
                    test_tensor = torch.tensor([1.0]).cuda()
                    torch.cuda.synchronize()

                    # Check for sufficient GPU memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    if gpu_memory < 2 * 1024 ** 3:  # Less than 2GB
                        logger.warning(f"⚠️ GPU memory low ({gpu_memory // 1024 ** 3}GB), using CPU")
                        return "cpu"

                    logger.info(f"✅ CUDA available with {gpu_memory // 1024 ** 3}GB memory")
                    return "cuda"

                except (RuntimeError, AttributeError) as e:
                    logger.warning(f"⚠️ CUDA initialization failed: {str(e)[:100]}, falling back to CPU")
                    return "cpu"

            # MPS (Apple Silicon)
            if device in ["auto", "mps"] and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    test_tensor = torch.tensor([1.0]).to("mps")
                    logger.info("✅ MPS (Apple Silicon) available")
                    return "mps"
                except Exception:
                    pass

            return "cpu"

        def _estimate_model_size(self) -> int:
            """📏 Estimate model size in bytes for memory planning."""
            try:
                model_path = Path(self.config.model_name)
                if model_path.exists():
                    return model_path.stat().st_size

                # Config/env-driven search roots. Set AMB_MODELS_DIR in
                # .env (or the environment) to override the default.
                # Never hardcode a specific machine's home directory.
                search_roots = [
                    os.environ.get("AMB_MODELS_DIR", ""),
                    os.environ.get("AMB_MODEL_DIR", ""),
                    str(Path.cwd() / "models"),
                    str(Path.cwd() / "model"),
                    str(Path.home() / "models"),
                ]
                for root in search_roots:
                    if not root:
                        continue
                    candidate = Path(root) / model_path.name
                    if candidate.exists():
                        return candidate.stat().st_size
            except Exception:
                pass

            # Default estimate for 3B Q4_K_M model
            return 2 * 1024 ** 3  # 2GB default

        def _resolve_threads(self) -> int:
            """
            ⚙️ Resolve optimal thread count for CPU inference.
            Not critical for HF, but provided for consistency.
            """
            try:
                cpu_count = os.cpu_count() or 4
                # Use 75% of available cores for CPU, 50% for GPU to avoid contention
                if self._device == "cpu":
                    threads = max(2, int(cpu_count * 0.75))
                else:
                    threads = max(2, int(cpu_count * 0.5))
                logger.info(f"✅ Resolved {threads} threads for {self._device}")
                return threads
            except Exception:
                return 4

        # ------------------------------------------------------------------
        # Context recovery (delegates to ModelManager)
        # ------------------------------------------------------------------
        def can_recover_context(self, current_pressure: float) -> bool:
            """
            Check if context recovery is possible.
            For HF, recovery is simply a hint to increase context window if memory permits.
            """
            try:
                mem = psutil.virtual_memory()
                return mem.percent < 60
            except Exception:
                return False

        async def attempt_context_recovery(self):
            """
            Safely attempt to recover context window via ModelManager.
            Called after successful tool executions.
            """
            try:
                await self.context_manager.attempt_context_recovery()
            except Exception as e:
                logger.warning(f"Context recovery attempt failed: {e}")

        # ------------------------------------------------------------------
        # Initialization
        # ------------------------------------------------------------------
        async def _initialize_impl(self) -> None:
            """
            ⚡ Load tokenizer and model with adaptive device placement.
            Includes comprehensive logging and error handling.
            """
            logger.info(f"🔧 Initializing HFAdapter on device: {self._device}")

            # Memory monitoring
            try:
                import psutil
                self._system_memory = psutil.virtual_memory()
                self._has_psutil = True
                logger.info("✅ psutil available for memory monitoring")
            except ImportError:
                self._has_psutil = False
                logger.warning("⚠️ psutil not installed – memory monitoring limited")

            # Model size estimate (for logging)
            model_size_gb = self._estimate_model_size()
            logger.info(f"📦 Estimated model size: {model_size_gb:.2f}GB")

            try:
                # Load tokenizer
                logger.info(f"🔤 Loading tokenizer for {self.config.model_name}...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    use_fast=True,
                    trust_remote_code=self.config.trust_remote_code
                )
                if not self._tokenizer.pad_token:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                logger.info("✅ Tokenizer loaded")

                # Determine dtype and device map
                if self._device == "cuda":
                    dtype = torch.float16
                    device_map = "auto" if torch.cuda.is_available() else None
                elif self._device == "mps":
                    dtype = torch.float32
                    device_map = None
                else:
                    dtype = torch.float32
                    device_map = None

                # Load model
                logger.info(f"🔧 Loading model (this may take a while)...")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=self.config.trust_remote_code,
                    low_cpu_mem_usage=self.config.low_cpu_mem_usage
                )

                # Manual device placement if not using device_map
                if device_map is None:
                    try:
                        self._model = self._model.to(self._device)
                        self._model.eval()
                        torch.set_grad_enabled(False)
                        logger.info(f"✅ Model moved to {self._device}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to move model to {self._device}: {e}, forcing CPU")
                        self._model = self._model.to("cpu")
                        self._device = "cpu"

                # Generation config
                self._generation_config = GenerationConfig.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=self.config.trust_remote_code
                )
                logger.info("✅ Generation config loaded")

                # Negotiate context window with ModelManager
                logger.info("🔄 Starting adaptive context negotiation...")
                negotiated_ctx = self.context_manager.negotiate_context_window(
                    model_name=self.config.model_name,
                    config_requested=self.config.context_length
                )
                self._initial_context = negotiated_ctx

                try:
                    model_max_pos = getattr(self._model.config, 'max_position_embeddings', None)
                    if model_max_pos and model_max_pos > 0:
                        # Respect the negotiated ceiling — don't exceed what
                        # negotiate_context_window decided is safe.
                        actual_ctx = min(model_max_pos, negotiated_ctx)
                        logger.info(
                            f"📌 HF model max_position_embeddings={model_max_pos} "
                            f"→ capped to {actual_ctx} by negotiated ceiling"
                        )
                    else:
                        actual_ctx = negotiated_ctx
                        logger.info(f"ℹ️ max_position_embeddings not found, using negotiated {actual_ctx}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not read max_position_embeddings: {e}")
                    actual_ctx = negotiated_ctx

                self.context_manager.set_actual_n_ctx(actual_ctx)
                logger.info(f"✅ HFAdapter context locked at {actual_ctx} tokens")

                logger.info(f"✅ HFAdapter initialized on {self._device}")

            except Exception as e:
                logger.error(f"❌ Failed to initialize HFAdapter: {e}", exc_info=True)
                raise

        # ------------------------------------------------------------------
        # Token counting
        # ------------------------------------------------------------------
        def _count_tokens(self, text: str) -> int:
            """
            Count tokens accurately using the loaded tokenizer.
            """
            if self._tokenizer:
                try:
                    return len(self._tokenizer.encode(text))
                except Exception:
                    pass
            # Fallback
            return max(1, int(len(text.split()) * 1.3))

        @property
        def supports_grammar(self):
            return False

        # ------------------------------------------------------------------
        # Generation
        # ------------------------------------------------------------------
        @async_retry(times=2, delay=1.0)
        async def generate(
                self,
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                timeout: Optional[float] = None
        ) -> str:
            """
            📝 Generate text with adaptive context management, token budgeting,
            and prompt compression.
            """
            # Shutdown checks
            if self._shutting_down or self._shutdown_flag:
                logger.warning("⚠️ Generation cancelled: adapter is shutting down")
                return "ERROR: Adapter shutting down"

            # Memory pressure check
            try:
                mem = psutil.virtual_memory()
                if mem.available < 500 * 1024 * 1024:  # less than 500 MB free
                    logger.error("🚨 Critical low memory – aborting generation")
                    return "ERROR: System memory too low to generate safely"
            except ImportError:
                pass

            await self.context_manager.adapt_to_memory_pressure()

            # Ensure initialized
            if not self._initialized:
                try:
                    await self.initialize()
                except Exception as e:
                    logger.error(f"❌ Model initialization failed: {e}")
                    return f"ERROR: Model initialization failed: {str(e)[:100]}"

            # ------------------------------------------------------------------
            # 1. TOKEN BUDGET CALCULATION & PROMPT ADAPTATION
            # ------------------------------------------------------------------
            max_tokens = int(max_tokens if max_tokens is not None else (self.config.max_tokens if self.config.max_tokens is not None else 256))
            temperature = float(temperature if temperature is not None else (self.config.temperature if self.config.temperature is not None else 0.7))
            temperature = max(1e-8, min(temperature, 2.0))

            # Count prompt tokens accurately
            try:
                inputs = self._tokenizer(prompt, return_tensors="pt", truncation=False)
                prompt_tokens = inputs.input_ids.shape[1]
            except Exception as e:
                logger.warning(f"⚠️ Tokenization failed: {e}, using word estimate")
                prompt_tokens = int(len(prompt.split()) * 1.3)

            token_budget = self.context_manager.calculate_token_budget(
                prompt_text=prompt,
                estimated_prompt_tokens=prompt_tokens,
                max_generation_tokens=max_tokens,
                tokens_per_sec=self.measured_tokens_per_sec
            )

            # Log warnings
            for w in token_budget.warnings:
                logger.warning(f"⚠️ {w}")

            # Adjust max_tokens based on budget
            max_tokens = token_budget.recommended_max_gen

            # If the prompt itself doesn't fit, compress it
            prompt_to_use = prompt
            compressed_applied = False
            if not token_budget.is_feasible:
                logger.warning("⚠️ Prompt exceeds context – compressing...")
                physical = getattr(self.context_manager, '_actual_model_n_ctx', 0)
                compress_base = physical if physical > 0 else token_budget.total_budget
                target_tokens = int(compress_base * 0.6)
                logger.info(f"✂️ HF compress target: {target_tokens} tokens (base={compress_base})")

                prompt_to_use = self.context_manager.adapt_prompt_to_context(
                    prompt,
                    target_tokens=target_tokens
                )
                compressed_applied = True

                # Re‑evaluate budget
                try:
                    new_inputs = self._tokenizer(prompt_to_use, return_tensors="pt", truncation=False)
                    new_prompt_tokens = new_inputs.input_ids.shape[1]
                except Exception:
                    new_prompt_tokens = int(len(prompt_to_use.split()) * 1.3)

                token_budget = self.context_manager.calculate_token_budget(
                    prompt_text=prompt_to_use,
                    estimated_prompt_tokens=new_prompt_tokens,
                    max_generation_tokens=max_tokens,
                    tokens_per_sec=self.measured_tokens_per_sec
                )
                max_tokens = token_budget.recommended_max_gen

            logger.info(
                f"Generation config: max_tokens={max_tokens}, "
                f"temperature={temperature:.2f}, compressed={compressed_applied}"
            )

            # ------------------------------------------------------------------
            # 2. TIMEOUT MANAGEMENT
            # ------------------------------------------------------------------
            use_timeout = token_budget.recommended_timeout if token_budget else (
                    timeout or self.config.timeout or 120.0)

            # ------------------------------------------------------------------
            # 3. EXECUTE GENERATION IN THREAD
            # ------------------------------------------------------------------
            def _call():
                """Synchronous generation call."""
                try:
                    if self._shutting_down or self._shutdown_flag:
                        return "ERROR: Generation cancelled during shutdown"

                    # Prepare inputs (move to same device as model)
                    model_device = next(self._model.parameters()).device
                    inputs = self._tokenizer(prompt_to_use, return_tensors="pt").to(model_device)

                    with torch.no_grad():
                        outputs = self._model.generate(
                            **inputs,
                            max_new_tokens=max(1, max_tokens),
                            temperature=temperature,
                            do_sample=temperature > 0.0,
                            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                            eos_token_id=self._tokenizer.eos_token_id,
                        )

                    # Decode only the new tokens
                    input_length = inputs.input_ids.shape[1]
                    generated_tokens = outputs[0][input_length:]
                    return self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                except Exception as e:
                    if self._shutting_down or self._shutdown_flag:
                        logger.warning(f"⚠️ Generation interrupted: {str(e)[:100]}")
                        return "ERROR: Generation cancelled"
                    raise

            future = self._executor.submit(_call)
            with self._pending_lock:
                self._pending_futures.add(future)
                future.add_done_callback(lambda f: self._pending_futures.discard(f))

            try:
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=use_timeout
                )
                self._last_used = time.time()
                return result
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Generation timed out after {use_timeout:.1f}s")
                return "ERROR: Generation timed out"
            except asyncio.CancelledError:
                logger.warning("⚠️ Generation cancelled")
                return "ERROR: Generation cancelled"
            except Exception as e:
                logger.error(f"❌ Generation failed: {str(e)[:200]}")
                return f"ERROR: {str(e)[:200]}"
            finally:
                with self._pending_lock:
                    self._pending_futures.discard(future)

        # ------------------------------------------------------------------
        # Shutdown
        # ------------------------------------------------------------------
        def shutdown(self, timeout: float = 10.0) -> bool:
            """
            🛑 Graceful shutdown: wait for pending generations and free GPU memory.
            """
            logger.info("🛑 HFAdapter shutdown initiated")
            super().shutdown(timeout)

            # Wait for pending futures
            with self._pending_lock:
                futures = list(self._pending_futures)

            not_done = set()
            if futures:
                try:
                    done, not_done = concurrent.futures.wait(
                        futures,
                        timeout=timeout,
                        return_when=concurrent.futures.ALL_COMPLETED
                    )
                    if not_done:
                        logger.warning(f"⚠️ {len(not_done)} generation(s) still running after timeout")
                    else:
                        logger.info("✅ All pending generations completed")
                except Exception as e:
                    logger.error(f"❌ Error waiting for futures: {e}")

            # Shutdown executor
            # Signal the worker to stop. Do NOT wait indefinitely: if a
            # generation is stuck inside a C call (torch CUDA sync,
            # safetensors mmap) it cannot be interrupted from Python,
            # and blocking here would just delay the caller. The worker
            # is a raw daemon thread, so interpreter exit will not join
            # it either.
            logger.info("🛑 Signalling HFAdapter worker to stop ...")
            try:
                exited = self._executor.shutdown(wait=True, timeout=2.0)
                if exited:
                    logger.info("✅ HFAdapter worker thread exited cleanly")
                else:
                    logger.warning(
                        "⚠️ HFAdapter worker thread still inside a C call "
                        "after 2s. It is a daemon thread and will not "
                        "block process exit; leaving it to the OS."
                    )
            except Exception as e:
                logger.error(f"❌ Error signalling HFAdapter worker: {e}")

            # Free GPU memory
            if hasattr(self, '_model') and self._model is not None:
                try:
                    self._model = self._model.to("cpu")
                    del self._model
                    self._model = None
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("✅ Model moved to CPU and cache cleared")
                except Exception as e:
                    logger.error(f"❌ Error during model cleanup: {e}")

            logger.info("✅ HFAdapter shutdown complete")
            return len(not_done) == 0

        # ------------------------------------------------------------------
        # Health and metrics
        # ------------------------------------------------------------------
        def is_healthy(self) -> bool:
            """💚 Check if model is loaded and device accessible."""
            return self._initialized and self._model is not None

        def get_metrics(self) -> Dict[str, Any]:
            """📊 Get adapter metrics."""
            base = super().get_metrics()
            base.update({
                "device": self._device,
                "tokenizer_loaded": self._tokenizer is not None,
                "tokens_per_sec": self.measured_tokens_per_sec,
            })
            return base

if _OPTIONAL_IMPORTS['llama_cpp']:
    from llama_cpp import Llama, LlamaGrammar


    class LlamaCppAdapter(BaseLLMAdapter):
        """
        🤖 Production-ready llama.cpp adapter with robust GPU/CPU detection and fallback.
        Now with automatic chat template detection for instruction‑tuned models
        (Llama‑3, Qwen, Gemma, etc.) and a startup benchmark to measure tokens/sec.
        """

        def __init__(self, config: ModelConfig):
            super().__init__(config)
            self._llm = None
            self.measured_tokens_per_sec = None
            self._device = self._resolve_device()
            self._n_gpu_layers = self._resolve_gpu_layers()
            self._n_threads = self._resolve_threads()

            # New: chat format handling
            self.chat_format = config.chat_template  # may be None
            self.use_chat = False  # set after detection/loading

            self._pending_futures = set()
            self._pending_lock = RLock()

            # Single-worker pool driven by a raw daemon thread. Unlike
            # ThreadPoolExecutor, the worker is not registered in
            # `concurrent.futures.thread._threads_queues`, so the atexit
            # handler `_python_exit` will not try to join it. A stuck
            # llama_decode therefore cannot prevent interpreter exit.
            self._executor = DaemonWorkerPool(name=f"llama_{Path(config.model_name).stem}")

            # For Qwen2.5 LLM
            if "qwen2.5" in config.model_name.lower() or "qwen" in config.model_name.lower():
                self._is_qwen = True
                self._context_shift = 512
            else:
                self._is_qwen = False
                self._context_shift = 0

            # Size cache for adaptive timeouts
            self._size_cache = {}
            self._size_cache_timestamp = {}
            self._size_cache_max_age = 300

        # ------------------------------------------------------------------
        # Device and resource detection (unchanged except logging)
        # ------------------------------------------------------------------
        def _resolve_device(self) -> str:
            """⚙️ Safely resolve compute device with robust detection and fallback."""
            device = self.config.device.lower()

            if device in ["auto", "cuda"]:
                try:
                    # Check if CUDA is theoretically available
                    if not torch.cuda.is_available():
                        logger.info("ℹ️ CUDA not available, falling back to CPU")
                        return "cpu"

                    # Test actual CUDA functionality
                    test_tensor = torch.tensor([1.0]).cuda()
                    torch.cuda.synchronize()

                    # Check GPU memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    model_size = self._estimate_model_size()

                    # Only use GPU if we have sufficient memory
                    if gpu_memory > (model_size * 1.5):  # 1.5x model size as buffer
                        logger.info(f"✅ CUDA available with {gpu_memory // 1024 ** 3}GB memory")
                        return "cuda"
                    else:
                        logger.warning(f"⚠️ GPU memory insufficient ({gpu_memory // 1024 ** 3}GB), using CPU")
                        return "cpu"

                except (RuntimeError, AttributeError) as e:
                    logger.warning(f"⚠️ CUDA initialization failed: {str(e)[:100]}, falling back to CPU")
                    return "cpu"

            # Check for MPS (Apple Silicon)
            if device in ["auto", "mps"] and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    test_tensor = torch.tensor([1.0]).to("mps")
                    logger.info("✅ MPS (Apple Silicon) available")
                    return "mps"
                except Exception:
                    pass

            # Default to CPU
            return "cpu"

        def _estimate_model_size(self) -> int:
            """📏 Estimate model size in bytes for memory planning."""
            try:
                model_path = Path(self.config.model_name)
                if model_path.exists():
                    return model_path.stat().st_size

                # Try common locations
                _roots = [
                    os.environ.get("AMB_MODELS_DIR", ""),
                    os.environ.get("AMB_MODEL_DIR", ""),
                    str(Path.cwd() / "models"),
                    str(Path.cwd() / "model"),
                    str(Path.home() / "models"),
                ]
                possible_paths = [self.config.model_name] + [
                    str(Path(r) / model_path.name) for r in _roots if r
                ]

                for path in possible_paths:
                    if Path(path).exists():
                        return Path(path).stat().st_size

            except Exception:
                pass

            # Default estimate for 3B Q4_K_M model
            return 2 * 1024 ** 3  # 2GB default

        def _resolve_threads(self) -> int:
            """⚙️ Resolve optimal thread count for the model."""
            try:
                cpu_count = os.cpu_count() or 4

                # Check if config specifies n_threads explicitly
                config_threads = getattr(self.config, 'n_threads', None)
                if config_threads and config_threads > 0:
                    logger.info(f"✅ Using config-specified n_threads: {config_threads}")
                    return min(config_threads, cpu_count)

                if "7b" in self.config.model_name.lower():
                    # 7B models need careful threading
                    if self._device == "cpu":
                        threads = min(6, max(2, int(cpu_count * 0.75)))  # Use 75% of available
                    else:  # GPU
                        threads = min(4, max(2, int(cpu_count * 0.5)))  # Use 50% for GPU
                else:
                    # Other models
                    if self._device == "cpu":
                        threads = max(4, int(cpu_count * 0.9))
                    else:
                        threads = max(2, int(cpu_count * 0.5))

                logger.info(f"✅ Resolved {threads} threads for {self._device} on {cpu_count} available")
                return threads
            except Exception as e:
                logger.warning(f"⚠️ Thread resolution failed: {e}, using default 4")
                return 4

        def _resolve_gpu_layers(self) -> int:
            """⚙️ Determine optimal GPU layers based on device and model."""
            is_qwen = "qwen" in self.config.model_name.lower()

            if self._device != "cuda":
                return 0  # No GPU layers on CPU

            if not torch.cuda.is_available():
                return 0

            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                model_size = self._estimate_model_size()

                logger.info(f"GPU memory available: {gpu_memory / (1024 ** 3):.1f}GB")

                # Determine layer allocation
                if gpu_memory > (model_size * 2.5):
                    # Plenty of VRAM: offload all layers
                    return -1
                elif gpu_memory > (model_size * 1.8):
                    # Good VRAM: offload most layers
                    if is_qwen:
                        return min(32, int((gpu_memory * 0.7) // (model_size / 32)))
                    else:
                        return min(35, int((gpu_memory * 0.7) // (model_size / 35)))
                elif gpu_memory > model_size:
                    # Tight VRAM: offload some layers
                    if is_qwen:
                        return min(16, int((gpu_memory * 0.5) // (model_size / 32)))
                    else:
                        return min(18, int((gpu_memory * 0.5) // (model_size / 35)))
                else:
                    # Limited VRAM: minimal offloading
                    logger.warning(
                        f"⚠️ GPU memory ({gpu_memory / (1024 ** 3):.1f}GB) < model size ({model_size / (1024 ** 3):.1f}GB)")
                    return 0

            except Exception as e:
                logger.warning(f"⚠️ GPU layer resolution failed: {e}")
                return 0

        # ------------------------------------------------------------------
        # NEW: Chat format detection
        # ------------------------------------------------------------------
        def _detect_chat_format(self) -> Optional[str]:
            """
            🔍 Detect the appropriate chat template from the model filename.
            Returns a string like "llama-3", "qwen" or None if unknown.
            """
            name = self.config.model_name.lower()
            if "llama-3" in name or "llama3" in name:
                return "llama-3"
            if "qwen" in name:
                return "qwen"
            if "gemma" in name:
                return "gemma"
            # Add more mappings as new models appear
            return None

        # ------------------------------------------------------------------
        # Startup benchmark (new)
        # ------------------------------------------------------------------
        async def _run_startup_benchmark(self):
            """🔥 Comprehensive benchmark to measure real generation speed."""
            logger.info("🔥 Running startup benchmark (this may take ~30s)...")
            test_prompts = [
                ("short", "What is AI?"),
                ("medium", "Explain machine learning with examples."),
                ("long", "Detailed explanation of transformer architecture. " * 50),
            ]
            results = {}
            for name, prompt in test_prompts:
                times = []
                for _ in range(3):
                    start = time.time()
                    try:
                        if self.use_chat:
                            self._llm.create_chat_completion(
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=50,
                                temperature=0.0
                            )
                        else:
                            self._llm.create_completion(prompt, max_tokens=50, temperature=0.0)
                        elapsed = time.time() - start
                        times.append(elapsed)
                    except Exception as e:
                        logger.warning(f"Benchmark failed for {name}: {e}")
                if times:
                    avg_time = sum(times) / len(times)
                    results[name] = avg_time
                    logger.info(f"  {name}: avg {avg_time:.2f}s")
            if results:
                prompt_tokens = len(test_prompts[2][1].split()) * 1.3
                gen_tokens = 50
                total_time = results.get("long")
                if total_time is None:
                    logger.warning(
                        "Benchmark: 'long' prompt never succeeded; cannot estimate "
                        "tokens/sec. measured_tokens_per_sec stays at %r.",
                        self.measured_tokens_per_sec,
                    )
                else:
                    denom = total_time - (prompt_tokens / (self.measured_tokens_per_sec or 10.0))
                    if denom > 1.0:
                        self.measured_tokens_per_sec = gen_tokens / denom
                        logger.info(
                            "✅ Benchmark complete. Estimated speed: %.1f tokens/sec",
                            self.measured_tokens_per_sec,
                        )
                    else:
                        logger.warning(
                            "Benchmark: denominator %.2f s too small; discarding "
                            "estimate. measured_tokens_per_sec stays at %r.",
                            denom, self.measured_tokens_per_sec,
                        )

        # ------------------------------------------------------------------
        # Initialization (modified to include chat format)
        # ------------------------------------------------------------------
        async def _initialize_impl(self) -> None:
            """
            ⚡ Initialize llama.cpp model with adaptive context management.

            PRODUCTION FEATURES:
            - Adaptive context window negotiation
            - System capability detection
            - Intelligent memory management
            - Graceful degradation
            - Comprehensive logging
            - Recovery mechanisms
            - Chat template auto‑detection
            """

            logger.info(f"🔧 Initializing llama.cpp adapter: {self.config.provider}")

            # Get context manager singleton
            ctx_mgr = get_model_manager()
            self.context_manager = ctx_mgr

            # Log system information
            logger.info(
                f"🖥️  System Information: Capability: {ctx_mgr.system_metrics.capability_level.value}, Total RAM: {ctx_mgr.system_metrics.total_memory_gb:.1f}GB,"
                f"Available RAM: {ctx_mgr.system_metrics.available_memory_gb:.1f}GB, CPU Cores: {ctx_mgr.system_metrics.cpu_cores}, CPU Threads: {ctx_mgr.system_metrics.cpu_threads},"
                f"CPU Freq: {ctx_mgr.system_metrics.cpu_freq_ghz:.2f}GHz, Has CUDA: {ctx_mgr.system_metrics.has_cuda}, Has MPS: {ctx_mgr.system_metrics.has_mps}"
            )

            # ====================================================================
            # 1. SYSTEM METRICS & MEMORY SETUP
            # ====================================================================

            try:
                import psutil
                self._system_memory = psutil.virtual_memory()
                self._available_ram = self._system_memory.available
                self._has_psutil = True
                logger.info(f"✅ psutil available for memory monitoring")
            except ImportError:
                self._has_psutil = False
                self._available_ram = 8 * 1024 ** 3  # Conservative estimate
                logger.warning(f"⚠️ psutil not installed – using conservative memory estimates")

            # ====================================================================
            # 2. MODEL FILE RESOLUTION
            # ====================================================================

            model_path = Path(self.config.model_name)

            if not model_path.exists():
                logger.info(f"🔍 Model file not found at {model_path}, searching common locations...")

                _roots = [
                    os.environ.get("AMB_MODELS_DIR", ""),
                    os.environ.get("AMB_MODEL_DIR", ""),
                    str(Path.cwd() / "models"),
                    str(Path.cwd() / "model"),
                    str(Path.home() / "models"),
                ]
                possible_paths = [self.config.model_name] + [
                    str(Path(r) / model_path.name) for r in _roots if r
                ]

                found = False
                for location in possible_paths:
                    loc_path = Path(location)
                    if loc_path.exists():
                        model_path = loc_path
                        logger.info(f"✅ Found model at: {model_path}")
                        found = True
                        break

                if not found:
                    raise FileNotFoundError(
                        f"❌ Model file not found: {self.config.model_name}\n"
                        f"Searched locations:\n" +
                        "\n".join(f"  - {p}" for p in possible_paths)
                    )

            # Store model metadata
            self._model_size_bytes = model_path.stat().st_size if model_path.exists() else 2 * 1024 ** 3
            self._model_size_gb = self._model_size_bytes / (1024 ** 3)

            logger.info(f"📦 Model size: {self._model_size_gb:.2f}GB")

            # ====================================================================
            # 3. MODEL & SYSTEM CLASSIFICATION
            # ====================================================================

            is_7b = "7b" in self.config.model_name.lower()
            is_qwen = "qwen" in self.config.model_name.lower()
            self._is_qwen = is_qwen  # Store for later use in generate()

            logger.info(f"Model classification: 7B={is_7b}, Qwen={is_qwen}")

            device = self._device
            logger.info(f"Device set to: {device}")

            # ====================================================================
            # 4. CHAT FORMAT DETECTION (NEW)
            # ====================================================================
            if self.config.chat_template:
                self.chat_format = self.config.chat_template
                logger.info(f"✅ Using configured chat format: {self.chat_format}")
            else:
                self.chat_format = self._detect_chat_format()
                if self.chat_format:
                    logger.info(f"✅ Auto‑detected chat format: {self.chat_format}")
                else:
                    logger.info("ℹ️ No chat format detected, using raw completion mode")

            self.use_chat = self.chat_format is not None

            # ====================================================================
            # 5. ADAPTIVE CONTEXT NEGOTIATION
            # ====================================================================

            logger.info(f"🔄 Starting adaptive context negotiation...")

            negotiated_ctx = ctx_mgr.negotiate_context_window(
                model_name=self.config.model_name,
                config_requested=self.config.context_length
            )

            # Build context candidates
            candidates = self._get_context_candidates(negotiated_ctx)

            logger.info(
                f"📊 Context Strategy: Model: {Path(self.config.model_name).name}, Model size: {self._model_size_gb:.2f}GB, Available RAM: {self._available_ram / (1024 ** 3):.1f}GB,"
                f"Negotiated context: {negotiated_ctx}, Candidates to try: {candidates}"
            )

            # ====================================================================
            # 6. THREAD & GPU CONFIGURATION
            # ====================================================================

            self._n_threads = self._resolve_threads()
            self._n_gpu_layers = self._resolve_gpu_layers()

            logger.info(
                f"⚙️  Configuration: Threads: {self._n_threads}, GPU layers: {self._n_gpu_layers}, Device: {device}"
            )

            # ====================================================================
            # 7. MODEL LOADING WITH ADAPTIVE FALLBACK (modified to pass chat_format)
            # ====================================================================

            def _try_load(model_path_str: str, n_ctx_val: int, chat_fmt: Optional[str]) -> bool:
                """Attempt to load model with specific context window and chat format."""

                if self._shutting_down or self._shutdown_flag:
                    logger.warning("⚠️ Load aborted: adapter shutting down")
                    return False

                try:
                    logger.info(f"Attempting load with n_ctx={n_ctx_val}, chat_format={chat_fmt}...")

                    self._llm = Llama(
                        model_path=str(model_path_str),
                        n_ctx=n_ctx_val,
                        n_threads=self._n_threads,
                        n_threads_batch=self._n_threads,
                        n_gpu_layers=self._n_gpu_layers,
                        verbose=False,
                        seed=42,
                        use_mlock=True,
                        chat_format=chat_fmt,  # <-- crucial addition
                    )

                    # Health test
                    logger.info(f"Running health test...")
                    if self.use_chat:
                        test_result = self._llm.create_chat_completion(
                            messages=[{"role": "user", "content": "test"}],
                            max_tokens=1,
                            temperature=0.0,
                            stop=["\n"]
                        )
                    else:
                        test_result = self._llm.create_completion(
                            "test",
                            max_tokens=1,
                            temperature=0.0,
                            stop=["\n"]
                        )

                    success = test_result and "choices" in test_result

                    if success:
                        logger.info(f"✅ Health test passed")
                    else:
                        logger.info(f"❌ Health test failed")

                    return success

                except Exception as e:
                    self._last_load_exception = e
                    logger.error(f"Load exception: {str(e)[:200]}")
                    return False

            # Try loading with candidates
            loaded = False
            actual_context = None
            last_exception = None

            for attempt, n_ctx in enumerate(candidates, 1):
                try:
                    logger.info(f"🔧 Load attempt {attempt}/{len(candidates)}: n_ctx={n_ctx}")

                    success = await asyncio.get_running_loop().run_in_executor(
                        None,
                        _try_load,
                        str(model_path),
                        n_ctx,
                        self.chat_format  # pass the detected/configured format
                    )

                    if success:
                        self._model_loaded = True
                        actual_context = n_ctx
                        self.config.context_length = n_ctx

                        self._initial_context = actual_context
                        self._initial_max_tokens = self.config.max_tokens

                        # ✅ CRITICAL: Store context in model object
                        if hasattr(self._llm, 'n_ctx'):
                            self._llm.n_ctx = n_ctx

                        logger.info(
                            f"✅ Model loaded successfully! ->  Context window: {actual_context} tokens, Device: {device}, GPU layers: {self._n_gpu_layers}, Threads: {self._n_threads}, Seed: 42 (reproducible)"
                        )
                        loaded = True
                        break
                    else:
                        last_exception = getattr(self, "_last_load_exception", Exception("Unknown"))
                        logger.info(
                            f"⚠️ Load with n_ctx={n_ctx} failed: {str(last_exception)[:100]}\n"
                            f"   Trying next candidate..."
                        )

                except asyncio.CancelledError:
                    logger.warning(f"🛑 Load cancelled at attempt {attempt}")
                    raise
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"⚠️ Exception during load attempt {attempt}: {str(e)[:200]}\n"
                        f"   Continuing with next candidate..."
                    )
                    continue

            # ====================================================================
            # 8. LOAD RESULT HANDLING
            # ====================================================================

            if not loaded:
                error_msg = (
                    f"❌ Failed to load model with any context value\n"
                    f"   Tried contexts: {candidates}\n"
                    f"   Last error: {str(last_exception)[:200]}\n"
                    f"   \n"
                    f"Troubleshooting:\n"
                    f"  1. Check model file exists: {model_path}\n"
                    f"  2. Verify sufficient disk space and RAM\n"
                    f"  3. Increase system swap space\n"
                    f"  4. Try reducing context_length in config.yaml\n"
                    f"  5. Try reducing n_threads setting\n"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # ====================================================================
            # 9. POST-LOAD VALIDATION
            # ====================================================================

            if not hasattr(self._llm, 'n_ctx'):
                logger.warning(f"⚠️ Model doesn't expose n_ctx attribute")
                self._llm.n_ctx = actual_context

            # Verify context actually set
            actual_model_ctx = getattr(self._llm, 'n_ctx', actual_context)
            if actual_model_ctx != actual_context:
                logger.warning(
                    f"⚠️ Context mismatch! Requested {actual_context} but model has {actual_model_ctx}"
                )
                actual_context = actual_model_ctx
                self.config.context_length = actual_context

            self.context_manager.set_actual_n_ctx(actual_context)

            # Final log
            logger.info(
                f"🎉 Initialization Complete: Model: {Path(model_path).name}, Context: {actual_context} tokens,"
                f"Size: {self._model_size_gb:.2f}GB, System: {ctx_mgr.system_metrics.capability_level.value.upper()} -> Ready for inference"
            )

        # ------------------------------------------------------------------
        # Context recovery (restored from original)
        # ------------------------------------------------------------------
        def can_recover_context(self, current_pressure: float) -> bool:
            """Check if we can recover context window from reduced state."""
            try:
                mem = psutil.virtual_memory()
                # Recover if memory pressure < 60% AND we reduced context before
                return (mem.percent < 60 and
                        hasattr(self, '_initial_context') and
                        self.context_manager.context_window < self._initial_context)
            except Exception:
                return False

        async def attempt_context_recovery(self):
            """Safely restore context window if memory allows."""
            try:
                mem = psutil.virtual_memory()
                if self.can_recover_context(mem.percent):
                    logger.info("📈 Recovering context window")
                    # Gradually increase back toward initial
                    new_ctx = min(
                        self._initial_context,
                        int(self.context_manager.context_window * 1.1)
                    )
                    self.context_manager.context_window = new_ctx
                    logger.info(f"✓ Context restored to {new_ctx}")
            except Exception as e:
                logger.warning(f"Context recovery attempt failed: {e}")

        # ------------------------------------------------------------------
        # Context candidates (unchanged)
        # ------------------------------------------------------------------
        def _get_context_candidates(self, preferred: int) -> List[int]:
            """🎯 Get context window candidates in priority order."""
            is_qwen = "qwen" in self.config.model_name.lower()
            is_7b = "7b" in self.config.model_name.lower()

            # Base candidates adjusted for model and system
            if is_qwen and is_7b:
                # Qwen 7B: conservative for CPU, aggressive for GPU
                if self._device == "cuda":
                    candidates = [preferred, 16384, 8192, 4096, 2048, 1024]
                else:
                    candidates = [preferred, 8192, 4096, 2048, 1024]
            else:
                # Other models: more flexible
                candidates = [preferred, 16384, 8192, 4096, 2048, 1024]

            # Filter and deduplicate
            seen = set()
            filtered = []
            for c in candidates:
                if c and c >= 1024 and c not in seen:
                    filtered.append(c)
                    seen.add(c)

            return filtered

        @property
        def supports_grammar(self):
            """Indicate that this adapter can use GBNF grammar."""
            return True

        # ------------------------------------------------------------------
        # Generation (modified to use chat completion when appropriate)
        # ------------------------------------------------------------------
        @async_retry(times=2, delay=1.0)
        async def generate(
                self,
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                timeout: Optional[float] = None,
                grammar: Optional[str] = None
        ) -> str:
            """
            📝 Generate text with adaptive context management.

            PRODUCTION FEATURES:
            - Adaptive token budgeting
            - Intelligent prompt compression
            - Real-time memory monitoring
            - System-aware timeouts
            - Comprehensive error handling
            - Token-aware safety caps
            - Chat completion support for instruction‑tuned models
            """

            # SHUTDOWN & STATE CHECKS
            if self._shutting_down or self._shutdown_flag:
                logger.warning("⚠️ Generation cancelled: adapter shutting down")
                return "ERROR: Adapter shutting down"

            # MEMORY PRESSURE CHECK
            try:
                await self.context_manager.adapt_to_memory_pressure()
            except Exception as e:
                logger.warning(f"Memory adaptation failed: {e}")

            # MODEL INITIALIZATION
            if not self._initialized:
                try:
                    await self.initialize()
                except Exception as e:
                    logger.error(f"❌ Model initialization failed: {e}")
                    return f"ERROR: Model initialization failed: {str(e)[:100]}"

            if not self._model_loaded:
                return "ERROR: Model not loaded"

            max_tokens = int(max_tokens if max_tokens is not None else (self.config.max_tokens if self.config.max_tokens is not None else 256))
            temperature = float(temperature if temperature is not None else (self.config.temperature if self.config.temperature is not None else 0.7))
            temperature = max(0.1, min(temperature, 1.5))

            logger.info(f"Generation params: max_tokens={max_tokens}, temperature={temperature:.2f}")

            ctx_mgr = getattr(self, 'context_manager', get_model_manager())

            # TOKEN BUDGET CALCULATION
            token_budget: Optional[TokenMetrics] = None

            try:
                token_budget = ctx_mgr.calculate_token_budget(
                    prompt_text=prompt,
                    estimated_prompt_tokens=None,  # Auto-calculate
                    max_generation_tokens=max_tokens,
                    tokens_per_sec=self.measured_tokens_per_sec
                )

                # Use recommended generation tokens
                max_tokens = token_budget.recommended_max_gen

                # Log warnings if any
                if token_budget.warnings:
                    for warning in token_budget.warnings:
                        logger.warning(f"⚠️ {warning}")

                logger.info(f"Token budget calculated: {token_budget}")

            except Exception as e:
                logger.warning(f"⚠️ Token budget calculation failed: {e}, using conservative defaults")
                max_tokens = min(max_tokens, 256)
                token_budget = None

            # ====================================================================
            # ADAPTIVE TOKENIZATION & OVERFLOW HANDLING
            # ====================================================================

            actual_prompt_tokens = None
            prompt_to_use = prompt
            compressed_applied = False

            if self._llm is not None:
                try:
                    # Get actual token count from model
                    prompt_tokens_raw = self._llm.tokenize(prompt.encode('utf-8', errors='ignore'))
                    actual_prompt_tokens = len(prompt_tokens_raw)

                    # Get actual context window from model
                    context_limit = getattr(self._llm, 'n_ctx', None)
                    if context_limit is None:
                        context_limit = self.config.context_length or 2048

                    logger.info(f"Context from model: {context_limit}, Prompt tokens: {actual_prompt_tokens}")

                    # Safety parameter checks
                    if context_limit < 1024:
                        logger.warning(f"⚠️ Context too small ({context_limit}), using 1024")
                        context_limit = 1024

                    if context_limit > 131072:
                        logger.warning(f"⚠️ Context unreasonably large ({context_limit}), capping at 131072")
                        context_limit = 131072

                    # Calculate available space
                    safety_buffer = max(100, int(context_limit * 0.1))  # 10% safety
                    max_allowed = context_limit - actual_prompt_tokens - safety_buffer

                    logger.info(
                        f"Token space calculation: Context: {context_limit} Prompt: {actual_prompt_tokens} Safety buffer: {safety_buffer} Available: {max_allowed}"
                    )

                    # ============================================================
                    # CRITICAL OVERFLOW HANDLING
                    # ============================================================

                    if max_allowed < 10:
                        # CRITICAL: Prompt too large - MUST compress
                        logger.error(
                            f"🚨 CRITICAL TOKEN OVERFLOW!\n"
                            f"  Context: {context_limit} tokens\n"
                            f"  Prompt: {actual_prompt_tokens} tokens\n"
                            f"  Safety buffer: {safety_buffer} tokens\n"
                            f"  Available: {max_allowed} tokens"
                        )

                        # Compress prompt to fit in 60% of context
                        target_tokens = int(context_limit * 0.6)
                        logger.info(f"✂️ Compressing prompt to fit {target_tokens} tokens...")

                        prompt_to_use = ctx_mgr.adapt_prompt_to_context(
                            prompt,
                            target_tokens=target_tokens
                        )

                        # Re-tokenize compressed prompt
                        try:
                            new_prompt_tokens = len(self._llm.tokenize(prompt_to_use.encode('utf-8', errors='ignore')))
                            actual_prompt_tokens = new_prompt_tokens
                            max_allowed = context_limit - actual_prompt_tokens - safety_buffer
                            compressed_applied = True

                            logger.info(
                                f"✂️ Compression successful:\n"
                                f"  New prompt tokens: {actual_prompt_tokens}\n"
                                f"  Available for generation: {max_allowed}"
                            )
                        except Exception as e:
                            logger.error(f"❌ Compression tokenization failed: {e}")
                            max_allowed = context_limit - actual_prompt_tokens - safety_buffer

                        # Final check
                        if max_allowed < 10:
                            # Even after compression, critical situation
                            max_tokens = 1
                            logger.warning(f"⚠️ Even after compression, very limited space - limiting to 1 token")
                        else:
                            max_tokens = min(max_tokens, max_allowed)
                            logger.info(f"📊 Post-compression generation tokens: {max_tokens}")

                    elif max_tokens > max_allowed:
                        # Adjust max tokens to fit available space
                        old_max = max_tokens
                        max_tokens = max_allowed
                        logger.info(f"✂️ Reduced generation tokens: {old_max} → {max_tokens}")

                except Exception as e:
                    logger.warning(f"⚠️ Tokenization/overflow check failed: {e}")
                    logger.warning(f"   Using conservative fallback: max_tokens = 256")
                    max_tokens = min(max_tokens, 256)

            # ====================================================================
            # TIMEOUT MANAGEMENT - SYSTEM AWARE
            # ====================================================================
            use_timeout = token_budget.recommended_timeout if token_budget else (
                    timeout or self.config.timeout or 120.0)
            logger.info(
                f"Generation config: Device: {self._device} Max tokens: {max_tokens} Temperature: {temperature:.2f}s Timeout: {use_timeout:.1f}s Prompt compressed: {compressed_applied}"
            )

            # ====================================================================
            # GENERATION EXECUTION (modified to handle chat completions)
            # ====================================================================
            def _call():
                """Synchronous generation call with safety checks."""
                try:
                    # Final shutdown check
                    if self._shutting_down or self._shutdown_flag:
                        logger.warning("⚠️ Generation interrupted: shutdown in progress")
                        return "ERROR: Generation cancelled during shutdown"

                    # Prepare stop tokens
                    stop_tokens = ["\n", "###", "<|endoftext|>", "</s>", "<|eot_id|>"]
                    if self._is_qwen:
                        stop_tokens.extend(["\n\n", "<|im_end|>", "<|end|>", "<|eom|>"])

                    logger.info(f"Starting generation with {len(prompt_to_use)} char prompt...")

                    # Create grammar object if provided
                    grammar_obj = None
                    if grammar is not None:
                        try:
                            grammar_obj = LlamaGrammar.from_string(grammar)
                        except Exception as e:
                            logger.error(f"Failed to create grammar object: {e}")
                            # Continue without grammar rather than failing completely
                            grammar_obj = None

                    if self.use_chat:
                        # ------------------------------------------------------------
                        # CHAT COMPLETION MODE
                        # ------------------------------------------------------------
                        messages = [{"role": "user", "content": prompt_to_use}]
                        response = self._llm.create_chat_completion(
                            messages=messages,
                            max_tokens=max(1, int(max_tokens)),
                            temperature=temperature,
                            stop=stop_tokens,
                            grammar=grammar_obj  # pass the object, not the string
                        )
                        if not response or "choices" not in response or not response["choices"]:
                            raise RuntimeError("Empty or invalid chat response from model")
                        text = response["choices"][0].get("message", {}).get("content", "")
                    else:
                        # ------------------------------------------------------------
                        # RAW COMPLETION MODE
                        # ------------------------------------------------------------
                        response = self._llm.create_completion(
                            prompt_to_use,
                            max_tokens=max(1, int(max_tokens)),
                            temperature=temperature,
                            echo=False,
                            stop=stop_tokens,
                            grammar=grammar_obj  # pass the object, not the string
                        )
                        if not response or "choices" not in response or not response["choices"]:
                            raise RuntimeError("Empty or invalid completion response from model")
                        text = response["choices"][0].get("text", "")

                    if not text:
                        logger.warning("⚠️ Model generated empty text")
                        return "ERROR: Model generated no text"

                    return text.strip()

                except Exception as e:
                    # Check if this is a shutdown cancellation
                    if self._shutting_down or self._shutdown_flag:
                        logger.warning(f"⚠️ Generation interrupted: {str(e)[:100]}")
                        return "ERROR: Generation cancelled"
                    # Log and re-raise for retry decorator
                    logger.error(f"❌ Generation call failed: {str(e)[:200]}")
                    raise

            # ====================================================================
            # ASYNC EXECUTION WITH TIMEOUT
            # ====================================================================

            future = self._executor.submit(_call)

            # Track the future
            with self._pending_lock:
                self._pending_futures.add(future)
                future.add_done_callback(lambda f: self._pending_futures.discard(f))

            try:
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=use_timeout
                )
                return result
            except asyncio.TimeoutError:
                # Do NOT cancel the future – it's already running.
                # The thread will continue, but we will wait for it in shutdown.
                logger.error("⏱️ Generation timed out")
                return "ERROR: Generation timed out"
            except asyncio.CancelledError:
                logger.warning("⚠️ Generation cancelled (asyncio)")
                return "ERROR: Generation cancelled"
            except Exception as e:
                logger.error(f"❌ Generation failed with exception: {str(e)[:200]}")
                return f"ERROR: {str(e)[:200]}"
            finally:
                # Ensure future is removed if not already done
                with self._pending_lock:
                    self._pending_futures.discard(future)

        # ------------------------------------------------------------------
        # Shutdown (unchanged)
        # ------------------------------------------------------------------
        def shutdown(self, timeout: float = 10.0) -> bool:
            """🛑 Wait for pending generations and release executor gracefully."""
            logger.info("🛑 LlamaCppAdapter shutdown initiated")

            # 1. Prevent new tasks (set flags via base class)
            super().shutdown(timeout)

            # 2. Wait for already‑submitted futures
            logger.info(f"🛑 LlamaCppAdapter: waiting up to {timeout}s for pending generations...")
            with self._pending_lock:
                futures = list(self._pending_futures)

            not_done = set()
            if futures:
                try:
                    done, not_done = concurrent.futures.wait(
                        futures,
                        timeout=timeout,
                        return_when=concurrent.futures.ALL_COMPLETED
                    )
                    if not_done:
                        logger.warning(f"⚠️ {len(not_done)} generation(s) still running after timeout")
                    else:
                        logger.info("✅ All pending generations completed")
                except Exception as e:
                    logger.error(f"❌ Error waiting for futures: {e}")

            # 3. Signal the worker to stop. Do NOT wait indefinitely:
            #    if a generation call is stuck inside a C library call
            #    (llama_decode, mmap, CUDA sync) it cannot be
            #    interrupted from Python, and blocking here would just
            #    delay the caller. The worker is a raw daemon thread,
            #    so interpreter exit will not join it either.
            logger.info("🛑 Signalling LlamaCppAdapter worker to stop ...")
            try:
                exited = self._executor.shutdown(wait=True, timeout=2.0)
                if exited:
                    logger.info("✅ Adapter worker thread exited cleanly")
                else:
                    logger.warning(
                        "⚠️ Adapter worker thread still inside a C call "
                        "after 2s. It is a daemon thread and will not "
                        "block process exit; leaving it to the OS."
                    )
            except Exception as e:
                logger.error(f"❌ Error signalling adapter worker: {e}")

            # 4. Force cleanup of model and memory
            if hasattr(self, '_llm') and self._llm is not None:
                try:
                    # Explicitly delete the model reference
                    del self._llm
                    self._llm = None
                    logger.info("✅ Model reference cleared")
                except Exception as e:
                    logger.error(f"❌ Error clearing model: {e}")

            logger.info("✅ LlamaCppAdapter shutdown complete")
            return len(not_done) == 0

        # ------------------------------------------------------------------
        # Health and metrics (unchanged)
        # ------------------------------------------------------------------
        def is_healthy(self) -> bool:
            """💚 Check if model is healthy and ready for inference."""
            return self._initialized and self._model_loaded and self._llm is not None

        def get_metrics(self) -> Dict[str, Any]:
            """📊 Get comprehensive model metrics."""
            base_metrics = super().get_metrics()
            base_metrics.update({
                "device": self._device,
                "gpu_layers": self._n_gpu_layers,
                "threads": self._n_threads,
                "model_path": self.config.model_name,
                "use_chat": self.use_chat,
                "chat_format": self.chat_format,
                "tokens_per_sec": self.measured_tokens_per_sec,
            })
            return base_metrics


if _OPTIONAL_IMPORTS['gpt4all']:
    class GPT4AllAdapter(BaseLLMAdapter):
        """
        🤖 Production‑grade GPT4All adapter with:
        - Adaptive context management (via ModelManager)
        - Token‑aware prompt compression
        - Graceful shutdown and pending request tracking
        - Robust error handling and timeouts
        - Health checks and metrics
        - Memory pressure monitoring and adaptation
        - Startup benchmark (optional)
        """

        def __init__(self, config: ModelConfig):
            super().__init__(config)
            self._model = None
            # Single-worker daemon-thread pool. Same rationale as in
            # HFAdapter and LlamaCppAdapter: ThreadPoolExecutor workers
            # are joined at interpreter exit by
            # `concurrent.futures.thread._python_exit`, and a stuck
            # `GPT4All.generate()` would block shutdown indefinitely.
            self._executor = DaemonWorkerPool(
                name=f"gpt4all_{Path(config.model_name).stem}"
            )
            self._pending_futures = set()
            self._pending_lock = threading.RLock()
            self.context_manager = get_model_manager()  # singleton

            # Device handling (GPT4All can run on CPU or GPU if backend supports)
            self._device = self._resolve_device()
            self._n_threads = self._resolve_threads()

            # Store initial values for potential recovery
            self._initial_context = None  # will be set after loading
            self._initial_max_tokens = self.config.max_tokens

            # Optional: measure tokens/sec (can be estimated later)
            self.measured_tokens_per_sec = None

            # Size cache for adaptive timeouts (optional)
            self._size_cache = {}
            self._size_cache_timestamp = {}
            self._size_cache_max_age = 300

        # ------------------------------------------------------------------
        # Device and resource detection (simplified for GPT4All)
        # ------------------------------------------------------------------
        def _resolve_device(self) -> str:
            """
            ⚙️ Resolve compute device for GPT4All.
            GPT4All primarily uses CPU, but we respect user setting.
            """
            device = self.config.device.lower()
            if device == "cuda" or device == "gpu":
                # GPT4All can use GPU if compiled with support; we assume user knows
                logger.info("ℹ️ Using GPU for GPT4All (ensure backend supports it)")
                return "gpu"
            return "cpu"

        def _estimate_model_size(self) -> float:
            """
            📏 Estimate model size in GB based on file size.
            """
            try:
                model_path = Path(self.config.model_name)
                if model_path.exists():
                    size_bytes = model_path.stat().st_size
                    return size_bytes / (1024 ** 3)
                # Try common locations
                _roots = [
                    os.environ.get("AMB_MODELS_DIR", ""),
                    os.environ.get("AMB_MODEL_DIR", ""),
                    str(Path.cwd() / "models"),
                    str(Path.cwd() / "model"),
                    str(Path.home() / "models"),
                ]
                possible_paths = [self.config.model_name] + [
                    str(Path(r) / model_path.name) for r in _roots if r
                ]
                for path in possible_paths:
                    if Path(path).exists():
                        size_bytes = Path(path).stat().st_size
                        return size_bytes / (1024 ** 3)
            except Exception:
                pass
            # Default fallback (e.g., 4GB)
            return 4.0

        def _resolve_threads(self) -> int:
            """
            ⚙️ Resolve optimal thread count for CPU inference.
            """
            try:
                cpu_count = os.cpu_count() or 4
                # Use 75% of available cores for CPU
                threads = max(2, int(cpu_count * 0.75))
                logger.info(f"✅ Resolved {threads} threads for GPT4All")
                return threads
            except Exception:
                return 4

        # ------------------------------------------------------------------
        # Context recovery (delegates to ModelManager)
        # ------------------------------------------------------------------
        def can_recover_context(self, current_pressure: float) -> bool:
            """
            Check if context recovery is possible.
            For GPT4All, recovery is simply a hint to increase context window if memory permits.
            """
            try:
                mem = psutil.virtual_memory()
                return mem.percent < 60
            except Exception:
                return False

        async def attempt_context_recovery(self):
            """
            Safely attempt to recover context window via ModelManager.
            Called after successful tool executions.
            """
            try:
                await self.context_manager.attempt_context_recovery()
            except Exception as e:
                logger.warning(f"Context recovery attempt failed: {e}")

        # ------------------------------------------------------------------
        # Startup benchmark (optional)
        # ------------------------------------------------------------------
        async def _run_startup_benchmark(self):
            """🔥 Quick benchmark to estimate generation speed (tokens/sec)."""
            logger.info("🔥 Running startup benchmark (this may take a few seconds)...")
            test_prompts = [
                ("short", "What is AI?"),
                ("medium", "Explain machine learning with examples."),
                ("long", "Detailed explanation of transformer architecture. " * 50),
            ]
            results = {}
            for name, prompt in test_prompts:
                times = []
                for _ in range(3):
                    start = time.time()
                    try:
                        # Use a simple generation with low token count
                        if hasattr(self._model, 'generate') and callable(self._model.generate):
                            self._model.generate(prompt, max_tokens=20, temp=0.0)
                        else:
                            self._model(prompt)
                        elapsed = time.time() - start
                        times.append(elapsed)
                    except Exception as e:
                        logger.warning(f"Benchmark failed for {name}: {e}")
                if times:
                    avg_time = sum(times) / len(times)
                    results[name] = avg_time
                    logger.info(f"  {name}: avg {avg_time:.2f}s")
            if results:
                prompt_tokens = len(test_prompts[2][1].split()) * 1.3
                gen_tokens = 20
                total_time = results.get("long")
                if total_time is None:
                    logger.warning(
                        "Benchmark: 'long' prompt never succeeded; cannot estimate "
                        "tokens/sec. measured_tokens_per_sec stays at %r.",
                        self.measured_tokens_per_sec,
                    )
                else:
                    denom = total_time - (prompt_tokens / (self.measured_tokens_per_sec or 10.0))
                    if denom > 1.0:
                        self.measured_tokens_per_sec = gen_tokens / denom
                        logger.info(
                            "✅ Benchmark complete. Estimated speed: %.1f tokens/sec",
                            self.measured_tokens_per_sec,
                        )
                    else:
                        logger.warning(
                            "Benchmark: denominator %.2f s too small; discarding "
                            "estimate. measured_tokens_per_sec stays at %r.",
                            denom, self.measured_tokens_per_sec,
                        )

        # ------------------------------------------------------------------
        # Initialization
        # ------------------------------------------------------------------
        async def _initialize_impl(self) -> None:
            """
            ⚡ Locate and load the GPT4All model file.
            Searches common locations if the given path does not exist.
            """
            logger.info(f"🔧 Initializing GPT4All adapter: {self.config.provider}")

            # Memory monitoring
            try:
                import psutil
                self._system_memory = psutil.virtual_memory()
                self._has_psutil = True
                logger.info("✅ psutil available for memory monitoring")
            except ImportError:
                self._has_psutil = False
                logger.warning("⚠️ psutil not installed – memory monitoring limited")

            # Model size estimate
            model_size_gb = self._estimate_model_size()
            logger.info(f"📦 Estimated model size: {model_size_gb:.2f}GB")

            # Locate model file
            model_path = Path(self.config.model_name)
            if not model_path.exists():
                logger.info(f"🔍 Model file not found at {model_path}, searching common locations...")
                _roots = [
                    os.environ.get("AMB_MODELS_DIR", ""),
                    os.environ.get("AMB_MODEL_DIR", ""),
                    str(Path.cwd() / "models"),
                    str(Path.cwd() / "model"),
                    str(Path.home() / "models"),
                ]
                possible_paths = [self.config.model_name] + [
                    str(Path(r) / model_path.name) for r in _roots if r
                ]
                found = False
                for loc in possible_paths:
                    if Path(loc).exists():
                        model_path = Path(loc)
                        logger.info(f"✅ Found GPT4All model at: {model_path}")
                        found = True
                        break
                if not found:
                    raise FileNotFoundError(
                        f"❌ GPT4All model file not found: {self.config.model_name}\n"
                        f"Searched: {possible_paths}"
                    )

            # Load model
            try:
                logger.info(f"🔧 Loading GPT4All model from {model_path}...")
                # Use device and n_threads if supported by the backend
                if self._device == "gpu" and hasattr(GPT4All, 'DEFAULT_BACKEND'):
                    # GPT4All can use GPU via llama.cpp backend; we pass device setting
                    self._model = GPT4All(
                        model_name=model_path.name,
                        model_path=str(model_path.parent),
                        allow_download=False,
                        device=self._device,
                        n_threads=self._n_threads
                    )
                else:
                    # CPU fallback
                    self._model = GPT4All(
                        model_name=model_path.name,
                        model_path=str(model_path.parent),
                        allow_download=False,
                        n_threads=self._n_threads
                    )
                logger.info(f"✅ GPT4All model loaded successfully from: {model_path}")

                # Negotiate context window with ModelManager
                logger.info("🔄 Starting adaptive context negotiation...")
                negotiated_ctx = self.context_manager.negotiate_context_window(
                    model_name=self.config.model_name,
                    config_requested=self.config.context_length
                )
                self._initial_context = negotiated_ctx
                self.context_manager.set_actual_n_ctx(negotiated_ctx)

                logger.info(f"✅ GPT4All context locked at {negotiated_ctx} tokens")

            except Exception as e:
                logger.error(f"❌ Failed to initialize GPT4All model: {e}", exc_info=True)
                raise

        # ------------------------------------------------------------------
        # Token counting (GPT4All doesn't expose tokenizer, use fallback)
        # ------------------------------------------------------------------
        def _count_tokens(self, text: str) -> int:
            """
            Count tokens using word‑based estimate (GPT4All lacks native tokenizer).
            """
            return max(1, int(len(text.split()) * 1.3))

        @property
        def supports_grammar(self):
            return False

        # ------------------------------------------------------------------
        # Generation
        # ------------------------------------------------------------------
        @async_retry(times=2, delay=1.0)
        async def generate(
                self,
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                timeout: Optional[float] = None
        ) -> str:
            """
            📝 Generate text using GPT4All with adaptive token management.
            """
            # Shutdown checks
            if self._shutting_down or self._shutdown_flag:
                logger.warning("⚠️ Generation cancelled: adapter is shutting down")
                return "ERROR: Adapter shutting down"

            # Memory pressure check
            try:
                mem = psutil.virtual_memory()
                if mem.available < 500 * 1024 * 1024:  # less than 500 MB free
                    logger.error("🚨 Critical low memory – aborting generation")
                    return "ERROR: System memory too low to generate safely"
            except ImportError:
                pass

            await self.context_manager.adapt_to_memory_pressure()

            # Ensure initialized
            if not self._initialized:
                try:
                    await self.initialize()
                except Exception as e:
                    logger.error(f"❌ Model initialization failed: {e}")
                    return f"ERROR: Model initialization failed: {str(e)[:100]}"

            # ------------------------------------------------------------------
            # 1. TOKEN BUDGET CALCULATION & PROMPT ADAPTATION
            # ------------------------------------------------------------------
            max_tokens = int(max_tokens if max_tokens is not None else (self.config.max_tokens if self.config.max_tokens is not None else 256))
            temperature = float(temperature if temperature is not None else (self.config.temperature if self.config.temperature is not None else 0.7))
            temperature = max(1e-8, min(temperature, 2.0))

            # Estimate prompt tokens (word‑based)
            estimated_tokens = self._count_tokens(prompt)

            token_budget = self.context_manager.calculate_token_budget(
                prompt_text=prompt,
                estimated_prompt_tokens=estimated_tokens,
                max_generation_tokens=max_tokens,
                tokens_per_sec=self.measured_tokens_per_sec
            )

            # Log warnings
            for w in token_budget.warnings:
                logger.warning(f"⚠️ {w}")

            # Adjust max_tokens based on budget
            max_tokens = token_budget.recommended_max_gen

            # If the prompt itself doesn't fit, compress it
            prompt_to_use = prompt
            compressed_applied = False
            if not token_budget.is_feasible:
                logger.warning("⚠️ Prompt exceeds context – compressing...")
                physical = getattr(self.context_manager, '_actual_model_n_ctx', 0)
                compress_base = physical if physical > 0 else token_budget.total_budget
                target_tokens = int(compress_base * 0.6)
                logger.info(f"✂️ GPT4All compress target: {target_tokens} tokens (base={compress_base})")

                prompt_to_use = self.context_manager.adapt_prompt_to_context(
                    prompt,
                    target_tokens=target_tokens
                )
                compressed_applied = True

                # Re‑evaluate budget
                new_estimated = self._count_tokens(prompt_to_use)
                token_budget = self.context_manager.calculate_token_budget(
                    prompt_text=prompt_to_use,
                    estimated_prompt_tokens=new_estimated,
                    max_generation_tokens=max_tokens,
                    tokens_per_sec=self.measured_tokens_per_sec
                )
                max_tokens = token_budget.recommended_max_gen

            logger.info(
                f"Generation config: max_tokens={max_tokens}, "
                f"temperature={temperature:.2f}, compressed={compressed_applied}"
            )

            # ------------------------------------------------------------------
            # 2. TIMEOUT MANAGEMENT
            # ------------------------------------------------------------------
            use_timeout = token_budget.recommended_timeout if token_budget else (
                    timeout or self.config.timeout or 120.0)

            # ------------------------------------------------------------------
            # 3. EXECUTE GENERATION IN THREAD
            # ------------------------------------------------------------------
            def _call():
                """Synchronous GPT4All generation call."""
                try:
                    if self._shutting_down or self._shutdown_flag:
                        return "ERROR: Generation cancelled during shutdown"

                    # GPT4All's generate() may accept max_tokens/temperature
                    if hasattr(self._model, 'generate') and callable(self._model.generate):
                        try:
                            return self._model.generate(
                                prompt_to_use,
                                max_tokens=max_tokens,
                                temp=temperature
                            )
                        except TypeError:
                            # Fallback: call without extra arguments
                            return self._model.generate(prompt_to_use)
                    else:
                        # Old API: __call__ method
                        return self._model(prompt_to_use)
                except Exception as e:
                    if self._shutting_down or self._shutdown_flag:
                        logger.warning(f"⚠️ Generation interrupted: {str(e)[:100]}")
                        return "ERROR: Generation cancelled"
                    raise

            future = self._executor.submit(_call)
            with self._pending_lock:
                self._pending_futures.add(future)
                future.add_done_callback(lambda f: self._pending_futures.discard(f))

            try:
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=use_timeout
                )
                self._last_used = time.time()
                return result
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Generation timed out after {use_timeout:.1f}s")
                return "ERROR: Generation timed out"
            except asyncio.CancelledError:
                logger.warning("⚠️ Generation cancelled")
                return "ERROR: Generation cancelled"
            except Exception as e:
                logger.error(f"❌ Generation failed: {str(e)[:200]}")
                return f"ERROR: {str(e)[:200]}"
            finally:
                with self._pending_lock:
                    self._pending_futures.discard(future)

        # ------------------------------------------------------------------
        # Shutdown
        # ------------------------------------------------------------------
        def shutdown(self, timeout: float = 10.0) -> bool:
            """
            🛑 Graceful shutdown: wait for pending generations and release resources.
            """
            logger.info("🛑 GPT4AllAdapter shutdown initiated")
            super().shutdown(timeout)

            # Wait for pending futures
            with self._pending_lock:
                futures = list(self._pending_futures)

            not_done = set()
            if futures:
                try:
                    done, not_done = concurrent.futures.wait(
                        futures,
                        timeout=timeout,
                        return_when=concurrent.futures.ALL_COMPLETED
                    )
                    if not_done:
                        logger.warning(f"⚠️ {len(not_done)} generation(s) still running after timeout")
                    else:
                        logger.info("✅ All pending generations completed")
                except Exception as e:
                    logger.error(f"❌ Error waiting for futures: {e}")

            # Shutdown executor
            # Signal the worker to stop with a bounded wait. See the
            # matching comment in HFAdapter.shutdown for the rationale.
            logger.info("🛑 Signalling GPT4AllAdapter worker to stop ...")
            try:
                exited = self._executor.shutdown(wait=True, timeout=2.0)
                if exited:
                    logger.info("✅ GPT4AllAdapter worker thread exited cleanly")
                else:
                    logger.warning(
                        "⚠️ GPT4AllAdapter worker thread still inside a C "
                        "call after 2s. It is a daemon thread and will not "
                        "block process exit; leaving it to the OS."
                    )
            except Exception as e:
                logger.error(f"❌ Error signalling GPT4AllAdapter worker: {e}")

            # Clear model reference
            if hasattr(self, '_model') and self._model is not None:
                try:
                    del self._model
                    self._model = None
                    logger.info("✅ Model reference cleared")
                except Exception as e:
                    logger.error(f"❌ Error clearing model: {e}")

            logger.info("✅ GPT4AllAdapter shutdown complete")
            return len(not_done) == 0

        # ------------------------------------------------------------------
        # Health and metrics
        # ------------------------------------------------------------------
        def is_healthy(self) -> bool:
            """💚 Check if model is loaded."""
            return self._initialized and self._model is not None

        def get_metrics(self) -> Dict[str, Any]:
            """📊 Get adapter metrics."""
            base = super().get_metrics()
            base.update({
                "model_path": self.config.model_name,
                "device": self._device,
                "threads": self._n_threads,
                "tokens_per_sec": self.measured_tokens_per_sec,
            })
            return base


class LocalLLMCloudAdapter(BaseLLMAdapter):
    """
    🌐 LLM adapter for the local-llm-cloud inference server (v2.4).

    Priority:
      1. localllm.LocalLLMAdapter   — v1.0.0 native library (preferred).
                                       Full async streaming, circuit breaker,
                                       batch_generate, TokenMetricsCompat —
                                       purpose-built for ai_agent.py.
      2. llm_cloud_adapter.py       — hand-rolled fallback when localllm is
                                       not on sys.path.

    Config (ModelConfig fields):
      model_name  — GGUF path OR model-id already loaded on server
      base_url    — e.g. "http://localhost:8080"  (or env LLM_CLOUD_HOST/PORT)
      api_key     — optional (or env LLM_CLOUD_API_KEY)
      max_tokens  — 0 = unlimited (server default)
      temperature — generation temperature
      timeout     — per-request timeout in seconds
    """

    def __init__(self, config: 'ModelConfig') -> None:
        super().__init__(config)
        self._io_pool = DaemonWorkerPool(name="llm_cloud_io")
        self._inner:       object = None   # localllm.LocalLLMAdapter instance
        self._client:      object = None   # llm_cloud_adapter fallback client
        self._model_id:    str    = ""
        self._hw_profile:  object = None
        self._use_localllm: bool  = _OPTIONAL_IMPORTS.get('localllm', False)

    @property
    def supports_grammar(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    async def _initialize_impl(self) -> None:
        if self._use_localllm:
            await self._init_with_localllm()
        else:
            await self._init_with_fallback()

    async def _init_with_localllm(self) -> None:
        """
        The 'localllm' native library path is not used by this build.
        Delete this method entirely if you never intend to add it back;
        it exists only to give a clear error if someone sets
        `_OPTIONAL_IMPORTS['localllm'] = True` without also adding a
        `_localllm_create_adapter` helper.
        """
        raise RuntimeError(
            "localllm path requested but _localllm_create_adapter is "
            "not defined in this build. Either supply that helper and "
            "unpin _OPTIONAL_IMPORTS['localllm'], or set the model "
            "provider to 'local_llm_cloud' and rely on the "
            "llm_cloud_adapter.py fallback."
        )

    async def _init_with_fallback(self) -> None:
        """Initialise using llm_cloud_adapter.py when localllm is unavailable."""
        try:
            from amb_sentinel.llm.llm_cloud_adapter import get_client, probe_server
        except ImportError as exc:
            raise LLMTransportError(
                f"Neither localllm nor llm_cloud_adapter is available: {exc}"
            ) from exc

        host = os.environ.get("LLM_CLOUD_HOST", "localhost")
        try:
            port = int(os.environ.get("LLM_CLOUD_PORT", "8080"))
        except ValueError:
            port = 8080

        # Override from base_url if provided
        base_url = getattr(self.config, 'base_url', '') or ''
        if base_url:
            import urllib.parse as _up
            _p = _up.urlparse(base_url)
            host = _p.hostname or host
            port = _p.port    or port

        # Wait for the server with exponential backoff instead of
        # a single 5 s probe. Total wait is configurable via the
        # LLM_CLOUD_WAIT_S env var (default 300 s), which is generous
        # enough for a cold local-llm-server to load a Q4 GGUF on APU
        # hardware.
        _wait_total_s = float(os.environ.get("LLM_CLOUD_WAIT_S", "300"))
        _wait_first   = 1.0
        _wait_cap     = 15.0
        _t0 = time.monotonic()
        _delay = _wait_first
        while True:
            if probe_server(host, port, timeout=5.0):
                break
            _elapsed = time.monotonic() - _t0
            if _elapsed >= _wait_total_s:
                raise LLMTransportError(
                    f"local-llm-cloud not reachable at {host}:{port} "
                    f"after {_wait_total_s:.0f}s of probing. "
                    f"Start the local-llm-server first (see the Local-LLM "
                    f"README) and retry.  To extend the wait, set "
                    f"LLM_CLOUD_WAIT_S=<seconds> in the environment."
                )
            _sleep_s = min(_delay, _wait_cap, _wait_total_s - _elapsed)
            logger.warning(
                "local-llm-cloud not ready at %s:%s (probe failed, "
                "elapsed %.1fs) — retrying in %.1fs",
                host, port, _elapsed, _sleep_s,
            )
            await asyncio.sleep(_sleep_s)
            _delay = min(_delay * 2.0, _wait_cap)

        api_key = (getattr(self.config, 'api_key', '') or
                   os.environ.get("LLM_CLOUD_API_KEY", ""))
        self._client = get_client(host, port, api_key=api_key,
                                  timeout=self.config.timeout or 300.0)

        # Resolve the model id against the LIVE /v1/models list, not
        # against a string-normalised guess. A file-name hint like
        # "Qwen3-8B-Q4_K_M.gguf" resolves to "qwen3-8b"; an ambiguous
        # hint raises LLMModelNotReady and the caller sees a clear error.
        available_ids: List[str] = []
        try:
            available_ids = [
                m.get("id") for m in self._client.list_models()
                if m.get("id")
            ]
        except Exception as exc:
            logger.warning(
                "LocalLLMCloudAdapter: /v1/models unreachable (%s); "
                "falling back to string-normalised model id — this may "
                "fail with 503 on the first real call.", exc,
            )

        self._model_id = _resolve_model_id(
            self.config.model_name or "", available=available_ids,
        )
        # Persist the resolved id back onto the config so logs and
        # metrics stop reporting the *hint* ("qwen3-8b") when the
        # server is actually serving a different model ("qwen3-4b").
        # The uploaded system_metric.log shows exactly this bug:
        # model_name="qwen3-8b", model_id="qwen3-4b" after the 8B→4B swap.
        if self._model_id and available_ids and self._model_id in available_ids:
            self.config.model_name = self._model_id

        if available_ids and self._model_id not in available_ids:
            logger.warning(
                "⚠️ LocalLLMCloudAdapter: model '%s' is not in the "
                "server's /v1/models list (loaded: %s). Generation will "
                "fail with 503 until it's ready — set autoload: true for "
                "this profile in local-llm-server's config.yaml, or POST "
                "/v1/models/load before the agent runs.",
                self._model_id, available_ids,
            )
        elif available_ids:
            logger.info(
                "LocalLLMCloudAdapter: resolved model hint %r -> %r "
                "(server has %d loaded)",
                self.config.model_name, self._model_id, len(available_ids),
            )

        try:
            self._hw_profile = self._client.get_hardware_profile()
            from amb_sentinel.llm.model_manager import get_backend_orchestrator
            orch = get_backend_orchestrator()
            orch._hw_profile = self._hw_profile
            orch.apply_hw_profile_to_manager()
        except Exception as exc:
            logger.warning("LocalLLMCloudAdapter(fallback): hw profile: %s", exc)

        logger.info("✅ LocalLLMCloudAdapter: ready via fallback (model=%s)",
                    self._model_id)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    async def generate(
            self,
            prompt:      str,
            max_tokens:  Optional[int]   = None,
            temperature: Optional[float] = None,
            timeout:     Optional[float] = None,
    ) -> str:
        if self._shutting_down or self._shutdown_flag:
            raise LLMTransportError("adapter is shutting down")

        # ── Path 1: localllm native adapter ──────────────────────────────
        if self._use_localllm and self._inner is not None:
            try:
                return await self._inner.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
            except LLMError:
                raise
            except Exception as exc:
                logger.error("LocalLLMCloudAdapter(localllm): %s", exc)
                raise LLMTransportError(f"localllm path failed: {exc}") from exc

        # ── Path 2: llm_cloud_adapter.py fallback ────────────────────────
        if self._client is None:
            raise LLMTransportError("LocalLLMCloudAdapter not initialised")

        try:
            from amb_sentinel.llm.llm_cloud_adapter import LLMCloudUnavailable, LLMCloudTimeout
        except ImportError as exc:
            raise LLMTransportError("llm_cloud_adapter not available") from exc

        # NOTE: use `is not None` rather than `or` — max_tokens=0 and
        # temperature=0.0 are both valid, meaningful values (0 tokens =
        # "unlimited" server-side; temperature=0 = greedy decoding). The old
        # `x or default` pattern silently replaced 0 with the fallback
        # because 0 is falsy in Python.
        _max = max_tokens if max_tokens is not None else self.config.max_tokens
        _max = 512 if _max is None else int(_max)

        _temp = temperature if temperature is not None else getattr(self.config, 'temperature', 0.3)
        _temp = 0.3 if _temp is None else float(_temp)

        _timo = float(timeout if timeout is not None else (self.config.timeout or 300.0))

        # force_no_think for Qwen3 on APU profiles
        if (self._hw_profile and
                getattr(self._hw_profile, 'force_no_think', False) and
                "<think>" not in prompt and
                not prompt.rstrip().endswith("<think>\n</think>")):
            prompt = prompt + "\n<think>\n</think>"

        # get_running_loop() is correct inside a coroutine; the
        # deprecated get_event_loop() is being removed in Python 3.12+.
        loop = asyncio.get_running_loop()

        def _sync_gen():
            return self._client.generate(
                self._model_id, prompt,
                max_tokens=_max, temperature=_temp, timeout=_timo)

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _sync_gen), timeout=_timo + 5.0)
            if result.error:
                # Server returned a structured error inside an otherwise
                # successful HTTP response. Classify it so the caller can
                # choose the correct retry policy.
                _err_lower = str(result.error).lower()
                if "timeout" in _err_lower or "timed out" in _err_lower:
                    raise LLMServerTimeout(f"server error: {result.error}")
                if "not ready" in _err_lower or "loading" in _err_lower:
                    raise LLMModelNotReady(f"server error: {result.error}")
                raise LLMBadReply(f"server error: {result.error}")
            return result.text
        except (LLMCloudTimeout, asyncio.TimeoutError) as exc:
            raise LLMServerTimeout(f"Timeout after {_timo:.0f}s") from exc
        except LLMCloudUnavailable as exc:
            raise LLMTransportError(
                f"local-llm-cloud unavailable: {exc}"
            ) from exc
        except LLMError:
            raise
        except Exception as exc:
            logger.error(
                "LocalLLMCloudAdapter(fallback): %s", exc, exc_info=True
            )
            raise LLMTransportError(
                f"Unexpected error: {exc}"
            ) from exc

    async def chat(
            self,
            messages: List[Dict[str, str]],
            *,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            timeout: Optional[float] = None,
            no_think: Optional[bool] = None,
            est_prompt_tokens: int = 0,
            deadline_s: Optional[float] = None,
            decode_tps: Optional[float] = None,
            prefill_tps: Optional[float] = None,
            **kw,
    ):
        """
        Pass `messages` through unchanged. The server applies its own chat
        template (Qwen ChatML for qwen3-8b, etc.).

        Historical note: the pre-A11 path flattened all messages into a
        Mistral [INST] block inside a single user turn, which the Qwen
        server then wrapped again — the model saw literal [INST] tags
        inside a Qwen turn and refused to emit schema-shaped JSON. The
        adaptive runtime (amb_runtime.AgentRun) sends structured messages
        and never hits that path.
        """
        if decode_tps is not None:
            self._decode_tps = float(decode_tps)
        if prefill_tps is not None:
            self._prefill_tps = float(prefill_tps)

        if self._shutting_down or self._shutdown_flag:
            raise LLMTransportError("adapter is shutting down")
        if self._client is None:
            raise LLMTransportError("LocalLLMCloudAdapter not initialised")

        _max = max_tokens if max_tokens is not None else (self.config.max_tokens or 512)
        _temp = temperature if temperature is not None else 0.2
        # ── Compute a deadline that actually fits this prompt on this hardware ──
        if deadline_s is not None:
            _deadline = float(deadline_s)
        else:
            # Prefer the server-provided tps; fall back to the adapter's config.
            _tps = getattr(self, "_decode_tps", None) or \
                   getattr(self._hw_profile, "decode_tps", None) or 2.0
            _pf = getattr(self, "_prefill_tps", None) or \
                  getattr(self._hw_profile, "prefill_tps", None) or (_tps * 2.0)

            # est_prompt_tokens is passed in by AgentRun._do_chat.
            # If it is 0 (direct call), fall back to a coarse char/4 estimate.
            _prompt_toks = int(est_prompt_tokens) or max(1, len(str(messages)) // 4)

            from amb_sentinel.llm.amb_llm_core import estimate_deadline_s
            _deadline = estimate_deadline_s(
                prompt_tokens=_prompt_toks,
                max_new_tokens=int(_max),
                decode_tps=float(_tps),
                prefill_tps=float(_pf),
                safety=1.6,
                floor=120.0,  # never less than 2 min, even for tiny calls
                ceil=7200.0,
            )

        # Bound the outer wait_for by the same deadline (+ small grace).
        _wait_s = _deadline + 30.0
        _sync_timeout = max(30.0, _wait_s - 5.0)

        def _sync():
            return self._client.chat(
                messages,
                max_tokens=int(_max),
                temperature=float(_temp),
                no_think=no_think,
                est_prompt_tokens=int(est_prompt_tokens),
                deadline_s=float(_sync_timeout),
            )

        future = self._io_pool.submit(_sync)
        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=_wait_s,
            )
        except asyncio.TimeoutError as exc:
            # We deliberately do NOT cancel the future: `submit` returns a
            # concurrent.futures.Future whose worker we cannot interrupt from
            # Python. Leaving it lets the worker drain in the background; the
            # pool is a single serial worker, so the next call will not start
            # until this one returns (or the underlying HTTP timeout fires).
            raise LLMServerTimeout(
                f"chat timeout after {_deadline:.0f}s (deadline_s was honoured)"
            ) from exc

    async def chat_stream_iter(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
        no_think: Optional[bool] = None,
        est_prompt_tokens: int = 0,
        first_token_s: Optional[float] = None,
        idle_s: Optional[float] = None,
        **kw,
    ):
        """
        Yield text chunks from the server's SSE stream.

        Only the llm_cloud_adapter fallback path supports streaming —
        the localllm path is not implemented (raises NotImplementedError).

        This method exists so `AgentRun._do_chat` can prefer streaming
        when `profile.allow_streaming` is True. It is NOT required for
        the default non-streaming path.
        """
        if self._use_localllm:
            raise NotImplementedError(
                "chat_stream_iter is only implemented on the "
                "llm_cloud_adapter fallback path"
            )
        if self._client is None:
            raise LLMTransportError("LocalLLMCloudAdapter not initialised")

        _max = max_tokens if max_tokens is not None else (self.config.max_tokens or 512)
        _temp = temperature if temperature is not None else 0.2

        if (self._hw_profile and
                getattr(self._hw_profile, 'force_no_think', False) and
                no_think is None):
            no_think = True

        async for chunk in self._client.astream(
            self._model_id,
            # astream expects a single prompt — join messages into one
            # string. If you need true multi-turn, extend astream to take
            # a messages list (the underlying ChatClient.chat_stream_iter
            # already does).
            prompt="\n\n".join(
                f"### {m.get('role','user')}\n{m.get('content','')}"
                for m in messages
            ),
            max_tokens=int(_max),
            temperature=float(_temp),
            stop=list(stop) if stop else None,
            no_think=no_think,
            first_token_s=first_token_s,
            idle_s=idle_s,
        ):
            yield chunk

    def transport_metrics(self) -> Dict[str, Any]:
        """Expose ChatClient metrics for AIModelService._collect_metrics."""
        if self._client is not None and hasattr(self._client, "metrics"):
            try:
                return self._client.metrics()
            except Exception:
                return {}
        return {}

    async def batch_generate(
            self,
            prompts:     List[str],
            max_tokens:  Optional[int]   = None,
            temperature: Optional[float] = None,
            timeout:     Optional[float] = None,
    ) -> List[str]:
        """Batch — localllm handles concurrency natively."""
        if self._use_localllm and self._inner is not None:
            try:
                return await self._inner.batch_generate(
                    prompts, max_tokens=max_tokens,
                    temperature=temperature, timeout=timeout)
            except Exception as exc:
                logger.error("LocalLLMCloudAdapter: batch_generate error: %s", exc)
        # Sequential fallback
        tasks = [
            self.generate(p, max_tokens=max_tokens,
                          temperature=temperature, timeout=timeout)
            for p in prompts
        ]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    # ------------------------------------------------------------------
    # Health / metrics / shutdown
    # ------------------------------------------------------------------
    def is_healthy(self) -> bool:
        if self._use_localllm and self._inner is not None:
            return self._initialized and (
                getattr(self._inner, 'is_healthy', lambda: True)()
            )
        return self._initialized and self._client is not None

    def get_metrics(self) -> Dict[str, Any]:
        base = super().get_metrics()
        base.update({
            "backend":  "localllm" if self._use_localllm else "llm_cloud_adapter",
            "model_id": self._model_id,
        })
        if self._use_localllm and self._inner is not None:
            try:
                base.update(getattr(self._inner, 'get_metrics', lambda: {})())
            except Exception:
                pass
        return base

    def shutdown(self, timeout: float = 10.0) -> bool:
        super().shutdown(timeout)
        if self._use_localllm and self._inner is not None:
            try:
                self._inner.shutdown()
            except Exception:
                pass
        return True


# ------------------------------------------------------------------
# Module-level helpers used by LocalLLMCloudAdapter
# ------------------------------------------------------------------
def _build_base_url_from_env() -> str:
    """Build base_url from LLM_CLOUD_HOST / LLM_CLOUD_PORT env vars."""
    host = os.environ.get("LLM_CLOUD_HOST", "localhost")
    port = os.environ.get("LLM_CLOUD_PORT", "8080")
    return f"http://{host}:{port}"


def _resolve_model_id(
    model_name: str,
    available: Optional[List[str]] = None,
) -> str:
    """
    Match `model_name` (a GGUF path, a bare id, or empty) against the list
    of ids the server is actually serving.

    Priority:
      1. LLM_CLOUD_MODEL_ID env var (explicit override)
      2. Exact match in `available`
      3. Normalised match against `available` (via amb_llm_core.resolve_model_id)
      4. If `available` has exactly one entry, use it
      5. String-normalised fallback of `model_name` (or "default")

    Callers that have already fetched /v1/models MUST pass the list as
    `available` — the G5 guarantee depends on this. Callers that have not
    yet fetched it should pass `available=None` and rely on ChatClient.
    ensure_model() to do the real resolution later.
    """
    env_id = os.environ.get("LLM_CLOUD_MODEL_ID", "")
    if env_id:
        return env_id
    if not model_name:
        return (available[0] if available else "default")

    p = Path(model_name)
    if p.suffix.lower() == ".gguf" or "/" in model_name:
        stem = p.stem.replace(".", "-")
    else:
        stem = model_name

    if _AMB_LLM_CORE_AVAILABLE and available:
        try:
            from amb_sentinel.llm.amb_llm_core import resolve_model_id as _amb_resolve
            return _amb_resolve(stem, list(available))
        except Exception:
            pass
    elif _AMB_LLM_CORE_AVAILABLE:
        # No live list: fall through to string normalisation only.
        try:
            from amb_sentinel.llm.amb_llm_core import resolve_model_id as _amb_resolve
            return _amb_resolve(stem, [stem])
        except Exception:
            pass
    return stem


class DummyAdapter(BaseLLMAdapter):
    """🤖 Enhanced fallback adapter for testing and when no real backend is available."""

    async def _initialize_impl(self) -> None:
        """⚡ Initialize dummy adapter."""
        logger.warning("⚠️ Using DummyAdapter - no real LLM backend")
        await asyncio.sleep(0.1)

    @property
    def supports_grammar(self):
        return False

    async def generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            timeout: Optional[float] = None
    ) -> str:
        """📝 Generate dummy response with simulated delay."""

        # Add this at the beginning of each adapter's generate() method:
        if hasattr(self, '_shutting_down') and self._shutting_down:
            logger.warning("⚠️ Generation cancelled: adapter is shutting down")
            return "ERROR: Adapter shutting down"

        if hasattr(self, '_shutdown_flag') and self._shutdown_flag:
            logger.warning("⚠️ Generation cancelled: adapter is shutting down")
            return "ERROR: Adapter shutting down"

        logger.info(f"🤖 DummyAdapter generating for prompt: {prompt[:100]}...")
        await asyncio.sleep(0.5)

        return json.dumps({
            "threat_level": "INFORMATIONAL",
            "confidence": 0.0,
            "categories": ["test"],
            "entities": [],
            "reasoning": "Dummy response - no real model available",
            "action_required": False,
            "urgency": "none",
            "metadata": {"source": "dummy_adapter"}
        })


# ---------------------------------------------------------------------
# Enhanced Embeddings & VectorStore (keep as is)
# ---------------------------------------------------------------------
class EmbeddingAdapter:
    """🔤 Enhanced text embeddings with better error handling and caching."""

    def __init__(self,
                 model_name: str = "",
                 cache_size: int = 10000,
                 cache_dir: str = "",
                 device: str = "",
                 timeout: int = 30,
                 enable_download: bool = False,
                 **kwargs
                 ):

        self.model_name = model_name
        self.cache_size = cache_size
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.device = device
        self.enable_download = enable_download

        self._model = None
        self._lock = RLock()
        self._cache = {}
        # [FIX-DEQUE] Was list[]. list.pop(0) is O(n) — with cache_size=10 000
        # that is 10 000 pointer shifts on every eviction. deque.popleft() is O(1).
        self._cache_order = deque()

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)

                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_dir,
                    local_files_only=not self.enable_download,
                )
                logger.info(f"✅ Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load sentence-transformers: {e}")
                self._model = None
        else:
            logger.warning("⚠️ sentence-transformers not available, using fallback embeddings")

        if self._model is not None:
            self.embed = self._embed_with_model
            self.degraded = False
        else:
            self.embed = self._fallback_embed
            self.degraded = True

    def _embed_with_model(self, texts: List[str]) -> List[List[float]]:
        """🔤 Generate embeddings using sentence-transformers with caching."""
        if not texts:
            return []

        cached_results = []
        uncached_texts = []
        uncached_indices = []

        with self._lock:
            for i, text in enumerate(texts):
                if text in self._cache:
                    cached_results.append(self._cache[text])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

        if uncached_texts:
            try:
                new_embeddings = self._model.encode(
                    uncached_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                ).tolist()

                with self._lock:
                    for text, embedding in zip(uncached_texts, new_embeddings, strict=False):
                        self._cache[text] = embedding
                        self._cache_order.append(text)

                        if len(self._cache) > self.cache_size:
                            oldest = self._cache_order.popleft()  # O(1) deque
                            del self._cache[oldest]
            except Exception as e:
                logger.error(f"❌ Embedding generation failed: {e}")
                new_embeddings = self._fallback_embed(uncached_texts)
        else:
            new_embeddings = []

        all_results = [None] * len(texts)
        cache_idx = 0
        new_idx = 0

        for i in range(len(texts)):
            if i in uncached_indices:
                all_results[i] = new_embeddings[new_idx]
                new_idx += 1
            else:
                all_results[i] = cached_results[cache_idx]
                cache_idx += 1

        return all_results

    def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        """🔤 Simple deterministic embedding fallback with better distribution."""
        import hashlib

        embeddings = []
        for text in texts:
            hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()

            embedding = []
            for i in range(384):
                hash_part = hash_bytes[i % 32]
                value = (hash_part / 127.5) - 1.0
                embedding.append(value)

            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]

            embeddings.append(embedding)

        return embeddings

    def get_cache_stats(self) -> Dict[str, Any]:
        """📊 Get cache statistics."""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "cache_hits": sum(1 for _ in self._cache),
                "model_loaded": self._model is not None
            }


class VectorStore:
    """🗄️ Enhanced vector storage with better error handling and performance."""

    def __init__(
            self,
            db_manager: Optional[DatabaseManager] = None,
            connection_name: str = "primary",
            table_name: str = "embeddings",
            embedding_dim: int = 384,
            cache_size: int = 1000,
            pgvector_enabled: bool = True,
            pgvector_dimension: int = 384,
            pgvector_index_type: str = "ivfflat",
            pgvector_lists: int = 100,
            pgvector_probes: int = 10,
            pgvector_operator: str = "vector_cosine_ops",
            degraded_embeddings: bool = False,
            allow_degraded_writes: bool = True,
    ):
        self.db_manager = db_manager
        self.connection_name = connection_name
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.cache_size = cache_size
        self.in_memory_store = {}
        self._initialized = False
        self._cache = {}
        self._cache_order = deque()  # [DEQUE] O(1) popleft vs O(n) pop(0)
        self._lock = RLock()
        self.pgvector_enabled = pgvector_enabled
        self.pgvector_dimension = pgvector_dimension
        self.degraded_embeddings = degraded_embeddings
        self.allow_degraded_writes = allow_degraded_writes
        self.pgvector_index_type = pgvector_index_type
        self.pgvector_lists = pgvector_lists
        self.pgvector_probes = pgvector_probes
        self.pgvector_operator = pgvector_operator

        if self.db_manager and self.pgvector_enabled:
            self._init_pgvector()

    def _init_pgvector(self):
        """⚡ Initialize pgvector extension and table.

        For the default ``embeddings`` table, the schema is owned by
        ``src/amb_sentinel/data/schema.sql`` and applied by
        ``AIModelService._create_database_tables()`` (which calls
        ``amb_sentinel.db.apply_schema``). Duplicating CREATE TABLE here
        would drift the two definitions on future changes, so we skip it
        for the default table. Custom table names still get their own
        table created here.
        """
        if not self.db_manager or not self.pgvector_enabled:
            return
        if self.table_name == "embeddings":
            self._initialized = True
            logger.debug("VectorStore: default 'embeddings' table managed by schema.sql")
            return

        # Validate all config-derived SQL identifiers before any DDL.
        # _db_safe_ident is imported at module level from tools.dbtools;
        # the except branch is an inline fallback for unusual import environments.
        try:
            _safe = _db_safe_ident   # module-level import from tools.dbtools
        except NameError:
            import re as _re
            _IDENT_RE_VS = _re.compile(r'^[A-Za-z_][A-Za-z0-9_$]{0,62}$')
            def _safe(n):
                if not isinstance(n, str) or not _IDENT_RE_VS.match(n):
                    raise ValueError(f"Unsafe SQL identifier: {n!r}")
                return '"' + n.replace('"', '""') + '"'

        try:
            safe_table     = _safe(self.table_name)
            safe_idx_emb   = _safe(f"idx_{self.table_name}_embedding")
            safe_idx_src   = _safe(f"idx_{self.table_name}_source")
            safe_idx_cre   = _safe(f"idx_{self.table_name}_created")
        except ValueError as exc:
            logger.error("❌ VectorStore: unsafe table_name in config: %s", exc)
            self._initialized = False
            return

        # Whitelist pgvector index types
        _VALID_INDEX_TYPES = {"ivfflat", "hnsw", "flat"}
        if self.pgvector_index_type not in _VALID_INDEX_TYPES:
            logger.error("❌ VectorStore: invalid pgvector_index_type '%s'. "
                         "Allowed: %s", self.pgvector_index_type, _VALID_INDEX_TYPES)
            self.pgvector_index_type = "ivfflat"

        # Whitelist pgvector distance operators
        _VALID_OPERATORS = {"vector_cosine_ops", "vector_l2_ops", "vector_ip_ops"}
        if self.pgvector_operator not in _VALID_OPERATORS:
            logger.error("❌ VectorStore: invalid pgvector_operator '%s'. "
                         "Using vector_cosine_ops.", self.pgvector_operator)
            self.pgvector_operator = "vector_cosine_ops"

        # Validate numeric params
        dim    = max(1, int(self.pgvector_dimension))
        lists  = max(1, int(self.pgvector_lists))
        probes = max(1, int(self.pgvector_probes))

        # ------------------------------------------------------------------
        # 1. CHECK IF PGVECTOR EXTENSION IS ALREADY INSTALLED
        # ------------------------------------------------------------------
        def _run_safe(query: str, params: tuple = ()):
            """Safely execute a query and return a standardised result dict."""
            try:
                result = self.db_manager.run(self.connection_name, query, *params)
                # If result is already a dict with 'ok' key, trust it
                if isinstance(result, dict) and 'ok' in result:
                    return result
                # Otherwise wrap success
                return {"ok": True, "result": result, "error": None}
            except Exception as e:
                logger.info(f"DB query failed: {e}")
                return {"ok": False, "error": str(e), "exception": e}

        # ------------------------------------------------------------------
        # 2. VERIFY EXTENSION PRESENCE
        # ------------------------------------------------------------------
        ext_check = _run_safe(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector';"
        )
        extension_exists = False
        if ext_check.get("ok") and ext_check.get("result"):
            # If result is a list of rows, check if non‑empty
            rows = ext_check["result"]
            if isinstance(rows, list) and len(rows) > 0:
                extension_exists = True
            elif isinstance(rows, (tuple, list)) and rows:
                extension_exists = True
            elif rows:  # maybe a single dict
                extension_exists = True

        # ------------------------------------------------------------------
        # 3. ATTEMPT TO CREATE EXTENSION (IF NOT ALREADY PRESENT)
        # ------------------------------------------------------------------
        if not extension_exists:
            logger.info("🔧 pgvector extension not found – attempting to create it...")
            create_result = _run_safe("CREATE EXTENSION IF NOT EXISTS vector;")
            if not create_result.get("ok"):
                error_msg = create_result.get("error", "unknown error")
                logger.error(f"❌ Failed to create pgvector extension: {error_msg}")
                logger.info("ℹ️ Switching to in‑memory cache only – this is safe, but vector search will be local.")
                self._initialized = False
                return
            else:
                logger.info("✅ pgvector extension created successfully.")
        else:
            logger.info("✅ pgvector extension already present – continuing.")

        # ------------------------------------------------------------------
        # 4. ensure embeddings table exists
        # ------------------------------------------------------------------
        create_table_sql = (
            f"CREATE TABLE IF NOT EXISTS {safe_table} (\n"
            f"    id TEXT PRIMARY KEY,\n"
            f"    embedding vector({dim}),\n"
            f"    payload JSONB,\n"
            f"    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n"
            f"    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n"
            f"    source TEXT,\n"
            f"    content_hash TEXT\n"
            f");"
        )
        table_result = _run_safe(create_table_sql)
        if not table_result.get("ok"):
            logger.error(f"❌ Failed to create embeddings table: {table_result.get('error')}")
            logger.info("ℹ️ Switching to in‑memory cache only – table creation failed.")
            self._initialized = False
            return
        logger.info(f"✅ Embeddings table '{self.table_name}' ready.")

        # ------------------------------------------------------------------
        # 5. CREATE VECTOR INDEX (BEST EFFORT – FAILURES ARE NON‑FATAL)
        # ------------------------------------------------------------------
        index_sql = (
            f"CREATE INDEX IF NOT EXISTS {safe_idx_emb} "
            f"ON {safe_table} USING {self.pgvector_index_type} "
            f"(embedding {self.pgvector_operator}) WITH (lists = {lists});"
        )
        index_result = _run_safe(index_sql)
        if index_result.get("ok"):
            logger.info(f"✅ Vector index created (type={self.pgvector_index_type}, lists={self.pgvector_lists})")
        else:
            logger.warning(f"⚠️ Could not create vector index: {index_result.get('error')}")
            # Non‑fatal – we can still query without the index (slower)

        # ------------------------------------------------------------------
        # 6. SET STATISTICS TARGET (OPTIONAL, PERMANENT)
        # ------------------------------------------------------------------
        if probes > 0:
            stats_sql = (
                f"ALTER TABLE {safe_table} "
                f"ALTER embedding SET STATISTICS {probes};"
            )
            stats_result = _run_safe(stats_sql)
            if stats_result.get("ok"):
                logger.info(f"✅ Statistics target set to {probes}")
            else:
                logger.info(f"ℹ️ For optimal recall, set 'ivfflat.probes = {probes}' in your session.")

        # ------------------------------------------------------------------
        # 7. CREATE AUXILIARY INDEXES (NON‑CRITICAL)
        # ------------------------------------------------------------------
        aux_indexes = [
            (f"CREATE INDEX IF NOT EXISTS {safe_idx_src} ON {safe_table}(source);",
             "source index"),
            (f"CREATE INDEX IF NOT EXISTS {safe_idx_cre} ON {safe_table}(created_at DESC);",
             "created_at index")
        ]
        for sql, desc in aux_indexes:
            res = _run_safe(sql)
            if not res.get("ok"):
                logger.info(f"ℹ️ Could not create {desc}: {res.get('error')}")

        # ------------------------------------------------------------------
        # 8. MARK AS INITIALISED
        # ------------------------------------------------------------------
        self._initialized = True
        logger.info(
            f"✅ Vector store fully initialised with pgvector "
            f"(dim={self.pgvector_dimension}, index={self.pgvector_index_type}, "
            f"lists={self.pgvector_lists})"
        )

    @retry(times=3, delay=1.0, exceptions=(Exception,))
    def upsert(self, id: str, embedding: List[float], payload: Dict[str, Any]):
        """💾 Insert or update embedding with payload and caching."""
        if self.degraded_embeddings and not self.allow_degraded_writes:
            logger.warning(
                "VectorStore: refusing to upsert id=%s — embedding backend "
                "is degraded and ai_service.embeddings.allow_degraded_writes "
                "is false. Set it to true to allow fallback (hash) vectors.",
                id,
            )
            return

        with self._lock:
            self._cache[id] = {"embedding": embedding, "payload": payload}
            self._cache_order.append(id)

            if len(self._cache) > self.cache_size:
                oldest = self._cache_order.popleft()  # O(1) deque
                del self._cache[oldest]

        if self.db_manager and self._initialized:
            try:
                result = self.db_manager.execute(self.connection_name, f"""
                    INSERT INTO {self.table_name} (id, embedding, payload, updated_at, 
                                                  source, content_hash)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        payload = EXCLUDED.payload,
                        updated_at = CURRENT_TIMESTAMP,
                        source = EXCLUDED.source,
                        content_hash = EXCLUDED.content_hash;
                """, (
                    id,
                    embedding,
                    json.dumps(payload),
                    payload.get("source", "unknown"),
                    payload.get("content_hash", "")
                ), fetch="none")

                if not result.get("ok"):
                    logger.warning(f"⚠️ Failed to upsert to vectorstore: {result.get('error')}")

            except Exception as e:
                logger.error(f"❌ Failed to upsert to vectorstore: {e}")
        else:
            self.in_memory_store[id] = {
                "embedding": embedding,
                "payload": payload,
                "updated_at": time.time()
            }

    @retry(times=2, delay=1.0, exceptions=(Exception,))
    def query(
            self,
            embedding: List[float],
            top_k: int = 5,
            similarity_threshold: float = 0.7,
            source: Optional[str] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """🔍 Query similar embeddings with enhanced filtering and caching."""
        cached_results = []
        if len(self._cache) > 0:
            cached_results = self._query_cache(embedding, top_k, similarity_threshold, source)

        db_results = []
        if self.db_manager and self._initialized:
            db_results = self._query_database(embedding, top_k, similarity_threshold, source)

        all_results = self._combine_results(cached_results, db_results, top_k)
        return all_results[:top_k]

    def _query_cache(self, embedding: List[float], top_k: int,
                     similarity_threshold: float, source: Optional[str]) -> List[Tuple[str, float, Dict[str, Any]]]:
        """🔍 Query in-memory cache."""
        results = []

        with self._lock:
            for id, data in self._cache.items():
                if source and data["payload"].get("source") != source:
                    continue

                similarity = self._cosine_similarity(embedding, data["embedding"])
                if similarity >= similarity_threshold:
                    results.append((id, similarity, data["payload"]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _query_database(self, embedding: List[float], top_k: int,
                        similarity_threshold: float, source: Optional[str]) -> List[Tuple[str, float, Dict[str, Any]]]:
        """🗄️ Query PostgreSQL database via DatabaseManager."""
        try:
            query = f"""
                SELECT id, 1 - (embedding <=> %s) as similarity, payload
                FROM {self.table_name}
                WHERE 1 - (embedding <=> %s) >= %s
            """
            params = [embedding, embedding, similarity_threshold]

            if source:
                query += " AND source = %s"
                params.append(source)

            query += " ORDER BY embedding <=> %s LIMIT %s;"
            params.extend([embedding, top_k])

            result = self.db_manager.execute(self.connection_name, query, params, fetch="all")
            if not result.get("ok"):
                logger.error(f"❌ Database query failed: {result.get('error')}")
                return []

            return [(row[0], float(row[1]), row[2]) for row in result.get("result", [])]

        except Exception as e:
            logger.error(f"❌ Database query failed: {e}")
            return []

    def _combine_results(self, cached_results: List, db_results: List, top_k: int) -> List:
        """🔄 Combine and deduplicate results from cache and database."""
        seen_ids = set()
        combined = []

        for result in cached_results:
            if result[0] not in seen_ids:
                combined.append(result)
                seen_ids.add(result[0])

        for result in db_results:
            if result[0] not in seen_ids:
                combined.append(result)
                seen_ids.add(result[0])

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """📐 Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_stats(self) -> Dict[str, Any]:
        """📊 Get vector store statistics."""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "memory_store_size": len(self.in_memory_store),
                "pgvector_initialized": self._initialized,
                "embedding_dim": self.embedding_dim
            }


# ---------------------------------------------------------------------
# Enhanced Memory Store (keep as is)
# ---------------------------------------------------------------------
class MemoryStore(ABC):
    """🧠 Abstract base class for conversation memory with enhanced features."""

    @abstractmethod
    def add_message(self, conversation_id: str, speaker: str, text: str,
                    metadata: Optional[Dict[str, Any]] = None):
        pass

    @abstractmethod
    def get_history(
            self,
            conversation_id: str,
            limit: int = 20,
            speaker: Optional[str] = None,
            since: Optional[float] = None
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        pass

    @abstractmethod
    def clear(self, conversation_id: str):
        pass

    @abstractmethod
    def get_conversation_ids(self) -> List[str]:
        pass

    @abstractmethod
    def cleanup_old(self, older_than_days: int = 7):
        pass


class InMemoryMemory(MemoryStore):
    """🧠 Thread-safe in-memory memory store with enhanced features."""

    def __init__(self, max_messages_per_conversation: int = 1000, max_conversations: int = 1000):
        self.store: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
        self._lock = RLock()
        self.max_messages_per_conversation = max_messages_per_conversation
        self.max_conversations = max_conversations

    def add_message(
            self,
            conversation_id: str,
            speaker: str,
            text: str,
            metadata: Dict[str, Any] = None
    ):
        """📝 Add message with timestamp and cleanup old conversations."""
        metadata = metadata or {}
        metadata['timestamp'] = time.time()
        metadata['speaker'] = speaker

        with self._lock:
            if conversation_id not in self.store:
                if len(self.store) >= self.max_conversations:
                    oldest_id = min(self.store.items(),
                                    key=lambda x: x[1][-1][2].get('timestamp', 0) if x[1] else 0)[0]
                    del self.store[oldest_id]
                    logger.info(f"🗑️ Removed oldest conversation: {oldest_id}")

                self.store[conversation_id] = []

            self.store[conversation_id].append((speaker, text, metadata))

            if len(self.store[conversation_id]) > self.max_messages_per_conversation:
                self.store[conversation_id] = self.store[conversation_id][-self.max_messages_per_conversation:]
                logger.info(f"✂️ Trimmed conversation {conversation_id}")

    def get_history(
            self,
            conversation_id: str,
            limit: int = 20,
            speaker: Optional[str] = None,
            since: Optional[float] = None
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """📜 Get conversation history with filtering."""
        with self._lock:
            messages = self.store.get(conversation_id, [])

            filtered = []
            for speaker_name, text, meta in messages:
                if speaker and speaker_name != speaker:
                    continue
                if since and meta.get('timestamp', 0) < since:
                    continue
                filtered.append((speaker_name, text, meta))

            return filtered[-limit:]

    def clear(self, conversation_id: str):
        """🧹 Clear specific conversation."""
        with self._lock:
            if conversation_id in self.store:
                del self.store[conversation_id]

    def get_conversation_ids(self) -> List[str]:
        """📋 Get list of all conversation IDs."""
        with self._lock:
            return list(self.store.keys())

    def cleanup_old(self, older_than_days: int = 7):
        """🧹 Clean up conversations older than specified days."""
        cutoff = time.time() - (older_than_days * 24 * 3600)

        with self._lock:
            to_remove = []
            for conv_id, messages in self.store.items():
                if messages:
                    last_timestamp = messages[-1][2].get('timestamp', 0)
                    if last_timestamp < cutoff:
                        to_remove.append(conv_id)

            for conv_id in to_remove:
                del self.store[conv_id]

            if to_remove:
                logger.info(f"🧹 Cleaned up {len(to_remove)} old conversations")

    # Add to InMemoryMemory class in ai_agent.py

    def checkpoint_execution_state(
            self,
            task_id: str,
            state: Dict[str, Any]
    ) -> bool:
        """Save execution state for recovery."""
        try:
            # Store as conversation checkpoint
            self.add_message(
                task_id,
                "system",
                json.dumps(state),
                metadata={
                    'type': 'execution_checkpoint',
                    'timestamp': time.time()
                }
            )
            return True
        except Exception as e:
            logger.error(f"❌ Checkpoint failed: {e}")
            return False

    def get_execution_checkpoint(
            self,
            task_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve last execution checkpoint."""
        try:
            history = self.get_history(task_id, limit=100)
            for speaker, text, meta in reversed(history):
                if meta.get('type') == 'execution_checkpoint':
                    return json.loads(text)
        except Exception:
            pass
        return None

    def get_completed_steps(self, task_id: str) -> List[str]:
        """Get list of successfully completed steps."""
        completed = []
        history = self.get_history(task_id, limit=1000)
        for speaker, text, meta in history:
            if meta.get('step_status') == 'completed':
                completed.append(meta.get('step_id'))
        return completed


if _OPTIONAL_IMPORTS['redis']:
    class RedisMemory(MemoryStore):
        """🔴 Redis-backed memory store with enhanced connection pooling and error handling."""

        def __init__(self, config: RedisConfig, max_messages: int = 1000):
            self.config = config
            self.max_messages = max_messages
            self._pool = None
            self._lock = RLock()
            self._initialize_pool()

        def _initialize_pool(self):
            """🔌 Initialize Redis connection pool with retry logic."""
            try:
                self._pool = redis.ConnectionPool(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    max_connections=self.config.max_connections,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    decode_responses=True
                )
                client = redis.Redis(connection_pool=self._pool)
                client.ping()
                logger.info("✅ Redis memory store initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Redis connection pool: {e}")
                self._pool = None

        def _get_client(self) -> Optional[redis.Redis]:
            """🔌 Get Redis client with error handling."""
            if not self._pool:
                self._initialize_pool()
                if not self._pool:
                    return None

            try:
                return redis.Redis(connection_pool=self._pool)
            except Exception as e:
                logger.error(f"❌ Failed to get Redis client: {e}")
                return None

        @retry(times=3, delay=1.0, exceptions=(redis.RedisError,))
        def add_message(
                self,
                conversation_id: str,
                speaker: str,
                text: str,
                metadata: Dict[str, Any] = None
        ):
            """📝 Add message to Redis with atomic operations."""
            metadata = metadata or {}
            metadata.update({
                'timestamp': time.time(),
                'speaker': speaker
            })

            client = self._get_client()
            if not client:
                raise ConnectionError("Redis connection not available")

            key = f"ai_memory:{conversation_id}"
            message_data = json.dumps({
                "speaker": speaker,
                "text": text,
                "metadata": metadata
            })

            with client.pipeline() as pipe:
                pipe.rpush(key, message_data)
                pipe.ltrim(key, -self.max_messages, -1)
                pipe.expire(key, 86400 * 30)
                pipe.execute()

        @retry(times=2, delay=1.0, exceptions=(redis.RedisError,))
        def get_history(
                self,
                conversation_id: str,
                limit: int = 20,
                speaker: Optional[str] = None,
                since: Optional[float] = None
        ) -> List[Tuple[str, str, Dict[str, Any]]]:
            """📜 Get conversation history from Redis with filtering."""
            client = self._get_client()
            if not client:
                return []

            key = f"ai_memory:{conversation_id}"

            try:
                messages = client.lrange(key, -limit, -1)
                results = []

                for msg in messages:
                    try:
                        data = json.loads(msg)

                        if speaker and data.get("speaker") != speaker:
                            continue

                        msg_timestamp = data.get("metadata", {}).get("timestamp", 0)
                        if since and msg_timestamp < since:
                            continue

                        results.append((
                            data.get("speaker", ""),
                            data.get("text", ""),
                            data.get("metadata", {})
                        ))
                    except (json.JSONDecodeError, KeyError):
                        continue

                return results
            except redis.RedisError as e:
                logger.error(f"❌ Redis error fetching history: {e}")
                return []

        def clear(self, conversation_id: str):
            """🧹 Clear specific conversation from Redis."""
            client = self._get_client()
            if not client:
                return

            key = f"ai_memory:{conversation_id}"
            try:
                client.delete(key)
            except redis.RedisError as e:
                logger.error(f"❌ Redis error clearing conversation: {e}")

        def get_conversation_ids(self) -> List[str]:
            """📋 Get list of all conversation IDs from Redis."""
            client = self._get_client()
            if not client:
                return []

            try:
                pattern = "ai_memory:*"
                keys = client.keys(pattern)
                return [key.decode().replace("ai_memory:", "") if isinstance(key, bytes)
                        else key.replace("ai_memory:", "") for key in keys]
            except redis.RedisError as e:
                logger.error(f"❌ Redis error getting conversation IDs: {e}")
                return []

        def cleanup_old(self, older_than_days: int = 7):
            """🧹 Clean up old conversations from Redis."""
            client = self._get_client()
            if not client:
                return

            cutoff = time.time() - (older_than_days * 24 * 3600)
            pattern = "ai_memory:*"

            try:
                keys = client.keys(pattern)
                for key in keys:
                    try:
                        last_msg = client.lrange(key, -1, -1)
                        if last_msg:
                            data = json.loads(last_msg[0])
                            last_timestamp = data.get("metadata", {}).get("timestamp", 0)

                            if last_timestamp < cutoff:
                                client.delete(key)
                    except (json.JSONDecodeError, KeyError):
                        continue
            except redis.RedisError as e:
                logger.error(f"❌ Redis error during cleanup: {e}")


class AdaptiveConcurrency:
    """
    Dynamically adjusts the number of concurrent operations based on
    system memory and CPU load. Uses a condition variable to manage
    waiting tasks without losing them when the limit changes.
    """

    def __init__(self, initial_limit: int, min_limit: int = 1,
                 mem_high_threshold: float = 80.0, cpu_high_threshold: float = 80.0,
                 adjustment_factor: float = 0.8):
        """
        Args:
            initial_limit: Starting concurrency limit.
            min_limit: Minimum allowed concurrency (never go below this).
            mem_high_threshold: Memory % above which we start reducing.
            cpu_high_threshold: CPU % above which we start reducing.
            adjustment_factor: Factor to multiply current limit by when reducing.
        """
        self._limit = initial_limit
        self._min_limit = min_limit
        self._mem_high = mem_high_threshold
        self._cpu_high = cpu_high_threshold
        self._factor = adjustment_factor

        self._count = 0  # currently active acquisitions
        self._condition = asyncio.Condition()
        self._adjusting = False  # prevent concurrent adjustments

    @property
    def limit(self) -> int:
        return self._limit

    async def acquire(self):
        """
        Acquire a permit, waiting if at limit. Adjusts the limit before waiting.
        """
        # Before waiting, check if we should adjust down
        await self._maybe_adjust()

        async with self._condition:
            while self._count >= self._limit:
                await self._condition.wait()
            self._count += 1

    async def release(self):
        async with self._condition:
            self._count -= 1
            self._condition.notify(1)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()

    async def _maybe_adjust(self):
        """Check system load and reduce limit if needed."""
        # Avoid concurrent adjustments
        if self._adjusting:
            return
        self._adjusting = True
        try:
            mem = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent(interval=0.1)  # quick sample

            if mem > self._mem_high or cpu > self._cpu_high:
                new_limit = max(self._min_limit, int(self._limit * self._factor))
                if new_limit != self._limit:
                    logger.warning(
                        f"📉 Reducing concurrency: {self._limit} → {new_limit} "
                        f"(mem={mem:.1f}%, cpu={cpu:.1f}%)"
                    )
                    async with self._condition:
                        self._limit = new_limit
                        # Notify all waiters so they can recheck the new limit
                        self._condition.notify_all()
        finally:
            self._adjusting = False

class ExecutionBackupRetrier:
    """
    Periodically scans the backup directory and retries inserting execution backups
    that previously failed. Deletes the file on successful insertion.
    """

    def __init__(self, service, backup_dir: Path, check_interval: int = 3600):
        self.service = service
        self.backup_dir = Path(backup_dir)
        self.check_interval = check_interval
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.logger = logging.getLogger("BackupRetrier")

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self.logger.info(f"🔁 Backup retrier started, checking every {self.check_interval}s")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.logger.info("🛑 Backup retrier stopped")

    async def _run_loop(self):
        while self._running:
            try:
                await self._process_backups()
            except Exception as e:
                self.logger.error(f"Error in backup processing: {e}")
            await asyncio.sleep(self.check_interval)

    async def _process_backups(self):
        if not self.service.db_manager or self.service._shutdown:
            return

        backup_files = list(self.backup_dir.glob("exec_*.json"))
        if not backup_files:
            return

        self.logger.info(f"Found {len(backup_files)} execution backup(s) to retry")

        for file in backup_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)

                # Reconstruct the original result structure
                execution_id = data.get("conversation_id")
                instruction = data.get("instruction")
                result = data.get("result", {})
                timestamp = data.get("timestamp")

                # Attempt to re-insert using the same logic as _log_agent_execution_to_db
                success = await self._retry_insert(execution_id, instruction, result)
                if success:
                    file.unlink()  # Delete on success
                    self.logger.info(f"✅ Successfully re‑inserted backup {file.name}")
                else:
                    self.logger.warning(f"⚠️ Backup {file.name} still failing, will retry later")
            except Exception as e:
                self.logger.error(f"Error processing backup {file.name}: {e}")

    async def _retry_insert(self, execution_id: str, instruction: str, result: Dict) -> bool:
        """
        Re‑insert a single execution record using the database manager.
        Returns True on success, False otherwise.
        """
        # Prepare parameters (same as in _log_agent_execution_to_db)
        metrics = result.get('metrics', {})
        tools_used = list(set(
            step.get('tool') for step in result.get('steps', [])
            if step.get('type') == 'tool_execution' and step.get('tool')
        ))
        params = (
            execution_id,
            execution_id,
            instruction[:500],
            "completed" if result.get('success') else "failed",
            json.dumps(result.get('result', '')),
            result.get('success', False),
            metrics.get('total_duration', 0),
            metrics.get('iterations', 0),
            metrics.get('tokens_used', 0),
            len(result.get('steps', [])),
            json.dumps(tools_used),
            result.get('error', ''),
            json.dumps(result.get('metrics', {}))
        )

        try:
            res = self.service.db_manager.execute(
                "primary",
                """
                INSERT INTO agent_executions 
                (execution_id, conversation_id, instruction, status, result, success,
                 total_duration, iterations, tokens_used, steps_count, tools_used,
                 error_message, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                params,
                fetch="none"
            )
            return res.get("ok", False)
        except Exception as e:
            self.logger.error(f"Retry insert failed: {e}")
            return False


@dataclass
class ToolCall:
    tool_name: str
    args: Dict[str, Any]
    priority: int = 5
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class ExecutionResult:
    tool_name: str
    success: bool
    result: Any
    duration: float
    error: Optional[str] = None
    tokens_used: int = 0
    insight: Optional[str] = None

class ToolIntelligenceEngine:
    def __init__(self):
        self.metrics: Dict[str, ToolPerformanceMetrics] = defaultdict(
            lambda: ToolPerformanceMetrics(tool_name="")
        )
        self.tool_to_goals: Dict[str, List[str]] = defaultdict(list)
        self.goal_to_tools: Dict[str, List[str]] = defaultdict(list)

    def record_success(self, tool_name: str, duration: float, goal_type: str):
        metrics = self.metrics[tool_name]
        metrics.success_count += 1
        metrics.avg_duration = (
                (metrics.avg_duration * (metrics.success_count - 1) + duration)
                / metrics.success_count
        )
        metrics.last_used = time.time()
        if goal_type and goal_type not in metrics.recommended_for:
            metrics.recommended_for.append(goal_type)

    def record_failure(self, tool_name: str, reason: str):
        self.metrics[tool_name].failure_count += 1
        self.metrics[tool_name].failure_reasons.append(reason)
        self.metrics[tool_name].last_used = time.time()

    def filter_tools_for_goal(
            self,
            goal: MicroGoal,
            all_tools: List[str],
            tokens_remaining: int,
            memory_percent: float
    ) -> List[Tuple[str, float]]:
        scored_tools = []
        for tool_name in all_tools:
            score = 0.0
            if tool_name in goal.required_tools:
                score += 100
            elif tool_name in goal.optional_tools:
                score += 50
            metrics = self.metrics[tool_name]
            score += metrics.success_rate * 30
            if metrics.is_healthy:
                score += 20
            else:
                score -= 20
            if metrics.avg_duration > 0:
                speed_score = max(0, 20 - (metrics.avg_duration / 10))
                score += speed_score
            if any(x in tool_name.lower() for x in ['search', 'query', 'get']):
                score += 15
            if memory_percent > 80:
                if any(x in tool_name.lower() for x in ['download', 'video', 'transcribe']):
                    score -= 50
            scored_tools.append((tool_name, score))
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [(t, s) for t, s in scored_tools if s > 0][:10]

# ---------------------------------------------------------------------
# Enhanced Tools & Agent Executor
# ---------------------------------------------------------------------
@dataclass
class Tool:
    """🛠️ Enhanced tool definition with validation and metrics."""
    name: str
    func: Callable[..., Any]
    description: str = ""
    timeout: float = 10.0
    retries: int = 2
    validate_args: bool = True
    schema: Optional[Dict[str, Any]] = None
    rate_limit: Optional[float] = None
    operation_type: Optional[str] = None  # e.g., "network", "disk", "compute", "mixed"
    cost_per_unit: Optional[float] = None  # e.g., seconds per MB for downloads
    provides: Optional[List[str]] = None  # Names of outputs this tool produces
    consumes: Optional[List[str]] = None  # Names of inputs this tool requires (from other tools)

    def __post_init__(self):
        """🔍 Validate tool configuration."""
        if self.timeout <= 0:
            raise ValueError(f"Tool timeout must be positive, got {self.timeout}")

        if self.retries < 0:
            raise ValueError(f"Tool retries must be non-negative, got {self.retries}")


class ToolRegistry:
    """📚 Enhanced registry for managing tools with metrics and rate limiting."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._lock = RLock()
        self._metrics: Dict[str, Dict[str, Any]] = {}
        self._rate_limiters: Dict[str, Dict[str, float]] = {}

    def register(self, tool: Tool):
        """📝 Register a tool with metrics tracking."""
        with self._lock:
            if tool.name in self.tools:
                logger.warning(f"⚠️ Tool '{tool.name}' already registered, overwriting")

            tool.registry = self
            self.tools[tool.name] = tool
            self._metrics[tool.name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0.0,
                "last_call": None
            }

            if tool.rate_limit:
                self._rate_limiters[tool.name] = {
                    "last_call": 0.0,
                    "min_interval": 1.0 / tool.rate_limit
                }

            logger.info(f"✅ Registered tool: {tool.name}")

    def register_function(self, func: Callable, **kwargs):
        tool = Tool(
            name=kwargs.get("name", func.__name__),
            func=func,
            description=kwargs.get("description", func.__doc__ or ""),
            timeout=kwargs.get("timeout", 10.0),
            retries=kwargs.get("retries", 2),
            validate_args=kwargs.get("validate_args", True),
            schema=kwargs.get("schema"),
            rate_limit=kwargs.get("rate_limit"),
            operation_type=kwargs.get("operation_type"),
            cost_per_unit=kwargs.get("cost_per_unit"),
            provides=kwargs.get("provides"),
            consumes=kwargs.get("consumes")
        )
        self.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """🔍 Get tool by name with access tracking."""
        tool = self.tools.get(name)
        if tool:
            with self._lock:
                self._metrics[name]["calls"] += 1
                self._metrics[name]["last_call"] = time.time()
        return tool

    def list_tools(self) -> List[Dict[str, Any]]:
        """📋 List all registered tools with metrics."""
        tools_info = []
        with self._lock:
            for name, tool in self.tools.items():
                metrics = self._metrics.get(name, {})
                tools_info.append({
                    "name": tool.name,
                    "description": tool.description,
                    "timeout": tool.timeout,
                    "retries": tool.retries,
                    "calls": metrics.get("calls", 0),
                    "successes": metrics.get("successes", 0),
                    "failures": metrics.get("failures", 0)
                })
        return tools_info

    def _check_rate_limit(self, tool_name: str) -> bool:
        """⏱️ Check if tool is within rate limits."""
        if tool_name not in self._rate_limiters:
            return True

        limiter = self._rate_limiters[tool_name]
        now = time.time()
        elapsed = now - limiter["last_call"]

        if elapsed < limiter["min_interval"]:
            logger.warning(f"⚠️ Rate limit exceeded for tool {tool_name}")
            return False

        limiter["last_call"] = now
        return True

    def record_success(self, tool_name: str, duration: float):
        """✅ Record successful tool execution."""
        with self._lock:
            if tool_name in self._metrics:
                self._metrics[tool_name]["successes"] += 1
                self._metrics[tool_name]["total_time"] += duration

    def record_failure(self, tool_name: str, duration: float):
        """❌ Record failed tool execution."""
        with self._lock:
            if tool_name in self._metrics:
                self._metrics[tool_name]["failures"] += 1
                self._metrics[tool_name]["total_time"] += duration


class InstructionValidator:
    """
    Comprehensive instruction validation with:
    - Length and basic safety checks
    - Contradiction detection (mutually exclusive actions)
    - Unrealistic/forbidden action patterns (configurable)
    - Tool existence verification
    - Parameter completeness (using tool schemas)
    - Circular goal dependency detection
    - Feasibility warnings (non‑blocking)
    """

    # Built‑in forbidden patterns (can be extended via config)
    DEFAULT_FORBIDDEN_PATTERNS = [
        (r'(?:hack|crack|breach).*(?:secure|protected|encrypted)',
         "Attempting to hack/breach secure systems is not supported"),
        (r'(?:remove|delete|destroy).*(?:firewall|security|antivirus)',
         "Cannot remove security controls"),
        (r'(?:bypass|circumvent).*(?:authentication|login|password)',
         "Authentication bypass is not allowed"),
        (r'(?:inject|exploit).*(?:sql|xss|command)',
         "Security exploit attempts are forbidden"),
    ]

    # Mutually exclusive action pairs (if both appear, instruction is invalid)
    MUTUALLY_EXCLUSIVE = [
        (r'delete|remove|destroy', r'save|store|keep'),
        (r'encrypt', r'decrypt'),
        (r'increase|raise', r'decrease|lower'),
    ]

    def __init__(self, tool_registry=None, forbidden_patterns: Optional[List[Tuple[str, str]]] = None):
        self.tool_registry = tool_registry
        # Start with built‑in default patterns
        self.forbidden_patterns = list(self.DEFAULT_FORBIDDEN_PATTERNS)

        # Append custom patterns from config
        if forbidden_patterns:
            for pattern, reason in forbidden_patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                    self.forbidden_patterns.append((compiled, reason))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")

        # logger = logging.getLogger("InstructionValidator")

    @classmethod
    def validate(cls, instruction: str, tool_registry=None) -> Tuple[bool, Optional[str]]:
        """
        Public validation method. Returns (is_valid, error_message).
        If error_message is None, validation passed.
        """
        validator = cls(tool_registry)
        return validator._validate(instruction)

    def _validate(self, instruction: str) -> Tuple[bool, Optional[str]]:
        # 1. Basic length checks
        instr = instruction.strip()
        if len(instr) < 20:
            return False, "Instruction too short (minimum 20 characters)"
        if len(instr) > 20000:  # generous upper limit
            return False, "Instruction too long (maximum 20000 characters)"

        instr_lower = instr.lower()

        # 2. Check for obvious safety violations (original patterns)
        original_forbidden = [
            r'exec\s*\(', r'eval\s*\(', r'__import__',
            r'os\.system', r'subprocess\.',
            r'open\s*\(\s*["\']/', r'os\.remove', r'os\.unlink',
            r'password\s*=\s*["\'][^"\']*["\']',  # ← improved
            r'(?:my_?)?password\s*=',  # ← added
            r'credential\s*=\s*["\']',
            r'private\s*key\s*=\s*["\']',
            r'secret\s*key\s*=\s*["\']',
            r'rm\s+-rf',
            r'format\s+c:',
            r'del\s+/s\s+/q',
            r'sudo', r'chmod\s+777',
            r'chown\s+-R',
            r'DROP\s+TABLE', r'DELETE\s+FROM', r'TRUNCATE',
            r'socket\.socket', r'raw_socket',
        ]
        for pattern in original_forbidden:
            if re.search(pattern, instr_lower):
                return False, f"Instruction contains dangerous pattern: {pattern}"

        # 3. Check for mutually exclusive actions
        for pat_a, pat_b in self.MUTUALLY_EXCLUSIVE:
            has_a = re.search(pat_a, instr_lower)
            has_b = re.search(pat_b, instr_lower)
            if has_a and has_b:
                return False, f"Contradictory instructions: '{pat_a}' and '{pat_b}' cannot both be present"

        # 4. Check for unrealistic/forbidden actions
        for pattern, reason in self.forbidden_patterns:
            if re.search(pattern, instr_lower):
                return False, reason

        # 5. Extract goals/steps for further analysis
        goals = self._extract_goals(instr)
        goal_titles = [g['title'] for g in goals]

        # 6. Check for duplicate goals (possible circular dependency)
        if len(goal_titles) != len(set(goal_titles)):
            return False, "Duplicate goals detected – may cause circular execution"

        # 7. If tool registry is available, validate tool names mentioned in the instruction
        if self.tool_registry:
            mentioned_tools = self._extract_tool_names(instr)
            unknown = [t for t in mentioned_tools if not self.tool_registry.get(t)]
            if unknown:
                return False, f"Unknown tool(s) referenced: {', '.join(unknown)}"

        # 8. Parameter completeness (optional – can be a warning rather than error)
        #    For now, we'll only produce a warning log, not block execution.
        param_issues = self._check_parameter_completeness(instr_lower, goals)
        if param_issues:
            logger.warning(f"Instruction may be incomplete: {param_issues}")

        # 9. Feasibility hints (e.g., extremely long instruction may cause timeout)
        word_count = len(instr.split())
        if word_count > 2000:
            logger.warning("Very long instruction – may exceed context or time out")

        return True, None

    def _extract_goals(self, instruction: str) -> List[Dict]:
        """Extract numbered goals/steps from instruction."""
        goals = []
        # Simple regex for lines starting with number and a period
        for line in instruction.split('\n'):
            line = line.strip()
            m = re.match(r'^(\d+)\.\s+(.*)', line)
            if m:
                goals.append({'number': int(m.group(1)), 'title': m.group(2).strip()})
        return goals

    def _extract_tool_names(self, instruction: str) -> Set[str]:
        """Find all registered tool names mentioned in the instruction."""
        if not self.tool_registry:
            return set()
        # Use a pre‑compiled regex for all tool names (case‑insensitive)
        tool_names = list(self.tool_registry.tools.keys())
        # Escape names to avoid regex issues, then join with |
        pattern = re.compile(r'\b(' + '|'.join(re.escape(name) for name in tool_names) + r')\b', re.I)
        matches = pattern.findall(instruction)
        return set(matches)

    def _check_parameter_completeness(self, instr_lower: str, goals: List[Dict]) -> Optional[str]:
        """
        Check if actions that require parameters (e.g., download needs a URL)
        have those parameters mentioned in the instruction.
        Returns a warning string or None.
        """
        issues = []
        # Very basic – just check for keywords
        if re.search(r'download|fetch', instr_lower) and not re.search(r'url|link|source', instr_lower):
            issues.append("Download/fetch actions usually need a URL or source")
        if re.search(r'analyze|process', instr_lower) and not re.search(r'file|data|content', instr_lower):
            issues.append("Analyze/process actions usually need a data source")
        if issues:
            return "; ".join(issues)
        return None



# =====================================================================
# EXECUTION STATE MANAGEMENT
# =====================================================================

class ExecutionPhase(Enum):
    """Execution phases for state machine."""
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    PROGRESS_TRACKING = "progress_tracking"
    COMPLETION = "completion"
    ERROR = "error"


class GoalStatus(Enum):
    """Goal completion status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class TaskGoal:
    """Represents a single task goal with tracking."""
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: int = 5  # 1-10, higher = more important
    suggested_tools: List[str] = field(default_factory=list)
    completed_at: Optional[float] = None
    completed_by: Optional[str] = None  # which tool completed it
    attempts: int = 0
    max_attempts: int = 3
    success_metric: Optional[str] = None  # how to measure success
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_completed(self, tool_name: str = ""):
        """Mark goal as completed."""
        self.status = GoalStatus.COMPLETED
        self.completed_at = time.time()
        self.completed_by = tool_name

    def mark_in_progress(self):
        """Mark goal as in progress."""
        self.status = GoalStatus.IN_PROGRESS
        self.attempts += 1

    def mark_failed(self):
        """Mark goal as failed."""
        if self.attempts >= self.max_attempts:
            self.status = GoalStatus.FAILED
        else:
            self.status = GoalStatus.PENDING

    def is_achievable(self) -> bool:
        """Check if goal can still be attempted."""
        return self.status in [GoalStatus.PENDING, GoalStatus.IN_PROGRESS]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "status": self.status.value,
            "priority": self.priority,
            "attempts": self.attempts,
            "suggested_tools": self.suggested_tools,
            "completed_at": self.completed_at
        }


class AdaptiveBackoff:
    """
    Exponential backoff with jitter for retrying failed operations.
    Tracks consecutive failures and provides an async wait method.
    """

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0,
                 max_consecutive: int = 5, name: str = "default"):
        """
        Args:
            base_delay: Initial delay (seconds) after first failure.
            max_delay: Maximum delay (seconds) to wait.
            max_consecutive: Max consecutive failures before giving up.
            name: Identifier for logging.
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_consecutive = max_consecutive
        self.name = name
        self._consecutive_failures = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def record_success(self):
        """Reset failure count after a success."""
        if self._consecutive_failures > 0:
            logger.info(f"Backoff '{self.name}' reset after success")
        self._consecutive_failures = 0

    def record_failure(self):
        """Increment failure count."""
        self._consecutive_failures += 1
        logger.info(f"Backoff '{self.name}' failures: {self._consecutive_failures}")

    def should_continue(self) -> bool:
        """Return True if we haven't exceeded max consecutive failures."""
        return self._consecutive_failures < self.max_consecutive

    async def wait(self):
        """
        Wait with exponential backoff + jitter.
        Does nothing if there are no failures.
        """
        if self._consecutive_failures == 0:
            return

        # Calculate exponential delay
        exp_delay = self.base_delay * (2 ** (self._consecutive_failures - 1))
        delay = min(exp_delay, self.max_delay)

        # Add jitter (±10%)
        jitter = random.uniform(-delay * 0.1, delay * 0.1)
        total_delay = delay + jitter

        logger.info(f"Backoff '{self.name}': waiting {total_delay:.2f}s "
                    f"(failures={self._consecutive_failures})")
        await asyncio.sleep(max(0, total_delay))


@dataclass
class ExecutionContext:
    """Execution context for agent runs."""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_instruction: str = ""
    goals: List[TaskGoal] = field(default_factory=list)
    current_phase: ExecutionPhase = ExecutionPhase.INITIALIZATION
    iteration: int = 0
    max_iterations: int = 10
    start_time: float = field(default_factory=time.time)
    timeout: float = 600.0  # 10 minutes default
    success: bool = False
    error_message: str = ""

    def is_timed_out(self) -> bool:
        """Check if execution has timed out."""
        return (time.time() - self.start_time) > self.timeout

    def get_elapsed_time(self) -> float:
        """Get elapsed execution time."""
        return time.time() - self.start_time

    def get_remaining_time(self) -> float:
        """Get remaining execution time."""
        return max(0, self.timeout - self.get_elapsed_time())

    def progress_percentage(self) -> float:
        """Get execution progress as percentage."""
        if not self.goals:
            return 0.0
        completed = sum(1 for g in self.goals if g.status == GoalStatus.COMPLETED)
        return (completed / len(self.goals)) * 100

    def goals_summary(self) -> Dict[str, int]:
        """Get summary of goal statuses."""
        return {
            "total": len(self.goals),
            "completed": sum(1 for g in self.goals if g.status == GoalStatus.COMPLETED),
            "in_progress": sum(1 for g in self.goals if g.status == GoalStatus.IN_PROGRESS),
            "pending": sum(1 for g in self.goals if g.status == GoalStatus.PENDING),
            "failed": sum(1 for g in self.goals if g.status == GoalStatus.FAILED)
        }


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation, failures counted
    OPEN = "open"  # Failing, no executions allowed
    HALF_OPEN = "half_open"  # Testing if problem resolved


class ToolCircuitBreaker:
    """
    Circuit breaker for tools to prevent repeated execution of failing operations.
    Thread‑safe, with configurable thresholds and cooldown.
    """

    def __init__(self,
                 failure_threshold: int = 3,
                 reset_timeout: float = 300.0,
                 half_open_max_calls: int = 1,
                 exclude_exceptions: Optional[Set[type]] = None):
        """
        Args:
            failure_threshold: Number of consecutive failures before opening circuit.
            reset_timeout: Seconds to wait before moving from OPEN to HALF_OPEN.
            half_open_max_calls: Number of test calls allowed in HALF_OPEN state.
            exclude_exceptions: Exception types that should NOT open the circuit.
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.exclude_exceptions = exclude_exceptions or set()

        self._lock = threading.RLock()
        self._failure_counts: Dict[str, int] = {}  # tool_name -> count
        self._circuit_state: Dict[str, CircuitState] = {}  # tool_name -> state
        self._open_time: Dict[str, float] = {}  # when circuit was opened
        self._half_open_calls: Dict[str, int] = {}  # number of test calls made

        # logger = logging.getLogger("CircuitBreaker")

    def can_execute(self, tool_name: str) -> bool:
        """
        Check if a tool can be executed now.
        Returns True if allowed, False if circuit is open.
        """
        with self._lock:
            state = self._circuit_state.get(tool_name, CircuitState.CLOSED)

            if state == CircuitState.CLOSED:
                return True

            if state == CircuitState.OPEN:
                # Check if reset timeout has elapsed
                opened_at = self._open_time.get(tool_name)
                if opened_at and (time.time() - opened_at) > self.reset_timeout:
                    # Move to HALF_OPEN and allow a test call
                    self._circuit_state[tool_name] = CircuitState.HALF_OPEN
                    self._half_open_calls[tool_name] = 0
                    logger.info(f"🔄 Circuit breaker for '{tool_name}' moved to HALF_OPEN")
                    return True
                return False

            if state == CircuitState.HALF_OPEN:
                # Allow a limited number of test calls
                calls = self._half_open_calls.get(tool_name, 0)
                if calls < self.half_open_max_calls:
                    return True
                else:
                    # Too many test calls already in flight; reject
                    return False

        return False  # unreachable

    def record_success(self, tool_name: str):
        """Record a successful execution. Resets failures and closes circuit."""
        with self._lock:
            self._failure_counts.pop(tool_name, None)
            old_state = self._circuit_state.get(tool_name, CircuitState.CLOSED)
            if old_state != CircuitState.CLOSED:
                logger.info(f"✅ Circuit breaker for '{tool_name}' CLOSED (success)")
            self._circuit_state.pop(tool_name, None)
            self._open_time.pop(tool_name, None)
            self._half_open_calls.pop(tool_name, None)

    def record_failure(self, tool_name: str, exception: Optional[Exception] = None):
        """Record a failure. Increment count and possibly open circuit."""
        with self._lock:
            # If exception is in exclude list, do not count it
            if exception and type(exception) in self.exclude_exceptions:
                logger.warning(f"Ignoring excluded exception {type(exception).__name__} for '{tool_name}'")
                return

            current_state = self._circuit_state.get(tool_name, CircuitState.CLOSED)

            if current_state == CircuitState.HALF_OPEN:
                # A failure in half‑open immediately re‑opens the circuit
                self._circuit_state[tool_name] = CircuitState.OPEN
                self._open_time[tool_name] = time.time()
                logger.warning(f"🚫 Circuit breaker for '{tool_name}' OPEN (failed in half‑open)")
                return

            # In CLOSED state, increment failure count
            count = self._failure_counts.get(tool_name, 0) + 1
            self._failure_counts[tool_name] = count

            if count >= self.failure_threshold:
                self._circuit_state[tool_name] = CircuitState.OPEN
                self._open_time[tool_name] = time.time()
                logger.warning(f"🚫 Circuit breaker for '{tool_name}' OPEN after {count} failures")
            else:
                logger.warning(f"Circuit breaker for '{tool_name}' failures: {count}/{self.failure_threshold}")


class PromptStrategy(Enum):
    """Different strategies for prompting the LLM based on context."""
    MINIMAL = "minimal"  # Ultra-short, emergency fallback
    SINGLE_STEP = "single_step"  # One action at a time (safest)
    FOCUS_GOAL = "focus_goal"  # Focused on current goal only
    MULTI_STEP = "multi_step"  # Plan next 2-3 steps
    FULL_PLAN = "full_plan"  # Generate complete plan
    BATCH = "batch"  # Multiple parallel actions


class GoalPhase(Enum):
    """Goal phase categorization."""
    INITIALIZATION = "initialization"  # Setup, validation
    COLLECTION = "collection"  # Data gathering
    PROCESSING = "processing"  # Transform, extract
    ANALYSIS = "analysis"  # Analyze, assess
    STORAGE = "storage"  # Save to DB
    REPORTING = "reporting"  # Generate reports
    OPERATIONAL = "operational"  # Guidelines, constraints
    METRIC = "metric"  # Success criteria


class ExecutionMode(Enum):
    FAST_SYNC = "fast_sync"
    BATCH_PARALLEL = "batch_parallel"
    GUIDED_SEQUENTIAL = "guided"
    ADAPTIVE = "adaptive"


@dataclass
class ToolPerformanceMetrics:
    tool_name: str
    success_count: int = 0
    failure_count: int = 0
    avg_duration: float = 0.0
    last_used: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)
    recommended_for: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def is_healthy(self) -> bool:
        return self.success_rate >= 0.6 or (time.time() - self.last_used) > 3600


@dataclass
class MicroGoal:
    goal_id: str
    title: str
    description: str
    priority: int = 5
    required_tools: List[str] = field(default_factory=list)
    optional_tools: List[str] = field(default_factory=list)
    estimated_tokens: int = 100
    success_metric: Optional[str] = None
    completed: bool = False
    attempts: int = 0
    max_attempts: int = 3


@dataclass
class StructuredGoal:
    """More detailed goal structure with phase info."""
    goal_id: str
    phase: GoalPhase
    title: str
    description: str
    priority: int
    depends_on: List[str] = field(default_factory=list)  # Other goal IDs
    is_mandatory: bool = True
    suggested_tools: List[str] = field(default_factory=list)
    estimated_tokens: int = 0


class SmartTimeoutCalculator:
    """
    Compute a safe per-iteration LLM timeout from measured hardware speed.

    Problem solved
    --------------
    The original AgentExecutor hard-coded ``timeout_per_iteration = 120 s``.
    On CPU-only hardware running Mistral-7B Q4 at ~2 tok/s, generating just
    370 tokens takes ~185 s — always exceeding 120 s, so every generation
    was cancelled before it could finish.

    Formula
    -------
    timeout = (max_gen_tokens / tokens_per_sec) * SAFETY_FACTOR
    Result is clamped to [MIN_TIMEOUT, MAX_TIMEOUT].

    Usage in AgentExecutor.__init__
    --------------------------------
    Replace the hard-coded ``self.timeout_per_iteration = timeout_per_iteration``
    with a call to this calculator (see AgentExecutor.__init__ for the exact
    integration point).
    """

    # Conservative defaults for CPU-only Mistral-7B Q4 (measured from logs)
    DEFAULT_TOKENS_PER_SEC: float = 2.0
    MIN_TIMEOUT: float = 300.0  # 5 minutes minimum
    MAX_TIMEOUT: float = 3600.0  # 1 hour maximum
    SAFETY_FACTOR: float = 2.2  # generous margin for variable CPU load

    @classmethod
    def calculate(
            cls,
            max_gen_tokens: int = 512,
            tokens_per_sec: Optional[float] = None,
            safety_factor: Optional[float] = None,
    ) -> float:
        """
        Return a per-iteration LLM timeout in seconds.

        Parameters
        ----------
        max_gen_tokens : int
            Maximum tokens the model may generate per call
            (set to ``AgentExecutor.max_total_tokens``).
        tokens_per_sec : float | None
            Measured generation speed.  Pass ``service.measured_tokens_per_sec``
            from the startup benchmark.  Falls back to DEFAULT_TOKENS_PER_SEC
            when None or zero.
        safety_factor : float | None
            Override for the 2.2× safety margin.
        """
        tps = tokens_per_sec or cls.DEFAULT_TOKENS_PER_SEC
        if tps <= 0:
            tps = cls.DEFAULT_TOKENS_PER_SEC
        sf = safety_factor or cls.SAFETY_FACTOR
        raw = (max_gen_tokens / tps) * sf
        result = max(cls.MIN_TIMEOUT, min(cls.MAX_TIMEOUT, raw))
        logger.info(
            f"⏱️  SmartTimeout: {max_gen_tokens} gen-tokens / {tps:.1f} tok/s "
            f"× {sf:.1f} safety = {raw:.0f}s → clamped to {result:.0f}s"
        )
        return result

# =====================================================================
# TOOL FAILURE TRACKER — blacklist tools that repeatedly return nothing
# =====================================================================

class ToolFailureTracker:
    """
    Track per-tool failure counts within a single agent execution run.

    A "failure" is any tool call that:
      - Returns an error dict  (ok=False / error key set)
      - Returns 0 results     (count=0, links=[], entries=[], etc.)
      - Returns None / empty string

    After ``max_failures`` consecutive failures, the tool is marked
    EXHAUSTED and will be:
      1. Excluded from the list of tools offered to the model on the
         next call.
      2. Reported in the exhaustion summary so the agent loop can
         prefer a different approach.

    This breaks the "LLM calls same failing tool forever" loop.
    """

    # Tools the LLM commonly over-uses but that require live internet access
    # that may not be available — penalised more aggressively.
    _SEARCH_TOOLS: frozenset = frozenset({
        "search_links", "search_with_dorks",
        "find_file_links", "find_hidden_resources",
    })

    def __init__(self, max_failures: int = 2):
        self.max_failures = max_failures
        self._failures: Dict[str, int] = {}        # tool → failure count
        self._successes: Dict[str, int] = {}       # tool → success count
        self._logger = logging.getLogger("AgentExecutor")

    def record_result(self, tool_name: str, result: Any) -> bool:
        """
        Inspect *result* and record success or failure.

        Returns True if the result is considered a success, False otherwise.
        """
        is_ok = self._is_successful(result)
        if is_ok:
            self._successes[tool_name] = self._successes.get(tool_name, 0) + 1
            # A real success resets the failure counter for that tool
            self._failures.pop(tool_name, None)
            return True
        else:
            self._failures[tool_name] = self._failures.get(tool_name, 0) + 1
            count = self._failures[tool_name]
            # Search tools get exhausted after 1 failure (saves iterations)
            threshold = 1 if tool_name in self._SEARCH_TOOLS else self.max_failures
            if count >= threshold:
                self._logger.warning(
                    f"🚫 Tool '{tool_name}' EXHAUSTED after {count} empty/failed "
                    f"result(s) — removed from LLM tool list."
                )
            return False

    def is_exhausted(self, tool_name: str) -> bool:
        """Return True if this tool should not be shown to the LLM."""
        count = self._failures.get(tool_name, 0)
        threshold = 1 if tool_name in self._SEARCH_TOOLS else self.max_failures
        return count >= threshold

    def get_exhausted(self) -> List[str]:
        """Return list of all currently exhausted tool names."""
        return [
            t for t in self._failures
            if self.is_exhausted(t)
        ]

    def reset(self) -> None:
        """Clear all tracking state (called at start of each arun())."""
        self._failures.clear()
        self._successes.clear()

    @staticmethod
    def _is_successful(result: Any) -> bool:
        """Heuristic: did the tool return genuinely useful data?"""
        if result is None:
            return False
        if isinstance(result, bool):
            return result
        if isinstance(result, str):
            return len(result.strip()) > 20
        if isinstance(result, (list, tuple)):
            return len(result) > 0
        if isinstance(result, dict):
            # Explicit error flag
            err = result.get("error")
            if err and err is not False and err != "":
                return False
            if result.get("ok") is False:
                return False
            # Empty data containers
            for key in ("links", "urls", "results", "entries", "items",
                        "files", "rows", "records", "data", "chunks"):
                val = result.get(key)
                if isinstance(val, (list, tuple)) and len(val) == 0:
                    # All data keys empty — likely an empty result
                    pass
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    return True
            # count / total fields
            count = (result.get("count") or result.get("total_count")
                     or result.get("total") or result.get("row_count") or 0)
            if count and int(count) > 0:
                return True
            # text/content present
            for key in ("text", "content", "html", "transcript",
                        "extracted_text", "body", "fetched_content"):
                if result.get(key) and len(str(result[key])) > 20:
                    return True
            # If ok=True but nothing useful, treat as failed
            if result.get("ok") is True:
                # Only useful if there's actual content or a non-zero count
                return False
            # success=True flag
            if result.get("success") is True:
                return True
        return True  # Unknown types — assume ok


# =====================================================================
# AGENT STATE — cross-iteration structured context
# =====================================================================

class AgentStateExtractor:
    """
    Extract structured, semantic key-value pairs from every tool result and
    accumulate them in a persistent ``state`` dict that survives across all
    iterations of one agent run.

    Design goals
    ------------
    * **Never crash the agent loop** — every extraction is wrapped in try/except.
    * **Flat dict, actionable keys** — key names match tool parameter names so
      the LLM can copy them verbatim (e.g. ``file_path``, ``url``).
    * **Bounded size** — lists capped at 10 items, strings at 1 000 chars.
    * **Rendered compactly** — ``render()`` produces a ≤ 800 char block
      that the agent loop injects as a dedicated "KNOWN FACTS" section,
      guaranteed to survive prompt packing unlike the truncated history tail.

    Slot reference (populated by tool group)
    -----------------------------------------
    search / link tools      → url, urls[N], page_links[N]
    parse_feed               → url, urls[N]
    http_fetch               → fetched_content, text, last_url, last_status
    scrape_page              → scraped_content, text, page_links[N]
    download_file            → file_path, downloaded_file, downloaded_url
    download_multiple        → file_path, file_paths[N]
    extract_text_from_*      → extracted_text, text, file_path
    chunk_text               → chunks[N], chunk_count, text
    extract_and_chunk_file   → extracted_text, text, chunks[N], chunk_count
    batch_extract_files      → batch_texts[N], text
    validate_url             → validated_url, url_valid
    query_database / db_*    → db_results[N], db_row_count, db_rowcount, db_first_row
    get_analysis_history /
      search_vectorstore     → analysis_results[N], analysis_count
    video_info               → video_title, video_url
    download_video           → video_file_path, file_path
    transcribe_*             → transcript, text
    transcribe_and_index /
      transcribe_url_to_*    → indexed_chunks
    get_system_status /
      *health_check          → system_ok
    (all tools)              → last_ok
    """

    # Ordered list of keys rendered first (most actionable for the LLM)
    _PRIORITY_KEYS: List[str] = [
        "url", "urls",
        "file_path", "file_paths",
        "validated_url",
        "extracted_text", "text",
        "db_results", "db_row_count", "db_rowcount", "db_first_row",
        "analysis_results", "analysis_count",
        "transcript", "fetched_content", "scraped_content",
        "chunks", "chunk_count",
        "video_file_path", "video_url", "video_title",
        "indexed_chunks",
        "system_ok", "last_ok",
    ]

    @staticmethod
    def extract(tool_name: str, result: Any, logger: logging.Logger) -> Dict[str, Any]:
        """
        Return a flat dict of semantic slots populated from *result*.

        Parameters
        ----------
        tool_name : Registered name of the tool that produced *result*.
        result    : Raw return value from the tool (any type).
        logger    : Logger used for non-fatal warnings.
        """
        if not isinstance(result, dict):
            return {}

        out: Dict[str, Any] = {}

        try:
            # ── Search / link tools ─────────────────────────────────────────
            if tool_name in (
                "search_links", "search_with_dorks", "find_file_links",
                "find_hidden_resources", "extract_links_from_page", "expand_links",
            ):
                raw = (result.get("links") or result.get("urls") or
                       result.get("results") or [])
                urls = []
                for item in raw[:10]:
                    u = item if isinstance(item, str) else (
                        item.get("url") or item.get("href") or item.get("link", "") if isinstance(item, dict) else ""
                    )
                    if u:
                        urls.append(str(u))
                if urls:
                    out["urls"] = urls
                    out["url"]  = urls[0]

            # ── parse_feed / parse_feeds_batch ──────────────────────────────
            elif tool_name in ("parse_feed", "parse_feeds_batch"):
                # parse_feed returns: {ok, url, title, entries:[{title,link,...}], count}
                # parse_feeds_batch returns: {ok, results:[per-feed-dicts], total_count, ...}
                all_entries = []
                if tool_name == "parse_feeds_batch":
                    # Aggregate entries from all per-feed result dicts
                    for sub in (result.get("results") or []):
                        if isinstance(sub, dict):
                            all_entries.extend(sub.get("entries") or sub.get("items") or [])
                else:
                    all_entries = result.get("entries") or result.get("items") or []
                    # Defensive: if result IS a list of entry dicts
                    if not all_entries and isinstance(result, list):
                        all_entries = result

                feed_urls = []
                for e in all_entries[:20]:
                    if isinstance(e, dict):
                        u = e.get("link") or e.get("url") or e.get("href")
                        if u:
                            feed_urls.append(str(u))
                if feed_urls:
                    out["urls"] = feed_urls[:10]
                    out["url"]  = feed_urls[0]
                # Also expose raw count for prompt context
                count = result.get("count") or result.get("total_count") or len(feed_urls)
                if count:
                    out["feed_entry_count"] = count

            # ── http_fetch ──────────────────────────────────────────────────
            elif tool_name == "http_fetch":
                content = result.get("content") or result.get("text") or result.get("body") or ""
                if content:
                    out["fetched_content"] = str(content)[:1000]
                    out["text"]            = str(content)[:500]
                if result.get("url"):
                    out["last_url"] = str(result["url"])
                if result.get("status_code") is not None:
                    out["last_status"] = result["status_code"]

            # ── scrape_page / browser_scrape / scrape_async ────────────────
            elif tool_name in ("scrape_page", "browser_scrape", "scrape_async"):
                content = (result.get("content") or result.get("text") or
                           result.get("markdown") or "")
                if content:
                    out["scraped_content"] = str(content)[:1000]
                    out["text"]            = str(content)[:500]
                page_links = result.get("links") or []
                if page_links:
                    out["page_links"] = [str(l) for l in page_links[:5]]

            # ── scrape_batch / scrape_batch_async ──────────────────────────
            elif tool_name in ("scrape_batch", "scrape_batch_async"):
                rlist = result.get("results") or []
                texts = []
                for r in rlist:
                    c = r.get("content") or r.get("text") or "" if isinstance(r, dict) else ""
                    if c:
                        texts.append(str(c)[:200])
                if texts:
                    out["batch_texts"] = texts
                    out["text"]        = texts[0]

            # ── download_file ───────────────────────────────────────────────
            elif tool_name == "download_file":
                path = (result.get("path") or result.get("file_path") or
                        result.get("local_path") or result.get("filename"))
                if path:
                    out["file_path"]       = str(path)
                    out["downloaded_file"] = str(path)
                if result.get("url"):
                    out["downloaded_url"] = str(result["url"])

            # ── download_multiple ───────────────────────────────────────────
            elif tool_name == "download_multiple":
                rlist = result.get("results") or []
                paths = []
                for r in rlist:
                    if isinstance(r, dict):
                        p = r.get("path") or r.get("file_path") or r.get("local_path")
                        if p:
                            paths.append(str(p))
                if paths:
                    out["file_paths"] = paths
                    out["file_path"]  = paths[0]

            # ── extract_text_from_* / extract_with_ocr ─────────────────────
            elif tool_name in (
                "extract_text_from_file", "extract_text_from_pdf",
                "extract_text_from_docx", "extract_text_from_spreadsheet",
                "extract_text_from_textfile", "extract_with_ocr",
            ):
                text = result.get("text") or result.get("content") or ""
                if text:
                    out["extracted_text"] = str(text)[:1000]
                    out["text"]           = str(text)[:500]
                if result.get("file_path"):
                    out["file_path"] = str(result["file_path"])

            # ── chunk_text ──────────────────────────────────────────────────
            elif tool_name == "chunk_text":
                chunks = result.get("chunks") or result.get("result") or []
                if isinstance(chunks, list) and chunks:
                    out["chunks"]      = [str(c)[:200] for c in chunks[:10]]
                    out["chunk_count"] = len(chunks)
                    out["text"]        = str(chunks[0])[:300]

            # ── extract_and_chunk_file ──────────────────────────────────────
            elif tool_name == "extract_and_chunk_file":
                text   = result.get("text") or result.get("content") or ""
                chunks = result.get("chunks") or []
                if text:
                    out["extracted_text"] = str(text)[:1000]
                    out["text"]           = str(text)[:500]
                if isinstance(chunks, list) and chunks:
                    out["chunks"]      = [str(c)[:200] for c in chunks[:10]]
                    out["chunk_count"] = len(chunks)
                if result.get("file_path"):
                    out["file_path"] = str(result["file_path"])

            # ── batch_extract_files ─────────────────────────────────────────
            elif tool_name == "batch_extract_files":
                rlist = result.get("results") or []
                texts = []
                for r in rlist:
                    t = r.get("text") or r.get("content") or "" if isinstance(r, dict) else ""
                    if t:
                        texts.append(str(t)[:200])
                if texts:
                    out["batch_texts"] = texts
                    out["text"]        = texts[0]

            # ── validate_url ────────────────────────────────────────────────
            elif tool_name == "validate_url":
                valid = bool(result.get("valid") or result.get("ok"))
                out["url_valid"] = valid
                if valid and result.get("url"):
                    out["validated_url"] = str(result["url"])

            # ── query_database / db_query / db_execute / db_query_one ──────
            elif tool_name in (
                "query_database", "db_query", "db_execute",
                "db_query_one", "db_list_tables",
            ):
                rows = result.get("results") or result.get("result") or []
                if isinstance(rows, list):
                    out["db_results"]   = rows[:10]
                    out["db_row_count"] = len(rows)
                    if rows:
                        out["db_first_row"] = rows[0]
                # rowcount for INSERT/UPDATE/DELETE
                rc = result.get("rowcount")
                if rc is None and isinstance(result.get("result"), dict):
                    rc = result["result"].get("rowcount")
                if rc is not None:
                    out["db_rowcount"] = rc

            # ── get_analysis_history / search_vectorstore ───────────────────
            elif tool_name in ("get_analysis_history", "search_vectorstore"):
                items = (result.get("results") or result.get("items") or
                         result.get("threats") or result.get("data") or [])
                if isinstance(items, list) and items:
                    out["analysis_results"] = items[:5]
                    out["analysis_count"]   = len(items)

            # ── video_info ──────────────────────────────────────────────────
            elif tool_name == "video_info":
                if result.get("title"):
                    out["video_title"] = str(result["title"])
                url = result.get("url") or result.get("webpage_url")
                if url:
                    out["video_url"] = str(url)

            # ── download_video ──────────────────────────────────────────────
            elif tool_name == "download_video":
                path = result.get("path") or result.get("file_path")
                if path:
                    out["video_file_path"] = str(path)
                    out["file_path"]       = str(path)

            # ── transcribe_audio_or_video / transcribe_multiple ─────────────
            elif tool_name in ("transcribe_audio_or_video", "transcribe_multiple"):
                text = result.get("text") or result.get("transcript") or ""
                if text:
                    out["transcript"] = str(text)[:1000]
                    out["text"]       = str(text)[:500]

            # ── transcribe_and_index / transcribe_url_to_vectorstore ─────────
            elif tool_name in ("transcribe_and_index", "transcribe_url_to_vectorstore"):
                if result.get("chunks_indexed") is not None:
                    out["indexed_chunks"] = result["chunks_indexed"]

            # ── system / health tools ────────────────────────────────────────
            elif tool_name in (
                "get_system_status", "db_health_check", "download_health_check",
            ):
                ok = bool(
                    result.get("ok") or
                    result.get("healthy") or
                    result.get("status") == "ok"
                )
                out["system_ok"] = ok

        except Exception as exc:
            # Never crash the agent loop because of state extraction
            logger.warning(f"⚠️ AgentStateExtractor.extract({tool_name}): {exc}")

        # Universal: top-level ok flag
        if "ok" in result:
            out["last_ok"] = bool(result["ok"])

        return out

    @staticmethod
    def render(state: Dict[str, Any], max_chars: int = 800) -> str:
        """
        Render *state* as a compact, LLM-readable string.

        Format::
            url: https://premiumtimesng.com/...
            urls[0]: https://premiumtimesng.com/article1
            urls[1]: https://saharareporters.com/article2
            file_path: /tmp/downloads/report.pdf
            text: "Armed bandits attacked three villages…"

        The block is always ≤ *max_chars* characters.
        """
        if not state:
            return "(none)"

        lines: List[str] = []
        seen: set = set()

        def _add(key: str, val: Any) -> None:
            nonlocal lines
            if isinstance(val, list):
                for i, item in enumerate(val[:10]):
                    s = str(item)
                    if len(s) > 150:
                        s = s[:150] + "…"
                    lines.append(f"  {key}[{i}]: {s}")
            else:
                s = str(val)
                if len(s) > 200:
                    s = s[:200] + "…"
                lines.append(f"  {key}: {s}")

        # Priority keys first
        for k in AgentStateExtractor._PRIORITY_KEYS:
            if k in state:
                seen.add(k)
                _add(k, state[k])

        # Remaining keys
        for k, v in state.items():
            if k not in seen:
                _add(k, v)

        block = "\n".join(lines)
        if len(block) > max_chars:
            block = block[:max_chars] + "\n  …(truncated)"
        return block or "(none)"

# =====================================================================
# ENHANCED AgentExecutor CLASS
# =====================================================================
class AgentExecutor:
    """
    🏆 PRODUCTION-READY Agent Executor V2 - NOW COMPLETE

    Comprehensive implementation featuring:
    - Smart instruction decomposition into micro-goals
    - Advanced multi-strategy prompt engineering
    - Adaptive execution based on context
    - Tool dependency resolution
    - Parallel tool execution
    - Full async/await patterns
    - Circuit breaker & rate limiting
    - Comprehensive safety validation
    - Database integration
    - Reflection-based self-improvement
    - Graceful prompt compression
    - Complete execution history tracking
    """

    def __init__(
            self,
            llm: BaseLLMAdapter,
            tool_registry: ToolRegistry,
            instruction: str = "",
            max_iterations: int = 12,
            max_total_tokens: int = 4096,
            timeout: float = 1800.0,
            planner_temperature: float = 0.0,
            enable_reflection: bool = True,
            enable_safety_checks: bool = True,
            trace_enabled: bool = True,
            enable_goal_tracking: bool = True,
            timeout_per_iteration: float = 120.0,
            max_concurrent_tools: int = 5,
            enable_db_logging: bool = True,
            backoff_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize AgentExecutorV2 with comprehensive configuration."""
        # --- Core LLM & Tool Components ---
        self.llm = llm
        self.tool_registry = tool_registry
        self.instruction = instruction

        # --- Execution Configuration ---
        self.max_iterations = max_iterations
        self.max_total_tokens = max_total_tokens
        self.timeout = timeout
        self.planner_temperature = max(1e-8, planner_temperature)

        # Compute per-iteration LLM timeout from hardware speed.
        #
        # Root cause: the original default was 120 s.  On CPU-only hardware
        # running Mistral-7B Q4 at ~2 tok/s, generating just 370 tokens takes
        # ~185 s, so every generation was cancelled before it finished.
        #
        # The caller may pass the measured benchmark speed via
        # ``timeout_per_iteration`` (keeping the API backwards-compatible).
        # If the caller passes the old default (120.0) or any value ≤ 120,
        # we override it with a properly calculated value.  If a caller
        # intentionally passes a large custom value (e.g. from config), we
        # respect it as long as it is > 120 s.
        if timeout_per_iteration <= 120.0:
            # Old default passed — compute the correct value from hardware speed.
            # AdaptiveModelManager may have measured tokens/sec via benchmark;
            # try to read it, fall back to the conservative default (2 tok/s).
            try:
                mm = get_model_manager()
                measured_tps = getattr(mm, "measured_tokens_per_sec", None)
            except Exception:
                measured_tps = None
            self.timeout_per_iteration = SmartTimeoutCalculator.calculate(
                max_gen_tokens=max_total_tokens,
                tokens_per_sec=measured_tps,
            )
        else:
            # Caller supplied a deliberate large timeout — honour it.
            self.timeout_per_iteration = timeout_per_iteration

        # --- Feature Flags ---
        self.enable_reflection = enable_reflection
        self.enable_safety_checks = enable_safety_checks
        self.trace_enabled = trace_enabled
        self.enable_goal_tracking = enable_goal_tracking
        self.enable_db_logging = enable_db_logging

        # --- Prompt & Parsing Engines ---
        self.tool_engine = ToolIntelligenceEngine()

        # --- Logging & Monitoring ---
        self.logger = logging.getLogger("AgentExecutor")
        self.start_time = time.time()
        self.tokens_used = 0
        self.iteration_count = 0
        self.current_execution_id = str(uuid.uuid4())
        self.conversation_id = str(uuid.uuid4())

        # --- Goal & Execution Tracking ---
        self.micro_goals: List[MicroGoal] = []
        self.completed_goals: List[MicroGoal] = []
        self.failed_goals: List[MicroGoal] = []
        self.execution_log: List[Dict[str, Any]] = []
        self.execution_trace: List[Dict[str, Any]] = []
        self.step_count = 0
        self.execution_history: List[str] = []

        # --- Concurrency & Rate Limiting ---
        self.max_concurrent_tools = max_concurrent_tools
        self.request_semaphore = AdaptiveConcurrency(
            initial_limit=max_concurrent_tools,
            min_limit=1,
            mem_high_threshold=80.0,
            cpu_high_threshold=80.0,
            adjustment_factor=0.8
        )
        self.tool_breaker = ToolCircuitBreaker(
            failure_threshold=3,
            reset_timeout=300,
            half_open_max_calls=1,
            exclude_exceptions={TimeoutError, asyncio.TimeoutError}
        )

        # --- Retry & Backoff Strategy ---
        self.backoff_config = backoff_config or {
            "base_delay": 1.0,
            "max_delay": 60.0,
            "max_consecutive": 5
        }
        self.llm_backoff = AdaptiveBackoff(
            base_delay=self.backoff_config.get("base_delay", 1.0),
            max_delay=self.backoff_config.get("max_delay", 60.0),
            max_consecutive=self.backoff_config.get("max_consecutive", 5),
            name="LLM"
        )

        # --- State Management ---
        self._running = False
        self._stop_flag = False
        self._shutdown = False
        self._service = None
        self._lock = RLock()

        # --- Model Manager Integration ---
        self.model_manager = get_model_manager()

        # --- Current Execution State ---
        self.current_context: Optional[ExecutionContext] = None
        self.goals: List[TaskGoal] = []
        self._iteration_progress: Dict[int, float] = {}

        # --- Error Tracking ---
        self.last_tool_error: Optional[str] = None
        self.error_counts: Dict[str, int] = defaultdict(int)

        # --- Tool Failure Tracker ---
        # Tracks per-tool failure counts for THIS run.  Exhausted tools are
        # removed from the LLM's tool list and listed in the AVOID section
        # of the prompt so the LLM knows to try a different approach.
        self._failure_tracker = ToolFailureTracker(max_failures=2)

        # --- Performance Metrics ---
        self.metrics = {
            "total_iterations": 0,
            "total_tools_executed": 0,
            "total_tools_failed": 0,
            "total_tokens_used": 0,
            "avg_tool_duration": 0.0,
            "avg_iteration_duration": 0.0,
            "goals_completed": 0,
            "goals_failed": 0,
        }

        self.logger.info(
            f"✅ AgentExecutor initialized: "
            f"instruction_len={len(instruction)}, "
            f"max_iterations={max_iterations}, "
            f"max_tokens={max_total_tokens}, "
            f"timeout={timeout}s"
        )

        # Initialize size cache
        self._init_size_cache()

    # =====================================================================
    # PUBLIC API: MAIN EXECUTION ENTRY POINTS
    # =====================================================================

    def set_service(self, service):
        """Set reference to parent service for database access."""
        self._service = service
        self.logger.info("✅ Service reference set for database logging")



    # ─────────────────────────────────────────────────────────────────
    # Engine selection: adaptive (default) or legacy
    # ─────────────────────────────────────────────────────────────────
    _VALID_ENGINES = ("adaptive",)

    @classmethod
    def _resolve_engine_mode(cls) -> str:
        """
        The legacy engine was removed with A11. AGENT_ENGINE=legacy now
        resolves to 'adaptive' with a warning, so an old environment
        variable does not silently change behaviour.
        """
        raw = os.environ.get("AGENT_ENGINE", "adaptive").strip().lower()
        if raw == "legacy":
            logging.getLogger("AgentExecutor").warning(
                "AGENT_ENGINE=legacy is no longer supported (the legacy "
                "engine was removed in A11). Falling back to 'adaptive'."
            )
            return "adaptive"
        if raw not in cls._VALID_ENGINES:
            logging.getLogger("AgentExecutor").warning(
                "Unknown AGENT_ENGINE=%r; expected one of %s. "
                "Defaulting to 'adaptive'.",
                raw, cls._VALID_ENGINES,
            )
        return "adaptive"

    def _model_hint(self, limits: Any = None) -> str:
        """
        Best-effort model identifier for CapabilityProfile.

        Priority:
          1. `limits.model_id` / `limits.model_name` (server is authoritative)
          2. `self.llm.model` / `self.llm.config.model_name` / `self.llm._model_id`
          3. `self.model_manager.model_name`
          4. "" — let CapabilityProfile apply its own default
        """
        # 1. Server-reported id wins when we have it.
        if limits is not None:
            for attr in ("model_id", "model_name", "id"):
                v = getattr(limits, attr, None)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        # 2. Adapter-reported name.
        for attr in ("model", "model_name", "_model_id"):
            v = getattr(self.llm, attr, None)
            if isinstance(v, str) and v.strip():
                return v.strip()

        cfg = getattr(self.llm, "config", None)
        if cfg is not None:
            v = getattr(cfg, "model_name", None)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # 3. Model manager.
        v = getattr(self.model_manager, "model_name", None)
        if isinstance(v, str) and v.strip():
            return v.strip()

        # 4. Give up cleanly — do NOT hardcode a specific model.
        return ""

    async def arun(
            self,
            instruction: Optional[str] = None,
            context: Optional[str] = None,
            timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Public entry point — dispatches to the configured engine."""
        agent_instruction = instruction or self.instruction
        if not agent_instruction:
            raise ValueError("No instruction provided for agent execution")

        return await self._arun_adaptive(agent_instruction, timeout=timeout)

    async def _arun_adaptive(
            self,
            agent_instruction: str,
            timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        adaptive engine.

        Uses amb_runtime.AgentRun, which:
          - parses every reply via amb_actions.parse_action (A6)
          - holds a per-prompt token budget from the server's own limits
            instead of a cumulative run counter (A11)
          - carries a ledger with arguments, memoizes duplicate (tool, args),
            resolves $rN handles, downgrades tier on repeated format failures,
            writes per-step checkpoints (A7)
          - builds tool cards from signatures and hides create_*/check_* tools (A12)
        """
        from amb_sentinel.agent.amb_runtime import AgentRun, CapabilityProfile
        from amb_sentinel.agent.amb_actions import build_specs

        try:
            from amb_sentinel.tools.netstate import wait_for_network
            if not await asyncio.to_thread(wait_for_network, 20.0, 4.0):
                self.logger.warning(
                    "🌐 Network unreachable at cycle start — network tools "
                    "will fast-fail; DB-only steps still run."
                )
        except Exception:
            pass

        # ── 1. capability profile from the live server ─────────────────────
        limits = None
        try:
            client = getattr(self.llm, "_client", None)
            if client is not None and hasattr(client, "aruntime_limits"):
                limits = await client.aruntime_limits()
            elif client is not None and hasattr(client, "runtime_limits"):
                # sync facade — safe to call from the main loop because
                # it routes through _BRIDGE internally.
                loop = asyncio.get_running_loop()
                limits = await loop.run_in_executor(None, client.runtime_limits)
        except Exception as e:
            self.logger.warning(
                "runtime_limits unavailable (%s: %s); using safe defaults",
                type(e).__name__, e,
            )

        _hint = self._model_hint(limits)

        # One resolver decides every LLM number: env > config llm.overrides >
        # SERVER > fallback. The table (with the source of each value) is
        # logged so nothing is silently assumed.
        from amb_sentinel import llm_config as _lc
        _cfg = self._config_dict()
        _rc = _lc.resolve(_cfg, limits)
        _rc.warnings.extend(_lc.find_conflicts(_cfg, _rc))
        for _line in _rc.table().splitlines():
            _logger_method = (
                self.logger.warning
                if _line.lstrip().startswith("⚠")
                else self.logger.info
            )
            _logger_method("🔧 %s", _line)

        profile = None
        if limits is not None:
            try:
                profile = CapabilityProfile.build(
                    model_hint=_hint, **_rc.profile_kwargs(),
                )
            except Exception as e:
                self.logger.warning(
                    f"CapabilityProfile.from_limits failed ({e}); "
                    f"falling back to CapabilityProfile.build()"
                )

        if profile is None:
            _build_kwargs = {
                "n_ctx": getattr(self.model_manager, "context_window", 4096) or 4096,
                "decode_tps": 3.0,
            }
            if _hint:
                _build_kwargs["model_hint"] = _hint
            profile = CapabilityProfile.build(**_build_kwargs)

        # ── 2. wrap tools, injecting `service` where the sig requires it ───
        import functools
        callables: Dict[str, Any] = {}
        for n, t in self.tool_registry.tools.items():
            fn = getattr(t, "func", None)
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                callables[n] = fn
                continue
            if "service" in sig.parameters:
                if self._service is not None:
                    fn = functools.partial(fn, service=self._service)
                else:
                    self.logger.warning(f"Skipping '{n}': requires service, none set")
                    continue
            # db_* tools need a valid connection name. Small models
            # frequently pass the schema example ("value") or the
            # first list item when there is only one connection.
            # Default to "primary" when the caller's name is unknown.
            if n.startswith("db_") and "name" in sig.parameters:
                _svc = self._service
                _db = getattr(_svc, "db_manager", None) if _svc is not None else None

                def _with_default_conn(_fn=fn, _db=_db):
                    @functools.wraps(_fn)
                    def _w(*a, **kw):
                        known = set(getattr(_db, "_conns", {}) or {}) if _db is not None else set()
                        nm = kw.get("name")
                        if (not nm or nm not in known) and "primary" in known and not a:
                            self.logger.warning(
                                "db tool '%s': unknown connection %r → using 'primary'",
                                _fn.__name__, nm,
                            )
                            kw["name"] = "primary"
                        return _fn(*a, **kw)
                    return _w
                fn = _with_default_conn()
            callables[n] = fn

        # ── 3. tool specs — hide factory/internal, apply bundled overrides ─
        specs = build_specs(
            callables,
            hide_prefixes=("create_", "check_", "get_default_"),
        )
        # Apply ai_service.exposed_tools allow/deny filter BEFORE handing
        # the specs to AgentRun. This drops the ~50 factory/utility tools
        # that eat prompt budget on an 8B at ~6 tok/s prefill.
        _ai_cfg = _cfg.get("ai_service")
        _exposed = _ai_cfg.get("exposed_tools") or {}
        _allow = set(_exposed.get("allow") or [])
        _deny_prefixes = tuple(_exposed.get("deny_prefixes") or ())
        if _allow or _deny_prefixes:
            _before = len(specs)
            specs = {
                n: sp for n, sp in specs.items()
                if (_allow and n in _allow or not _allow)
                and not (_deny_prefixes and n.startswith(_deny_prefixes))
            }
            callables = {n: callables[n] for n in specs if n in callables}
            self.logger.info(
                "specs filtered by ai_service.exposed_tools: %d → %d",
                _before, len(specs),
            )
            if len(specs) > 20:
                self.logger.warning(
                    "Still exposing %d tools to the LLM (>20). "
                    "Tighten ai_service.exposed_tools.allow in config.yaml.",
                    len(specs),
                )

        # ── 4. budgets ─────────────────────────────────────────────────────
        # The profile now carries its own latency estimate, so we can size
        # max_llm_calls against the actual hardware instead of a constant.
        wall_clock_s = float(timeout or self.timeout or 3600.0)

        # Reserve 25% of the wall clock for tool execution, checkpoint I/O,
        # repair turns, and jitter. The remaining 75% is the LLM budget.
        _LLM_TIME_FRACTION = 0.75
        _llm_time_s = wall_clock_s * _LLM_TIME_FRACTION

        # Realistic per-call estimate. A typical agent turn carries ~1000
        # prompt tokens (see ledger's peak_prompt=1051) and emits a short
        # JSON action (20–64 tokens), NOT the full prompt_budget /
        # reserve_out the profile advertises as upper bounds. The old
        # formula divided by the upper bound, producing 1687 s/call when
        # real was ~90 s — which then collapsed max_llm_calls to the floor
        # of 4 and aborted 6-step cycles after ~400 s of a 3600 s budget.
        _typ_prompt = min(profile.prompt_budget, 1200)
        _typ_out    = min(profile.reserve_out, 128)
        _per_call_s = max(
            20.0,
            _typ_prompt / max(profile.prefill_tps, 0.5)
            + _typ_out / max(profile.decode_tps, 0.2),
        )
        # Blend with what the transport has actually measured (≥3 samples).
        try:
            _tm = self.llm.transport_metrics() or {}
            if int(_tm.get("successful", 0)) >= 3 and float(_tm.get("avg_latency_s", 0)) > 0:
                _per_call_s = 0.5 * _per_call_s + 0.5 * float(_tm["avg_latency_s"])
        except Exception:
            pass

        _min_calls = 8
        _max_calls_cap = max(_min_calls, int(self.max_iterations) * 8)
        _est_calls = int(_llm_time_s / _per_call_s) if _per_call_s > 0 else _min_calls
        max_llm_calls = max(_min_calls, min(_est_calls, _max_calls_cap))

        if _est_calls < _min_calls:
            self.logger.warning(
                "⚠️ wall_clock_s=%.0fs allows only %d LLM call(s) at the "
                "profile's estimated %.0fs per call. The run may abort before "
                "all steps complete; the checkpoint will let the next cycle "
                "resume. Consider raising ai_service.agent_execution_interval "
                "or lowering ai_service.max_agent_iterations.",
                wall_clock_s, _est_calls, _per_call_s,
            )

        self.logger.info(
            "🧠 Profile: size_tier=%s prompt_budget=%d reserve_out=%d "
            "decode_tps=%.2f prefill_tps=%.2f est_call=%.0fs | "
            "wall=%.0fs est_calls=%d max_llm_calls=%d",
            profile.tier, profile.prompt_budget, profile.reserve_out,
            profile.decode_tps, profile.prefill_tps, _per_call_s,
            wall_clock_s, _est_calls, max_llm_calls,
        )

        # ── 5. run ─────────────────────────────────────────────────────────
        checkpoint_dir = paths.checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Observability: every decision of this run goes to
        # agent_events.jsonl, agent_trace.log, PostgreSQL, and the ring buffer.
        import uuid as _uuid
        _run_id = _uuid.uuid4().hex[:8]
        _svc_db = (
            getattr(self._service, "db_manager", None)
            if self._service is not None else None
        )
        try:
            from amb_sentinel.observability import RunRecorder
            recorder = RunRecorder(
                _run_id, db=_svc_db,
                instruction=agent_instruction, model=_hint,
            )
        except Exception as _e:      # never let logging break the agent
            self.logger.warning(
                "RunRecorder unavailable (%s); continuing without it", _e,
            )
            from amb_sentinel.observability import NullRecorder
            recorder = NullRecorder()
        self._recorder = recorder

        run = AgentRun(
            chat=self.llm.chat,
            tools=callables,
            profile=profile,
            instruction=agent_instruction,
            specs=specs,
            max_llm_calls=max_llm_calls,
            wall_clock_s=wall_clock_s,
            checkpoint_dir=str(checkpoint_dir),
            run_id=_run_id,
            recorder=recorder,
            contracts=((_cfg.get("ai_service") or {}).get("step_contracts")) or None,
        )
        self._current_run = run

        # Every step needs at least a tool turn plus a finishing turn,
        # plus slack. `_min_calls = 8` above only floors the *global*
        # count; this floors it against the actual plan length.
        run.max_llm_calls = max(run.max_llm_calls, 2 * len(run.steps) + 2)

        # Floor the wall clock by the plan size × per-call latency × safety.
        # `run.max_llm_calls` is now the real ceiling (floored by step
        # count above). If the initial wall_clock_s cannot fit those calls
        # at the server's measured speed, raise it. Otherwise the run
        # would abort mid-step and have to resume on the next cycle.
        _needed_wall_s = float(run.max_llm_calls) * _per_call_s * 1.5
        if _needed_wall_s > run.wall_clock_s:
            _old_wall_s = run.wall_clock_s
            run.wall_clock_s = _needed_wall_s
            run._deadline = time.monotonic() + _needed_wall_s
            self.logger.warning(
                "⏱️ Wall clock raised %.0fs → %.0fs to fit %d calls × "
                "%.0fs/call × 1.5 safety (was capped by agent_execution_interval)",
                _old_wall_s, _needed_wall_s, run.max_llm_calls, _per_call_s,
            )

        # A hung model call must not consume the whole cycle. Cap each
        # call at 4× the per-call estimate, bounded to [120, 900] s.
        run.per_call_cap_s = max(120.0, min(900.0, 4.0 * _per_call_s))

        if self._stop_flag:
            run.cancel()

        self.logger.info(
            f"🧠 Adaptive engine: tier={profile.tier} "
            f"prompt_budget={profile.prompt_budget} max_tools={profile.max_tools} "
            f"max_llm_calls={max_llm_calls} wall={wall_clock_s:.0f}s "
            f"specs={len(specs)} tools={len(callables)}"
        )

        try:
            with recorder:                       # exposes the run id to tools
                res = await run.run()
        except asyncio.CancelledError:
            run.cancel()
            raise
        finally:
            self._current_run = None

        # Independent post-condition: what did this run REALLY leave in the DB?
        _verified_rows = None
        try:
            from amb_sentinel.agent.step_gate import verify_rows_in_db
            _tbl = (
                (_cfg.get("data_analyzer") or {}).get("raw_data_table")
                or "raw_data"
            )
            _verified_rows = (
                verify_rows_in_db(_svc_db, _tbl, res.run_id)
                if _svc_db is not None else None
            )
        except Exception:
            pass
        self.logger.info(
            "🗄 DB verification run=%s: rows in %s tagged with this run = %s "
            "(tool-reported writes = %s)",
            res.run_id,
            ((_cfg.get("data_analyzer") or {}).get("raw_data_table")) or "raw_data",
            _verified_rows,
            getattr(recorder, "rows_written", "?"),
        )

        # ── 6. metrics ─────────────────────────────────────────────────────
        self.iteration_count = res.llm_calls
        self.metrics["total_iterations"] = res.llm_calls
        self.metrics["total_tools_executed"] = len(res.ledger)
        self.metrics["total_tools_failed"] = sum(
            1 for e in res.ledger if e.get("status") == "failed"
        )
        self.metrics["total_tokens_used"] = res.prompt_tokens_peak
        # Every autonomous cycle must update the service counters the
        # metrics job reads, or system_metric.log keeps reporting
        # requests_total=0 / agent_executions=0 (which is exactly what
        # the uploaded log shows for the whole session).
        self.metrics["requests_total"] = self.metrics.get("requests_total", 0) + 1
        self.metrics["agent_executions"] = self.metrics.get("agent_executions", 0) + 1
        if res.success:
            self.metrics["requests_successful"] = self.metrics.get("requests_successful", 0) + 1
        else:
            self.metrics["requests_failed"] = self.metrics.get("requests_failed", 0) + 1

        self.metrics["last_request"] = time.time()

        # The metrics job reads the SERVICE's dict, not the executor's — bump both.
        _sm = (
            getattr(self._service, "metrics", None)
            if self._service is not None else None
        )
        if isinstance(_sm, dict):
            _sm["requests_total"] = _sm.get("requests_total", 0) + 1
            _sm["agent_executions"] = _sm.get("agent_executions", 0) + 1
            _key = "requests_successful" if res.success else "requests_failed"
            _sm[_key] = _sm.get(_key, 0) + 1
            _sm["last_request"] = time.time()

        self.logger.info(
            f"{'✅' if res.success else '❌'} Adaptive run: "
            f"success={res.success} steps={res.step_status} "
            f"llm_calls={res.llm_calls} peak_prompt={res.prompt_tokens_peak}"
        )

        if not res.success:
            self.logger.warning("⚠️ Adaptive run notes: %s", "; ".join(res.notes[:6]) or "(none)")

        return {
            "result": res.answers,
            "success": res.success,
            "steps": res.ledger,
            "step_status": res.step_status,
            "metrics": res.metrics,
            "notes": res.notes,
            "run_id": res.run_id,
            "cancelled": res.cancelled,
            "verified_rows": _verified_rows,
        }

    def _config_dict(self) -> Dict[str, Any]:
        """The parsed config.yaml (service copy if available, else from disk)."""
        c = getattr(self._service, "config", None) if self._service is not None else None
        if isinstance(c, dict) and c:
            return c
        try:
            from amb_sentinel.dbsetup import load_config
            return load_config(None)[0]
        except Exception:
            return {}

    async def start_continuous_execution(self, interval: int = 600):
        """
        Start continuous autonomous execution loop.

        Args:
            interval: Seconds between execution cycles
        """

        self._running = True
        self._stop_flag = False
        self.logger.info(f"🚀 Starting continuous execution (interval: {interval}s)")

        iteration = 0
        retry_schedule = [60, 300, 900]
        consecutive_failures = 0

        while self._running and not self._stop_flag:
            iteration += 1
            self.logger.info(f"\n{'=' * 80}\n🔄 Execution cycle #{iteration}\n{'=' * 80}")

            cycle_ok = False
            try:
                result = await self.arun()
                cycle_ok = bool(result.get("success", False))
                self.logger.info(
                    f"✅ Cycle #{iteration} complete: success={cycle_ok}"
                )
            except asyncio.CancelledError:
                self.logger.warning("🛑 Execution cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ Cycle #{iteration} failed: {e}", exc_info=True)
                self.error_counts["cycle_errors"] += 1

            if cycle_ok:
                consecutive_failures = 0
                wait_s = interval
            else:
                wait_s = retry_schedule[min(consecutive_failures, len(retry_schedule) - 1)]
                consecutive_failures += 1
                try:
                    from amb_sentinel.tools.netstate import known_offline, wait_for_network
                    if known_offline():
                        # Return as soon as DNS works; no blind 60/300/900 s.
                        await asyncio.to_thread(wait_for_network, 900.0, 10.0)
                        wait_s = 15
                except Exception:
                    pass
                self.logger.info(
                    f"⚠️ Cycle #{iteration} failed — retrying in {wait_s}s "
                    f"(failure streak: {consecutive_failures})"
                )

            if not self._stop_flag:
                self.logger.info(f"⏳ Waiting {wait_s}s for next cycle...")
                for _ in range(wait_s):
                    if self._stop_flag:
                        break
                    await asyncio.sleep(1)

        self.logger.info("🛑 Continuous execution stopped")
        self._running = False

    def stop_continuous_execution(self):
        """Signal continuous execution to stop gracefully."""
        self._stop_flag = True
        self._running = False
        _r = getattr(self, "_current_run", None)
        if _r is not None:
            try:
                _r.cancel()
                self.logger.info("🛑 Cancelling in-flight AgentRun")
            except Exception:
                pass
        self.logger.info("🛑 Stop signal sent to continuous execution")

    # =====================================================================
    # PROMPT ENGINEERING & STRATEGY SELECTION
    # =====================================================================

    def _select_prompt_strategy(self, context: ExecutionContext) -> PromptStrategy:
        """
        Select the best prompt strategy based on execution context.

        PRODUCTION DECISION TREE:
        1. If stuck (many iterations with 0 progress) → MINIMAL
        2. If low tokens left → SINGLE_STEP
        3. If many goals pending → FOCUS_GOAL
        4. If low time remaining → SINGLE_STEP
        5. If GPU available & good progress → MULTI_STEP
        6. Default → FOCUS_GOAL (safest for CPU)
        """
        summary = context.goals_summary()
        progress = context.progress_percentage()
        time_left = context.get_remaining_time()
        tokens_left = max(0, self.model_manager.context_window - self.tokens_used)

        # RULE 1: Emergency - stuck with no progress
        if context.iteration > 3 and progress == 0:
            self.logger.warning("🚨 Agent stuck with 0 progress - switching to MINIMAL")
            return PromptStrategy.MINIMAL

        # RULE 2: Token pressure
        if tokens_left < 2000:
            self.logger.info(f"⚠️ Low tokens ({tokens_left}) - switching to SINGLE_STEP")
            return PromptStrategy.SINGLE_STEP

        # RULE 3: Time pressure
        if time_left < 120:  # Less than 2 minutes
            self.logger.info(f"⏰ Low time ({time_left}s) - switching to SINGLE_STEP")
            return PromptStrategy.SINGLE_STEP

        # RULE 4: Too many goals
        if summary['pending'] > 10:
            self.logger.info(f"📊 Many goals ({summary['pending']}) - using FOCUS_GOAL")
            return PromptStrategy.FOCUS_GOAL

        # RULE 5: Good progress, can plan ahead
        if progress > 50 and time_left > 300:
            self.logger.info(f"✅ Good progress, planning ahead - using MULTI_STEP")
            return PromptStrategy.MULTI_STEP

        # DEFAULT: Safe choice for most scenarios
        self.logger.info("Using default FOCUS_GOAL strategy")
        return PromptStrategy.FOCUS_GOAL

    def _generate_example_tool_call(self, goal: TaskGoal, tool_registry: ToolRegistry) -> Optional[str]:
        """Generate a concrete example JSON tool call based on the current goal."""
        if not goal or not goal.suggested_tools:
            return None
        # Pick the first suggested tool that is registered
        for tool_name in goal.suggested_tools:
            tool = tool_registry.get(tool_name)
            if tool and tool.schema:
                # Build example args from schema
                example_args = {}
                for param_name, param_info in tool.schema.items():
                    if isinstance(param_info, dict):
                        if param_info.get('required') and param_info.get('type') == 'string':
                            example_args[param_name] = "example_value"
                        elif param_info.get('type') == 'integer':
                            example_args[param_name] = 10
                        elif param_info.get('type') == 'boolean':
                            example_args[param_name] = True
                        elif param_info.get('type') == 'array':
                            example_args[param_name] = ["item1", "item2"]
                    else:
                        # Simple schema (legacy)
                        if param_info == "string":
                            example_args[param_name] = "example_value"
                        elif param_info == "integer":
                            example_args[param_name] = 10
                if example_args:
                    return json.dumps({"tool": tool_name, "args": example_args})
        return None

    def _extract_key_values(self, tool_name: str, result: Any) -> Dict[str, Any]:
        """
        Convenience wrapper: delegates to AgentStateExtractor.extract().

        Called from arun() after every tool execution.  Returns a flat dict of
        semantic slots that are merged into the run-scoped ``state`` dict and
        injected into every subsequent prompt as STATE CONTEXT.

        All extraction logic lives in AgentStateExtractor so it can be tested
        independently and reused by subclasses.
        """
        return AgentStateExtractor.extract(tool_name, result, self.logger)

    def _render_state_block(self, state: Dict[str, Any]) -> str:
        """
        Convenience wrapper: delegates to AgentStateExtractor.render().

        Produces a compact (≤ 800 char) human- and LLM-readable string of the
        current agent state, suitable for injection into a prompt or log entry.
        """
        return AgentStateExtractor.render(state)

    def _resolve_dependencies(
            self,
            tool_calls: List[ToolCall]
    ) -> Tuple[List[List[int]], Dict[str, int]]:
        """Resolve tool dependencies and return execution stages."""
        n = len(tool_calls)
        graph = defaultdict(list)
        indeg = [0] * n
        output_providers = {}

        # Collect outputs from tool definitions (not from calls)
        for idx, call in enumerate(tool_calls):
            tool = self.tool_registry.get(call.tool_name)
            if tool and hasattr(tool, 'provides') and tool.provides:
                for out in tool.provides:
                    if out in output_providers:
                        self.logger.warning(f"Duplicate output '{out}'")
                    output_providers[out] = idx

        # Add edges based on consumed inputs
        for idx, call in enumerate(tool_calls):
            tool = self.tool_registry.get(call.tool_name)
            if tool and hasattr(tool, 'consumes') and tool.consumes:
                for needed in tool.consumes:
                    if needed not in output_providers:
                        # Non-fatal: just skip this dependency
                        continue
                    prod_idx = output_providers[needed]
                    if prod_idx not in graph[idx]:  # Avoid duplicates
                        graph[prod_idx].append(idx)
                        indeg[idx] += 1

        # Topological sort
        queue = deque([i for i in range(n) if indeg[i] == 0])
        stages = []
        while queue:
            stage = []
            for _ in range(len(queue)):
                node = queue.popleft()
                stage.append(node)
                for nb in graph[node]:
                    indeg[nb] -= 1
                    if indeg[nb] == 0:
                        queue.append(nb)
            stages.append(stage)

        if sum(indeg) > 0:
            self.logger.warning("⚠️ Cycle detected in tool dependencies, executing sequentially")
            return [[i] for i in range(n)], output_providers

        return stages, output_providers

    def _substitute_args(self, args: Dict, output_values: Dict) -> Dict:
        new_args = {}
        for k, v in args.items():
            if isinstance(v, str) and v.startswith('{{') and v.endswith('}}'):
                key = v[2:-2].strip()
                if key in output_values:
                    new_args[k] = output_values[key]
                else:
                    self.logger.warning(f"Placeholder '{key}' not found in outputs, keeping as is")
                    new_args[k] = v
            else:
                new_args[k] = v
        return new_args

    async def _execute_stages(
            self,
            tool_calls: List[ToolCall],
            stages: List[List[int]]
    ) -> List[Any]:
        results = [None] * len(tool_calls)
        output_values = {}

        for stage_num, stage_indices in enumerate(stages):
            stage_calls = [tool_calls[i] for i in stage_indices]
            self.logger.info(f"⚡ Executing stage {stage_num + 1} with {len(stage_calls)} tool(s)")

            # Substitute args
            substituted_calls = []
            for call in stage_calls:
                new_args = self._substitute_args(call.args, output_values)
                new_call = ToolCall(
                    tool_name=call.tool_name,
                    args=new_args,
                    priority=call.priority,
                    timeout=call.timeout,
                    retry_count=call.retry_count,
                    max_retries=call.max_retries
                )
                substituted_calls.append(new_call)

            stage_results = await self._execute_tool_batch(substituted_calls)

            # Store results and update output_values
            for idx, call, res in zip(stage_indices, stage_calls, stage_results, strict=False):
                results[idx] = res
                tool = self.tool_registry.get(call.tool_name)
                if tool and tool.provides:
                    if isinstance(res, dict):
                        for out in tool.provides:
                            if out in res:
                                output_values[out] = res[out]
                            else:
                                output_values[out] = res
                    else:
                        if tool.provides:
                            output_values[tool.provides[0]] = res

        return results

    async def _execute_tool_batch(self, tool_calls: List[ToolCall]) -> List[Any]:
        """Execute a batch of independent tool calls concurrently."""
        if not tool_calls:
            return []

        async def _run_one(call: ToolCall) -> Any:
            async with self.request_semaphore:
                return await self._execute_tool_adaptive(call)

        results = await asyncio.gather(
            *[_run_one(call) for call in tool_calls],
            return_exceptions=True
        )
        return results

    async def _execute_tool(self, tool_call: ToolCall) -> ExecutionResult:
        start_time = time.time()
        tool_name = tool_call.tool_name
        try:
            tool = self.tool_registry.get(tool_name)
            if not tool:
                return ExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    duration=0,
                    error=f"Tool not found: {tool_name}"
                )

            if not self.tool_breaker.can_execute(tool_name):
                return ExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    duration=0,
                    error="Tool circuit breaker is open"
                )

            self.logger.info(f"⚡ Executing tool: {tool_name}")
            async with self.request_semaphore:
                result = await asyncio.wait_for(
                    self._call_tool_async(tool, tool_call.args),
                    timeout=tool_call.timeout
                )

            duration = time.time() - start_time
            success = self._is_result_successful(result)

            if success:
                self.tool_breaker.record_success(tool_name)
                self.metrics["total_tools_executed"] += 1
            else:
                self.tool_breaker.record_failure(tool_name)
                self.metrics["total_tools_failed"] += 1

            return ExecutionResult(
                tool_name=tool_name,
                success=success,
                result=result,
                duration=duration,
                insight=self._extract_insight(result)
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.logger.error(f"⏱️ Tool timeout: {tool_name} ({tool_call.timeout}s)")
            self.tool_breaker.record_failure(tool_name)
            self.metrics["total_tools_failed"] += 1
            return ExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                duration=duration,
                error=f"Timeout after {tool_call.timeout}s"
            )

        except asyncio.CancelledError:
            self.logger.info(f"🛑 Tool '{tool_name}' cancelled")
            raise

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"❌ Tool execution failed: {e}")
            self.tool_breaker.record_failure(tool_name, exception=e)
            self.metrics["total_tools_failed"] += 1
            return ExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                duration=duration,
                error=str(e)[:200]
            )

    async def _execute_tool_adaptive(self, tool_call: ToolCall) -> Any:
        args = tool_call.args or {}
        tool_name = tool_call.tool_name
        tool = self.tool_registry.get(tool_name)

        # 1. Unknown tool — bail out before touching .name.
        if tool is None:
            self.logger.warning(f"⛔ Unknown tool '{tool_name}'")
            return {"error": f"Unknown tool '{tool_name}'"}

        if not self.tool_breaker.can_execute(tool_name):
            self.logger.warning(f"⛔ Tool '{tool_name}' is circuit‑broken")
            return {"error": f"Tool '{tool_name}' temporarily unavailable"}

        # Estimate timeout. May issue a HEAD request for http/download tools;
        # a failure there must not abort the whole call, so wrap it.
        try:
            final_timeout = await self._estimate_base_timeout(tool, args)
        except Exception as exc:
            self.logger.warning(
                f"⚠️ Timeout estimation for '{tool_name}' failed ({exc}); using 60s"
            )
            final_timeout = 60.0

        self.logger.info(f"🕐 Tool '{tool_name}' timeout: {final_timeout:.1f}s")

        fn = getattr(tool, "func", None)
        if not callable(fn):
            self.logger.error(f"❌ Tool '{tool_name}' has no callable func")
            return {"error": f"Tool '{tool_name}' is not callable"}

        try:
            if inspect.iscoroutinefunction(fn):
                # Real coroutine — await it directly. Running an async
                # function through run_in_executor only *builds* the
                # coroutine object in the worker thread and returns it as
                # the result; the coroutine never runs and is later GC'd
                # un-awaited (same failure mode as ChatClient.runtime_limits).
                result = await asyncio.wait_for(fn(**args), timeout=final_timeout)
            else:
                # Sync callable — offload to a worker thread so a blocking
                # tool (subprocess, sync HTTP lib, large file I/O) does not
                # stall the event loop.
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: fn(**args)),
                    timeout=final_timeout,
                )
            self.tool_breaker.record_success(tool_name)
            return result

        except asyncio.TimeoutError:
            # A hang is a real failure — record it so repeated hangs open
            # the circuit. Previously this path returned a partial result
            # without ever touching the breaker.
            self.tool_breaker.record_failure(tool_name)
            self.logger.warning(
                f"⏱️ Tool '{tool_name}' timed out after {final_timeout:.1f}s"
            )
            return self._create_partial_result(tool, args, final_timeout)

        except asyncio.CancelledError:
            # Caller cancelled (client disconnect, shutdown). Not the tool's
            # fault — do not pollute the breaker. Re-raise so the caller's
            # cancellation semantics are preserved.
            self.logger.info(f"🛑 Tool '{tool_name}' cancelled")
            raise

        except Exception as e:
            self.tool_breaker.record_failure(tool_name, exception=e)
            self.logger.error(f"❌ Tool '{tool_name}' failed: {e}")
            return {
                "error": f"Tool execution failed: {str(e)}",
                "tool": tool_name,
                "args": args,
            }

    async def _estimate_base_timeout(self, tool: Tool, args: Dict) -> float:
        """Estimate safe timeout for a tool call."""
        tool_defaults = {
            "http_fetch": 30,
            "http_fetch_batch": 60,
            "download_file": 300,
            "download_multiple": 600,
            "scrape_page": 60,
            "scrape_batch": 120,
            "transcribe_audio_or_video": 600,
            "transcribe_multiple": 1200,
            "extract_text_from_file": 120,
            "batch_extract_files": 300,
            "video_info": 30,
            "download_video": 600,
            "extract_video_thumbnail": 60,
            "search_links": 60,
            "search_with_dorks": 120,
            "find_file_links": 90,
        }
        base = tool_defaults.get(tool.name, 60)

        if tool.cost_per_unit is not None:
            size_mb = await self._estimate_input_size_async(tool.name, args)
            if size_mb is not None:
                return max(10, size_mb * tool.cost_per_unit * 1.1)

        if tool.operation_type == "batch":
            count = self._extract_batch_count(args)
            if count:
                return max(10, base * count)

        return base

    async def _estimate_input_size_async(self, tool_name: str, args: Dict) -> Optional[float]:
        """Estimate data size (MB) from tool arguments."""
        if tool_name in ("http_fetch", "download_file") and "url" in args:
            url = args["url"]
            now = asyncio.get_running_loop().time()

            if url in self._size_cache:
                age = now - self._size_cache_timestamp.get(url, 0)
                if age < self._size_cache_max_age:
                    return self._size_cache[url]

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.head(url, timeout=5.0, allow_redirects=True) as resp:
                        content_length = resp.headers.get('Content-Length')
                        if content_length:
                            size_mb = int(content_length) / (1024 * 1024)
                            self._size_cache[url] = size_mb
                            self._size_cache_timestamp[url] = now
                            return size_mb
            except Exception:
                pass

        return None

    def _extract_batch_count(self, args: Dict) -> Optional[int]:
        """Extract number of items in a batch operation."""
        if "urls" in args and isinstance(args["urls"], (list, tuple)):
            return len(args["urls"])
        if "paths" in args and isinstance(args["paths"], (list, tuple)):
            return len(args["paths"])
        return None

    def _create_partial_result(self, tool: Tool, args: Dict, timeout: float) -> Any:
        """Return graceful partial result when tool times out."""
        return {
            "error": f"Tool '{tool.name}' timed out after {timeout:.1f}s",
            "partial": True,
            "args": args,
            "suggestion": "Consider reducing input size or using more powerful system."
        }

    async def _call_tool_async(self, tool, args: Dict[str, Any]) -> Any:
        fn = tool.func
        if inspect.iscoroutinefunction(fn):
            return await fn(**args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(**args))

    # =====================================================================
    # SAFETY & VALIDATION
    # =====================================================================

    def _filter_available_tools(self, goal: MicroGoal) -> List[str]:
        """Filter tools appropriate for a goal."""
        try:
            memory_percent = psutil.virtual_memory().percent
            all_tools = list(self.tool_registry.tools.keys())

            ranked_tools = self.tool_engine.filter_tools_for_goal(
                goal,
                all_tools,
                self._tokens_remaining(),
                memory_percent
            )

            filtered = [t[0] for t in ranked_tools[:10]]
            self.logger.info(f"📋 Filtered to {len(filtered)} tools")
            return filtered

        except Exception as e:
            self.logger.error(f"❌ Tool filtering failed: {e}")
            return list(self.tool_registry.tools.keys())[:5]

    def _get_next_goal(self) -> Optional[MicroGoal]:
        """Get next incomplete goal."""
        for goal in self.micro_goals:
            if not goal.completed:
                return goal
        return None

    def _validate_tool_safety(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Validate tool call safety before execution."""
        tool_name = tool_call.tool_name.lower()

        dangerous_tools = {
            "shell", "exec", "system", "eval", "subprocess",
            "os.system", "__import__", "compile", "open_file"
        }
        if tool_name in dangerous_tools:
            return {"allowed": False, "reason": f"Dangerous tool: {tool_name}"}

        dangerous_patterns = [
            r'rm\s+-rf', r'format\s+c:', r'del\s+/s\s+/q',
            r'drop\s+table', r'truncate\s+table', r'delete\s+from',
            r'chmod\s+777', r'chown\s+-R', r'exec\s*\(', r'eval\s*\(',
            r'__import__', r'subprocess\.', r'os\.system', r'os\.remove',
            r'password\s*=\s*["\']', r'private\s+key', r'secret\s+key',
        ]

        for arg_name, arg_value in tool_call.args.items():
            if isinstance(arg_value, str):
                for pattern in dangerous_patterns:
                    if re.search(pattern, arg_value, re.IGNORECASE):
                        return {"allowed": False, "reason": f"Dangerous pattern: {pattern}"}

        return {"allowed": True, "reason": "Tool appears safe"}

    def _check_tool_safety(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Check tool safety."""
        return self._validate_tool_safety(tool_call)

    def _validate_tool_payload(self, payload: dict) -> Tuple[bool, str]:
        """Validate tool payload structure."""
        if not isinstance(payload, dict):
            return False, "Payload not a dictionary"
        if "tool" not in payload:
            return False, "Missing 'tool' field"
        if not isinstance(payload["tool"], str):
            return False, "'tool' must be string"
        if "args" not in payload:
            payload["args"] = {}
        if not isinstance(payload["args"], dict):
            return False, "'args' must be dictionary"
        return True, ""

    # =====================================================================
    # REFLECTION & ADAPTATION
    # =====================================================================

    async def _should_reflect(self, iteration: int, progress: float) -> bool:
        """Decide if reflection is useful now."""
        if iteration < 2:
            return False
        if progress > 80:
            return False
        if time.time() - self.start_time < 30:
            return False
        # Check if progress stalled
        if iteration > 0:
            prev_progress = self._iteration_progress.get(iteration - 1, 0)
            if progress <= prev_progress:
                return True  # Progress stalled - reflect
        return iteration % 5 == 0  # Reflect every 5 iterations

    async def _reflect(
            self,
            instruction: str,
            history: str,
            steps: List[Dict],
            goal_summary: Dict[str, int]
    ) -> Optional[str]:
        """Reflect on execution progress and suggest improvements."""
        if not self.enable_reflection:
            return None

        reflection_prompt = f"""Analyze this agent's progress on a task.

                            INSTRUCTION: {instruction}

                            GOAL PROGRESS: {goal_summary['completed']}/{goal_summary['total']} goals complete

                            RECENT ACTIONS (last 3):
                            {json.dumps(steps[-3:], indent=2)}

                            ANALYSIS QUESTIONS:
                            1. Is the agent making progress toward the goals?
                            2. Are better tools or approaches available?
                            3. What should the agent do next?
                            4. Are there any failures to address?

                            PROVIDE CONCISE ACTIONABLE FEEDBACK (2-3 sentences max):"""

        try:
            reflection = await self.llm.generate(
                prompt=reflection_prompt,
                max_tokens=150,
                temperature=0.3,
                timeout=30.0
            )
            return reflection.strip()
        except Exception as e:
            self.logger.error(f"❌ Reflection failed: {e}")
            return None

    # =====================================================================
    # LLM INTERACTION
    # =====================================================================
    async def _call_llm_with_retry(
            self,
            prompt: str,
            max_retries: int = 3,
            backoff: Optional[AdaptiveBackoff] = None
    ) -> str:
        """
        Call the LLM with intelligent retry strategy.

        Two changes from the original:

        1. Uses ``self.timeout_per_iteration`` which is now computed by
           SmartTimeoutCalculator instead of being hard-coded at 120 s.
           On CPU-only Mistral-7B Q4 @ 2 tok/s the correct value is ~563 s.

        2. Logs the approximate prompt token count before each call so the
           token budget situation is always visible in the log (previously the
           budget contradiction between model_manager and actual context was
           invisible, making diagnosis very hard).

        3. Performs a quick sanity check after receiving a response: if the
           response contains no ``{`` character it is unlikely to be valid JSON,
           so we warn and retry rather than returning a string that will fail
           ``amb_actions.parse_action`` on the next step.

        4. Caps max_tokens to remaining budget (max_total_tokens - tokens_used)
           to avoid exceeding the overall token budget.

        5. Catches the typed LLMError subclasses raised by the
           adapters (LLMServerTimeout, LLMTransportError, LLMModelNotReady,
           LLMError) and treats them the same as any retryable failure —
           exponential backoff, then retry. Previously these exceptions
           would propagate straight out of _call_llm_with_retry and abort
           the entire arun() iteration with a generic Exception, which the
           outer loop then swallowed as "cycle failed". With this fix the
           loop keeps retrying on transport/server errors as intended.
        """
        # Log prompt size so token budget is transparent in the logs
        if self.model_manager.token_counter:
            prompt_tok = self.model_manager.token_counter.count(prompt)
        else:
            prompt_tok = int(len(prompt.split()) * 1.3)
        ctx = getattr(self.model_manager, "context_window", 4096)
        self.logger.info(
            f"📐 Prompt tokens: ~{prompt_tok} / {ctx}  "
            f"(generation budget: ~{ctx - prompt_tok - 100} tokens)  "
            f"timeout: {self.timeout_per_iteration:.0f}s"
        )

        # Calculate safe max_tokens based on remaining budget
        remaining_budget = self.max_total_tokens - self.tokens_used
        # Ensure at least 100 tokens for generation, and never exceed 2048 to avoid huge generations
        safe_max_tokens = max(100, min(remaining_budget, 2048))

        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    self.llm.generate(
                        prompt=prompt,
                        max_tokens=safe_max_tokens,
                        temperature=self.planner_temperature
                    ),
                    timeout=self.timeout_per_iteration
                )

                if not response or not response.strip():
                    self.logger.warning(f"Attempt {attempt + 1}: Empty response")
                    if backoff:
                        backoff.record_failure()
                        await backoff.wait()
                    continue

                # Sanity check: expect JSON-like content
                if "{" not in response:
                    self.logger.warning(
                        f"Attempt {attempt + 1}: Response contains no JSON "
                        f"(first 80 chars): {response[:80]!r}"
                    )
                    if backoff:
                        backoff.record_failure()
                        if attempt < max_retries - 1:
                            await backoff.wait()
                    continue

                self.logger.info(
                    f"✅ LLM response (attempt {attempt + 1}): {response[:120]!r}"
                )
                return response

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Attempt {attempt + 1}: Timeout after "
                    f"{self.timeout_per_iteration:.0f}s"
                )
                if backoff:
                    backoff.record_failure()
                    if attempt < max_retries - 1:
                        await backoff.wait()
                continue

            # Typed LLM errors — treat transport / server / model-not-ready
            # as retryable, back off, and continue. Without this catch the
            # exception propagates out and aborts the whole arun() iteration.
            except LLMServerTimeout as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}: Server timeout: {e}"
                )
                if backoff:
                    backoff.record_failure()
                    if attempt < max_retries - 1:
                        await backoff.wait()
                continue

            except LLMTransportError as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}: Transport error: {e}"
                )
                if backoff:
                    backoff.record_failure()
                    if attempt < max_retries - 1:
                        await backoff.wait()
                continue

            except LLMModelNotReady as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}: Model not ready: {e}"
                )
                if backoff:
                    backoff.record_failure()
                    if attempt < max_retries - 1:
                        await backoff.wait()
                continue

            except LLMError as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}: {type(e).__name__}: {e}"
                )
                if backoff:
                    backoff.record_failure()
                    if attempt < max_retries - 1:
                        await backoff.wait()
                continue

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}: {e}")
                if backoff:
                    backoff.record_failure()
                    if attempt < max_retries - 1:
                        await backoff.wait()
                continue

        return "ERROR: Unable to get valid LLM response after retries"

    # =====================================================================
    # RESULT COMPILATION & REPORTING
    # =====================================================================

    async def _generate_completion_report(self) -> str:
        """Generate report when all goals completed."""
        summary = self.current_context.goals_summary()
        elapsed = self.current_context.get_elapsed_time()

        report = f"""
                🎉 EXECUTION COMPLETED SUCCESSFULLY

                GOAL COMPLETION SUMMARY:
                - Total Goals: {summary['total']}
                - Completed: {summary['completed']}
                - Success Rate: {(summary['completed'] / summary['total']) * 100:.0f}%

                EXECUTION METRICS:
                - Duration: {elapsed:.1f}s
                - Iterations: {self.iteration_count}
                - Tokens Used: {self.tokens_used}
                - Conversation ID: {self.conversation_id}

                COMPLETED GOALS:
                """
        for goal in self.goals:
            if goal.status == GoalStatus.COMPLETED:
                report += f"\n✅ {goal.title}"
                if goal.completed_by:
                    report += f" (by {goal.completed_by})"

        return report

    def _generate_loop_completion_summary(self) -> str:
        """Generate summary when max iterations reached."""
        summary = self.current_context.goals_summary()
        progress = self.current_context.progress_percentage()

        return f"""
            ⏱️ EXECUTION REACHED MAX ITERATIONS

            FINAL STATUS:
            - Progress: {progress:.0f}%
            - Goals Completed: {summary['completed']}/{summary['total']}
            - Iterations Used: {self.iteration_count}/{self.max_iterations}
            - Duration: {self.current_context.get_elapsed_time():.1f}s

            NEXT STEPS:
            Continue execution to complete remaining goals in next iteration.
            """

    def _create_success_result(self) -> Dict[str, Any]:
        """Compile execution success result."""
        duration = time.time() - self.start_time
        self.metrics["avg_iteration_duration"] = duration / max(1, self.iteration_count)

        return {
            "success": len(self.completed_goals) > 0,
            "execution_id": self.current_execution_id,
            "conversation_id": self.conversation_id,
            "instruction": self.instruction,
            "goals": {
                "total": len(self.micro_goals),
                "completed": len(self.completed_goals),
                "failed": len(self.failed_goals),
                "pending": len([g for g in self.micro_goals if not g.completed])
            },
            "metrics": {
                "total_duration": duration,
                "iterations": self.iteration_count,
                "tokens_used": self.tokens_used,
                "progress_percentage": self._calculate_progress_percentage(),
                **self.metrics
            },
            "execution_log": self.execution_log,
            "errors": dict(self.error_counts)
        }

    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Compile execution error result."""
        duration = time.time() - self.start_time

        return {
            "success": False,
            "execution_id": self.current_execution_id,
            "conversation_id": self.conversation_id,
            "error": error,
            "metrics": {
                "total_duration": duration,
                "iterations": self.iteration_count,
                "tokens_used": self.tokens_used,
                **self.metrics
            },
            "errors": dict(self.error_counts)
        }

    # =====================================================================
    # UTILITY METHODS
    # =====================================================================

    def _summarize_result(self, result: Any) -> str:
        """
        Produce a compact human-readable summary of a tool result for the history.
        The LLM reads this to understand what was accomplished.
        """
        if result is None:
            return "None"
        if isinstance(result, str):
            return f"text ({len(result)} chars)" if len(result) > 30 else result
        if isinstance(result, list):
            return f"list with {len(result)} items"
        if isinstance(result, dict):
            parts = []
            # Counts
            for key in ("count", "total_count", "total", "row_count"):
                v = result.get(key)
                if v is not None:
                    parts.append(f"{key}={v}")
                    break
            # List lengths
            for key in ("links", "urls", "entries", "results", "items",
                        "files", "rows", "records", "chunks"):
                v = result.get(key)
                if isinstance(v, list) and v:
                    parts.append(f"{len(v)} {key}")
                    break
            # Text content
            for key in ("text", "content", "transcript", "extracted_text",
                        "body", "fetched_content", "html"):
                v = result.get(key)
                if v and len(str(v)) > 10:
                    parts.append(f"{key}={len(str(v))} chars")
                    break
            # URL stored
            if result.get("url"):
                parts.append(f"url={str(result['url'])[:60]}")
            if parts:
                return ", ".join(parts)
            return str(result)[:80]
        return str(result)[:80]

    def _is_result_successful(self, result: Any) -> bool:
        """Evaluate if tool result indicates success."""
        if result is None:
            return False
        if isinstance(result, dict):
            _err2 = result.get("error")
            if _err2 is not None and _err2 is not False and _err2 != "":
                return False
            if result.get("success") is True:
                return True
            if any(k in result for k in ["data", "content", "results", "items"]):
                return bool(result.get("data") or result.get("content"))
        if isinstance(result, str):
            return len(result) > 10 and "error" not in result.lower()
        if isinstance(result, list):
            return len(result) > 0
        return True

    def _extract_insight(self, result: Any) -> Optional[str]:
        """Extract key insight from tool result."""
        if isinstance(result, dict):
            if "insight" in result:
                return result["insight"]
            if "summary" in result:
                return result["summary"]
        if isinstance(result, str) and len(result) > 0:
            return result[:150]
        return None

    def _is_timed_out(self) -> bool:
        """Check if execution has timed out."""
        return (time.time() - self.start_time) > self.timeout

    def _time_remaining(self) -> float:
        """Get remaining execution time."""
        return max(0, self.timeout - (time.time() - self.start_time))

    def _tokens_remaining(self) -> int:
        """Get remaining token budget."""
        return max(0, self.max_total_tokens - self.tokens_used)

    def _calculate_progress_percentage(self) -> float:
        """Calculate execution progress as percentage."""
        if not self.micro_goals:
            return 0.0
        completed = len(self.completed_goals)
        return (completed / len(self.micro_goals)) * 100

    def _log_step(
            self,
            step_type: str,
            description: str,
            success: Optional[bool],
            duration: float = 0.0
    ):
        """Log execution step."""
        self.step_count += 1
        log_entry = {
            "step_id": self.step_count,
            "type": step_type,
            "description": description,
            "success": success,
            "duration": duration,
            "timestamp": time.time()
        }
        self.execution_log.append(log_entry)
        if self.trace_enabled:
            self.execution_trace.append(log_entry)

    # =====================================================================
    # DATABASE INTEGRATION
    # =====================================================================

    async def _log_agent_execution_to_db(self, result: Dict[str, Any]):
        """Persist execution result to database."""
        if not self._service or not hasattr(self._service, 'db_manager'):
            return

        db = self._service.db_manager
        if not db:
            return

        try:
            query = """
                INSERT INTO agent_executions 
                (execution_id, conversation_id, instruction, status, result, success,
                 total_duration, iterations, tokens_used, steps_count, tools_used,
                 error_message, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (execution_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
            """

            tools_used = list(set(
                log["description"] for log in self.execution_log
                if log["type"] == "TOOL_SUCCESS"
            ))

            params = (
                self.current_execution_id,
                self.conversation_id,
                self.instruction[:500],
                "completed" if result.get("success") else "failed",
                json.dumps(result),
                result.get("success", False),
                result.get("metrics", {}).get("total_duration", 0),
                self.iteration_count,
                self.tokens_used,
                len(self.execution_log),
                json.dumps(tools_used),
                result.get("error", ""),
                json.dumps(result.get("metrics", {}))
            )

            db_result = db.execute("primary", query, params, fetch="none")
            if db_result.get("ok"):
                self.logger.info(f"✅ Execution persisted: {self.current_execution_id}")
            else:
                self.logger.error(f"❌ Database persist failed")

        except Exception as e:
            self.logger.error(f"❌ Error persisting: {e}")

    async def _write_execution_backup(self, result: Dict[str, Any]):
        """Write execution result to local backup file."""
        try:
            backup_dir = paths.backup_dir()
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = backup_dir / f"exec_{self.conversation_id}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "execution_id": self.current_execution_id,
                    "conversation_id": self.conversation_id,
                    "instruction": self.instruction,
                    "result": result,
                    "timestamp": time.time()
                }, f, indent=2)
            self.logger.info(f"💾 Execution backed up to {filename}")
        except Exception as e:
            self.logger.error(f"❌ Failed to write backup: {e}")

    # =====================================================================
    # HEALTH & MONITORING
    # =====================================================================

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of last execution."""
        return {
            "execution_id": self.current_execution_id,
            "conversation_id": self.conversation_id,
            "iterations": self.iteration_count,
            "tokens_used": self.tokens_used,
            "goals_completed": len(self.completed_goals),
            "goals_total": len(self.micro_goals),
            "progress": self._calculate_progress_percentage(),
            "tools_used": len([log for log in self.execution_log if log["type"] == "TOOL_SUCCESS"]),
            "errors": dict(self.error_counts),
            "metrics": self.metrics
        }

    def is_healthy(self) -> bool:
        """Check executor health."""
        return (
                self.llm is not None
                and self.tool_registry is not None
                and not self._shutdown
        )

    # =====================================================================
    # INTERNAL SIZE CACHE FOR ADAPTIVE TIMEOUTS
    # =====================================================================

    def _init_size_cache(self):
        """Initialize size cache for adaptive timeouts."""
        self._size_cache: Dict[str, float] = {}
        self._size_cache_timestamp: Dict[str, float] = {}
        self._size_cache_max_age = 300  # 5 minutes


# ═══════════════════════════════════════════════════════════════════════════
#  DataAnalyzer — Inline, fully-configurable, production-grade analysis engine
#  v1.0.0 — Replaces ThreatAnalyzer completely.
#
#  Domain-agnostic: security / social / financial / health / research / generic
#  Driven 100% by config.yaml → data_analyzer section.
#  No external file, no hard-coded table or column names.
#
#  Architecture:
#  - Background thread runs _loop() on configurable interval
#  - _loop() → asyncio.run(_run_batch_async())
#  - Batch fetched from raw_data_table where analyzed=FALSE (via stream_query if available)
#  - Each row analysed in a bounded async semaphore (max_concurrent workers)
#  - Result saved to analysis_results_table; raw row marked analyzed=TRUE
#  - Circuit breaker: 5 consecutive LLM failures → 120s cooldown
#  - Per-item timeout enforced via asyncio.wait_for
#  - Exponential retry with configurable max_retries
#  - Full metrics: items_analysed, items_failed, avg_item_ms, circuit_breaker state
# ═══════════════════════════════════════════════════════════════════════════

# ── Backward-compat: code that imports ThreatLevel / AnalysisResult still works ─
class ThreatLevel(Enum):
    CRITICAL      = "critical"
    HIGH          = "high"
    MEDIUM        = "medium"
    LOW           = "low"
    INFORMATIONAL = "informational"


@dataclass
class AnalysisResult:
    """🔍 Enhanced result of threat analysis with validation."""
    content: str
    threat_level: ThreatLevel
    confidence: float
    entities: List[Dict[str, Any]]
    categories: List[str]
    reasoning: str
    action_required: bool
    urgency: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """🔍 Validate analysis result."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

        if self.urgency not in ["immediate", "high", "medium", "low", "none"]:
            raise ValueError(f"Invalid urgency level: {self.urgency}")


class DataAnalyzer:
    """
    🔍 DataAnalyzer v1.0.0 — Inline, fully-configurable, production-grade analysis engine.

    Domain-agnostic: security / social / financial / health / research / generic
    100% config-driven via config.yaml → data_analyzer section.

    Architecture:
    - Async coroutine runs _run_batch_async() on configurable interval
    - Batch fetched from raw_data_table WHERE analyzed=FALSE (ordered by config col)
    - Each row analysed in a bounded asyncio.Semaphore (max_concurrent workers)
    - Result saved to analysis_results_table; raw row marked analyzed=TRUE
    - Circuit breaker: N consecutive LLM failures → cooldown pause
    - Per-item timeout enforced via asyncio.wait_for
    - Exponential retry with configurable max_retries
    - Full metrics: items_analysed, items_failed, avg_item_ms, circuit_breaker state

    Backward compat: ThreatAnalyzer = DataAnalyzer alias defined at module level.
    """

    # ── Domain keyword maps ──────────────────────────────────────────────────
    _DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "security":  ["threat", "attack", "exploit", "vulnerability", "malware",
                      "phishing", "cve", "ransomware", "intrusion", "incident",
                      "kidnap", "ransom", "bandit", "insecurity", "crime",
                      "terrorism", "militia", "fraud", "breach", "hacker"],
        "social":    ["social media", "sentiment", "opinion", "tweet", "post",
                      "comment", "disinformation", "propaganda", "narrative",
                      "influence", "fake news", "misinformation"],
        "financial": ["fraud", "money laundering", "trading", "financial", "payment",
                      "scam", "banking", "cryptocurrency", "market manipulation",
                      "embezzlement", "ponzi", "investment"],
        "health":    ["health", "disease", "outbreak", "epidemic", "pandemic",
                      "medical", "patient", "symptom", "treatment", "vaccine",
                      "drug abuse", "mortality"],
        "research":  ["research", "academic", "paper", "study", "publication",
                      "journal", "finding", "discovery", "hypothesis",
                      "experiment", "dataset"],
    }

    # ── Domain-specific prompt templates ────────────────────────────────────
    _DOMAIN_PROMPTS: Dict[str, str] = {
        "security": (
            "You are a professional security threat analyst. Analyze the following content "
            "for security threats, attacks, vulnerabilities, crimes, and risks. "
            "Return ONLY valid JSON (no extra text).\n\n"
        ),
        "social": (
            "You are a social intelligence analyst. Analyze the following content for "
            "sentiment, disinformation, influence operations, and social risks. "
            "Return ONLY valid JSON (no extra text).\n\n"
        ),
        "financial": (
            "You are a financial intelligence analyst. Analyze the following content for "
            "fraud, financial crimes, market manipulation, and monetary risks. "
            "Return ONLY valid JSON (no extra text).\n\n"
        ),
        "health": (
            "You are a public health intelligence analyst. Analyze the following content "
            "for disease outbreaks, health risks, and epidemiological signals. "
            "Return ONLY valid JSON (no extra text).\n\n"
        ),
        "research": (
            "You are a research intelligence analyst. Analyze the following content for "
            "key findings, methodologies, risks, and significant insights. "
            "Return ONLY valid JSON (no extra text).\n\n"
        ),
        "generic": (
            "You are a data intelligence analyst. Analyze the following content and "
            "extract key intelligence signals, risks, and actionable insights. "
            "Return ONLY valid JSON (no extra text).\n\n"
        ),
    }

    # ── Default prompt body (appended after domain preamble) ─────────────────
    _PROMPT_BODY = """\
Content: {content}

Context:
- Source: {source_type}
- Location: {location}
- Language: {language}
- Timestamp: {timestamp}
- Content Length: {content_length} characters
- Domain: {domain}

Analysis Guidelines:
1. Assess overall severity / threat level
2. Identify relevant categories for this domain
3. Extract key entities (locations, names, organizations, IPs, etc.)
4. Provide confidence score (0.0–1.0)
5. Determine if immediate action is required
6. Set appropriate urgency level
7. Estimate affected population / area if applicable
8. Rate source credibility (0.0–1.0)

Required JSON Response Format:
{{
    "threat_level": "CRITICAL|HIGH|MEDIUM|LOW|INFORMATIONAL",
    "severity_score": 0.0,
    "confidence": 0.0,
    "categories": ["category1", "category2"],
    "entities": [
        {{"type": "entity_type", "value": "entity_value", "score": 0.0}}
    ],
    "keywords": ["keyword1", "keyword2"],
    "reasoning": "Detailed analysis explaining the assessment...",
    "action_required": true,
    "urgency": "immediate|high|medium|low|none",
    "location": "Identified location or region",
    "affected_population": "Estimated affected group",
    "incident_count": 0,
    "source_credibility": 0.0,
    "metadata": {{}},
    "warnings": ["warning1"]
}}

Important: Return ONLY the JSON object, nothing else.

Analysis:"""

    # ── Default config (used when config.yaml has no data_analyzer section) ─
    _DEFAULT_CONFIG: Dict[str, Any] = {
        "enabled": True,
        "interval": 1200,
        "batch_size": 10,
        "max_concurrent": 3,
        "max_retries": 3,
        "timeout": 90,
        "circuit_breaker_threshold": 5,
        "circuit_breaker_cooldown": 120,
        "raw_data_table": "raw_data",
        "raw_data_id_col": "id",
        "raw_data_content_col": "content",
        "raw_data_analyzed_col": "analyzed",
        "raw_data_source_col": "source_type",
        "raw_data_order_col": "id",
        "analysis_results_table": "analysis_results",
        "domain": "auto",
        "temperature": 0.1,
        "max_tokens": 512,
        "prompt_template": "",
    }

    # ────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        llm: "BaseLLMAdapter",
        tool_registry: Optional["ToolRegistry"] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: float = 60.0,
        enable_validation: bool = True,
        da_config: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.3,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.enable_validation = enable_validation
        self.min_confidence = min_confidence

        # Load DataAnalyzer config (from constructor or defaults; service config
        # is merged in set_service() once the service is available)
        self._da_cfg: Dict[str, Any] = dict(self._DEFAULT_CONFIG)
        if da_config:
            self._da_cfg.update(da_config)
            self.min_confidence = float(self._da_cfg.get("min_confidence", min_confidence))

        # Override from constructor args (backward-compat)
        self._da_cfg.setdefault("temperature", temperature)
        self._da_cfg.setdefault("max_tokens", max_tokens)
        self._da_cfg.setdefault("timeout", timeout)
        self._da_cfg["temperature"] = temperature
        self._da_cfg["max_tokens"] = max_tokens

        # Resolved at set_service() time
        self._domain: str = "generic"
        self._prompt_template: str = ""

        # Runtime state
        self._running = False
        self._stop_flag = False
        self._service = None

        # Metrics
        self._metrics: Dict[str, Any] = {
            "items_analyzed": 0,
            "items_failed": 0,
            "items_skipped": 0,
            "cycles": 0,
            "total_time_ms": 0.0,
            "avg_item_ms": 0.0,
            "cb_trips": 0,
            "last_cycle": None,
        }

        # Inline circuit breaker
        self._cb_failures = 0
        self._cb_state = "closed"   # closed | open | half_open
        self._cb_opened_at: float = 0.0

        self.logger = logging.getLogger("DataAnalyzer")

    # ── Service wiring ───────────────────────────────────────────────────────
    def set_service(self, service: Any) -> None:
        """Set reference to parent service; extract data_analyzer config from yaml."""
        self._service = service
        try:
            raw_cfg: Dict[str, Any] = getattr(service, "config", {}) or {}
            da_section: Dict[str, Any] = raw_cfg.get("data_analyzer", {})
            if da_section:
                self._da_cfg.update(da_section)
                self.logger.info("✅ DataAnalyzer config loaded from config.yaml")
            else:
                self.logger.warning("⚠️ No data_analyzer section in config.yaml — using defaults")
        except Exception as exc:
            self.logger.warning(f"⚠️ Could not read data_analyzer config: {exc}")

        # Detect domain from agent_instruction
        instruction: str = ""
        try:
            instruction = service.service_config.agent_instruction or ""
        except Exception:
            pass
        cfg_domain = self._da_cfg.get("domain", "auto")
        if cfg_domain == "auto":
            self._domain = self._detect_domain(instruction)
        else:
            self._domain = cfg_domain
        self.logger.info(f"🔍 DataAnalyzer domain: {self._domain}")

        # Build effective prompt template
        self._prompt_template = self._build_prompt_template()

        # Derive the LLM timeout from the live server's measured rates.
        # Same principle as CapabilityProfile: no hardcoded ceilings.
        #
        # The analyzer truncates content at 15_000 chars (in analyze())
        # before formatting it into the template. Estimate the worst case
        # prompt and out size, ask the resolver for the server's actual
        # decode/prefill rates, and derive a timeout with a 1.5x safety
        # margin. Never shrink a value the operator deliberately raised.
        try:
            from amb_sentinel.llm_config import (
                resolve as _lc_resolve,
                est_call_latency_s as _lc_est,
            )
            _rc = _lc_resolve(raw_cfg)

            _decode = float(_rc.get("decode_tps") or 2.0)
            # Server may not report prefill; use the same fallback the
            # capability profile uses (2× decode). Not a hardcoded
            # number: it's the documented fallback for "server didn't
            # measure".
            _prefill = float(_rc.get("prefill_tps") or 0.0) or (_decode * 2.0)

            # 15_000 chars / 3.0 chars-per-token ≈ 5_000 tokens of
            # content + ~1_500 tokens for the template body and the JSON
            # schema example. Both come from the analyzer's own code,
            # not from a magic number.
            _content_chars = 15_000
            _prompt_tokens = int(_content_chars / 3.0) + 1_500
            _out_tokens = int(self._da_cfg.get("max_tokens", 512))

            _est_s = _lc_est(_prompt_tokens, _out_tokens, _decode, _prefill)
            _required = _est_s * 1.5   # 50% safety margin for jitter

            _configured = float(self._da_cfg.get("timeout", 90))
            if _configured < _required:
                self._da_cfg["timeout"] = _required
                self.logger.warning(
                    "DataAnalyzer timeout auto-raised %.0fs → %.0fs "
                    "(server: %.2f decode / %.2f prefill tok/s, "
                    "est call %.0fs, ×1.5 safety)",
                    _configured, _required, _decode, _prefill, _est_s,
                )
            else:
                self.logger.info(
                    "DataAnalyzer timeout %.0fs "
                    "(server: %.2f decode / %.2f prefill tok/s, "
                    "est call %.0fs)",
                    _configured, _decode, _prefill, _est_s,
                )
        except Exception as _e:
            self.logger.warning(
                "Could not derive DataAnalyzer timeout from server: %s "
                "(using config value %ss)",
                _e, self._da_cfg.get("timeout"),
            )

    # ── Domain detection ─────────────────────────────────────────────────────
    def _detect_domain(self, text: str) -> str:
        """Detect domain from text by keyword frequency scoring."""
        if not text:
            return "generic"
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            scores[domain] = sum(1 for kw in keywords if kw in text_lower)
        best = max(scores, key=lambda d: scores[d])
        return best if scores[best] > 0 else "generic"

    def _build_prompt_template(self) -> str:
        """Build the effective prompt template."""
        custom = (self._da_cfg.get("prompt_template") or "").strip()
        if custom:
            return custom
        preamble = self._DOMAIN_PROMPTS.get(self._domain, self._DOMAIN_PROMPTS["generic"])
        return preamble + self._PROMPT_BODY

    # ── Circuit breaker ──────────────────────────────────────────────────────
    def _cb_is_open(self) -> bool:
        threshold = int(self._da_cfg.get("circuit_breaker_threshold", 5))
        cooldown  = float(self._da_cfg.get("circuit_breaker_cooldown", 120))
        if self._cb_state == "open":
            if time.time() - self._cb_opened_at >= cooldown:
                self._cb_state = "half_open"
                self.logger.info("⚡ CircuitBreaker → HALF_OPEN")
                return False
            return True
        return False

    def _cb_record_success(self) -> None:
        if self._cb_state in ("half_open", "open"):
            self.logger.info("⚡ CircuitBreaker → CLOSED (recovered)")
        self._cb_failures = 0
        self._cb_state = "closed"

    def _cb_record_failure(self) -> None:
        self._cb_failures += 1
        threshold = int(self._da_cfg.get("circuit_breaker_threshold", 5))
        if self._cb_failures >= threshold and self._cb_state == "closed":
            self._cb_state = "open"
            self._cb_opened_at = time.time()
            self._metrics["cb_trips"] += 1
            cooldown = self._da_cfg.get("circuit_breaker_cooldown", 120)
            self.logger.error(
                f"⚡ CircuitBreaker TRIPPED after {self._cb_failures} failures. "
                f"Cooldown: {cooldown}s"
            )

    # ── Metrics ──────────────────────────────────────────────────────────────
    def get_metrics(self) -> Dict[str, Any]:
        """Return copy of current runtime metrics."""
        return {
            **self._metrics,
            "domain": self._domain,
            "circuit_breaker": self._cb_state,
            "running": self._running,
        }

    # ── DDL helper ───────────────────────────────────────────────────────────
    def get_output_table_ddl(self) -> str:
        """Generate CREATE TABLE DDL from output_schema config."""
        table_raw = self._da_cfg.get("analysis_results_table", "analysis_results")
        schema = self._da_cfg.get("output_schema", {})
        if not schema:
            return ""
        try:
            table_safe = _db_safe_ident(table_raw)
        except ValueError as exc:
            self.logger.error(f"❌ Invalid table name in config: {exc}")
            return ""
        cols: List[str] = []
        pk_col = None
        for col_name, col_def in schema.items():
            if isinstance(col_def, dict):
                # Validate the column name
                try:
                    safe_col = _db_safe_ident(col_name)
                except ValueError:
                    self.logger.warning(f"⚠️ Skipping unsafe column name in output_schema: {col_name!r}")
                    continue
                col_type = col_def.get("type", "TEXT")
                default  = col_def.get("default", "")
                is_pk    = col_def.get("primary", False)
                parts = [f'    {safe_col} {col_type}']
                if is_pk:
                    parts.append("PRIMARY KEY")
                    pk_col = col_name
                if default:
                    parts.append(f"DEFAULT {default}")
                cols.append(" ".join(parts))
        col_str = ",\n".join(cols)
        return (
            f'CREATE TABLE IF NOT EXISTS {table_safe} (\n'
            f"{col_str}\n"
            f");"
        )

    # ── Public lifecycle ──────────────────────────────────────────────────────
    async def start_continuous_analysis(
        self,
        interval: int = 1200,
        batch_size: int = 10,
    ) -> None:
        """Start continuous analysis loop (called as asyncio.Task)."""
        eff_interval   = int(self._da_cfg.get("interval", interval))
        eff_batch_size = int(self._da_cfg.get("batch_size", batch_size))

        self._running = True
        self._stop_flag = False
        self.logger.info(
            f"🚀 DataAnalyzer starting: domain={self._domain} "
            f"interval={eff_interval}s batch={eff_batch_size} "
            f"concurrent={self._da_cfg.get('max_concurrent', 3)}"
        )

        while self._running and not self._stop_flag:
            cycle_start = time.time()
            self._metrics["cycles"] += 1

            try:
                await self._run_batch_async(eff_batch_size)
            except Exception as exc:
                self.logger.error(f"❌ Analysis cycle failed: {exc}", exc_info=True)

            self._metrics["last_cycle"] = datetime.now(timezone.utc).isoformat()
            elapsed = time.time() - cycle_start
            self.logger.info(
                f"✅ Analysis cycle #{self._metrics['cycles']} done "
                f"in {elapsed:.1f}s — "
                f"analyzed={self._metrics['items_analyzed']} "
                f"failed={self._metrics['items_failed']}"
            )

            # Interruptible sleep
            for _ in range(eff_interval):
                if self._stop_flag:
                    break
                await asyncio.sleep(1)

        self.logger.info("🛑 DataAnalyzer stopped")
        self._running = False

    def stop_continuous_analysis(self) -> None:
        """Signal the analysis loop to stop."""
        self._stop_flag = True
        self.logger.info("🛑 DataAnalyzer stop signal sent")

    # ── Batch processing ─────────────────────────────────────────────────────
    async def _run_batch_async(self, batch_size: int) -> None:
        """Fetch a batch and analyze each item with bounded concurrency."""
        if self._cb_is_open():
            self.logger.warning(
                f"⚡ CircuitBreaker OPEN — skipping cycle "
                f"(cooldown {self._da_cfg.get('circuit_breaker_cooldown', 120)}s)"
            )
            self._metrics["items_skipped"] += batch_size
            return

        items = await self._fetch_unanalyzed_content(batch_size)
        if not items:
            self.logger.info("ℹ️ No unanalyzed items found")
            return

        self.logger.info(f"🔍 Analyzing {len(items)} items (domain={self._domain})")
        max_concurrent = int(self._da_cfg.get("max_concurrent", 3))
        sem = asyncio.Semaphore(max_concurrent)

        tasks = [self._analyze_one_guarded(item, sem) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ok  = sum(1 for r in results if not isinstance(r, Exception) and r)
        err = len(results) - ok
        self.logger.info(f"✅ Batch done: {ok} OK, {err} errors")

    async def _analyze_one_guarded(
        self,
        item: Dict[str, Any],
        sem: asyncio.Semaphore,
    ) -> bool:
        """Retry-wrapped, semaphore-bounded single-item analysis.

        The `timeout` used here is resolved once, in `set_service()`, from
        the live server's measured decode/prefill rates. Nothing here reads
        a hard-coded ceiling.
        """
        max_retries  = int(self._da_cfg.get("max_retries", 3))
        item_timeout = float(self._da_cfg.get("timeout", 90))

        async with sem:
            for attempt in range(max_retries + 1):
                t0 = time.time()
                try:
                    result = await asyncio.wait_for(
                        self._analyze_one_sync(item),
                        timeout=item_timeout,
                    )
                    await self._save_analysis_result(result, item.get("id"))
                    elapsed_ms = (time.time() - t0) * 1000
                    self._metrics["items_analyzed"] += 1
                    n = self._metrics["items_analyzed"]
                    prev_avg = self._metrics["avg_item_ms"]
                    self._metrics["avg_item_ms"] = (prev_avg * (n - 1) + elapsed_ms) / n
                    self._cb_record_success()
                    return True

                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"⏱️ Item {item.get('id')} timed out (attempt {attempt+1}/{max_retries+1})"
                    )
                    self._cb_record_failure()
                except Exception as exc:
                    backoff = min(2 ** attempt, 30)
                    self.logger.warning(
                        f"⚠️ Item {item.get('id')} failed attempt {attempt+1}: {exc} "
                        f"(retry in {backoff}s)"
                    )
                    self._cb_record_failure()
                    if attempt < max_retries:
                        await asyncio.sleep(backoff)

            self._metrics["items_failed"] += 1
            return False

    async def _analyze_one_sync(self, item: Dict[str, Any]) -> "AnalysisResult":
        """Build prompt, call LLM, parse and return AnalysisResult."""
        content  = item.get("content", "")[:15_000]
        metadata = item.get("metadata", {})
        return await self.analyze(content, metadata)

    # ── Core analysis (public, used by agent too) ─────────────────────────────
    async def analyze(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AnalysisResult":
        """Analyze a single piece of content; returns AnalysisResult."""
        metadata = metadata or {}
        content_trunc = content[:15_000]
        if len(content) > 15_000:
            metadata["truncated"] = True
            metadata["original_length"] = len(content)

        prompt = self._prompt_template.format(
            content=content_trunc,
            source_type=metadata.get("source", "unknown"),
            location=metadata.get("location", "unknown"),
            language=metadata.get("language", "en"),
            timestamp=metadata.get("timestamp", datetime.now(timezone.utc).isoformat()),
            content_length=len(content_trunc),
            domain=self._domain,
        )

        try:
            raw = await asyncio.wait_for(
                self.llm.generate(
                    prompt=prompt,
                    max_tokens=int(self._da_cfg.get("max_tokens", 512)),
                    temperature=float(self._da_cfg.get("temperature", 0.1)),
                ),
                timeout=float(self._da_cfg.get("timeout", 90)),
            )
            self.logger.info(f"📝 LLM raw (first 400): {raw[:400]}")
            result = self._parse_response(raw, content, metadata)
            if self.enable_validation:
                vr = self._validate_analysis(result, content)
                if vr.get("warnings"):
                    result.warnings.extend(vr["warnings"])
            return result

        except asyncio.TimeoutError:
            self.logger.error("⏱️ LLM call timed out")
            return self._create_error_result(content, metadata, "Analysis timed out")
        except Exception as exc:
            self.logger.error(f"❌ Analysis failed: {exc}", exc_info=True)
            return self._create_error_result(content, metadata, str(exc))

    # ── Database helpers ──────────────────────────────────────────────────────
    async def _fetch_unanalyzed_content(self, limit: int) -> List[Dict[str, Any]]:
        """Fetch up to `limit` unanalyzed rows from the raw_data table."""
        if not self._db_available():
            return []

        # Validate all config-derived identifiers through _db_safe_ident before
        # interpolating into SQL — prevents SQL injection via a malicious config.
        try:
            t            = _db_safe_ident(self._da_cfg.get("raw_data_table",         "raw_data"))
            id_col_raw   = self._da_cfg.get("raw_data_id_col",       "id")
            id_col       = _db_safe_ident(id_col_raw)
            content_col  = _db_safe_ident(self._da_cfg.get("raw_data_content_col",  "content"))
            analyzed_col = _db_safe_ident(self._da_cfg.get("raw_data_analyzed_col", "analyzed"))
            source_col   = _db_safe_ident(self._da_cfg.get("raw_data_source_col",   "source_type"))
            order_col_raw = self._da_cfg.get("raw_data_order_col", "id")
            order_col    = _db_safe_ident(order_col_raw)
        except ValueError as exc:
            self.logger.error(f"❌ Invalid SQL identifier in data_analyzer config: {exc}")
            return []

        # Verify order_col exists (fallback to id)
        try:
            check = self._service.db_manager.execute(
                "primary",
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name=%s AND column_name=%s",
                [self._da_cfg.get("raw_data_table", "raw_data"), order_col_raw], fetch="all"
            )
            if not (check.get("ok") and check.get("result")):
                order_col = id_col
        except Exception:
            order_col = id_col

        query = (
            f'SELECT {id_col}, {content_col}, {source_col} '
            f'FROM {t} '
            f'WHERE {analyzed_col} = FALSE '
            f'ORDER BY {order_col} ASC '
            f'LIMIT %s'
        )
        # Column name keys for result row lookups (unquoted raw strings)
        id_col_raw      = self._da_cfg.get("raw_data_id_col",      "id")
        content_col_raw = self._da_cfg.get("raw_data_content_col", "content")
        source_col_raw  = self._da_cfg.get("raw_data_source_col",  "source_type")
        try:
            res = self._service.db_manager.execute("primary", query, [limit], fetch="all")
            if not res.get("ok"):
                self.logger.warning(f"⚠️ fetch_unanalyzed failed: {res.get('error')}")
                return []
            rows = []
            for row in res.get("result", []):
                rows.append({
                    "id":       row.get(id_col_raw),
                    "content":  row.get(content_col_raw, ""),
                    "metadata": {"source": row.get(source_col_raw, "unknown")},
                })
            return rows
        except Exception as exc:
            self.logger.error(f"❌ fetch_unanalyzed error: {exc}")
            return []

    async def _save_analysis_result(
        self,
        result: "AnalysisResult",
        content_id: Any,
    ) -> None:
        """Save AnalysisResult → analysis_results; mark raw row analyzed."""
        if not self._db_available():
            return

        # Validate config-derived identifiers before SQL interpolation
        try:
            t_raw        = _db_safe_ident(self._da_cfg.get("raw_data_table",          "raw_data"))
            id_col       = _db_safe_ident(self._da_cfg.get("raw_data_id_col",         "id"))
            analyzed_col = _db_safe_ident(self._da_cfg.get("raw_data_analyzed_col",   "analyzed"))
            t_out        = _db_safe_ident(self._da_cfg.get("analysis_results_table",  "analysis_results"))
        except ValueError as exc:
            self.logger.error(f"❌ Invalid SQL identifier in data_analyzer config: {exc}")
            return

        import hashlib as _hl
        content_hash = _hl.sha256(result.content.encode()).hexdigest()

        try:
            # ── Mark raw row analyzed ────────────────────────────────────────
            upd = self._service.db_manager.execute(
                "primary",
                f'UPDATE {t_raw} SET {analyzed_col}=TRUE WHERE {id_col}=%s',
                [content_id], fetch="none"
            )
            if not upd.get("ok"):
                self.logger.warning(f"⚠️ Could not mark row {content_id} analyzed")

            # ── Ensure output table exists ───────────────────────────────────
            ddl = self.get_output_table_ddl()
            if ddl:
                self._service.db_manager.execute("primary", ddl, [], fetch="none")

            # ── Build dynamic INSERT from output_schema ──────────────────────
            schema = self._da_cfg.get("output_schema", {})
            # Map of col_name → value (only columns in schema, skip DDL-only cols)
            _skip = {"created_at", "updated_at"}
            row_data: Dict[str, Any] = {
                "analysis_id":       result.analysis_id,
                "raw_data_id":       content_id,
                "content_hash":      content_hash,
                "content_preview":   result.content[:500],
                "domain":            self._domain,
                "threat_level":      result.threat_level.value,
                "severity_score":    result.metadata.get("severity_score", result.confidence),
                "confidence":        result.confidence,
                "categories":        json.dumps(result.categories),
                "entities":          json.dumps(result.entities),
                "keywords":          json.dumps(result.metadata.get("keywords", [])),
                "reasoning":         result.reasoning[:4000],
                "action_required":   result.action_required,
                "urgency":           result.urgency,
                "location":          result.metadata.get("location", ""),
                "affected_population": result.metadata.get("affected_population", ""),
                "incident_count":    int(result.metadata.get("incident_count", 0) or 0),
                "source_credibility":float(result.metadata.get("source_credibility", 0.5)),
                "warnings":          json.dumps(result.warnings),
                "metadata":          json.dumps(result.metadata),
            }
            # Only include columns declared in schema
            cols_to_insert = [
                c for c in schema if c not in _skip and c in row_data
            ]
            if not cols_to_insert:
                # Fallback: insert all available
                cols_to_insert = list(row_data.keys())

            # Validate every column name before interpolating into SQL
            try:
                safe_cols = [_db_safe_ident(c) for c in cols_to_insert]
            except ValueError as exc:
                self.logger.error(f"❌ Invalid column name in output_schema: {exc}")
                return

            col_list = ", ".join(safe_cols)
            ph_list  = ", ".join("%s" for _ in cols_to_insert)
            vals     = [row_data[c] for c in cols_to_insert]

            ins_sql = (
                f'INSERT INTO {t_out} ({col_list}) VALUES ({ph_list})\n'
                f'ON CONFLICT (analysis_id) DO UPDATE SET '
                + ", ".join(
                    f'{sc}=EXCLUDED.{sc}' for c, sc in zip(cols_to_insert, safe_cols, strict=False)
                    if c != "analysis_id"
                )
                + ', updated_at=NOW();'
            )
            ins = self._service.db_manager.execute("primary", ins_sql, vals, fetch="none")
            if ins.get("ok"):
                self.logger.info(f"✅ Saved analysis {result.analysis_id} for item {content_id}")
            else:
                self.logger.warning(f"⚠️ Save failed: {ins.get('error')}")

        except Exception as exc:
            self.logger.error(f"❌ _save_analysis_result error: {exc}", exc_info=True)

    def _db_available(self) -> bool:
        """Return True if database is accessible and service is live."""
        return (
            self._service is not None
            and hasattr(self._service, "db_manager")
            and self._service.db_manager is not None
            and not getattr(self._service, "_shutdown", False)
        )

    # ── Response parsing ──────────────────────────────────────────────────────
    def _parse_response(
        self,
        raw: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> "AnalysisResult":
        """Parse JSON from LLM response into AnalysisResult."""
        try:
            json_str = self._extract_json(raw)
            if not json_str:
                raise ValueError("No valid JSON found in LLM response")
            parsed = json.loads(json_str)

            required = ["threat_level", "confidence", "categories", "entities",
                        "reasoning", "action_required"]
            missing = [f for f in required if f not in parsed]
            if missing:
                raise ValueError(f"Missing fields: {missing}")

            # threat_level
            tl_str = str(parsed["threat_level"]).upper()
            try:
                threat_level = ThreatLevel(tl_str.lower())
            except ValueError:
                self.logger.warning(f"⚠️ Unknown threat_level '{tl_str}' → INFORMATIONAL")
                threat_level = ThreatLevel.INFORMATIONAL

            confidence = max(0.0, min(1.0, float(parsed["confidence"])))

            # Apply confidence threshold downgrade
            if confidence < self.min_confidence:
                # Downgrade: CRITICAL → HIGH, HIGH → MEDIUM, MEDIUM → LOW, LOW → INFORMATIONAL
                downgrade_map = {
                    ThreatLevel.CRITICAL: ThreatLevel.HIGH,
                    ThreatLevel.HIGH: ThreatLevel.MEDIUM,
                    ThreatLevel.MEDIUM: ThreatLevel.LOW,
                    ThreatLevel.LOW: ThreatLevel.INFORMATIONAL,
                    ThreatLevel.INFORMATIONAL: ThreatLevel.INFORMATIONAL,
                }
                new_level = downgrade_map.get(threat_level, threat_level)
                if new_level != threat_level:
                    logger.info(
                        f"Downgrading threat level from {threat_level.value} to {new_level.value} "
                        f"due to low confidence ({confidence:.2f} < {self.min_confidence})"
                    )
                    threat_level = new_level
                    # Add a warning (will be appended later)
                    # We'll add it after constructing result, but we can add to a temporary list

            categories = [str(c).lower().strip() for c in parsed.get("categories", []) if c]

            entities = []
            for ent in parsed.get("entities", []):
                if isinstance(ent, dict) and "type" in ent and "value" in ent:
                    entities.append({
                        "type":  str(ent["type"]),
                        "value": str(ent["value"]),
                        "score": float(ent.get("score", 0.0)),
                    })

            urgency = str(parsed.get("urgency", "none")).lower()
            valid_urgencies = {"immediate", "high", "medium", "low", "none"}
            if urgency not in valid_urgencies:
                urgency_map = {
                    ThreatLevel.CRITICAL:      "immediate",
                    ThreatLevel.HIGH:          "high",
                    ThreatLevel.MEDIUM:        "medium",
                    ThreatLevel.LOW:           "low",
                    ThreatLevel.INFORMATIONAL: "none",
                }
                urgency = urgency_map.get(threat_level, "none")

            # Merge extra fields into metadata
            extra_meta: Dict[str, Any] = {**metadata, **parsed.get("metadata", {})}
            for field_name in [
                "keywords", "location", "affected_population",
                "incident_count", "source_credibility", "severity_score",
            ]:
                if field_name in parsed:
                    extra_meta[field_name] = parsed[field_name]

            warnings = list(parsed.get("warnings", []))
            # Add downgrade warning if applicable
            if confidence < self.min_confidence and threat_level != ThreatLevel(tl_str.lower()):
                warnings.append(f"Threat level downgraded due to low confidence ({confidence:.2f})")

            return AnalysisResult(
                content=content,
                threat_level=threat_level,
                confidence=confidence,
                entities=entities,
                categories=categories,
                reasoning=str(parsed.get("reasoning", ""))[:4000],
                action_required=bool(parsed.get("action_required", False)),
                urgency=urgency,
                metadata=extra_meta,
                warnings=warnings,
            )

        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            self.logger.warning(f"⚠️ Response parse failed: {exc}")
            return self._create_error_result(content, metadata, f"Parse error: {exc}")

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract the first JSON object from a text blob."""
        text = text.replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            return None

    def _validate_analysis(
        self,
        result: "AnalysisResult",
        original_content: str,
    ) -> Dict[str, Any]:
        """Post-hoc consistency checks; returns dict with 'warnings' list."""
        warns: List[str] = []
        if result.confidence < self.min_confidence and result.threat_level in (
            ThreatLevel.CRITICAL, ThreatLevel.HIGH
        ):
            warns.append(f"High threat level with low confidence ({result.confidence:.2f})")
        if result.confidence > 0.85 and result.threat_level == ThreatLevel.INFORMATIONAL:
            warns.append("Very high confidence for INFORMATIONAL level")
        if len(result.reasoning) < 20:
            warns.append("Reasoning too brief")
        return {"valid": True, "warnings": warns}

    def _create_error_result(
        self,
        content: str,
        metadata: Dict[str, Any],
        error: str,
    ) -> "AnalysisResult":
        """Return a safe fallback AnalysisResult on any failure."""
        return AnalysisResult(
            content=content,
            threat_level=ThreatLevel.INFORMATIONAL,
            confidence=0.0,
            entities=[],
            categories=["analysis_error"],
            reasoning=f"Analysis failed: {error}",
            action_required=False,
            urgency="none",
            metadata={**metadata, "error": error},
            warnings=[f"Analysis error: {error}"],
        )

# ── Backward-compat alias (for any old code that still references ThreatAnalyzer) ─
ThreatAnalyzer = DataAnalyzer


# ---------------------------------------------------------------------
# Enhanced Model Factory (keep as is)
# ---------------------------------------------------------------------
class ModelFactory:
    """🏭 Production-ready factory for creating model adapters with graceful fallbacks."""

    @staticmethod
    def create_model(config: ModelConfig) -> BaseLLMAdapter:
        """🏭 Create model adapter with proper dependency checks and graceful fallbacks.

        Priority (when provider == 'auto' or 'local_llm_cloud'):
          1. LocalLLMCloudAdapter  — hardware-optimised cloud server
          2. LlamaCppAdapter       — direct llama-cpp-python
          3. HFAdapter             — HuggingFace transformers
          4. OpenAIAdapter         — OpenAI-compat endpoint
          5. DummyAdapter          — zero-dep fallback
        """
        provider = config.provider.lower().strip()

        # Enhanced provider mapping
        provider_map = {
            # local-llm-cloud aliases
            "local_llm_cloud": "local_llm_cloud",
            "llm_cloud":       "local_llm_cloud",
            "cloud":           "local_llm_cloud",
            # HuggingFace
            "huggingface":     "hf",
            "hugging_face":    "hf",
            "transformers":    "hf",
            # llama-cpp
            "llama":           "llama_cpp",
            "llamacpp":        "llama_cpp",
            "llama-cpp":       "llama_cpp",
            "llama.cpp":       "llama_cpp",
            # GPT4All
            "gpt4all":         "gpt4all",
            "gpt4all-j":       "gpt4all",
            # OpenAI
            "openai":          "openai",
            "chatgpt":         "openai",
            "gpt":             "openai",
        }

        provider_key = provider_map.get(provider, provider)

        # ── AUTO mode: use BackendOrchestrator to pick the best available ──
        if provider_key in ("auto", ""):
            try:
                from amb_sentinel.llm.model_manager import get_backend_orchestrator
                orch = get_backend_orchestrator()
                provider_key = orch.recommend_provider()
                logger.info("🏭 AUTO mode: BackendOrchestrator selected '%s'",
                            provider_key)
            except Exception as exc:
                logger.warning("BackendOrchestrator probe failed: %s — using llama_cpp", exc)
                provider_key = "llama_cpp"

        # Check if required imports are available
        import_checks = {
            "openai":     _OPTIONAL_IMPORTS.get('openai', False),
            "hf":         _OPTIONAL_IMPORTS.get('transformers', False),
            "llama_cpp":  _OPTIONAL_IMPORTS.get('llama_cpp', False),
            "gpt4all":    _OPTIONAL_IMPORTS.get('gpt4all', False),
            # local_llm_cloud: no local import needed — checks at init time
            "local_llm_cloud": True,
        }

        logger.info("🏭 Creating adapter: provider='%s' -> '%s', model='%s', device='%s'",
                    provider, provider_key, config.model_name, config.device)

        # Check for required dependencies
        if provider_key in import_checks and not import_checks[provider_key]:
            logger.warning("⚠️ Provider '%s' unavailable (missing dependencies). "
                           "Using fallback.", provider_key)
            return DummyAdapter(config)

        try:
            if provider_key == "local_llm_cloud":
                return LocalLLMCloudAdapter(config)

            elif provider_key == "openai":
                if not config.api_key and not config.model_name.startswith(
                        ("http://", "https://")):
                    logger.warning("⚠️ OpenAI adapter requires api_key or base_url. "
                                   "Using fallback.")
                    return DummyAdapter(config)
                return OpenAIAdapter(config)

            elif provider_key == "hf":
                return HFAdapter(config)

            elif provider_key == "llama_cpp":
                model_path = Path(config.model_name)
                if not model_path.exists():
                    # Config/env-driven search roots.
                    _roots = [
                        os.environ.get("AMB_MODELS_DIR", ""),
                        os.environ.get("AMB_MODEL_DIR", ""),
                        str(Path.cwd() / "models"),
                        str(Path.cwd() / "model"),
                        str(Path.home() / "models"),
                    ]
                    found = None
                    for _root in _roots:
                        if not _root:
                            continue
                        _cand = Path(_root) / model_path.name
                        if _cand.exists():
                            found = _cand
                            break
                    if found is None:
                        logger.warning(
                            "⚠️ Model file not found: %s "
                            "(searched AMB_MODELS_DIR, ./models, ./model, ~/models). "
                            "Using fallback.",
                            model_path,
                        )
                        return DummyAdapter(config)
                    config.model_name = str(found)
                    logger.info("🔍 Found model at: %s", found)

            elif provider_key == "gpt4all":
                model_path = Path(config.model_name)
                if not model_path.exists():
                    _roots = [
                        os.environ.get("AMB_MODELS_DIR", ""),
                        os.environ.get("AMB_MODEL_DIR", ""),
                        str(Path.cwd() / "models"),
                        str(Path.cwd() / "model"),
                        str(Path.home() / "models"),
                    ]
                    possible_paths = [config.model_name] + [
                        str(Path(r) / model_path.name) for r in _roots if r
                    ]
                    for location in possible_paths:
                        if Path(location).exists():
                            config.model_name = location
                            logger.info("🔍 Found GPT4All model at: %s", location)
                            break
                    else:
                        logger.warning("⚠️ GPT4All model not found: %s. Using fallback.",
                                       model_path)
                        return DummyAdapter(config)
                return GPT4AllAdapter(config)

            else:
                logger.warning("⚠️ Unknown provider '%s'. Using fallback.", provider)
                return DummyAdapter(config)

        except Exception as e:
            logger.error(f"❌ Failed to create adapter for '%s': %s", provider, e)
            return DummyAdapter(config)

    @staticmethod
    def detect_capabilities() -> Dict[str, Any]:
        """🔍 Detect hardware and software capabilities.

        Torch is optional; a core-only install (no [ml] extra) must be able
        to run health checks without it.
        """
        import importlib.util as _ilu
        _torch_ok = _ilu.find_spec("torch") is not None

        capabilities: Dict[str, Any] = {
            "cuda": False, "cuda_version": None, "mps": False,
            "cpu_count": os.cpu_count() or 4, "memory": {},
            "optional_imports": _OPTIONAL_IMPORTS.copy(),
            "recommended_device": "cpu",
            "torch_available": _torch_ok,
        }

        if _torch_ok:
            try:
                capabilities["cuda"] = torch.cuda.is_available()
                capabilities["cuda_version"] = (
                    torch.version.cuda if capabilities["cuda"] else None
                )
                capabilities["mps"] = (
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                )
            except Exception as exc:
                logger.warning(f"🔍 torch capability probe failed: {exc}")

        try:
            import psutil
            mem = psutil.virtual_memory()
            capabilities["memory"] = {
                "total_gb": round(mem.total / (1024 ** 3), 2),
                "available_gb": round(mem.available / (1024 ** 3), 2),
                "percent": mem.percent,
            }
        except ImportError:
            pass

        if capabilities["cuda"] and _torch_ok:
            try:
                capabilities["cuda_devices"] = torch.cuda.device_count()
                capabilities["cuda_names"] = [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
            except Exception as e:
                logger.warning(f"🔍 CUDA device info error: {e}")

        if capabilities["cuda"]:
            capabilities["recommended_device"] = "cuda"
        elif capabilities["mps"]:
            capabilities["recommended_device"] = "mps"

        logger.info(
            f"🔍 Capabilities: CUDA={capabilities['cuda']}, MPS={capabilities['mps']}, "
            f"CPU={capabilities['cpu_count']}, Recommended={capabilities['recommended_device']}, "
            f"torch={_torch_ok}"
        )
        return capabilities


class RealTimeMemoryMonitor:
    """
    Periodically triggers memory adaptation in the model manager.
    """

    def __init__(self, model_manager, check_interval: float = 5.0):
        self.model_manager = model_manager
        self.check_interval = check_interval
        self._monitoring = False
        self._task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger("MemoryMonitor")

    async def start(self):
        if self._monitoring:
            return
        self._monitoring = True
        self._task = asyncio.create_task(self._monitor_loop())
        self.logger.info("🔍 Real‑time memory monitor started")

    async def stop(self):
        self._monitoring = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.logger.info("🛑 Real‑time memory monitor stopped")

    async def _monitor_loop(self):
        while self._monitoring:
            try:
                await self.model_manager.adapt_to_memory_pressure()
            except Exception as e:
                self.logger.error(f"Memory monitor error: {e}")
            await asyncio.sleep(self.check_interval)


# ---------------------------------------------------------------------
# ENHANCED AIModelService (Main Orchestrator)
# ---------------------------------------------------------------------
class AIModelService:
    """
    🚀 Enhanced main service orchestrating all components with production features.
    Now properly runs AgentExecutor and DataAnalyzer concurrently and autonomously.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._raw_config = self.config
        self.service_config = self._parse_config()

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - we're being initialized outside async context
            # Don't try to create one - let the caller handle it
            self._loop = None

            # Add flag to track if we're in async context
        self._in_async_context = self._loop is not None

        # Core components
        self.db_manager: Optional[DatabaseManager] = None
        self.memory: Optional[MemoryStore] = None
        self.vectorstore: Optional[VectorStore] = None
        self.embedding_adapter: Optional[EmbeddingAdapter] = None
        self.models: Dict[str, BaseLLMAdapter] = {}
        self.scheduler: Optional[BackgroundScheduler] = None
        self.tool_registry = ToolRegistry()
        self.memory_monitor: Optional[RealTimeMemoryMonitor] = None

        # CORE COMPONENTS FOR AUTONOMOUS EXECUTION
        self.agent_executor: Optional[AgentExecutor] = None
        self.data_analyzer: Optional[DataAnalyzer] = None

        # Execution tasks
        self._agent_task: Optional[asyncio.Task] = None
        self._data_analyzer_task: Optional[asyncio.Task] = None

        self._warm_executor = None
        # Thread currently running adapter initialization (if any).
        #
        # Deliberately NOT a ThreadPoolExecutor worker. A pool worker is
        # registered in `concurrent.futures.thread._threads_queues`, whose
        # atexit handler joins every registered thread on interpreter exit.
        # A worker stuck inside a C call (llama.cpp GGUF load, HF
        # safetensors mmap) would then block process shutdown forever,
        # daemon flag notwithstanding. A raw `threading.Thread(daemon=True)`
        # bypasses that registry entirely; see `_run_adapter_init_async`.
        #
        # Retained for diagnostics only — never joined.
        self._adapter_init_thread: Optional[threading.Thread] = None
        # Root-cause capture for the A3 message and exit-code decision.
        # `_last_adapter_init_exc` is written by `_init_adapter_sync` when
        # adapter.initialize() raises; `_primary_load_exc` is the copy
        # that `_init_models_async` keeps for the primary provider.
        self._last_adapter_init_exc: Optional[BaseException] = None
        self._primary_load_exc: Optional[BaseException] = None

        # State management
        self.running = False
        self._shutdown = False
        self._lock = RLock()
        self.request_semaphore = AdaptiveConcurrency(
            initial_limit=self.service_config.max_concurrent_requests,
            min_limit=1,  # or a configurable value
            mem_high_threshold=80.0,
            cpu_high_threshold=80.0,
            adjustment_factor=0.8
        )

        self._start_time = datetime.now(timezone.utc).isoformat()

        # Metrics
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "avg_response_time": 0.0,
            "last_request": None,
            "agent_executions": 0,
            "data_analyses": 0,
            "components_running": {
                "agent_executor": False,
                "data_analyzer": False
            }
        }

        # Backup retrier — initialized AFTER _init_components() (which sets db_manager)
        self.backup_retrier = None

        self._init_lock = asyncio.Lock()
        # Initialize
        # Heavy initialization lives in initialize_async() so retry loops can
        # cooperate with the event loop instead of blocking it with time.sleep.
        # See the class docstring for the required calling convention.
        self._initialized = False

    async def initialize_async(self) -> None:
        """
        Perform async component initialization.

        Must be awaited before any other method on this service is used.
        `__init__` deliberately does no I/O and no waiting so that a caller
        who forgets to await this method gets a clear AttributeError on the
        first use, not a silently half-initialized service.

        Calling twice is a no-op (idempotent).
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            try:
                _t0 = time.monotonic()
                await self._init_components_async()
                logger.info("init phase 1 (components) %.1fs", time.monotonic() - _t0)
                _t0 = time.monotonic()
                self._validate_instructions()
                self._start_memory_monitor()
                logger.info("init phase 2 (validate+monitor) %.1fs", time.monotonic() - _t0)
                _t0 = time.monotonic()
                self.running = True
                logger.info("✅ AIModelService initialized successfully")

                if self.db_manager:
                    backup_dir = paths.backup_dir()
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    self.backup_retrier = ExecutionBackupRetrier(
                        service=self,
                        backup_dir=backup_dir,
                        check_interval=3600,
                    )
                    # Already inside the loop → create_task is safe.
                    asyncio.create_task(self.backup_retrier.start())
                    logger.info("✅ ExecutionBackupRetrier initialized")

                if (self.service_config.auto_run_agent
                        and self.service_config.enable_concurrent_execution):
                    self._start_autonomous_execution()

                self._initialized = True
            except Exception as e:
                logger.error(f"❌ Failed to initialize AIModelService: {e}", exc_info=True)
                self.shutdown()
                raise

    def initialize_sync(self) -> None:
        """
        Sync entry point for callers that are not already inside a loop.
        Raises if called from within a running loop — in that case the
        caller should `await service.initialize_async()` directly.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.initialize_async())
            return
        raise RuntimeError(
            "initialize_sync() called from within a running event loop; "
            "use `await service.initialize_async()` instead."
        )

    def _load_config(self) -> Dict[str, Any]:
        """📄 Load and validate configuration with enhanced error handling."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            # -----------------------------
            # Secure credential handling
            # -----------------------------
            # Mask credentials for logs and optionally prompt for runtime secrets
            # Minimal, TTY-safe interactive prompt. Does not write plaintext secrets to logs.
            import getpass

            def _mask_secret(v):
                # None -> None (not present), empty string -> "" (explicit empty), otherwise "<HIDDEN>"
                return None if v is None else ("<HIDDEN>" if v else "")

            # Postgres section could be located at config["database"]["postgresql"]
            db_section = (config.get("database") or {}).get("postgresql", {})
            if db_section:
                raw_pass = db_section.get("password")
                # Use sentinel "<PROMPT>" to request runtime prompt, or prompt if empty and interactive.
                if raw_pass == "<PROMPT>" or raw_pass in (None, ""):
                    # Try environment variables first (non-interactive)
                    env_pass = (os.environ.get("PGPASSWORD") or
                                os.environ.get("AMB_PG_PASSWORD") or
                                os.environ.get("DB_PASSWORD") or
                                os.environ.get("POSTGRES_PASSWORD") or "")
                    if env_pass:
                        db_section["password"] = env_pass
                        logger.info("✅ Postgres password loaded from environment variable")
                    elif sys.stdin.isatty():
                        try:
                            db_section["password"] = getpass.getpass("Postgres password (interactive): ")
                        except Exception:
                            logger.warning("⚠️ Could not prompt for Postgres password; continuing without it.")
                    else:
                        logger.warning("⚠️ Postgres password missing and no TTY to prompt; startup may fail.")
                # Masked log entry only
                logger.info(
                    f"✅ Postgres host={db_section.get('host')} user={db_section.get('user')} "
                    f"✅ password={_mask_secret(db_section.get('password'))}"
                )

            # Redis may be declared at database.redis or top-level "redis" section
            redis_section = (config.get("database") or {}).get("redis", {}) or config.get("redis", {})
            if redis_section:
                raw_rpass = redis_section.get("password")
                if raw_rpass == "<PROMPT>" or raw_rpass in (None, ""):
                    env_rpass = (os.environ.get("REDIS_PASSWORD")
                             or os.environ.get("AMB_REDIS_PASSWORD")
                             or os.environ.get("REDISCLI_AUTH") or "")
                    if env_rpass:
                        redis_section["password"] = env_rpass
                        logger.info("✅ Redis password loaded from environment variable")
                    elif sys.stdin.isatty():
                        try:
                            redis_section["password"] = getpass.getpass("Redis password (interactive): ")
                        except Exception:
                            logger.warning("⚠️ Could not prompt for Redis password; continuing without it.")
                    else:
                        logger.warning(
                            "⚠️ Redis password missing and no TTY to prompt; Redis features may not be available.")
                # The exact debug line you requested (keeps no secret text in logs)
                logger.info(
                    f"✅ Redis host={redis_section.get('host')} password_set={'yes' if redis_section.get('password') else 'no'}")

            # ── local-llm-cloud API key ─────────────────────────────
            # Same pattern as Postgres/Redis above: config.yaml never stores
            # this in plaintext. Populate it from LLM_CLOUD_API_KEY (see .env).
            # Leave both blank for LAN-only deployments where local-llm-server's
            # own api.key is also unset.
            for _model_cfg in (config.get("ai_service", {}).get("models", []) or []):
                if _model_cfg.get("provider") in ("local_llm_cloud", "llm_cloud", "cloud"):
                    raw_key = _model_cfg.get("api_key")
                    if raw_key in (None, "", "<PROMPT>"):
                        env_key = os.environ.get("LLM_CLOUD_API_KEY", "")
                        if env_key:
                            _model_cfg["api_key"] = env_key
                            logger.info("✅ local-llm-cloud API key loaded from LLM_CLOUD_API_KEY")
                        else:
                            logger.warning(
                                "⚠️ LLM_CLOUD_API_KEY not set — requests to local-llm-cloud will "
                                "go out with no API key. Fine if local-llm-server's api.key is also "
                                "empty; otherwise every generation call will fail with 403.")

            # Validate that top-level ai_service section exists (original validation)
            if not config.get("ai_service"):
                raise ValueError("Missing 'ai_service' section in configuration")

            if not config["ai_service"].get("models"):
                raise ValueError("No models configured in config.yaml")

            return config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}") from e
        except Exception as e:
            raise ValueError(f"Configuration error: {e}") from e

    @staticmethod
    def _resolve_embedding_cache_dir(configured: str, model_name: str) -> str:
        """Return an HF hub cache folder that actually contains *model_name*.
        Pure search — never hardcodes a machine path. Uses (in order):
        config.yaml, AMB_MODELS_DIR, HF_HOME, TRANSFORMERS_CACHE,
        paths.model_dir(), ./model, ./models, ~/models. Returns the first
        candidate that contains a ``models--<org>--<name>`` snapshot for
        the requested model, else falls back to *configured* or
        ``paths.model_dir()`` with a warning.
        """

        candidates: list[str] = []
        if configured:
            candidates.append(configured)

        for env in ("AMB_MODELS_DIR", "HF_HOME", "TRANSFORMERS_CACHE"):
            v = os.environ.get(env, "")
            if v:
                candidates.append(v)

        candidates.extend([
            str(paths.model_dir()),
            str(Path.cwd() / "model"),
            str(Path.cwd() / "models"),
            str(Path.home() / "models"),
        ])

        # HF hub stores `sentence-transformers/all-MiniLM-L6-v2` as
        # `models--sentence-transformers--all-MiniLM-L6-v2`.
        if not model_name:
            dir_names: list[str] = []
        elif "/" in model_name:
            dir_names = [f"models--{model_name.replace('/', '--')}"]
        else:
            dir_names = [
            f"models--sentence-transformers--{model_name}",
            f"models--{model_name}",
            ]

        for cand in candidates:
            p = Path(cand)
            if not p.exists():
                continue

            for d in dir_names:
                if (p / d).exists():
                    logger.info(
                        "embeddings: resolved cache_dir=%s (found %s)",
                        p, d,
                    )

                    return str(p)

        fallback = configured or (candidates[0] if candidates else str(paths.model_dir()))
        logger.warning(
            "embeddings: model %r not found in any of: %s — "
            "using %s as cache_dir (expect graceful degradation unless "
            "the model snapshot is placed there)",
            model_name, candidates, fallback,
        )

        return fallback

    def _parse_config(self) -> ServiceConfig:
        """🔍 Parse configuration into structured objects with enhanced validation."""
        ai_service = self.config.get("ai_service", {})
        database = self.config.get("database", {})
        redis_cfg = database.get("redis", {})
        embedding_cfg = self.config.get("ai_service", {}).get("embeddings", {})

        models = []
        for model_cfg in ai_service.get("models", []):
            try:
                models.append(ModelConfig(**model_cfg))
            except Exception as e:
                logger.error(f"❌ Failed to parse model config {model_cfg.get('provider', 'unknown')}: {e}")
                continue

        if not models:
            raise ValueError("No valid models configured")

        db_config = None
        if database.get("postgresql"):
            pg = database["postgresql"]
            try:
                # Extract pgvector sub‑config
                pgv = pg.get("pgvector", {})
                pgvector_config = PgVectorConfig(
                    enabled=pgv.get("enabled", True),
                    dimension=pgv.get("dimension", 384),
                    index_type=pgv.get("index_type", "ivfflat"),
                    lists=pgv.get("lists", 100),
                    probes=pgv.get("probes", 10),
                    operator=pgv.get("operator", "vector_cosine_ops"),
                )

                db_config = DatabaseConfig(
                    host=pg.get("host", "localhost"),
                    port=pg.get("port", 5432),
                    database=pg.get("database", ""),
                    user=pg.get("user", ""),
                    password=pg.get("password", ""),
                    min_connections=pg.get("min_connections", 1),
                    max_connections=pg.get("max_connections", 10),
                    timeout=pg.get("timeout", 10),
                    sslmode=pg.get("sslmode", "prefer"),
                    pgvector=pgvector_config,
                )
            except Exception as e:
                logger.error(f"❌ Failed to parse database config: {e}")

        redis_config = None
        if redis_cfg and _OPTIONAL_IMPORTS['redis']:
            try:
                redis_config = RedisConfig(
                    host=redis_cfg.get("host", "localhost"),
                    port=redis_cfg.get("port", 6379),
                    db=redis_cfg.get("db", 0),
                    password=redis_cfg.get("password"),
                    max_connections=redis_cfg.get("max_connections", 10),
                    socket_timeout=redis_cfg.get("socket_timeout", 5),
                    socket_connect_timeout=redis_cfg.get("socket_connect_timeout", 5),
                    retry_on_timeout=redis_cfg.get("retry_on_timeout", True)
                )
            except Exception as e:
                logger.error(f"❌ Failed to parse Redis config: {e}")

        download_dir = ai_service.get("download_dir")
        agent_instruction = ai_service.get("agent_instruction", "")

        # Resolve the HF hub cache folder that actually contains the
        # requested embedding model. Priority:
        # 1. `embeddings.cache_dir` from config.yaml, with ${VAR} expansion
        # 2. AMB_MODELS_DIR / HF_HOME / TRANSFORMERS_CACHE env vars
        # 3. paths.model_dir() and standard relative "model"/"models"
        # 4. paths.model_dir() as final fallback (best effort)
        # Config stays path-agnostic: `cache_dir: "${AMB_MODELS_DIR}"`.
        _emb_model_name = embedding_cfg.get("model", "") or ""
        _emb_cache_raw = embedding_cfg.get("cache_dir") or ""
        _emb_cache_expanded = (
            os.path.expandvars(_emb_cache_raw) if _emb_cache_raw else ""
        )
        _emb_cache_final = self._resolve_embedding_cache_dir(
            _emb_cache_expanded, _emb_model_name,
        )

        return ServiceConfig(
            models=models,
            primary_model=ai_service.get("primary_model", models[0].provider),
            agent_instruction=agent_instruction,
            database=db_config,
            redis=redis_config,
            health_check_interval=ai_service.get("health_check_interval", HEALTH_CHECK_INTERVAL),
            max_concurrent_requests=ai_service.get("max_concurrent_requests", MAX_CONCURRENT_REQUESTS),
            request_timeout=ai_service.get("request_timeout", REQUEST_TIMEOUT),
            embedding_model=embedding_cfg.get("model", ""),
            embedding_cache_dir=_emb_cache_final,
            embedding_cache_size=embedding_cfg.get("cache_size", 1000),
            embedding_enable_download=embedding_cfg.get("enable_download", False),
            embedding_timeout=embedding_cfg.get("timeout", 10),
            embedding_device=embedding_cfg.get("device", "auto"),
            embedding_similarity_threshold=embedding_cfg.get("similarity_threshold", 0.85),
            download_dir=download_dir,
            enable_monitoring=ai_service.get("enable_monitoring", True),
            metrics_port=ai_service.get("metrics_port", 9090),
            auto_run_agent=ai_service.get("auto_run_agent", True),
            max_agent_iterations=ai_service.get("max_agent_iterations", 10),
            agent_execution_interval=ai_service.get("agent_execution_interval", AGENT_EXECUTION_INTERVAL),
            data_analysis_interval=ai_service.get("data_analysis_interval", DATA_ANALYSIS_INTERVAL),
            enable_concurrent_execution=ai_service.get("enable_concurrent_execution", True),
            require_real_backend=ai_service.get("require_real_backend", True),
        )

    def _validate_instructions(self):
        # Create validator with custom patterns from config
        validator_cfg = self.config.get("ai_service", {}).get("validator", {})
        forbidden_patterns = validator_cfg.get("forbidden_patterns")
        self.instruction_validator = InstructionValidator(
            tool_registry=self.tool_registry,
            forbidden_patterns=forbidden_patterns
        )

        # Validate the agent instruction
        if self.service_config.agent_instruction:
            is_valid, error = self.instruction_validator.validate(self.service_config.agent_instruction)
            if not is_valid:
                raise ValueError(f"Invalid agent instruction: {error}")
            logger.info("✅ Agent instruction validated successfully")

    def _start_memory_monitor(self):
        if not self.service_config.enable_monitoring:
            return
        self.memory_monitor = RealTimeMemoryMonitor(
            model_manager=get_model_manager(),
            check_interval=2.0,
        )
        if self._in_async_context:
            asyncio.create_task(self.memory_monitor.start())
        else:
            logger.warning("No async context – memory monitor not started")

    def get_best_available_model(self) -> BaseLLMAdapter:
        """Return the healthiest available model, with comprehensive fallback."""
        if not self.models:
            logger.error("❌ No models available in registry")
            dummy_config = ModelConfig(provider="dummy", model_name="dummy")
            return DummyAdapter(dummy_config)

        # Ordered by preference
        candidates = [
            self.service_config.primary_model,
            'llama-cpp', 'llama_cpp',
            'huggingface', 'hf',
            'gpt4all', 'openai',
            'dummy'
        ]

        # Try to find a healthy model in order
        for model_key in candidates:
            if model_key in self.models:
                adapter = self.models[model_key]

                # Check health
                try:
                    if adapter.is_healthy():
                        logger.info(f"✅ Using model: {model_key}")
                        return adapter
                    else:
                        logger.warning(f"⚠️ Model {model_key} exists but not healthy yet")
                except Exception as e:
                    logger.warning(f"⚠️ Health check failed for {model_key}: {e}")
                    continue

        # Last resort: find ANY non-dummy model
        for model_key, adapter in self.models.items():
            if not isinstance(adapter, DummyAdapter):
                try:
                    logger.warning(f"⚠️ Using unverified model: {model_key}")
                    return adapter
                except Exception:
                    continue

        # Ultimate fallback
        if self.service_config.require_real_backend:
            raise RuntimeError(
                "A3: No healthy real backend available and "
                "ai_service.require_real_backend is true (default)."
            )
        logger.error("🚨 All models failed, returning DummyAdapter")
        if 'dummy' in self.models:
            return self.models['dummy']
        dummy_config = ModelConfig(provider="dummy", model_name="dummy")
        return DummyAdapter(dummy_config)

    async def _init_components_async(self) -> None:
        """Async component initialization.

        `_init_database` is offloaded to a daemon thread because its
        `DatabaseManager.add_connection` / `health_check` calls do blocking
        network I/O that could otherwise stall the loop or hang interpreter
        exit. `_init_memory`, `_init_embedding`, and `_init_vectorstore`
        perform no cooperative waiting and stay inline.

        `_init_models_async` handles its own thread offloading.
        """
        try:
            # Database connection is the only one that touches the network,
            # and DatabaseManager's connect + health_check are both blocking
            # sync calls. Run them on a throwaway daemon thread so a stalled
            # socket cannot (a) block the event loop or (b) hang interpreter
            # shutdown at `_python_exit` via the loop's default executor.
            await self._call_in_daemon_thread(
                self._init_database,
                timeout=30.0,
                label="db_init",
            )
            self._init_memory()
            self._init_embedding()
            self._init_vectorstore()
            await self._init_models_async()

            self._register_default_tools()

            download_dir = self.service_config.download_dir or str(DEFAULT_DOWNLOAD_DIR)
            self.tools_registrar = ToolsRegistrar(dest_download_dir=download_dir)

            self._register_tools()
            self._init_scheduler()

            await self._init_core_components_async()

            logger.info("✅ All components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}", exc_info=True)
            raise

    async def _init_core_components_async(self) -> None:
        """Initialize AgentExecutor and DataAnalyzer.

        Retry wait is `await asyncio.sleep(3.0)` — the event loop keeps
        running (memory monitor, backup retrier, external callers) during
        the backoff window. The previous `time.sleep(3)` froze the loop
        for up to 42 s in the all-fail path.
        """
        try:
            max_retries = 15

            for retry_count in range(max_retries):
                adapter = self.get_best_available_model()

                if adapter is None:
                    logger.error(
                        f"❌ No model adapter available "
                        f"(attempt {retry_count + 1}/{max_retries})"
                    )
                    if retry_count < max_retries - 1:
                        await asyncio.sleep(3.0)
                    continue

                if isinstance(adapter, DummyAdapter):
                    logger.error(
                        f"❌ Only DummyAdapter available "
                        f"(attempt {retry_count + 1}/{max_retries})"
                    )
                    if retry_count < max_retries - 1:
                        await asyncio.sleep(3.0)
                    continue

                logger.info(
                    f"✅ Using model: {adapter.config.provider} "
                    f"({adapter.config.model_name})"
                )

                _iters  = self.service_config.max_agent_iterations
                _floor  = 1800
                _per    = 600
                _cap    = self.service_config.agent_execution_interval
                _exec_timeout = max(_floor, min(_iters * _per + 300, _cap))
                logger.info(
                    f"⏱️  AgentExecutor timeout set to {_exec_timeout}s "
                    f"(iters={_iters}, per_iter_estimate={_per}s)"
                )

                self.agent_executor = AgentExecutor(
                    llm=adapter,
                    tool_registry=self.tool_registry,
                    instruction=self.service_config.agent_instruction,
                    max_iterations=self.service_config.max_agent_iterations,
                    max_total_tokens=getattr(
                        self.service_config, "max_total_tokens", 2048,
                    ),
                    timeout=_exec_timeout,
                )
                self.agent_executor.set_service(self)
                logger.info("✅ AgentExecutor initialized")

                self.data_analyzer = DataAnalyzer(
                    llm=adapter,
                    tool_registry=self.tool_registry,
                    temperature=0.1,
                    max_tokens=512,
                    timeout=self.service_config.request_timeout,
                    enable_validation=True,
                )
                self.data_analyzer.set_service(self)
                logger.info("✅ DataAnalyzer initialized")

                logger.info(
                    "✅ Core components (AgentExecutor and DataAnalyzer) "
                    "initialized successfully"
                )
                return

            logger.error(
                f"❌ Failed to initialize core components after "
                f"{max_retries} attempts"
            )
            raise RuntimeError(
                "No suitable model adapter available for core components"
            )
        except Exception as e:
            logger.error(
                f"❌ Failed to initialize core components: {e}", exc_info=True,
            )
            raise

    def _start_autonomous_execution(self):
        """🚀 Start autonomous execution of AgentExecutor and DataAnalyzer (guarded)."""
        if hasattr(self, '_agent_task') and self._agent_task and not self._agent_task.done():
            logger.warning("⚠️ Agent execution already running, skipping")
            return

        if hasattr(self, '_data_analyzer_task') and self._data_analyzer_task and not self._data_analyzer_task.done():
            logger.warning("⚠️ Data analysis already running, skipping")
            return

        try:
            if self.agent_executor and self.data_analyzer:
                try:
                    # Preferred: create tasks on running loop
                    self._agent_task = asyncio.create_task(
                        self.agent_executor.start_continuous_execution(
                            interval=self.service_config.agent_execution_interval
                        )
                    )
                    self._data_analyzer_task = asyncio.create_task(
                        self.data_analyzer.start_continuous_analysis(
                            interval=self.service_config.data_analysis_interval,
                            batch_size=10
                        )
                    )
                except RuntimeError:
                    # No running loop in this thread. Recover only if
                    # __init__ captured a live loop via get_running_loop()
                    # (i.e. AIModelService was constructed inside async
                    # context). Otherwise refuse to start the loops here —
                    # creating an ad-hoc loop from a sync context produces
                    # tasks that never run.
                    loop = getattr(self, "_loop", None)
                    if loop is None or not loop.is_running():
                        logger.warning(
                            "Autonomous execution skipped: no running event "
                            "loop. Construct AIModelService from an async "
                            "context (or call a start_* method from one) to "
                            "enable the agent and data analyzer."
                        )
                        return
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(
                            self.agent_executor.start_continuous_execution(
                                interval=self.service_config.agent_execution_interval
                            )
                        )
                    )
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(
                            self.data_analyzer.start_continuous_analysis(
                                interval=self.service_config.data_analysis_interval,
                                batch_size=10
                            )
                        )
                    )

                self.metrics["components_running"]["agent_executor"] = True
                self.metrics["components_running"]["data_analyzer"] = True

                logger.info("🚀 Started autonomous execution of AgentExecutor and DataAnalyzer")
        except Exception as e:
            logger.error(f"❌ Failed to start autonomous execution: {e}")

    def _init_database(self):
        """🗄️ Initialize database connection using DatabaseManager."""
        if not self.service_config.database:
            logger.info("ℹ️ No database configuration, skipping DB initialization")
            return

        cfg = self.service_config.database

        try:
            # Initialize DatabaseManager
            if not DBTOOLS_AVAILABLE or DatabaseManager is None:
                logger.warning("⚠️ Database tools not available. Install dbtools or check imports.")
                return

            self.db_manager = DatabaseManager()

            # Build PostgreSQL URI
            uri = f"postgresql://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.database}"

            # Add connection to DatabaseManager
            result = self.db_manager.add_connection("primary", uri, {
                "min_conn": cfg.min_connections,
                "max_conn": cfg.max_connections,
                "sslmode": cfg.sslmode,
                "connect_timeout": cfg.timeout
            })

            if not result.get("ok"):
                logger.warning(f"⚠️ Failed to add database connection: {result.get('error')}")
                self.db_manager = None
                return

            # Test connection
            health = self.db_manager.health_check("primary")
            if not health.get("ok"):
                logger.warning(f"⚠️ Database health check failed: {health.get('error')}")
                self.db_manager = None
                return

            logger.info("✅ Database connection initialized via DatabaseManager")

            # Create tables using DatabaseManager
            self._create_database_tables()

        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}", exc_info=True)
            self.db_manager = None

    def _create_database_tables(self):
        """Ensure the database schema matches src/amb_sentinel/data/schema.sql.

        DDL lives in exactly one place: `src/amb_sentinel/data/schema.sql`,
        applied by `amb_sentinel.db.apply_schema()`. This method no longer
        contains any CREATE TABLE statements of its own; the entire schema
        (including pgvector dim/lists from config) is rendered and applied
        idempotently.
        """
        if not self.db_manager:
            logger.warning("⚠️ No DatabaseManager available - skipping DB schema verification")
            return
        from amb_sentinel import db as _db_mod
        try:
            result = _db_mod.apply_schema(self.db_manager, self._raw_config or {})
        except Exception as e:
            logger.error(f"❌ DB schema apply raised: {e}", exc_info=True)
            return
        if not result.get("ok"):
            logger.error(f"❌ DB schema apply failed: {result.get('error')}")
            if result.get("hint"):
                logger.info(f"ℹ️ {result['hint']}")
            return
        logger.info(
            f"✅ DB schema ready "
            f"(pgvector dim={result.get('dimension')} lists={result.get('lists')})"
        )

    def _init_memory(self):
        """🧠 Initialize memory store with better fallback strategy."""
        try:
            if self.service_config.redis and _OPTIONAL_IMPORTS['redis']:
                self.memory = RedisMemory(self.service_config.redis)
                self.memory.add_message("test", "system", "connection test", {"test": True})
                logger.info("✅ Using Redis memory store")
            else:
                self.memory = InMemoryMemory()
                logger.info("✅ Using in-memory store")
        except Exception as e:
            logger.warning(f"⚠️ Memory initialization failed: {e}, falling back to in-memory")
            self.memory = InMemoryMemory()

    def _init_embedding(self):
        """🔤 Initialize embedding adapter."""
        try:
            self.embedding_adapter = EmbeddingAdapter(
                self.service_config.embedding_model,
                self.service_config.embedding_cache_size,
                self.service_config.embedding_cache_dir,
                self.service_config.embedding_device,
                self.service_config.embedding_timeout,
                self.service_config.embedding_enable_download,
            )
            _degraded = bool(
                getattr(self.embedding_adapter, "degraded",
                       self.embedding_adapter._model is None)
                )

            if _degraded:
                logger.warning(
                    "⚠️ Embedding adapter DEGRADED — model %r failed to load; "
                    "using deterministic hash fallback. Vectors will have "
                    "different semantics from real embeddings. "
                    "cache_dir=%s  download=%s. "
                    "Fix: ensure the model snapshot is present under "
                    "cache_dir, and set HF_HUB_OFFLINE=1 + "
                    "TRANSFORMERS_OFFLINE=1 in the service environment ",
                    self.service_config.embedding_model,
                    self.service_config.embedding_cache_dir,
                    self.service_config.embedding_enable_download,
                    )
            else:
                logger.info(
                    "✅ Embedding adapter initialized with model: %s",
                    self.service_config.embedding_model,
                )
        except Exception as e:
            logger.error(
                f"❌ Failed to initialize embedding adapter: {e}",
                exc_info = True,
            )
            self.embedding_adapter = None

    def _init_vectorstore(self):
        """🗄️ Initialize vector store."""
        try:
            pgv_enable = False
            pgv_dim = 384
            pgv_index = "ivfflat"
            pgv_lists = 100
            pgv_probes = 10
            pgv_operator = "vector_cosine_ops"

            if self.service_config.database:
                pgv = self.service_config.database.pgvector
                pgv_enable = pgv.enabled
                pgv_dim = pgv.dimension
                pgv_index = pgv.index_type
                pgv_lists = pgv.lists
                pgv_probes = pgv.probes
                pgv_operator = pgv.operator

            if self.db_manager and self.embedding_adapter:
                _degraded = bool(
                    getattr(self.embedding_adapter, "degraded", False)
                )
                _allow_degraded = bool(
                    (self._raw_config or {})
                    .get("ai_service", {})
                    .get("embeddings", {})
                    .get("allow_degraded_writes", False)
                )
                self.vectorstore = VectorStore(
                    db_manager=self.db_manager,
                    connection_name="primary",
                    table_name="embeddings",
                    embedding_dim=pgv_dim,
                    cache_size=1000,
                    pgvector_enabled=pgv_enable,
                    pgvector_dimension=pgv_dim,
                    pgvector_index_type=pgv_index,
                    pgvector_lists=pgv_lists,
                    pgvector_probes=pgv_probes,
                    pgvector_operator=pgv_operator,
                    degraded_embeddings=_degraded,
                    allow_degraded_writes=_allow_degraded,
                )
                logger.info("✅ Vector store initialized with DatabaseManager")
            elif self.embedding_adapter:
                self.vectorstore = VectorStore(
                    db_manager=None,
                    connection_name="primary",
                    table_name="embeddings",
                    embedding_dim=pgv_dim,
                    cache_size=1000,
                    pgvector_enabled=False,  # no DB, so disable
                )
                logger.info("✅ Vector store initialized in memory only (no database)")
            else:
                self.vectorstore = None
                logger.warning("⚠️ Vector store not available (missing embeddings)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            self.vectorstore = None

    async def _init_models_async(self) -> None:
        """
        Async model initialisation.

        Every blocking call — network probe, BackendOrchestrator probe,
        ModelFactory construction, adapter load — is offloaded to a
        worker thread via run_in_executor. Previously this method used
        ThreadPoolExecutor.submit(...).result(timeout=120) inline, which
        froze the event loop for up to two minutes while the primary
        adapter loaded.
        """
        logger.info(
            f"🔧 Initializing models... Primary: {self.service_config.primary_model}"
        )
        loop = asyncio.get_running_loop()

        # ====================================================================
        # local-llm-cloud reachability probe
        # ====================================================================
        if self.service_config.require_real_backend:
            _llm_cfg      = self.config.get("llm_cloud", {}) or {}
            _host         = _llm_cfg.get("host", "localhost")
            _port         = int(_llm_cfg.get("port", 8080))
            _api_key      = (os.environ.get("LLM_CLOUD_API_KEY", "")
                             or _llm_cfg.get("api_key", "") or "")
            _probe_timeout = float(_llm_cfg.get("probe_timeout_s", 3.0))

            def _probe() -> bool:
                try:
                    from amb_sentinel.llm.llm_cloud_adapter import probe_server
                    return probe_server(
                        _host, _port,
                        timeout=_probe_timeout,
                        api_key=_api_key,
                    )
                except Exception as exc:
                    logger.warning(
                        "A3: local-llm-cloud probe raised %s; treating as "
                        "unreachable.", exc,
                    )
                    return False

            try:
                _reachable = await loop.run_in_executor(None, _probe)
            except Exception as exc:
                logger.warning("A3: probe task failed: %s", exc)
                _reachable = False

            if not _reachable:
                raise RuntimeError(
                    f"A3: ai_service.require_real_backend=true and "
                    f"local-llm-cloud at {_host}:{_port} did not answer "
                    f"/health within {_probe_timeout}s. "
                    f"Start the server (see Local-LLM README), set "
                    f"LLM_CLOUD_WAIT_S to a larger value if it is cold-"
                    f"starting, or set ai_service.require_real_backend=false "
                    f"in config.yaml to allow the DummyAdapter for dev."
                )

            try:
                _llm_cfg = self.config.get("llm_cloud", {}) or {}
                logger.info(
                    "LLM endpoint: host=%s port=%s api_key_set=%s",
                    _llm_cfg.get("host", "localhost"),
                    _llm_cfg.get("port", 8080),
                    bool(_llm_cfg.get("api_key") or os.environ.get("LLM_CLOUD_API_KEY")),
                )
            except Exception:
                pass

        # ====================================================================
        # BackendOrchestrator: probe all backends, feed HW profile into
        # AdaptiveModelManager. Blocking (subprocess / network probes), so
        # off-loop. Failure is non-fatal.
        # ====================================================================
        orch = None

        def _probe_orchestrator():
            llm_cloud_cfg = {}
            if getattr(self, "_raw_config", None):
                llm_cloud_cfg = self._raw_config.get("llm_cloud", {})
            o = get_backend_orchestrator({"llm_cloud": llm_cloud_cfg})
            o.apply_hw_profile_to_manager()
            return o

        try:
            orch = await loop.run_in_executor(None, _probe_orchestrator)
            logger.info(
                "🔍 BackendOrchestrator ready: active=%s status=%s",
                orch._active.value if orch._active else "none",
                {b.value: s.value for b, s in orch._status.items()
                 if s.value != "unknown"},
            )
        except Exception as exc:
            logger.warning("BackendOrchestrator probe error (non-fatal): %s", exc)
            orch = None

        # ====================================================================
        # Resolve "auto" primary
        # ====================================================================
        primary_provider = self.service_config.primary_model
        if primary_provider in ("auto", "", None):
            if orch is not None:
                try:
                    primary_provider = orch.recommend_provider()
                    logger.info("🏭 AUTO primary model → '%s'", primary_provider)
                except Exception:
                    primary_provider = "llama_cpp"
            else:
                primary_provider = "llama_cpp"

        primary_loaded   = False
        primary_adapter  = None
        fallback_adapter = None

        # ====================================================================
        # STEP 1: load primary model
        # ====================================================================
        primary_config = None
        for model_config in self.service_config.models:
            if model_config.provider == primary_provider and model_config.enabled:
                primary_config = model_config
                break
        if primary_config is None:
            for model_config in self.service_config.models:
                if model_config.enabled:
                    primary_config = model_config
                    break

        if primary_config is None:
            logger.error(
                f"❌ Primary model '{primary_provider}' not found in config"
            )
        else:
            try:
                logger.info(f"🔧 Loading PRIMARY model: {primary_config.provider}")

                adapter = await loop.run_in_executor(
                    None, ModelFactory.create_model, primary_config,
                )

                if isinstance(adapter, DummyAdapter):
                    logger.error(
                        f"❌ Primary model {primary_config.provider} creation "
                        f"returned DummyAdapter"
                    )
                    logger.error(
                        "   This means dependencies are missing or model file "
                        "not found"
                    )
                    logger.error(f"   Model file: {primary_config.model_name}")
                    primary_loaded = False
                else:
                    try:
                        logger.info("⚙️  Initializing adapter (daemon thread)...")
                        result = await self._run_adapter_init_async(
                            adapter,
                            timeout=120.0,
                            label=f"adapter_init:{primary_config.provider}",
                        )
                        if not result:
                            logger.error(
                                "❌ Adapter initialization returned False"
                            )
                            self._primary_load_exc = self._last_adapter_init_exc
                            primary_loaded = False
                        else:
                            self.models[primary_config.provider] = adapter
                            if hasattr(adapter, "_run_startup_benchmark"):
                                # Fire-and-forget; safe here because
                                # we are inside the running loop. The
                                # wrapper captures any exception via
                                # logger.warning instead of letting it
                                # surface as "Task exception was never
                                # retrieved" at GC time.
                                asyncio.create_task(
                                    self._safe_benchmark(adapter),
                                    name=f"startup_benchmark:{primary_config.provider}",
                                )
                            primary_loaded  = True
                            primary_adapter = adapter
                            logger.info(
                                f"✅ PRIMARY model loaded successfully: "
                                f"{primary_config.provider}"
                            )
                    except asyncio.TimeoutError:
                        logger.error(
                            "❌ Adapter initialization timed out after 120s"
                        )
                        self._primary_load_exc = asyncio.TimeoutError("adapter init exceeded 120s")
                        primary_loaded = False
                    except Exception as exc:
                        logger.error(
                            f"❌ Failed to initialize primary model: {exc}",
                            exc_info=True,
                        )
                        self._primary_load_exc = exc
                        primary_loaded = False
            except Exception as exc:
                logger.error(
                    f"❌ Failed to create primary model "
                    f"{primary_config.provider}: {exc}",
                    exc_info=True,
                )
                self._primary_load_exc = exc
                primary_loaded = False

        # ====================================================================
        # STEP 2: fallback models if primary failed
        # ====================================================================
        if not primary_loaded:
            logger.warning(
                "⚠️ Primary model failed to load, trying fallback models..."
            )

            for model_config in self.service_config.models:
                # skip the *resolved* primary provider, not the
                # literal config value. When primary_model is "auto",
                # the old check compared "some_provider" != "auto" and
                # never skipped anything, so the same adapter was retried
                # as its own fallback.
                if model_config.provider == primary_provider:
                    continue
                if not model_config.enabled:
                    continue
                if model_config.provider in self.models:
                    continue

                try:
                    logger.info(
                        f"🔧 Trying fallback model: {model_config.provider}"
                    )
                    adapter = await loop.run_in_executor(
                        None, ModelFactory.create_model, model_config,
                    )

                    if isinstance(adapter, DummyAdapter):
                        logger.info(
                            f"⏭️  Skipping {model_config.provider} "
                            f"(dependencies missing)"
                        )
                        continue

                    try:
                        result = await self._run_adapter_init_async(
                            adapter,
                            timeout=60.0,
                            label=f"adapter_init:{model_config.provider}",
                        )
                        if result:
                            self.models[model_config.provider] = adapter
                            fallback_adapter = adapter
                            logger.info(
                                f"✅ Loaded fallback model: "
                                f"{model_config.provider}"
                            )
                            break
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"⚠️ Fallback model {model_config.provider} "
                            f"init timed out after 60s"
                        )
                        continue
                    except Exception as exc:
                        logger.warning(
                            f"⚠️ Fallback model {model_config.provider} "
                            f"init failed: {exc}"
                        )
                        continue

                except Exception as exc:
                    logger.warning(
                        f"⚠️ Failed to create fallback "
                        f"{model_config.provider}: {exc}"
                    )
                    continue

        # ====================================================================
        # STEP 3: final validation
        # ====================================================================
        if not self.models:
            if self.service_config.require_real_backend:
                _tried = [
                    m.provider for m in self.service_config.models
                    if m.enabled and m.provider != primary_provider
                ]
                _cause = self._primary_load_exc
                _cause_str = (
                    f"{type(_cause).__name__}: {_cause}"
                    if _cause is not None else "(none recorded)"
                    )

                # Distinguish config-class errors (fix config and restart)
                # from transport-class errors (retry with backoff).

                try:
                    from amb_sentinel.llm.llm_cloud_adapter import (
                            LLMCloudConfigError as _CfgErr,
                    )
                    _is_config_err = isinstance(_cause, _CfgErr)
                except ImportError:
                    _is_config_err = False

                _a3_msg = (
                        f"A3: no real LLM backend could be initialised.\n"
                        f"  resolved primary provider: {primary_provider!r}\n"
                        f"  configured (enabled): "
                        f"{[m.provider for m in self.service_config.models if m.enabled]}\n"
                        f"  attempted as fallback: {_tried}\n"
                        f"  probe result: reachable\n"
                        f"  root cause: {_cause_str}\n"
                        f"To allow DummyAdapter for dev, set "
                        f"ai_service.require_real_backend: false."
                        )

                if _is_config_err:
                    raise ConfigError(_a3_msg)
                raise RuntimeError(_a3_msg)
            logger.error(
                "❌ CRITICAL: No models loaded! Creating emergency DummyAdapter..."
            )
            dummy_config = ModelConfig(provider="dummy", model_name="dummy")
            self.models["dummy"] = DummyAdapter(dummy_config)
        else:
            logger.info(f"✅ Models initialized: {list(self.models.keys())}")
            primary_ok = self.service_config.primary_model in self.models
            logger.info(f"✅ Primary model available: {primary_ok}")

    def _init_adapter_sync(self, adapter: BaseLLMAdapter) -> bool:
        """Helper to initialize adapter synchronously (thread-safe)."""
        try:
            # Create a new event loop for this thread
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(adapter.initialize())
                # Cancel any remaining background tasks (e.g., the benchmark)
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                logger.info("✅ Adapter initialized in executor thread")
                return True
            finally:
                loop.close()

        except Exception as e:
            self._last_adapter_init_exc = e
            logger.error(f"❌ Sync init failed: {e}", exc_info=True)
            return False

    async def _run_adapter_init_async(
        self,
        adapter: BaseLLMAdapter,
        timeout: float,
        label: str = "adapter_init",
    ) -> bool:
        """
        Run `_init_adapter_sync` on a dedicated daemon thread and await
        the result, with a hard timeout.

        Background — why not `run_in_executor(None, ...)`
        -------------------------------------------------
        Adapter loading blocks inside a C library call:
          * llama.cpp's `Llama(...)` constructor (mmap + Q4 dequant),
          * HF's `from_pretrained` (safetensors mmap + tensor init),
          * CUDA context warmup.

        Python cannot interrupt such a call from another thread, and
        the underlying C library is under no obligation to poll a
        cancellation flag. `loop.run_in_executor(None, ...)` uses
        `concurrent.futures.ThreadPoolExecutor`, whose workers are:

          a) registered in `_threads_queues`, and
          b) spawned as non-daemon threads.

        At interpreter exit, `concurrent.futures.thread._python_exit`
        (registered with `atexit`) iterates `_threads_queues` and
        calls `t.join()` on every entry. A worker stuck in a C call
        that never returns therefore prevents the process from
        exiting — after the rest of the shutdown sequence has already
        completed cleanly. The `daemon=True` flag does *not* help
        here; it only affects the later daemon-thread kill phase,
        which runs *after* atexit.

        This helper sidesteps that:
          * the worker is a raw `threading.Thread(daemon=True)` and
            is therefore **not** registered in `_threads_queues`;
          * it is never joined;
          * on timeout this coroutine returns control immediately and
            the thread continues in the background. If the C call
            eventually returns, the thread exits quietly. If it never
            does, the OS reaps the daemon thread at process exit.

        The thread object is retained on `self._adapter_init_thread`
        so `shutdown_async` can log whether it is still alive. We
        never join it.

        Result bridge
        -------------
        `_init_adapter_sync` already spins up its own event loop in
        its own thread, so it is safe to call from any thread. Its
        return value is delivered back to the awaiting coroutine via
        `loop.call_soon_threadsafe` on the *original* loop.

        Timeout semantics
        -----------------
        On timeout this method raises `asyncio.TimeoutError` and the
        thread keeps running. Do not call this a second time for the
        same adapter while the previous thread is still alive: the
        adapter's underlying state is not guaranteed to tolerate
        concurrent `_init_adapter_sync` invocations. The caller in
        `_init_models_async` already guarantees serial invocations.
        """
        loop = asyncio.get_running_loop()
        done: asyncio.Future = loop.create_future()

        prev = self._adapter_init_thread
        if prev is not None and prev.is_alive():
            # Diagnostic only. A previous load was abandoned on timeout
            # and its thread is still inside a C call. We do not join
            # it; we just note that a second one is starting.
            logger.warning(
                "Previous adapter-init thread %r is still alive; "
                "spawning a new one (%s). The old thread will not "
                "be joined.",
                prev.name, label,
            )

        def _worker():
            try:
                ok = self._init_adapter_sync(adapter)
            except BaseException as exc:
                # Failure path. Deliver the exception, not a result.
                # Do NOT reference `ok` here: the exception happened
                # before it was assigned.
                def _set_exc(exc=exc):
                    try:
                        if not done.done():
                            done.set_exception(exc)
                    except asyncio.InvalidStateError:
                        pass

                loop.call_soon_threadsafe(_set_exc)
                return

            # Success path. Bind `ok` as a default so the closure
            # captures the value at definition time, and check
            # `done.done()` on the loop thread to avoid the
            # check-cancel-set race.
            def _set_result(v=ok):
                if not done.done():
                    done.set_result(v)

            loop.call_soon_threadsafe(_set_result)

        t = threading.Thread(
            target=_worker,
            name=f"{label}-{id(adapter):x}",
            daemon=True,
        )
        self._adapter_init_thread = t
        try:
            t.start()
        except RuntimeError as exc:
            logger.error("Failed to start adapter-init thread: %s", exc)
            self._adapter_init_thread = None
            raise

        try:
            return await asyncio.wait_for(done, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                "❌ %s exceeded %.0fs; daemon thread %r continues in "
                "the background and cannot be interrupted from Python. "
                "Process shutdown will not block on it.",
                label, timeout, t.name,
            )
            raise

    async def _call_in_daemon_thread(
        self,
        fn: Callable[[], Any],
        *,
        timeout: Optional[float] = None,
        label: str = "daemon_call",
    ) -> Any:
        """
        Run a synchronous callable on a throwaway daemon thread and
        await its return value.

        Unlike `loop.run_in_executor(None, fn)`, the thread is **not**
        registered in `concurrent.futures.thread._threads_queues`.
        That means `asyncio.run()`'s `shutdown_default_executor()` and
        the module-level `_python_exit` handler will not attempt to
        join it at interpreter exit. A blocking call inside `fn`
        (network round-trip with no timeout, mmap on slow storage)
        therefore cannot hang process shutdown.

        Use this for one-shot startup calls where spawning a full
        DaemonWorkerPool would be overkill. For repeated work, use
        `DaemonWorkerPool` directly.
        """
        loop = asyncio.get_running_loop()
        done: asyncio.Future = loop.create_future()

        def _worker() -> None:
            try:
                result = fn()
            except BaseException as exc:
                def _set_exc(exc=exc) -> None:
                    try:
                        if not done.done():
                            done.set_exception(exc)
                    except asyncio.InvalidStateError:
                        pass
                loop.call_soon_threadsafe(_set_exc)
                return

            def _set_result(v=result) -> None:
                try:
                    if not done.done():
                        done.set_result(v)
                except asyncio.InvalidStateError:
                    pass
            loop.call_soon_threadsafe(_set_result)

        t = threading.Thread(target=_worker, name=label, daemon=True)
        t.start()

        if timeout is None:
            return await done
        try:
            return await asyncio.wait_for(done, timeout=timeout)
        except asyncio.TimeoutError:
            # The caller's wait_for cancels the local future and gives up;
            # the thread keeps running in the background. It cannot be
            # interrupted from Python (it may be inside a C call or a
            # blocking socket read), and it is a raw daemon thread so
            # interpreter exit will not join it. Log so a stuck call is
            # visible in the log rather than silently ignored.
            logger.warning(
                "⏱️ %s exceeded %.1fs; daemon thread %r continues in "
                "the background and cannot be interrupted from Python. "
                "Process shutdown will not block on it.",
                label, timeout, t.name,
            )
            raise

    async def _safe_benchmark(self, adapter: BaseLLMAdapter) -> None:
        """
        Run the adapter's startup benchmark, swallowing any exception.

        `_run_startup_benchmark` is scheduled fire-and-forget from
        `_init_models_async`. A bare `asyncio.create_task(...)` on a
        coroutine that raises leaves the exception unretrieved until GC,
        at which point Python prints

            Task exception was never retrieved
            future: <Task finished ... exception=KeyError('long')>

        to stderr — which does not go through the logging framework and
        is easy to miss in a supervisor's log filter.

        Wrapping the coroutine here means every failure path goes through
        `logger.warning` and is therefore captured by the standard
        handlers. The benchmark is best-effort by design: a failure must
        never abort model loading or leave `measured_tokens_per_sec` in
        an inconsistent state.
        """
        try:
            await adapter._run_startup_benchmark()
        except asyncio.CancelledError:
            # Shutdown during benchmark — not an error. Re-raise so the
            # task's cancellation semantics are preserved.
            raise
        except Exception as exc:
            logger.warning(
                "Startup benchmark failed for %s: %s",
                type(adapter).__name__, exc,
            )

    def _init_scheduler(self):
        """⏰ Initialize background scheduler with enhanced jobs."""
        try:
            self.scheduler = BackgroundScheduler()

            self.scheduler.add_job(
                self._health_check,
                IntervalTrigger(seconds=self.service_config.health_check_interval),
                id="health_check",
                misfire_grace_time=30
            )

            self.scheduler.add_job(
                self._warm_models,
                IntervalTrigger(seconds=300),
                id="model_warming",
                misfire_grace_time=30
            )

            self.scheduler.add_job(
                self._cleanup_memory,
                IntervalTrigger(hours=24),
                id="memory_cleanup",
                misfire_grace_time=300
            )

            self.scheduler.add_job(
                self._collect_metrics,
                IntervalTrigger(seconds=60),
                id="metrics_collection",
                misfire_grace_time=30
            )

            self.scheduler.start()
            logger.info("✅ Background scheduler started with enhanced jobs")

        except Exception as e:
            logger.error(f"❌ Failed to initialize scheduler: {e}")
            self.scheduler = None

    def _register_default_tools(self):
        """🛠️ Register default tools with enhanced capabilities."""
        if self.vectorstore:
            self.tool_registry.register_function(
                self.search_vectorstore,
                name="search_vectorstore",
                description="Search vector store for similar content",
                timeout=10.0,
                retries=2,
                schema={
                    "query": {"type": "string", "required": True},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                    "similarity_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                }
            )

        self.tool_registry.register_function(
            self.get_system_status,
            name="get_system_status",
            description="Get comprehensive system status and health information",
            timeout=5.0,
            retries=1
        )

        if self.db_manager:
            self.tool_registry.register_function(
                self.get_analysis_history,
                name="get_analysis_history",
                description="Get recent analysis results from database",
                timeout=10.0,
                retries=1,
                schema={
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "threat_level": {"type": "string", "enum": ["critical", "high", "medium", "low", "informational"]}
                }
            )

        if self.db_manager:
            self.tool_registry.register_function(
                self.query_database,
                name="query_database",
                description="Execute SQL query on database (read-only)",
                timeout=15.0,
                retries=1,
                schema={
                    "query": {"type": "string", "required": True},
                    "parameters": {"type": "array", "items": {"type": "string"}}
                }
            )

        logger.info(f"✅ Registered {len(self.tool_registry.tools)} default tools")

    def _register_tools(self):
        """🛠️ Register external tools (pipeline and scraper) using ToolsRegistrar."""
        try:
            # Log all registered tools for debugging
            tool_list = [tool.name for tool in self.tool_registry.tools.values()]
            logger.info(f"🔧 Registered tools: {', '.join(tool_list)}")

            # Use ToolsRegistrar to register all tools
            self.tools_registrar.register_with_service(self)
            logger.info(f"✅ Total tools registered: {len(self.tool_registry.tools)}")

            total = len(self.tool_registry.tools)
            hidden = sum(
                1 for n in self.tool_registry.tools
                if n.startswith(("create_", "check_", "get_default_"))
            )
            logger.info(
                "tools: registered=%d exposed_to_llm=%d hidden_factories=%d",
                total, total - hidden, hidden,
            )

        except Exception as e:
            logger.error(f"❌ Tools registration failed: {e}", exc_info=True)
            # Don't raise - continue without tools

    async def run_agent_from_instruction(self) -> Dict[str, Any]:
        if not self.service_config.agent_instruction:
            logger.warning("⚠️ No agent instruction configured")
            return {"success": False, "error": "No agent instruction configured"}

        logger.info(f"🚀 Running agent: {self.service_config.agent_instruction[:100]}...")
        try:
            if not self.agent_executor:
                self.agent_executor = self.create_agent()
            self.metrics["requests_total"] += 1
            result = await self.agent_executor.arun(
                instruction=self.service_config.agent_instruction,
                timeout=self.service_config.request_timeout * 2,
            )
            self.metrics["agent_executions"] += 1
            logger.info(f"✅ Agent execution completed: success={result.get('success', False)}")
            return result
        except Exception as e:
            self.metrics["requests_failed"] += 1
            logger.error(f"❌ Agent execution failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def analyze_content(
            self,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        🔍 Enhanced content analysis with rate limiting and comprehensive error handling.
        """
        metadata = metadata or {}
        metadata.update({
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        start_time = time.time()

        async with self.request_semaphore:
            try:
                self.metrics["requests_total"] += 1
                self.metrics["last_request"] = time.time()

                adapter = self.get_best_available_model()
                if not adapter:
                    raise ValueError(f"Primary model '{self.service_config.primary_model}' not available")

                if not self.data_analyzer:
                    self.data_analyzer = DataAnalyzer(
                        llm=adapter,
                        tool_registry=self.tool_registry,
                        temperature=0.1,
                        max_tokens=512,
                        timeout=self.service_config.request_timeout,
                        enable_validation=True
                    )
                    self.data_analyzer.set_service(self)

                result = await asyncio.wait_for(
                    self.data_analyzer.analyze(content, metadata),
                    timeout=self.service_config.request_timeout
                )

                await self._persist_analysis_result(result)

                duration = time.time() - start_time
                self.metrics["requests_successful"] += 1
                self.metrics["data_analyses"] += 1
                self.metrics["avg_response_time"] = (
                                                            self.metrics["avg_response_time"] * (
                                                            self.metrics["requests_successful"] - 1) + duration
                                                    ) / self.metrics["requests_successful"]

                logger.info(f"✅ Analysis completed in {duration:.2f}s: {result.threat_level.value}")

                return result

            except asyncio.TimeoutError:
                duration = time.time() - start_time
                self.metrics["requests_failed"] += 1
                logger.error(f"⏱️ Analysis request timed out after {duration:.2f}s")

                return AnalysisResult(
                    content=content,
                    threat_level=ThreatLevel.INFORMATIONAL,
                    confidence=0.0,
                    entities=[],
                    categories=["timeout"],
                    reasoning=f"Analysis request timed out after {duration:.2f}s",
                    action_required=False,
                    metadata=metadata
                )

            except Exception as e:
                duration = time.time() - start_time
                self.metrics["requests_failed"] += 1
                logger.error(f"❌ Analysis failed after {duration:.2f}s: {e}", exc_info=True)

                return AnalysisResult(
                    content=content,
                    threat_level=ThreatLevel.INFORMATIONAL,
                    confidence=0.0,
                    entities=[],
                    categories=["error"],
                    reasoning=f"Analysis failed: {str(e)}",
                    action_required=False,
                    metadata=metadata
                )

    async def _persist_analysis_result(self, result: AnalysisResult):
        """💾 Persist analysis result to database with retry logic."""
        if not self.db_manager:
            logger.warning("⚠️ No database manager available, skipping persistence")
            return

        try:
            import hashlib
            content_hash = hashlib.sha256(result.content.encode()).hexdigest()

            query = """
                INSERT INTO analysis_results 
                (analysis_id, content_hash, content_preview, threat_level, confidence, 
                 categories, entities, reasoning, action_required, urgency, 
                 warnings, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (analysis_id) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    content_preview = EXCLUDED.content_preview,
                    threat_level = EXCLUDED.threat_level,
                    confidence = EXCLUDED.confidence,
                    categories = EXCLUDED.categories,
                    entities = EXCLUDED.entities,
                    reasoning = EXCLUDED.reasoning,
                    action_required = EXCLUDED.action_required,
                    urgency = EXCLUDED.urgency,
                    warnings = EXCLUDED.warnings,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP;
            """

            params = (
                result.analysis_id,
                content_hash,
                result.content[:500],
                result.threat_level.value,
                result.confidence,
                json.dumps(result.categories),
                json.dumps(result.entities),
                result.reasoning[:4000],
                result.action_required,
                result.urgency,
                json.dumps(result.warnings),
                json.dumps(result.metadata)
            )

            db_result = self.db_manager.execute("primary", query, params, fetch="none")
            if not db_result.get("ok"):
                logger.error(f"❌ Failed to persist analysis result: {db_result.get('error')}")
            else:
                logger.info(f"✅ Analysis result persisted: {result.analysis_id}")

        except Exception as e:
            logger.error(f"❌ Database error in _persist_analysis_result: {e}")

    def create_agent(self, max_iterations: int = 6, **kwargs) -> AgentExecutor:
        """🤖 Create an enhanced agent executor with registered tools and instruction."""
        adapter = self.models.get(self.service_config.primary_model)
        if not adapter:
            raise ValueError(f"No model available for agent (primary: {self.service_config.primary_model})")

        instruction = kwargs.pop("instruction", self.service_config.agent_instruction)

        agent = AgentExecutor(
            llm=adapter,
            tool_registry=self.tool_registry,
            instruction=instruction,
            max_iterations=max_iterations,
            planner_temperature=0.0,
            enable_reflection=True,
            enable_safety_checks=True,
            **kwargs
        )
        agent.set_service(self)
        return agent

    # Enhanced tool implementations
    def search_vectorstore(self, query: str, top_k: int = 5,
                           similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """🔍 Enhanced vector store search with better error handling."""
        if not self.vectorstore or not self.embedding_adapter:
            return {"error": "Vector store not available", "results": []}

        try:
            embeddings = self.embedding_adapter.embed([query])
            if not embeddings:
                return {"error": "Failed to generate embedding", "results": []}

            embedding = embeddings[0]

            results = self.vectorstore.query(
                embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )

            formatted_results = []
            for r in results:
                formatted_results.append({
                    "id": r[0],
                    "similarity": round(r[1], 4),
                    "content_preview": r[2].get("text_preview", "")[:200],
                    "source": r[2].get("source_type", "unknown"),
                    "metadata": r[2]
                })

            return {
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results)
            }

        except Exception as e:
            logger.error(f"❌ Vector store search failed: {e}", exc_info=True)
            return {"error": f"Search failed: {str(e)}", "results": []}

    def get_system_status(self) -> Dict[str, Any]:
        """📊 Get comprehensive system status with enhanced metrics."""
        model_status = {}
        for name, adapter in self.models.items():
            model_status[name] = {
                "healthy": adapter.is_healthy(),
                "metrics": adapter.get_metrics(),
                "type": adapter.__class__.__name__
            }

        components = {
            "database": self.db_manager is not None,
            "memory": self.memory is not None,
            "vectorstore": self.vectorstore is not None,
            "embeddings": self.embedding_adapter is not None,
            "scheduler": self.scheduler is not None and self.scheduler.running,
            "tools_registered": len(self.tool_registry.tools),
            "agent_executor": self.agent_executor is not None,
            "data_analyzer": self.data_analyzer is not None
        }

        # Add component running status
        components.update(self.metrics["components_running"])

        all_healthy = all(components.values()) and len(model_status) > 0
        status = "HEALTHY" if all_healthy else "DEGRADED"

        capabilities = ModelFactory.detect_capabilities()

        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service_uptime": getattr(self, "_start_time", None),
            "components": components,
            "models": model_status,
            "primary_model": self.service_config.primary_model,
            "tools_registered": len(self.tool_registry.tools),
            "agent_instruction_configured": bool(self.service_config.agent_instruction),
            "metrics": self.metrics,
            "capabilities": capabilities,
            "config": {
                "max_concurrent_requests": self.service_config.max_concurrent_requests,
                "request_timeout": self.service_config.request_timeout,
                "health_check_interval": self.service_config.health_check_interval,
                "auto_run_agent": self.service_config.auto_run_agent,
                "agent_execution_interval": self.service_config.agent_execution_interval,
                "data_analysis_interval": self.service_config.data_analysis_interval,
                "enable_concurrent_execution": self.service_config.enable_concurrent_execution
            }
        }

    def get_analysis_history(self, limit: int = 10,
                             threat_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """📜 Get recent analysis results with filtering using DatabaseManager."""
        if not self.db_manager:
            return []

        try:
            query = """
                SELECT analysis_id, content_hash, threat_level, confidence, 
                       categories, entities, reasoning, action_required,
                       urgency, warnings, metadata, analyzed_at
                FROM analysis_results
                WHERE 1=1
            """
            params = []

            if threat_level:
                query += " AND threat_level = %s"
                params.append(threat_level.lower())

            query += " ORDER BY analyzed_at DESC LIMIT %s;"
            params.append(limit)

            result = self.db_manager.execute("primary", query, params, fetch="all")
            if not result.get("ok"):
                logger.warning(f"⚠️ Failed to get analysis history: {result.get('error')}")
                return []

            return result.get("result", [])

        except Exception as e:
            logger.error(f"❌ Failed to get analysis history: {e}")
            return []

    def query_database(self, query: str, parameters: Optional[List[str]] = None) -> Dict[str, Any]:
        """🗄️ Execute SQL query on database (read-only for safety) using DatabaseManager."""
        if not self.db_manager:
            return {"error": "Database not available", "results": []}

        query_upper = query.upper()

        dangerous_keywords = ["DELETE", "DROP", "TRUNCATE", "UPDATE", "INSERT", "ALTER", "CREATE"]
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return {"error": f"Query contains dangerous keyword: {keyword}", "results": []}

        try:
            # Determine fetch type
            fetch_type = "all" if query_upper.strip().startswith("SELECT") else "none"

            # Execute via DatabaseManager
            result = self.db_manager.execute("primary", query, parameters, fetch=fetch_type)

            if not result.get("ok"):
                return {"error": result.get("error", "Unknown database error"), "results": []}

            if fetch_type == "all":
                return {
                    "success": True,
                    "row_count": len(result.get("result", [])),
                    "results": result.get("result", [])
                }
            else:
                rowcount = result.get("result", {}).get("rowcount", 0)
                return {
                    "success": True,
                    "message": "Query executed successfully",
                    "rowcount": rowcount
                }

        except Exception as e:
            logger.error(f"❌ Database query failed: {e}")
            return {"error": f"Query failed: {str(e)}", "results": []}

    def _health_check(self):
        """💚 Enhanced health checks on all components using DatabaseManager."""
        checks = {
            "models": len(self.models) > 0,
            "primary_model": self.service_config.primary_model in self.models,
            "scheduler": self.scheduler is not None and self.scheduler.running,
            "memory": self.memory is not None,
            "tool_registry": len(self.tool_registry.tools) > 0,
            "agent_executor": self.agent_executor is not None,
            "data_analyzer": self.data_analyzer is not None
        }

        if getattr(self, '_shutdown', False):
            return

        # Database health check using DatabaseManager
        if self.db_manager:
            try:
                health = self.db_manager.health_check("primary")
                checks["database"] = health.get("ok", False)
                if not checks["database"]:
                    logger.warning(f"⚠️ Database health check failed: {health.get('error')}")
            except Exception as e:
                checks["database"] = False
                logger.error(f"❌ Database health check failed: {e}")
        else:
            checks["database"] = False

        model_health = {}
        for name, adapter in self.models.items():
            try:
                model_health[name] = adapter.is_healthy()
            except Exception as e:
                model_health[name] = False
                logger.error(f"❌ Model {name} health check failed: {e}")

        checks["model_details"] = model_health

        all_healthy = all(checks.values()) and all(model_health.values())
        status = "HEALTHY" if all_healthy else "UNHEALTHY"

        health_info = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "model_health": model_health
        }

        logger.info(f"💚 Health check: {status}")

        if not all_healthy:
            logger.warning(f"⚠️ Unhealthy components: {json.dumps(health_info, indent=2)}")

        return health_info

    def _get_warm_executor(self) -> DaemonWorkerPool:
        """
        Lazy single-worker pool for warm-up work.

        Deliberately `DaemonWorkerPool`, not `ThreadPoolExecutor`: a
        warm-up that lands inside a C call (adapter.initialize on a
        slow model) must not become a thread that `_python_exit` tries
        to join at interpreter exit.
        """
        if self._warm_executor is None:
            self._warm_executor = DaemonWorkerPool(name="model_warm")
        return self._warm_executor

    def _warm_models(self):
        if getattr(self, '_shutdown', False):
            return

        for name, adapter in list(self.models.items()):
            if adapter.is_healthy():
                continue

            logger.info(f"🔥 Warming unhealthy model: {name}")
            try:
                # Try to initialize using the event loop if available
                loop = getattr(self, "_loop", None)
                if loop and loop.is_running():
                    # Submit to event loop from another thread
                    asyncio.run_coroutine_threadsafe(adapter.initialize(), loop)
                else:
                    # Run in thread using our dedicated executor
                    executor = self._get_warm_executor()
                    future = executor.submit(self._init_adapter_in_thread, adapter)
                    # Optional: wait a tiny bit to let it start, but we don't block
                    # We'll just let it run in background
            except Exception as e:
                logger.error(f"Failed to warm model {name}: {e}")

            # Small delay between attempts to avoid overwhelming system
            time.sleep(0.2)

    def _init_adapter_in_thread(self, adapter):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(adapter.initialize())
        except Exception as e:
            logger.warning(f"Adapter init in thread failed: {e}")
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def _cleanup_memory(self):
        """🧹 Clean up old memory entries."""
        if getattr(self, '_shutdown', False):
            return

        if self.memory:
            try:
                self.memory.cleanup_old(older_than_days=7)
                logger.info("✅ Memory cleanup completed")
            except Exception as e:
                logger.error(f"❌ Memory cleanup failed: {e}")

    def _collect_metrics(self):
        """📊 Collect and log system metrics.

        `tool_metrics` used to be `self.tool_registry.list_tools()`,
        which returns one entry per registered tool. With 149 tools and
        ~55 chars/description that was ~42 KB written to
        system_metric.log every 60 s. Per-tool stats belong in a
        one-shot startup banner, not in a periodic snapshot. Only the
        counts and per-model transport counters are kept here.
        """
        if getattr(self, '_shutdown', False):
            return

        tool_count = len(self.tool_registry.tools)
        failed_tools = sum(
            1 for name in self.tool_registry.tools
            if self.tool_registry._metrics.get(name, {}).get("failures", 0) > 0
        )

        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service_metrics": self.metrics.copy(),
            "tool_summary": {
                "registered": tool_count,
                "with_failures": failed_tools,
            },
            "model_metrics": {},
        }

        for name, adapter in self.models.items():
            try:
                metrics["model_metrics"][name] = adapter.get_metrics()
            except Exception as exc:
                metrics["model_metrics"][name] = {"error": str(exc)[:120]}
            if hasattr(adapter, "transport_metrics"):
                try:
                    metrics["model_metrics"][name]["transport"] = (
                        adapter.transport_metrics()
                    )
                except Exception:
                    pass

        system_logger = logging.getLogger("System_Metric")
        system_logger.info("📊 System metrics: %s",
                           json.dumps(metrics, indent=2, default=str))
        return metrics

    def _status_handler(self, signum, frame):
        """📊 Handle status request signal."""
        logger.info(f"📊 Received status request signal {signum}")
        status = self.get_system_status()
        logger.debug(json.dumps(status, indent=2))

    def shutdown(self):
        """
        Synchronous emergency shutdown for use during initialization failure.
        Stops the scheduler and closes database connections without awaiting.
        """
        logger.info("🛑 Performing emergency synchronous shutdown")
        if hasattr(self, 'scheduler') and self.scheduler and self.scheduler.running:
            try:
                self.scheduler.shutdown(wait=False)  # do not block
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")
        if hasattr(self, 'db_manager') and self.db_manager:
            try:
                self.db_manager.close_all()
            except Exception as e:
                logger.error(f"Error closing database: {e}")

        # Do not join the adapter-init daemon thread; just drop the
        # reference so __del__ and shutdown_async don't revisit it.
        _t = getattr(self, "_adapter_init_thread", None)
        if _t is not None and _t.is_alive():
            logger.warning(
                "🛑 Adapter-init thread %r still running at sync "
                "shutdown; leaving it to the OS.",
                _t.name,
            )
        self._adapter_init_thread = None
        self._shutdown = True

    async def shutdown_async(self):
        """
        🛑 PRODUCTION‑GRADE ASYNCIO SHUTDOWN (30s timeout).
        -----------------------------------------------------
        - Cancels and awaits all running async tasks.
        - Stops scheduler immediately.
        - Shuts down all model adapters with graceful timeouts.
        - Closes database connections.
        - Cleans up memory and forces garbage collection.
        - Logs every step with consistent formatting.
        - Blocks until all cleanup is done or timeout expires.

        Designed to be called from the main exception handler after
        KeyboardInterrupt, never from a synchronous signal handler.
        """

        try:
            if self._shutdown:
                return

            self._shutdown = True
            logger.info("🛑 Initiating AIModelService async shutdown...")
            shutdown_start = time.time()
            SHUTDOWN_TIMEOUT = 30
            loop = asyncio.get_running_loop()

            def _remaining():
                return max(0, SHUTDOWN_TIMEOUT - (time.time() - shutdown_start))

            # Stop memory monitor
            if self.memory_monitor:
                await self.memory_monitor.stop()
                self.memory_monitor = None

            # Stop backup retrier
            if self.backup_retrier:
                await self.backup_retrier.stop()

            # Stop warm executor. Bounded wait: warm-up may be inside a
            # C call we cannot interrupt, and the worker is a daemon
            # thread anyway, so waiting longer would only delay shutdown.
            if self._warm_executor:
                try:
                    exited = self._warm_executor.shutdown(
                        wait=True, timeout=2.0,
                    )
                    if not exited:
                        logger.warning(
                            "⚠️ Warm-up worker still running after 2s; "
                            "leaving it to the OS."
                        )
                except Exception as exc:
                    logger.error(f"Error stopping warm executor: {exc}")
                self._warm_executor = None

            # ----------------------------------------------------------------------
            # STEP 1: BLOCK SCHEDULER IMMEDIATELY – NO NEW JOBS
            # ----------------------------------------------------------------------
            if self.scheduler and self.scheduler.running:
                try:
                    logger.info("⏰ Pausing scheduler and removing all jobs...")
                    self.scheduler.pause()
                    self.scheduler.remove_all_jobs()
                    self.scheduler.shutdown(wait=False)  # ⚡ STOP IMMEDIATELY WITHOUT WAITING FOR JOBS

                    logger.info("✅ Scheduler paused and stopped")
                except Exception as e:
                    logger.error(f"Error stopping scheduler: {e}")
                self.scheduler = None

            # ----------------------------------------------------------------------
            # STEP 2: STOP AUTONOMOUS EXECUTION LOOPS (set flags, then cancel tasks)
            # ----------------------------------------------------------------------
            # Signal both components to exit their infinite loops
            if self.agent_executor:
                try:
                    self.agent_executor.stop_continuous_execution()
                    logger.info("🛑 AgentExecutor stop signal sent")
                except Exception as e:
                    logger.error(f"Error signalling AgentExecutor: {e}")

            if self.data_analyzer:
                try:
                    self.data_analyzer.stop_continuous_analysis()
                    logger.info("🛑 DataAnalyzer stop signal sent")
                except Exception as e:
                    logger.error(f"Error signalling DataAnalyzer: {e}")

            # ----------------------------------------------------------------------
            # STEP 3: CANCEL AND AWAIT ALL ASYNCIO TASKS (with timeout)
            # ----------------------------------------------------------------------
            remaining = _remaining()
            if remaining > 0:
                logger.info("🔄 Cancelling async tasks...")
                tasks_to_cancel = []

                if self._agent_task and not self._agent_task.done():
                    self._agent_task.cancel()
                    tasks_to_cancel.append(self._agent_task)

                if self._data_analyzer_task and not self._data_analyzer_task.done():
                    self._data_analyzer_task.cancel()
                    tasks_to_cancel.append(self._data_analyzer_task)

                if tasks_to_cancel:
                    # Wait for tasks to actually finish (or timeout)
                    _, pending = await asyncio.wait(tasks_to_cancel, timeout=min(remaining, 10.0))
                    if pending:
                        logger.warning(f"⚠️ {len(pending)} task(s) did not complete within timeout")
                    else:
                        logger.info("✅ All async tasks cancelled and finished")
                else:
                    logger.info("✅ No pending async tasks to cancel")

            self._agent_task = None
            self._data_analyzer_task = None

            # ----------------------------------------------------------------------
            # STEP 4: GRACEFULLY SHUT DOWN ALL MODEL ADAPTERS (in thread pool)
            # ----------------------------------------------------------------------
            remaining = _remaining()
            if remaining > 0 and self.models:
                logger.info("🤖 Shutting down all model adapters...")

                # Each adapter's shutdown is invoked on a *throwaway* daemon
                # thread rather than the loop's default executor. Rationale:
                #   * `loop.run_in_executor(None, ...)` uses ThreadPoolExecutor,
                #     whose workers are joined at interpreter exit by
                #     `concurrent.futures.thread._python_exit`. If any
                #     adapter's shutdown blocks (e.g. OpenAIAdapter joins its
                #     own pool with wait=True), that join cannot be interrupted
                #     and would hang the process at exit.
                #   * `_call_in_daemon_thread` uses a bare threading.Thread,
                #     which the atexit handler does not know about, so a stuck
                #     shutdown cannot block process exit.
                #   * The per-adapter timeout is enforced by `wait_for`, so a
                #     slow shutdown is bounded even in the pathological case.
                async def _shutdown_adapter(n, a):
                    try:
                        res = await self._call_in_daemon_thread(
                            lambda: a.shutdown(min(10.0, _remaining())),
                            timeout=min(10.0, _remaining()) + 2.0,
                            label=f"adapter_shutdown:{n}",
                        )
                        if not res:
                            logger.warning(
                                f"⚠️ Adapter {n} did not finish pending "
                                f"work within timeout"
                            )
                        else:
                            logger.info(f"✅ Adapter {n} shut down cleanly")
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"⚠️ Adapter {n} shutdown exceeded its "
                            f"timeout; leaving its worker to the OS."
                        )
                    except Exception as exc:
                        logger.error(f"❌ Error shutting down adapter {n}: {exc}")

                shutdown_tasks = [
                    _shutdown_adapter(name, adapter)
                    for name, adapter in self.models.items()
                ]
                if shutdown_tasks:
                    await asyncio.gather(*shutdown_tasks)

                self.models.clear()
                logger.info("✅ All model adapters removed")

            # ----------------------------------------------------------------------
            # STEP 4b: REPORT (NEVER JOIN) THE ADAPTER-INIT DAEMON THREAD
            # ----------------------------------------------------------------------
            # Adapter init is deliberately run on a bare daemon thread (see
            # `_run_adapter_init_async`). It cannot be interrupted from
            # Python if it is stuck inside a C library call, and — unlike a
            # ThreadPoolExecutor worker — it is not registered in
            # `_threads_queues`, so interpreter exit will not attempt to
            # join it. We only log here to make a stuck load visible; the
            # process can exit regardless.
            _t = getattr(self, "_adapter_init_thread", None)
            if _t is not None and _t.is_alive():
                logger.warning(
                    "⚠️ Adapter-init thread %r is still alive at "
                    "shutdown. It is a daemon thread stuck inside a C "
                    "call that cannot be interrupted from Python. "
                    "Shutdown will not join it; the OS will reap it "
                    "at process exit.",
                    _t.name,
                )
            self._adapter_init_thread = None

            # ----------------------------------------------------------------------
            # STEP 5: CLOSE DATABASE CONNECTIONS
            # ----------------------------------------------------------------------
            remaining = _remaining()
            if remaining > 1 and self.db_manager:
                try:
                    logger.info("🗄️ Closing database connections...")
                    # Same reasoning as the adapter shutdown above: route
                    # through a throwaway daemon thread so a stalled socket
                    # in `close_connection` cannot hang interpreter exit via
                    # the default executor's atexit join.
                    await self._call_in_daemon_thread(
                        self.db_manager.close_all,
                        timeout=5.0,
                        label="db_close",
                    )
                    logger.info("✅ Database connections closed")
                except asyncio.TimeoutError:
                    logger.warning(
                        "⚠️ Database close exceeded 5s; leaving its "
                        "worker to the OS."
                    )
                except Exception as e:
                    logger.error(f"Error closing database: {e}")
                self.db_manager = None

            # ----------------------------------------------------------------------
            # STEP 6: FORCE MEMORY CLEANUP
            # ----------------------------------------------------------------------
            try:
                logger.info("🧹 Running garbage collection...")
                import gc
                gc.collect()
                logger.info("✅ Garbage collection completed")
            except Exception as e:
                logger.error(f"Error during garbage collection: {e}")

            # ----------------------------------------------------------------------
            # FINAL STATE
            # ----------------------------------------------------------------------
            self.running = False
            self._shutdown = True

            shutdown_duration = time.time() - shutdown_start
            logger.info(f"✅ AIModelService async shutdown complete ({shutdown_duration:.1f}s)")
            if shutdown_duration > SHUTDOWN_TIMEOUT * 0.8:
                logger.warning(f"⚠️ Shutdown took {shutdown_duration:.1f}s (>80% of timeout)")

        except Exception as e:
            logger.error(f"💥 Unhandled exception in shutdown_async: {e}")
            raise


# ---------------------------------------------------------------------
# Main entry point with enhanced features
# ---------------------------------------------------------------------
async def main(argv: list[str] | None = None) -> int:
    """🚀 Enhanced main entry point for the AI model service."""
    import argparse

    parser = argparse.ArgumentParser(
        description="🚀 Enhanced AI Model Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config.yaml --log-level INFO
  %(prog)s --config /path/to/config.yaml --log-level DEBUG --no-agent
  %(prog)s --config config.yaml --run-agent-once
        """
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="📄 Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="📊 Set logging level (default: INFO)"
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="🤖 Disable auto agent execution"
    )
    parser.add_argument(
        "--run-agent-once",
        action="store_true",
        help="🚀 Run agent once with instruction and exit"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="💚 Run health check and exit"
    )

    args = parser.parse_args(argv)
    configure(args.log_level)

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")

    logging.getLogger("Main").setLevel(numeric_level)

    service = None
    shutdown_event = asyncio.Event()
    exit_code = 0

    # Signal handling. On POSIX, use the loop-aware API — the loop's signal
    # callback runs cleanly interleaved with bytecode, no cross-thread races.
    # Windows does not implement loop.add_signal_handler, so fall back to
    # signal.signal + loop.call_soon_threadsafe.
    loop = asyncio.get_running_loop()

    if sys.platform != "win32":
        loop.add_signal_handler(
            signal.SIGINT,
            lambda: (logger.info("👋 SIGINT received, shutting down..."),
                     shutdown_event.set()),
        )
        loop.add_signal_handler(
            signal.SIGTERM,
            lambda: (logger.info("👋 SIGTERM received, shutting down..."),
                     shutdown_event.set()),
        )
    else:
        def _win_handler(sig, _frame):
            logger.info(f"👋 Windows signal {sig} received, shutting down...")
            loop.call_soon_threadsafe(shutdown_event.set)

        signal.signal(signal.SIGINT, _win_handler)
        signal.signal(signal.SIGTERM, _win_handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _win_handler)

    try:
        logger.info(f"🚀 Starting AI Model Service with config: {args.config}")

        service = AIModelService(args.config)

        if args.no_agent:
            logger.info("🤖 Disabling agent auto-execution as requested")
            service.service_config.auto_run_agent = False

        # Heavy initialization runs here, inside the event loop, so retry
        # loops cooperate with the loop instead of blocking it.
        await service.initialize_async()

        if args.health_check:
            logger.info("💚 Running health check...")
            health = service._health_check() or {}
            logger.debug(json.dumps(health, indent=2))
            exit_code = 0 if health.get("status") == "HEALTHY" else 1
            return exit_code

        if args.run_agent_once:
            logger.info("🚀 Running agent once with instruction from config...")
            result = await service.run_agent_from_instruction()
            logger.debug(json.dumps(result, indent=2))
            exit_code = 0 if result.get("success") else 1
            return exit_code

        logger.info(f"✅ Service initialized successfully")
        logger.info(f"🤖 Primary model: {service.service_config.primary_model}")
        logger.info(f"🛠️ Available tools: {len(service.tool_registry.tools)}")
        if service.service_config.agent_instruction:
            logger.info(f"📝 Agent instruction configured: {service.service_config.agent_instruction[:100]}...")
        else:
            logger.warning("⚠️ No agent instruction configured in config.yaml")

        logger.info(f"⏰ Agent execution interval: {service.service_config.agent_execution_interval}s")
        logger.info(f"⏰ Data analysis interval: {service.service_config.data_analysis_interval}s")
        logger.info(f"⚡ Concurrent execution enabled: {service.service_config.enable_concurrent_execution}")

        # Keep service running
        logger.info("🏃 Service is running. Press Ctrl+C to stop.")

        # Wait for shutdown signal instead of infinite loop
        await shutdown_event.wait()

        logging.info("*" * 50)
        logger.info("👋 *** SHUTDOWN EVENT TRIGGERED - STARTING CLEAN SHUTDOWN ***")

    except KeyboardInterrupt:
        # This WILL catch Ctrl+C if raised before shutdown_event
        logger.info("👋 *** KEYBOARD INTERRUPT CAUGHT IN EXCEPT BLOCK ***")
    except ConfigError as e:
        logger.critical(
            f"💥 Configuration error (exit 78 / EX_CONFIG): {e}",
            exc_info = True,
        )
        exit_code = 78
    except Exception as e:
        logger.error(f"💥 Unhandled exception in main(): {e}", exc_info=True)
        exit_code = 1
    finally:
        if service and not service._shutdown:
            logger.info("🔴 Calling service.shutdown_async()...")
            try:
                await service.shutdown_async()
            except Exception as e:
                logger.error(f"❌ Error during shutdown: {e}", exc_info=True)
        if exit_code == 0:
            logger.info("✅ Main function exiting cleanly")
        else:
            logger.error("❌ Main function exiting with code %d", exit_code)

    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()) or 0)
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"💥 Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
