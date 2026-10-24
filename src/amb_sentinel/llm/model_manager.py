# model_manager.py
"""
Adaptive Context & Strategy Management for AI Agents

Features (v1.0.0):
- Hardware-aware execution strategy: full-plan vs iterative next-action
- Task complexity analysis (goal count, tool needs, token estimation)
- Dynamic timeout estimation based on CPU speed, prompt length, model size
- Intelligent prompt compression preserving core instructions
- Adaptive context window negotiation with fine-grained control
- Real-time memory monitoring and graceful degradation

"""

import logging
import os

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
import time
import threading
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from collections import deque
import gc
import asyncio

logger = logging.getLogger("ModelManager")


# ----------------------------------------------------------------------
# Enums & Data Classes
# ----------------------------------------------------------------------
class SystemCapability(Enum):
    ULTRA_LOW = "ultra_low"  # < 2GB
    VERY_LOW = "very_low"  # 2-4GB
    LOW = "low"  # 4-8GB
    MEDIUM = "medium"  # 8-16GB
    HIGH = "high"  # 16-32GB
    VERY_HIGH = "very_high"  # 32-64GB
    ULTRA_HIGH = "ultra_high"  # 64GB+


class ModelClass(Enum):
    TINY = "tiny"  # < 1B params
    SMALL = "small"  # 1-3B
    MEDIUM = "medium"  # 3-7B
    LARGE = "large"  # 7-13B
    VERY_LARGE = "very_large"  # 13-65B
    HUGE = "huge"  # 65B+


class ExecutionStrategy(Enum):
    FULL_PLAN = "full_plan"  # Generate multi-step plan at once
    ITERATIVE_NEXT_ACTION = "next_action"  # One step at a time
    HYBRID = "hybrid"  # Plan a few steps then iterate


@dataclass
class SystemMetrics:
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    cpu_cores: int
    cpu_threads: int
    cpu_freq_ghz: float
    has_cuda: bool
    has_mps: bool
    capability_level: SystemCapability
    memory_usage_percent: float
    # New: CPU performance estimation (tokens/sec approximate)
    estimated_cpu_tokens_per_sec: float = 10.0  # conservative default

    def __repr__(self):
        return (
            f"SystemMetrics(RAM={self.total_memory_gb:.1f}GB, "
            f"Available={self.available_memory_gb:.1f}GB, "
            f"Cores={self.cpu_cores}, Level={self.capability_level.value})"
        )


@dataclass
class ContextBudget:
    total_context: int
    system_reserve: int = 0
    prompt_allocation: int = 0
    generation_allocation: int = 0
    safety_buffer: int = 0


@dataclass
class TokenMetrics:
    prompt_tokens: int
    available_for_generation: int
    recommended_max_gen: int
    total_budget: int
    remaining: int
    efficiency_ratio: float
    is_feasible: bool
    warnings: List[str] = field(default_factory=list)
    # New: recommended timeout (seconds) based on hardware
    recommended_timeout: float = 120.0
    max_prompt_tokens: int = 0

    def __repr__(self):
        return (
            f"TokenMetrics(prompt={self.prompt_tokens}, "
            f"available_gen={self.available_for_generation}, "
            f"recommended={self.recommended_max_gen}, "
            f"timeout={self.recommended_timeout:.1f}s, "
            f"feasible={self.is_feasible})"
        )


@dataclass
class TaskComplexity:
    estimated_goals: int
    estimated_prompt_tokens: int
    required_tool_categories: List[str]
    instruction_text_sample: str
    complexity_score: float  # 0-1, higher = more complex
    recommended_strategy: ExecutionStrategy


# ----------------------------------------------------------------------
# TokenCounter Class
# ----------------------------------------------------------------------
class TokenCounter:
    def __init__(self, model_name: str = ""):
        self.model_name = model_name.lower()
        self._tiktoken_enc = None
        self._transformers_tokenizer = None
        self._init_tokenizer()
        self._cache = {}
        self._cache_order = []
        self._max_cache = 1000

    def _init_tokenizer(self):
        """
        Initialize the most accurate tokenizer available:
        1. Use tiktoken with model‑specific mapping.
        2. If tiktoken not available or model unknown, try transformers (if installed).
        3. Fallback to None (word‑based estimation).
        """
        # ----- TIKTOKEN (fast, lightweight) -----
        try:
            import tiktoken

            # Extended mapping: model family -> encoding
            encoding_map = {
                # OpenAI models
                "gpt-4": "cl100k_base",
                "gpt-3.5": "cl100k_base",
                "text-davinci": "p50k_base",
                # Local models that use similar tokenizers
                "llama": "cl100k_base",  # best approximate
                "mistral": "cl100k_base",
                "mixtral": "cl100k_base",
                "qwen": "cl100k_base",
                "gemma": "cl100k_base",
                "phi": "cl100k_base",
                "deepseek": "cl100k_base",
                "codellama": "cl100k_base",
                "wizardcoder": "cl100k_base",
                # Older models
                "code-davinci": "p50k_base",
                "davinci": "r50k_base",
                "curie": "r50k_base",
                "babbage": "r50k_base",
                "ada": "r50k_base",
            }

            # Find the first matching key
            for key, encoding_name in encoding_map.items():
                if key in self.model_name:
                    self._tiktoken_enc = tiktoken.get_encoding(encoding_name)
                    logger.info(f"✅ Using tiktoken encoding '{encoding_name}' for model '{self.model_name}'")
                    break
            else:
                # No specific match – use a safe default
                self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
                logger.info(f"ℹ️ No exact tiktoken match for '{self.model_name}'; using cl100k_base as fallback.")

        except ImportError:
            logger.warning("⚠️ tiktoken not installed – will try transformers next.")
            self._tiktoken_enc = None

        # ----- TRANSFORMERS (more accurate for many local models) -----
        if self._tiktoken_enc is None:
            try:
                from transformers import AutoTokenizer

                # Attempt to load the tokenizer for this model (may download)
                self._transformers_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_fast=True
                )
                logger.info(f"✅ Loaded transformers tokenizer for '{self.model_name}'")
            except Exception as e:
                logger.warning(f"⚠️ Could not load transformers tokenizer: {e}")
                self._transformers_tokenizer = None

        # If both failed, we'll rely on word‑based estimation.

    def count(self, text: str) -> int:
        """Return the number of tokens in the given text."""
        # Check cache first
        if text in self._cache:
            self._cache_order.remove(text)
            self._cache_order.append(text)
            return self._cache[text]

        # 1. Use tiktoken if available
        if self._tiktoken_enc is not None:
            try:
                tokens = len(self._tiktoken_enc.encode(text))
            except Exception:
                tokens = self._fallback_count(text)

        # 2. Use transformers tokenizer if available
        elif self._transformers_tokenizer is not None:
            try:
                tokens = len(self._transformers_tokenizer.encode(text))
            except Exception:
                tokens = self._fallback_count(text)

        # 3. Fallback to word‑based estimate
        else:
            tokens = self._fallback_count(text)

        # Update cache
        self._cache[text] = tokens
        self._cache_order.append(text)
        if len(self._cache) > self._max_cache:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return tokens

    def _fallback_count(self, text: str) -> int:
        """
        Word‑based fallback with an adaptive factor.
        Uses 1.3 for most models, but can be tuned.
        """
        # For known models with different character‑to‑token ratios,
        # you could adjust the factor. Here we keep it simple.
        factor = 1.3
        if 'llama' in self.model_name:
            factor = 1.2  # slightly more efficient
        return max(1, int(len(text.split()) * factor))

    def clear_cache(self):
        """Clear the internal token cache."""
        self._cache.clear()
        self._cache_order.clear()

    def shrink_cache(self, target_size: int = 100):
        """Reduce cache to at most target_size entries (keeps most recent)."""
        if len(self._cache) > target_size:
            # Keep only the most recent target_size entries
            keep = self._cache_order[-target_size:]
            new_cache = {k: self._cache[k] for k in keep}
            self._cache = new_cache
            self._cache_order = keep


# ----------------------------------------------------------------------
# Main Adaptive Model Manager (Singleton)
# ----------------------------------------------------------------------
class AdaptiveModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.system_metrics = self._detect_system()
        self.model_class: Optional[ModelClass] = None
        self.context_window: int = 2048
        self.model_name: str = "unknown"
        self._memory_check_interval = 5.0
        self._last_memory_check = 0.0
        self.token_counter: Optional[TokenCounter] = None
        self.max_generation_tokens: int = 1024  # can be overridden by config
        self._context_lock = threading.RLock()

        # Cache for estimated CPU speed (run quick benchmark later if needed)
        self._cpu_speed_benchmarked = False
        self._benchmark_lock = threading.Lock()

        self._memory_samples = deque(maxlen=10)  # last 10 memory readings
        self._last_reduction_time = 0.0  # timestamp of last reduction
        self._recovery_cooldown = 60.0  # seconds before attempting recovery
        self._monitoring_task = None  # background asyncio task
        self._monitor_interval = 5.0  # check every 5 seconds
        self._min_context = 1024
        self._initial_context = self.context_window  # store initial negotiated value
        self._initial_max_gen = self.max_generation_tokens

        self._actual_model_n_ctx: int = 0
        self._config_ceiling: int = 0
        self._min_viable_budget: int = 4096

        self._initialized = True
        logger.info(f"✅ ContextManager initialized: {self.system_metrics}")

    async def adapt_to_memory_pressure(self, force_recovery: bool = False):
        """
        Adapt context window based on current memory pressure.
        Uses moving average, gradual reduction/recovery, and triggers GC.
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            self._memory_samples.append(mem.percent)
            avg_memory = (
                sum(self._memory_samples) / len(self._memory_samples)
                if self._memory_samples else mem.percent
            )

            # Estimate model memory usage (RSS)
            model_memory_ratio = self._estimate_model_memory_usage()
            effective_pressure = avg_memory * 0.7 + (model_memory_ratio * 100) * 0.3

            with self._context_lock:
                old_ctx = self.context_window
                old_max_gen = self.max_generation_tokens

                physical = self._actual_model_n_ctx if self._actual_model_n_ctx > 0 else 0
                floor = min(self._min_viable_budget, physical) if physical > 0 else self._min_viable_budget

                # Only allow one reduction every 30 s.
                # Previously there was NO cooldown on reduction (only on recovery),
                # so adapt_to_memory_pressure() called every 5 s would reduce
                # max_gen 1024→921→828→745→670→603→542→487→438→394→354→318→286→256
                # in under 2 minutes.  At 256 tokens the LLM cannot finish a JSON
                # response, making every generation fail silently.
                _REDUCTION_COOLDOWN_S = 30.0
                _time_since_last = time.time() - self._last_reduction_time

                if effective_pressure > 85 and _time_since_last >= _REDUCTION_COOLDOWN_S:
                    reduction_factor = 0.9  # reduce by 10%
                    new_ctx = max(floor, int(old_ctx * reduction_factor))
                    # Raised min floor from 256 → 512.
                    # 256 tokens is not enough for a complete JSON tool-call response.
                    # 512 tokens is the practical minimum for Mistral [INST] JSON output.
                    new_max_gen = max(512, int(old_max_gen * reduction_factor))

                    if new_ctx != old_ctx or new_max_gen != old_max_gen:
                        logger.warning(
                            f"📉 High memory ({effective_pressure:.1f}%) – reducing soft budget "
                            f"{old_ctx} → {new_ctx} (floor={floor}), max_gen {old_max_gen} → {new_max_gen}"
                        )
                        self.context_window = new_ctx
                        self.max_generation_tokens = new_max_gen
                        self._last_reduction_time = time.time()
                        await self._trigger_garbage_collection()
                    else:
                        # Already at floor — log once so logs aren't flooded
                        if not getattr(self, "_floor_logged", False):
                            logger.warning(
                                f"⚠️ Memory pressure {effective_pressure:.1f}% — "
                                f"already at min floor (ctx={old_ctx}, max_gen={old_max_gen}). "
                                f"Cannot reduce further."
                            )
                            self._floor_logged = True
                elif effective_pressure <= 85:
                    self._floor_logged = False  # reset flag on recovery

                elif (effective_pressure < 75 and
                      (time.time() - self._last_reduction_time) > self._recovery_cooldown and
                      (self.context_window < self._initial_context or
                       self.max_generation_tokens < self._initial_max_gen)):

                    recovery_factor = 1.05  # increase by 5%
                    new_ctx = min(self._initial_context, int(old_ctx * recovery_factor))
                    new_max_gen = min(self._initial_max_gen, int(old_max_gen * recovery_factor))

                    if new_ctx != old_ctx or new_max_gen != old_max_gen:
                        logger.info(
                            f"📈 Memory recovered ({effective_pressure:.1f}%) – increasing context "
                            f"{old_ctx} → {new_ctx}, max_gen {old_max_gen} → {new_max_gen}"
                        )
                        self.context_window = new_ctx
                        self.max_generation_tokens = new_max_gen
        except Exception as e:
            logger.error(f"Error in memory adaptation: {e}")

    def _estimate_model_memory_usage(self) -> float:
        try:
            import psutil
            process = psutil.Process()
            rss = process.memory_info().rss / (1024 ** 3)
            total = psutil.virtual_memory().total / (1024 ** 3)
            system_ratio = rss / total if total > 0 else 0.0

            gpu_ratio = 0.0
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                if max_alloc > 0:
                    gpu_ratio = allocated / max_alloc
            # Combine (you may weight GPU more heavily if used)
            return max(system_ratio, gpu_ratio)
        except Exception:
            return 0.0

    async def _trigger_garbage_collection(self):
        """Run garbage collection in a thread to avoid blocking."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, gc.collect)
        logger.debug("🧹 Garbage collection triggered")

    def _clear_caches(self):
        """Clear all internal caches (token counts, compressed prompts)."""
        logger.info("🧹 Clearing all internal caches")
        if self.token_counter:
            self.token_counter.clear_cache()
        # If we add a compressed prompt cache later, clear it here
        import gc
        gc.collect()

    def _compress_caches(self):
        """Reduce cache sizes (e.g., trim to half)."""
        logger.info("🗜️ Compressing caches")
        if self.token_counter:
            # Reduce token cache to 500 entries (half of default 1000)
            self.token_counter.shrink_cache(target_size=500)
        # Similarly for other caches

    def reduce_context_window(self):
        """
        Reduce context window by 50% (but not below 1024) and
        reduce max generation tokens by 256 (but not below 512).
        This method is thread‑safe.
        """
        with self._context_lock:
            physical = self._actual_model_n_ctx if self._actual_model_n_ctx > 0 else 0
            floor = min(self._min_viable_budget, physical) if physical > 0 else self._min_viable_budget
            old_ctx = self.context_window
            new_ctx = max(floor, old_ctx // 2)
            self.context_window = new_ctx

            old_max = self.max_generation_tokens
            # [FIX-MEM-FLOOR] Floor raised to 512 (was 256) — 256 tokens is not
            # enough for a complete JSON response from Mistral-7B.
            new_max = max(512, old_max - 256)
            self.max_generation_tokens = new_max

        logger.info(f"📉 Context window reduced: {old_ctx} → {new_ctx} (floor={floor})")
        logger.info(f"📉 Max generation tokens reduced: {old_max} → {new_max}")

    # ====================================================================
    # System Detection (enhanced)
    # ====================================================================
    def _detect_system(self) -> SystemMetrics:
        try:
            # Memory detection
            try:
                mem = psutil.virtual_memory()
                total_mem_gb = mem.total / (1024 ** 3)
                available_mem_gb = mem.available / (1024 ** 3)
                used_mem_gb = mem.used / (1024 ** 3)
                mem_percent = mem.percent
            except Exception:
                total_mem_gb = 8.0
                available_mem_gb = 4.0
                used_mem_gb = 4.0
                mem_percent = 50.0

            # CPU detection
            try:
                cpu_count = os.cpu_count() or 4
                cpu_freq = psutil.cpu_freq()
                cpu_freq_ghz = (cpu_freq.current / 1000.0) if cpu_freq else 2.0
            except Exception:
                cpu_count = 4
                cpu_freq_ghz = 2.0

            # GPU detection
            has_cuda = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

            # Estimate CPU tokens/sec (crude but reasonable)
            # Base on CPU cores, frequency, and a factor
            # Could be improved with a quick benchmark later
            est_tokens_per_sec = cpu_count * cpu_freq_ghz * 1.5  # heuristic
            est_tokens_per_sec = max(5.0, min(50.0, est_tokens_per_sec))  # clamp

            capability = self._classify_capability(total_mem_gb)

            return SystemMetrics(
                total_memory_gb=total_mem_gb,
                available_memory_gb=available_mem_gb,
                used_memory_gb=used_mem_gb,
                cpu_cores=max(1, cpu_count // 2),
                cpu_threads=cpu_count,
                cpu_freq_ghz=cpu_freq_ghz,
                has_cuda=has_cuda,
                has_mps=has_mps,
                capability_level=capability,
                memory_usage_percent=mem_percent,
                estimated_cpu_tokens_per_sec=est_tokens_per_sec,
            )

        except Exception as e:
            logger.error(f"System detection failed: {e}")
            return SystemMetrics(
                total_memory_gb=4.0,
                available_memory_gb=2.0,
                used_memory_gb=2.0,
                cpu_cores=2,
                cpu_threads=4,
                cpu_freq_ghz=2.0,
                has_cuda=False,
                has_mps=False,
                capability_level=SystemCapability.VERY_LOW,
                memory_usage_percent=50.0,
                estimated_cpu_tokens_per_sec=10.0,
            )

    @staticmethod
    def _classify_capability(total_mem_gb: float) -> SystemCapability:
        if total_mem_gb < 2:
            return SystemCapability.ULTRA_LOW
        elif total_mem_gb < 4:
            return SystemCapability.VERY_LOW
        elif total_mem_gb < 8:
            return SystemCapability.LOW
        elif total_mem_gb < 16:
            return SystemCapability.MEDIUM
        elif total_mem_gb < 32:
            return SystemCapability.HIGH
        elif total_mem_gb < 64:
            return SystemCapability.VERY_HIGH
        else:
            return SystemCapability.ULTRA_HIGH

    # ====================================================================
    # Model Classification
    # ====================================================================
    def detect_model_class(self, model_name: str) -> ModelClass:
        name_lower = model_name.lower()
        if any(x in name_lower for x in ['tiny', '125m', '160m']):
            return ModelClass.TINY
        elif any(x in name_lower for x in ['small', '1b', '2b', '3b']):
            return ModelClass.SMALL
        elif any(x in name_lower for x in ['7b', 'qwen2.5-coder', 'mistral', 'qwen']):
            return ModelClass.MEDIUM
        elif any(x in name_lower for x in ['13b', 'llama-2-13b']):
            return ModelClass.LARGE
        elif any(x in name_lower for x in ['30b', '34b', '40b', '65b']):
            return ModelClass.VERY_LARGE
        elif any(x in name_lower for x in ['70b', '175b', '540b']):
            return ModelClass.HUGE
        else:
            return ModelClass.MEDIUM

    # ====================================================================
    # Task Complexity Analysis
    # ====================================================================
    def analyze_task_complexity(self, instruction: str) -> TaskComplexity:
        """
        Analyze instruction to estimate number of goals, required tools, etc.
        Uses heuristics and pattern matching.
        """
        # Count potential goals by looking for numbered lists, bullet points,
        # keywords like "Collect", "Analyze", etc.
        goal_patterns = [
            r'^\d+\.\s+(.+?)(?:\n|$)',  # 1. Goal
            r'^[-•]\s+(.+?)(?:\n|$)',  # - Goal
            r'(?:GOAL|TASK|STEP)\s*[:#]?\s*(.+?)(?:\n|$)',
            r'(?:Collect|Analyze|Process|Extract|Store|Save|Download|Fetch|Find|Search|Generate|Report|Create|Update)\s+',
        ]
        goals_found = set()
        for pat in goal_patterns:
            matches = re.finditer(pat, instruction, re.MULTILINE | re.IGNORECASE)
            for m in matches:
                if m.lastindex:
                    text = m.group(1).strip()
                else:
                    text = m.group(0).strip()
                if len(text) > 10:
                    goals_found.add(text[:50])
        estimated_goals = max(1, len(goals_found))

        # Estimate token count (rough)
        word_count = len(instruction.split())
        estimated_tokens = int(word_count * 1.3)

        # Comprehensive tool categories derived from registrar.py
        tool_keywords = {
            # Link / Search tools
            "link_search": ["search", "find", "discover", "locate", "dorks", "expand", "extract links"],
            "file_discovery": ["find file", "filetype", "pdf", "doc", "xls", "hidden resources"],

            # HTTP / Network
            "http": ["http", "fetch", "request", "url", "batch", "retry", "rate limit", "circuit breaker"],

            # Download
            "download": ["download", "save", "file", "multiple", "stats", "validate", "cleanup"],

            # Scraping
            "scrape": ["scrape", "extract content", "parse", "feed", "rss", "atom", "browser", "cache"],

            # Extraction & Text Processing
            "extraction": ["extract text", "pdf", "docx", "spreadsheet", "chunk", "ocr", "file processing"],

            # Video
            "video": ["video", "info", "thumbnail", "platform", "yt-dlp", "ffmpeg"],

            # Transcription
            "transcription": ["transcribe", "audio", "speech", "whisper", "word timestamps", "batch"],

            # Pipeline / Integration
            "pipeline": ["pipeline", "transcribe url", "index", "vectorstore", "end-to-end"],

            # Database
            "database": ["database", "query", "insert", "update", "connection", "sql", "vector"],

            # Utility / Security
            "utility": ["hash", "filename", "validate url", "sanitize", "cache dir", "log dir"],

            # Analysis (already present)
            "analysis": ["analyze", "threat", "risk", "confidence"],

            # Report / Summary
            "report": ["report", "summary", "generate"],

            # Factories (create_* tools)
            "factory": ["create", "config", "manager", "engine", "backend"],

            # Error / Exception
            "error": ["error", "exception", "circuit open", "rate limit exceeded"],
        }

        # Now identify required categories by checking instruction against each keyword list
        required = []
        instruction_lower = instruction.lower()
        for category, keywords in tool_keywords.items():
            if any(k in instruction_lower for keyword in keywords for k in keyword.split()):
                # careful: we need to match whole words or substrings appropriately
                # simplified: if any keyword substring appears
                if any(k in instruction_lower for keyword in keywords for k in keyword.split()):
                    required.append(category)
        # Deduplicate
        required = list(set(required))

        # Compute complexity score: combination of goal count, token count, required categories
        goal_score = min(1.0, estimated_goals / 20)  # 20 goals = max
        token_score = min(1.0, estimated_tokens / 5000)  # 5000 tokens = max
        category_score = min(1.0, len(required) / 5)  # 5 categories = max
        complexity = (goal_score * 0.4 + token_score * 0.3 + category_score * 0.3)

        # Decide strategy
        if self.system_metrics.has_cuda:
            # GPU: can handle full plan up to moderate complexity
            if complexity < 0.7:
                strategy = ExecutionStrategy.FULL_PLAN
            else:
                strategy = ExecutionStrategy.HYBRID
        else:
            # CPU: be conservative
            if complexity < 0.3:
                strategy = ExecutionStrategy.FULL_PLAN
            elif complexity < 0.6:
                strategy = ExecutionStrategy.HYBRID
            else:
                strategy = ExecutionStrategy.ITERATIVE_NEXT_ACTION

        return TaskComplexity(
            estimated_goals=estimated_goals,
            estimated_prompt_tokens=estimated_tokens,
            required_tool_categories=required,
            instruction_text_sample=instruction[:200],
            complexity_score=complexity,
            recommended_strategy=strategy,
        )

    def recommend_execution_strategy(
            self,
            instruction: str,
            model_name: str,
            remaining_goals: int = 0,
            time_remaining: float = float('inf'),
            progress: float = 0.0,
            tokens_remaining: int = float('inf')
    ) -> ExecutionStrategy:
        """
        Recommend execution strategy based on static system capabilities and dynamic progress.

        Args:
            instruction: Original task instruction (for complexity analysis)
            model_name: Name of the model being used
            remaining_goals: Number of goals still pending/in progress
            time_remaining: Seconds left before timeout
            progress: Current completion percentage (0-100)
            tokens_remaining: Tokens left in budget
        """
        # Base strategy from original logic (system capability + instruction complexity)
        base_strategy = self._base_strategy(instruction, model_name)

        # Dynamic adjustments
        if remaining_goals <= 1:
            # Only one goal left – full plan may be safe even if system is weak
            return ExecutionStrategy.FULL_PLAN

        if time_remaining < 60 and remaining_goals > 3:
            # Very little time, many goals – switch to iterative to at least start something
            return ExecutionStrategy.ITERATIVE_NEXT_ACTION

        if tokens_remaining < 500:
            # Low token budget – iterative is safer
            return ExecutionStrategy.ITERATIVE_NEXT_ACTION

        if progress > 80 and remaining_goals <= 2:
            # Near completion – full plan is fine
            return ExecutionStrategy.FULL_PLAN

        # Otherwise return the base strategy
        return base_strategy

    def _base_strategy(self, instruction: str, model_name: str) -> ExecutionStrategy:
        """Original strategy logic based on system capability and instruction analysis."""
        if self.system_metrics.capability_level in [SystemCapability.ULTRA_LOW, SystemCapability.VERY_LOW]:
            return ExecutionStrategy.ITERATIVE_NEXT_ACTION

        complexity = self.analyze_task_complexity(instruction)
        tools_needed = complexity.required_tool_categories

        # Tools that are inherently slow (I/O bound)
        high_latency_tools = {'http', 'download', 'scrape', 'video', 'transcription', 'database'}
        uses_high_latency = any(tool in high_latency_tools for tool in tools_needed)

        if uses_high_latency and len(tools_needed) > 2:
            return ExecutionStrategy.ITERATIVE_NEXT_ACTION

        if complexity.complexity_score < 0.3:
            return ExecutionStrategy.FULL_PLAN
        elif complexity.complexity_score < 0.6:
            return ExecutionStrategy.HYBRID
        else:
            return ExecutionStrategy.ITERATIVE_NEXT_ACTION

    async def attempt_context_recovery(self):
        """Async wrapper for sync recovery method."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._attempt_context_recovery_sync)

    def decompose_instruction(self, instruction: str, max_tokens_per_step: int = 1500) -> List[str]:
        """
        Split a long instruction into smaller sub-tasks for iterative execution.
        Simple splitting by numbered steps or paragraphs.
        """
        # Try to split by numbered lines
        lines = instruction.split('\n')
        sub_tasks = []
        current = []
        current_len = 0
        for line in lines:
            # If line looks like a new step (numbered or bullet), start new chunk
            if re.match(r'^\s*\d+\.\s+', line) or re.match(r'^\s*[-•]\s+', line):
                if current:
                    sub_tasks.append('\n'.join(current))
                    current = []
                    current_len = 0
            current.append(line)
            current_len += len(line)
            if current_len > max_tokens_per_step * 4:  # approx chars
                sub_tasks.append('\n'.join(current))
                current = []
                current_len = 0
        if current:
            sub_tasks.append('\n'.join(current))

        if not sub_tasks:
            sub_tasks = [instruction]
        return sub_tasks

    # ====================================================================
    # Context Window Negotiation (unchanged core, but we can add adaptive limits)
    # ====================================================================
    def negotiate_context_window(self, model_name: str, config_requested: Optional[int] = None) -> int:
        self.model_name = model_name
        self.model_class = self.detect_model_class(model_name)
        self.token_counter = TokenCounter(model_name)

        safe_ctx = self._calculate_safe_context(self.model_name)
        model_max_ctx = self._get_model_max_context()
        resource_limit = self._calculate_resource_limit()

        # All other computed values (model_max_ctx, resource_limit) can exceed
        # what the admin configured, but the admin's value wins.
        ceiling = config_requested if (config_requested and config_requested >= 1024) else None
        if ceiling:
            self._config_ceiling = ceiling
            logger.info(f"📌 Config ceiling enforced: {ceiling} tokens (will not exceed this)")

        candidates = [safe_ctx, model_max_ctx, resource_limit]
        if ceiling:
            # Add ceiling itself as a candidate and filter everything above it
            candidates.append(ceiling)
            candidates = [c for c in candidates if c and c >= 1024 and c <= ceiling]
        else:
            candidates = [c for c in candidates if c and c >= 1024]

        valid_candidates = sorted(set(candidates), reverse=True)

        final_ctx = valid_candidates[0] if valid_candidates else 2048
        self.context_window = final_ctx
        self._initial_context = final_ctx
        logger.info(
            f"📊 Context negotiation → FINAL: {final_ctx} tokens "
            f"(ceiling={ceiling}, safe={safe_ctx}, model_max={model_max_ctx})"
        )
        return final_ctx

    def set_actual_n_ctx(self, n_ctx: int) -> None:
        """
        Called by the adapter immediately after a SUCCESSFUL model load.

        Locks in the physical context limit that the model was loaded with.
        This value is NEVER modified by memory pressure adaptation —
        it represents what the model can actually handle, not what we prefer.

        Args:
            n_ctx: The actual n_ctx the model was loaded with (from _llm.n_ctx)
        """
        with self._context_lock:
            self._actual_model_n_ctx = n_ctx
            # Also update the soft budget to match, capped at config ceiling if set
            if self._config_ceiling > 0:
                self.context_window = min(n_ctx, self._config_ceiling)
            else:
                self.context_window = n_ctx
            self._initial_context = self.context_window
            self._initial_max_gen = self.max_generation_tokens
        logger.info(
            f"📌 Physical model n_ctx locked: {n_ctx} tokens | "
            f"Soft budget initialised: {self.context_window} tokens"
        )

    def _estimate_model_memory(self, model_name: str) -> float:
        """Return estimated memory in GB for the model weights."""
        name_lower = model_name.lower()
        # Rough estimates for Q4 quantised models (adjust as needed)
        if '70b' in name_lower:
            return 35.0
        if '30b' in name_lower or '34b' in name_lower:
            return 18.0
        if '13b' in name_lower:
            return 7.0
        if '7b' in name_lower:
            return 3.5
        if '3b' in name_lower:
            return 1.8
        if '1b' in name_lower or '2b' in name_lower:
            return 1.0
        # Fallback to a default
        return 2.0

    def _calculate_safe_context(self, model_name: str) -> int:
        available_gb = self.system_metrics.available_memory_gb
        model_mem_gb = self._estimate_model_memory(model_name)
        # Reserve model memory + 1GB OS buffer
        usable_mem_gb = max(0, available_gb - model_mem_gb - 1.0)
        # KV cache per token: for CPU, roughly 2 * layers * hidden_size * 2 bytes (Q4) ≈ 2KB per token for 3B
        # Use a conservative factor (bytes per token) based on model class
        if '3b' in model_name.lower():
            bytes_per_token = 2048  # 2KB per token
        elif '7b' in model_name.lower():
            bytes_per_token = 4096  # 4KB per token
        else:
            bytes_per_token = 1024  # default
        max_tokens = int((usable_mem_gb * (1024 ** 3)) / bytes_per_token)
        # Cap to reasonable absolute maximums
        sensible = [1024, 2048, 4096, 8192, 16384, 32768]
        safe_ctx = max(1024, max((v for v in sensible if v <= max_tokens), default=1024))
        return safe_ctx

    def _get_model_max_context(self) -> int:
        mapping = {
            ModelClass.TINY: 2048,
            ModelClass.SMALL: 4096,
            ModelClass.MEDIUM: 8192,
            ModelClass.LARGE: 16384,
            ModelClass.VERY_LARGE: 32768,
            ModelClass.HUGE: 65536,
        }
        return mapping.get(self.model_class, 8192)

    def _calculate_resource_limit(self) -> int:
        mem_ctx = int(self.system_metrics.total_memory_gb * 1024)
        core_bonus = self.system_metrics.cpu_cores * 512
        gpu_bonus = 4096 if self.system_metrics.has_cuda else 0
        total = mem_ctx + core_bonus + gpu_bonus
        return min(total, 65536)

    # ====================================================================
    # Token Budget Calculation (enhanced with timeout)
    # ====================================================================
    def calculate_token_budget(
            self,
            prompt_text: str,
            estimated_prompt_tokens: Optional[int] = None,
            max_generation_tokens: int = 1024,
            model_name: Optional[str] = None,
            tokens_per_sec: Optional[float] = None
    ) -> TokenMetrics:

        MIN_GEN_RESERVE = 100  # Guarantee at least 100 tokens for generation

        if estimated_prompt_tokens is None:
            counter = TokenCounter(model_name or self.model_name)
            estimated_prompt_tokens = counter.count(prompt_text)

        with self._context_lock:
            soft_budget = self.context_window  # memory-pressure-adapted soft limit
            physical_limit = self._actual_model_n_ctx if self._actual_model_n_ctx > 0 else soft_budget

        # --- Tier 1: physical feasibility check ---
        physical_safety = max(200, int(physical_limit * 0.05))
        physical_usable = physical_limit - physical_safety
        warnings = []
        is_feasible = True

        if estimated_prompt_tokens > physical_usable:
            # Prompt TRULY does not fit in the loaded model
            logger.error(
                f"🚨 TRUE overflow: prompt {estimated_prompt_tokens} > "
                f"physical capacity {physical_usable} (n_ctx={physical_limit})"
            )
            recommended_max_gen = 1
            is_feasible = False
            warnings.append("CRITICAL: Prompt exceeds physical model context — compress required")
            total_budget = physical_limit
            usable = physical_usable
            max_prompt_tokens = 0
            available_for_generation = physical_usable - estimated_prompt_tokens  # negative
        else:
            # --- Tier 2: soft generation budget ---
            # Ensure the soft budget is never smaller than what the prompt needs + 100 gen tokens
            min_needed = estimated_prompt_tokens + 100
            effective_budget = max(soft_budget, min_needed, self._min_viable_budget)
            # But never exceed physical limit
            effective_budget = min(effective_budget, physical_limit)

            system_reserve = max(0, int(effective_budget * 0.05))
            safety_buffer = max(100, int(effective_budget * 0.08))
            usable = effective_budget - system_reserve - safety_buffer
            max_prompt_tokens = max(0, usable - MIN_GEN_RESERVE)
            available_for_generation = usable - estimated_prompt_tokens
            total_budget = effective_budget

            if available_for_generation < 10:
                # Very tight but physically possible — generate minimal tokens
                recommended_max_gen = max(1, available_for_generation)
                warnings.append(f"⚠️ Tight budget: only {available_for_generation} gen tokens available")
            elif available_for_generation < 100:
                recommended_max_gen = max(1, available_for_generation // 2)
                warnings.append(f"Low generation space: {available_for_generation} tokens")
            else:
                # Clamp to max_generation_tokens and ensure not negative
                recommended_max_gen = min(max_generation_tokens, available_for_generation)

            if soft_budget < min_needed:
                warnings.append(
                    f"ℹ️ Soft budget ({soft_budget}) below prompt needs ({min_needed}) — "
                    f"using floor {effective_budget}. Physical limit: {physical_limit}."
                )

        remaining = usable - estimated_prompt_tokens - recommended_max_gen
        efficiency = (estimated_prompt_tokens + recommended_max_gen) / total_budget if total_budget else 0

        # TIMEOUT CALCULATION (FIXED)
        if tokens_per_sec is not None and tokens_per_sec > 0:
            gen_time = (recommended_max_gen / tokens_per_sec) * 2.5
            prompt_time = (estimated_prompt_tokens / (tokens_per_sec * 10)) * 1.3
        else:
            is_gpu = self.system_metrics.has_cuda
            if is_gpu:
                gen_time = (recommended_max_gen / 50.0) * 2.0
                prompt_time = (estimated_prompt_tokens / 500.0) * 1.5
            else:
                gen_time = (recommended_max_gen / self.system_metrics.estimated_cpu_tokens_per_sec) * 3.0
                prompt_time = (estimated_prompt_tokens / (self.system_metrics.estimated_cpu_tokens_per_sec * 5)) * 2.0

        total_time = prompt_time + gen_time
        recommended_timeout = total_time + 60.0
        max_timeout = getattr(self, 'max_timeout', 3600)
        recommended_timeout = min(recommended_timeout, max_timeout)

        return TokenMetrics(
            prompt_tokens=estimated_prompt_tokens,
            available_for_generation=available_for_generation,
            recommended_max_gen=recommended_max_gen,
            total_budget=total_budget,
            remaining=remaining,
            efficiency_ratio=efficiency,
            is_feasible=is_feasible,
            warnings=warnings,
            recommended_timeout=recommended_timeout,
            max_prompt_tokens=max_prompt_tokens,
        )

    def estimate_execution_tokens(
            self,
            remaining_goals: int,
            tokens_available: int,
            complexity_score: float
    ) -> Dict[str, Any]:
        """
        Estimate token requirements for remaining goals.
        Helps prevent mid-execution token exhaustion.
        """
        tokens_per_goal = tokens_available / max(1, remaining_goals)

        # Adjust based on complexity
        if complexity_score > 0.7:
            safety_factor = 0.6  # Use only 60% - complex tasks need buffer
        elif complexity_score > 0.4:
            safety_factor = 0.7
        else:
            safety_factor = 0.8

        safe_tokens_per_goal = int(tokens_per_goal * safety_factor)

        return {
            "total_available": tokens_available,
            "goals_remaining": remaining_goals,
            "tokens_per_goal": tokens_per_goal,
            "safe_tokens_per_goal": safe_tokens_per_goal,
            "safety_factor": safety_factor,
            "is_feasible": safe_tokens_per_goal >= 100  # Minimum 100 tokens per goal
        }

    def get_execution_strategy_for_context(
            self,
            remaining_goals: int,
            tokens_remaining: int,
            time_remaining: Optional[float],
            complexity_score: float,
            current_success_rate: float
    ) -> ExecutionStrategy:
        """
        Dynamically select best strategy for current execution context.
        Called during execution to switch strategies if needed.
        """
        tokens_per_goal = tokens_remaining / max(1, remaining_goals)

        # CRITICAL CONDITIONS: Force iterative
        if tokens_per_goal < 100 or tokens_remaining < 500:
            return ExecutionStrategy.ITERATIVE_NEXT_ACTION

        if time_remaining and time_remaining < 60 and remaining_goals > 1:
            return ExecutionStrategy.ITERATIVE_NEXT_ACTION

        # If we're doing well, can use fuller plan
        if current_success_rate > 0.8 and complexity_score < 0.5:
            return ExecutionStrategy.FULL_PLAN

        # Default to hybrid (safest for long tasks)
        return ExecutionStrategy.HYBRID

    # Add to AdaptiveModelManager class in model_manager.py after line ~500

    def evaluate_execution_health(
            self,
            goals_completed: int,
            goals_remaining: int,
            tokens_used: int,
            tokens_budget: int,
            execution_time_seconds: float,
            timeout_seconds: Optional[float] = None,
            current_strategy: ExecutionStrategy = ExecutionStrategy.FULL_PLAN
    ) -> Dict[str, Any]:
        """
        Real-time health evaluation during multi-step execution.
        CRITICAL: Called between goal steps to adjust strategy if needed.

        Returns recommendation for strategy change + tokens/time warnings.
        """
        health_report = {
            "should_switch_strategy": False,
            "recommended_strategy": current_strategy,
            "token_health": "ok",
            "time_health": "ok",
            "action": "continue"
        }

        # Calculate burn rates
        if execution_time_seconds > 0:
            tokens_per_second = tokens_used / execution_time_seconds
            goals_per_second = goals_completed / execution_time_seconds
        else:
            return health_report

        # PROJECT remaining execution
        if goals_per_second > 0:
            projected_time_for_remaining = goals_remaining / goals_per_second
        else:
            projected_time_for_remaining = float('inf')

        tokens_remaining = tokens_budget - tokens_used
        if goals_remaining > 0:
            tokens_per_remaining_goal = tokens_remaining / goals_remaining
        else:
            tokens_per_remaining_goal = tokens_remaining

        # TOKEN HEALTH CHECK
        if tokens_per_remaining_goal < 100:
            health_report["token_health"] = "critical"
            health_report["should_switch_strategy"] = True
            health_report["recommended_strategy"] = ExecutionStrategy.ITERATIVE_NEXT_ACTION
            health_report["action"] = "DEGRADE: Switch to iterative (token critical)"

            # Also reduce context
            self.reduce_context_window()

        elif tokens_per_remaining_goal < 200:
            health_report["token_health"] = "warning"
            if current_strategy == ExecutionStrategy.FULL_PLAN:
                health_report["should_switch_strategy"] = True
                health_report["recommended_strategy"] = ExecutionStrategy.HYBRID
                health_report["action"] = "DEGRADE: Switch to hybrid (token warning)"

        # TIME HEALTH CHECK (timeout risk)
        if timeout_seconds:
            time_headroom = timeout_seconds - projected_time_for_remaining
            if time_headroom < 30 and goals_remaining > 2:
                health_report["time_health"] = "critical"
                health_report["should_switch_strategy"] = True
                health_report["recommended_strategy"] = ExecutionStrategy.ITERATIVE_NEXT_ACTION
                health_report["action"] = "DEGRADE: Switch to iterative (timeout risk)"
            elif time_headroom < 120:
                health_report["time_health"] = "warning"

        return health_report

    def intelligent_compress(self, instruction: str, target_tokens: int) -> str:
        """
        Semantic‑aware prompt compression.
        Preserves critical lines, selectively includes important lines,
        and truncates only as a last resort.
        """
        if not hasattr(self, 'token_counter') or self.token_counter is None:
            # Fallback to simple truncation if token counter unavailable
            words = instruction.split()
            keep = int(target_tokens / 1.3)
            return ' '.join(words[:keep]) + '…'

        lines = instruction.split('\n')
        if not lines:
            return ""

        # 1. Score each line (higher = more important)
        scored_lines = []
        for line in lines:
            score = self._score_line_importance(line)
            token_count = self.token_counter.count(line)
            scored_lines.append((line, score, token_count))

        # 2. Separate into critical (score >= 2), important (score == 1), optional (score 0)
        critical = [(line, tokens) for line, score, tokens in scored_lines if score >= 2]
        important = [(line, tokens) for line, score, tokens in scored_lines if score == 1]
        optional = [(line, tokens) for line, score, tokens in scored_lines if score == 0]

        # 3. Always keep all critical lines (they are essential)
        result_lines = []
        total_tokens = 0
        for line, tokens in critical:
            result_lines.append(line)
            total_tokens += tokens

        # 4. If critical lines already exceed target, log warning and return them
        if total_tokens > target_tokens:
            logger.warning(
                f"🚨 Critical lines alone exceed target ({total_tokens} > {target_tokens}) – dropping all optional content")
            return '\n'.join(result_lines)

        remaining = target_tokens - total_tokens

        # 5. Add important lines in order, whole if possible
        for line, tokens in important:
            if tokens <= remaining:
                result_lines.append(line)
                remaining -= tokens
            else:
                # Important line doesn't fit whole – truncate it
                truncated = self._truncate_line_to_tokens(line, remaining)
                if truncated:
                    result_lines.append(truncated)
                remaining = 0
                break

        # 6. If still have remaining tokens, add optional lines (whole or truncated)
        if remaining > 0:
            for line, tokens in optional:
                if tokens <= remaining:
                    result_lines.append(line)
                    remaining -= tokens
                else:
                    truncated = self._truncate_line_to_tokens(line, remaining)
                    if truncated:
                        result_lines.append(truncated)
                    break

        return '\n'.join(result_lines)

    def _score_line_importance(self, line: str) -> int:
        """
        Score line importance: 2 = critical, 1 = important, 0 = optional.
        """
        line_lower = line.lower()
        # Critical patterns (must be preserved)
        critical_patterns = [
            r'tool\s*:', r'use\s*:', r'call\s*:',
            r'(?:if|then|when|else|otherwise)',
            r'must|required|critical',
            r'^(?:goal|objective|success|task|step)\s*[:#]',
            r'[🔗🌐📥🔍🚨🗄️⚙️🛠️✂️🎯🏭]',  # emoji markers
            r'^\d+\.\s+',  # numbered steps
        ]
        for pat in critical_patterns:
            if re.search(pat, line_lower):
                return 2

        # Important patterns (nice to have)
        important_patterns = [
            r'example|note|hint',
            r'priority|high',
            r'recommend',
        ]
        for pat in important_patterns:
            if re.search(pat, line_lower):
                return 1

        return 0

    def _truncate_line_to_tokens(self, line: str, max_tokens: int) -> str:
        """
        Truncate a line to fit within max_tokens, preserving as much text as possible.
        Uses token counter to find the right character length via binary search.
        Falls back to character estimation if token counter fails.
        """
        if not line or max_tokens <= 0:
            return ""

        # Quick win: if line already fits
        if self.token_counter.count(line) <= max_tokens:
            return line

        # Binary search for the longest substring that fits
        low, high = 1, len(line)
        best = 0
        while low <= high:
            mid = (low + high) // 2
            candidate = line[:mid]
            if self.token_counter.count(candidate) <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        if best > 0:
            return line[:best] + '…'
        else:
            # Even one character exceeds? Return a single character + ellipsis
            return line[:1] + '…'

    def _attempt_context_recovery_sync(self):
        """
        Synchronous context recovery (safe for daemon threads).
        Called after successful tool execution without async overhead.
        """
        try:
            mem = psutil.virtual_memory()

            # Only attempt recovery if memory is healthy AND sufficient time has passed
            if mem.percent < 50 and (time.time() - self._last_reduction_time) > self._recovery_cooldown:
                with self._context_lock:
                    if self.context_window < self._initial_context:
                        recovery_factor = 1.05
                        new_ctx = min(self._initial_context, int(self.context_window * recovery_factor))

                        if new_ctx > self.context_window:
                            logger.info(
                                f"📈 Context recovery: {self.context_window} → {new_ctx} tokens "
                                f"(Memory: {mem.percent:.1f}%)"
                            )
                            self.context_window = new_ctx

                            # Also attempt to recover generation tokens
                            if self.max_generation_tokens < self._initial_max_gen:
                                new_max_gen = min(
                                    self._initial_max_gen,
                                    int(self.max_generation_tokens * recovery_factor)
                                )
                                self.max_generation_tokens = new_max_gen
                                logger.info(f"📈 Max generation tokens: {new_max_gen}")
        except Exception as e:
            logger.debug(f"Sync recovery attempt failed: {e}")

    # ====================================================================
    # Intelligent Prompt Compression
    # ====================================================================
    def adapt_prompt_to_context(self, original_prompt: str, target_tokens: int) -> str:
        """Public method to compress a prompt – uses intelligent compression."""
        return self.intelligent_compress(original_prompt, target_tokens)

    # ====================================================================
    # Real-Time Monitoring (unchanged)
    # ====================================================================
    def check_memory_available(self, required_gb: float) -> bool:
        try:
            if time.time() - self._last_memory_check > self._memory_check_interval:
                self.system_metrics.available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
                self._last_memory_check = time.time()
            return self.system_metrics.available_memory_gb >= required_gb
        except Exception:
            return True

    # ====================================================================
    # Reporting
    # ====================================================================
    def get_full_report(self) -> Dict[str, Any]:
        return {
            "system": {
                "capability": self.system_metrics.capability_level.value,
                "total_memory_gb": round(self.system_metrics.total_memory_gb, 2),
                "available_memory_gb": round(self.system_metrics.available_memory_gb, 2),
                "cpu_cores": self.system_metrics.cpu_cores,
                "cpu_freq_ghz": round(self.system_metrics.cpu_freq_ghz, 2),
                "has_cuda": self.system_metrics.has_cuda,
                "has_mps": self.system_metrics.has_mps,
                "estimated_cpu_tokens_per_sec": round(self.system_metrics.estimated_cpu_tokens_per_sec, 1),
            },
            "model": {
                "name": self.model_name,
                "class": self.model_class.value if self.model_class else None,
                "context_window": self.context_window,
            },
            "timestamp": time.time(),
        }


# Singleton accessor
def get_model_manager() -> AdaptiveModelManager:
    return AdaptiveModelManager()


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION ADDITIONS v1.0.0 — Adaptive token budgeting, generation telemetry
# ─────────────────────────────────────────────────────────────────────────────

import threading as _mm_threading
import time as _mm_time
import collections as _mm_collections


class AdaptiveTokenBudget:
    """
    Dynamically adjust generation token limit based on system memory pressure.

    On a 13.5GB system running Mistral-7B Q4 (4GB model), available RAM
    shrinks as the KV cache grows. This class monitors psutil memory and
    reduces max_tokens when RAM falls below safety thresholds.

    Tiers:
    - RAM > 60% free  → full budget (max_tokens as configured)
    - RAM 40-60% free → 75% of budget
    - RAM 20-40% free → 50% of budget
    - RAM < 20% free  → 25% of budget (minimum safe operation)

    Usage::

        budget = AdaptiveTokenBudget(base_tokens=512)
        tokens = budget.current()   # → adjusted token count
    """

    def __init__(self, base_tokens: int = 512, min_tokens: int = 128):
        self._base = base_tokens
        self._min = min_tokens
        self._cache: dict = {"ts": 0.0, "pct": 50.0}
        self._lock = _mm_threading.Lock()
        self._ttl = 30.0  # re-check every 30s

    def _get_mem_pct_free(self) -> float:
        """Return percentage of RAM that is free (0-100)."""
        try:
            import psutil
            m = psutil.virtual_memory()
            return 100.0 - m.percent
        except ImportError:
            return 50.0

    def current(self) -> int:
        """Return adjusted max_tokens based on current memory pressure."""
        now = _mm_time.monotonic()
        with self._lock:
            if now - self._cache["ts"] > self._ttl:
                self._cache["pct"] = self._get_mem_pct_free()
                self._cache["ts"] = now
            pct_free = self._cache["pct"]

        if pct_free >= 60:
            factor = 1.0
        elif pct_free >= 40:
            factor = 0.75
        elif pct_free >= 20:
            factor = 0.50
        else:
            factor = 0.25

        return max(self._min, int(self._base * factor))

    def stats(self) -> dict:
        return {
            "base_tokens": self._base,
            "current_tokens": self.current(),
            "pct_ram_free": round(self._cache.get("pct", 50.0), 1),
        }


class GenerationTelemetry:
    """
    Track LLM generation performance across calls.

    Measures tokens/sec, prompt vs completion token ratio, and
    time-to-first-token (TTFT). Used by SmartTimeoutCalculator
    to refine its estimate of actual hardware speed.

    Usage::

        tel = GenerationTelemetry()
        tel.record(prompt_tokens=518, gen_tokens=320, elapsed_s=160.0)
        stats = tel.summary()   # → {"avg_tok_per_sec": 2.0, ...}
    """

    def __init__(self, window: int = 50):
        self._window = window
        self._lock = _mm_threading.Lock()
        self._records: "_mm_collections.deque" = _mm_collections.deque(maxlen=window)

    def record(self, prompt_tokens: int, gen_tokens: int, elapsed_s: float) -> None:
        if elapsed_s <= 0 or gen_tokens <= 0:
            return
        tps = gen_tokens / elapsed_s
        with self._lock:
            self._records.append({
                "prompt_tok": prompt_tokens,
                "gen_tok": gen_tokens,
                "elapsed_s": elapsed_s,
                "tok_per_sec": tps,
                "ts": _mm_time.time(),
            })

    def measured_tps(self) -> float:
        """Return EMA-weighted tokens/sec estimate (recent records weighted more)."""
        with self._lock:
            recs = list(self._records)
        if not recs:
            return 2.0  # safe default for CPU-only Mistral-7B Q4
        # EMA with alpha=0.3 (more weight to recent observations)
        ema = recs[0]["tok_per_sec"]
        for r in recs[1:]:
            ema = 0.3 * r["tok_per_sec"] + 0.7 * ema
        return max(0.5, round(ema, 2))

    def summary(self) -> dict:
        with self._lock:
            recs = list(self._records)
        if not recs:
            return {"samples": 0, "avg_tok_per_sec": 2.0, "ema_tok_per_sec": 2.0}
        tps_vals = [r["tok_per_sec"] for r in recs]
        return {
            "samples": len(recs),
            "avg_tok_per_sec": round(sum(tps_vals) / len(tps_vals), 2),
            "ema_tok_per_sec": self.measured_tps(),
            "min_tok_per_sec": round(min(tps_vals), 2),
            "max_tok_per_sec": round(max(tps_vals), 2),
            "last_elapsed_s": round(recs[-1]["elapsed_s"], 1),
            "last_gen_tokens": recs[-1]["gen_tok"],
        }


# Module-level singletons
_generation_telemetry = GenerationTelemetry(window=50)


def get_generation_telemetry() -> "GenerationTelemetry":
    """Return the global LLM generation telemetry tracker."""
    return _generation_telemetry


def get_telemetry_summary() -> dict:
    """Return LLM generation performance statistics."""
    return _generation_telemetry.summary()


# =============================================================================
# BACKEND ORCHESTRATOR  (1.0.0 — adaptive multi-backend selection)
# =============================================================================
# Probes every available inference backend in priority order and selects
# the best one.  Falls back seamlessly when the primary fails mid-session.
#
# Priority:
#   1. local_llm_cloud  — hardware-optimised cloud server
#   2. llama_cpp_direct — direct llama-cpp-python (existing path)
#   3. openai_compat    — any OpenAI-format HTTP endpoint
#   4. transformers     — HuggingFace transformers
#   5. numpy_sim        — zero-dep simulation (always available)
# =============================================================================

import enum as _enum


class BackendType(_enum.Enum):
    LOCAL_LLM_CLOUD = "local_llm_cloud"
    LLAMA_CPP_DIRECT = "llama_cpp_direct"
    OPENAI_COMPAT = "openai_compat"
    TRANSFORMERS = "transformers"
    NUMPY_SIM = "numpy_sim"


_BACKEND_PRIORITY = [
    BackendType.LOCAL_LLM_CLOUD,
    BackendType.LLAMA_CPP_DIRECT,
    BackendType.OPENAI_COMPAT,
    BackendType.TRANSFORMERS,
    BackendType.NUMPY_SIM,
]


class BackendStatus(_enum.Enum):
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"  # available but slower than expected


class BackendOrchestrator:
    """
    Probe every inference backend in priority order and select the best one.

    Thread-safe singleton.  Designed so ai_agent.py's ModelFactory can call
    ``BackendOrchestrator.recommend_provider(config)`` to get the optimal
    provider string rather than hard-coding "llama_cpp".

    Also feeds the detected hardware profile back into AdaptiveModelManager
    so token budgets and context windows are automatically calibrated to the
    actual hardware (APU 2 GB vs RTX 4090, etc.).
    """

    _instance: "Optional[BackendOrchestrator]" = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._initialized = False
                cls._instance = obj
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._status: Dict[BackendType, BackendStatus] = {
            b: BackendStatus.UNKNOWN for b in BackendType}
        self._hw_profile = None  # HardwareProfile from local-llm-cloud (if available)
        self._probe_lock = threading.Lock()
        self._last_probe: float = 0.0
        self._probe_ttl: float = 60.0  # re-probe every 60 s
        self._active: Optional[BackendType] = None
        self._initialized = True

    # ── Probing ───────────────────────────────────────────────────────────────

    def probe_all(self, cfg: dict = None) -> Dict[BackendType, BackendStatus]:
        """
        Check availability of every backend.  Caches results for _probe_ttl s.
        ``cfg`` is the parsed config.yaml dict (used to read host/port/api_key).
        """
        with self._probe_lock:
            if time.time() - self._last_probe < self._probe_ttl:
                return dict(self._status)
            cfg = cfg or {}

            # ── 1. local-llm-cloud ─────────────────────────────────────────
            try:
                from amb_sentinel.llm.llm_cloud_adapter import probe_server, get_client, LLMCloudUnavailable
                llc_cfg = cfg.get("llm_cloud", {})
                host = llc_cfg.get("host", "localhost")
                port = int(llc_cfg.get("port", 8080))
                # api_key was read from config but never passed to
                # get_client(), so this probe hit /v1/system and
                # /v1/drive/stats unauthenticated. Harmless when
                # local-llm-server has no api.key set, but if it does, this
                # silently 403's (see the bare `except Exception: pass` a
                # few lines down) and self._hw_profile just stays None.
                api_key = llc_cfg.get("api_key", "") or os.environ.get("LLM_CLOUD_API_KEY", "")
                if probe_server(host, port, timeout=2.0):
                    self._status[BackendType.LOCAL_LLM_CLOUD] = BackendStatus.AVAILABLE
                    # Cache hardware profile
                    try:
                        c = get_client(host, port, api_key=api_key)
                        self._hw_profile = c.get_hardware_profile()
                        logger.info("BackendOrchestrator: local-llm-cloud AVAILABLE "
                                    "(profile=%s tps=%.1f)",
                                    self._hw_profile.profile_name,
                                    self._hw_profile.target_tps)
                    except Exception:
                        pass
                else:
                    self._status[BackendType.LOCAL_LLM_CLOUD] = BackendStatus.UNAVAILABLE
            except ImportError:
                self._status[BackendType.LOCAL_LLM_CLOUD] = BackendStatus.UNAVAILABLE

            # ── 2. llama-cpp-python ────────────────────────────────────────
            try:
                import llama_cpp  # noqa: F401
                self._status[BackendType.LLAMA_CPP_DIRECT] = BackendStatus.AVAILABLE
            except ImportError:
                self._status[BackendType.LLAMA_CPP_DIRECT] = BackendStatus.UNAVAILABLE

            # ── 3. OpenAI-compat endpoint ──────────────────────────────────
            oai_cfg = cfg.get("llm_cloud", {}).get("openai_compat", {})
            base_url = (oai_cfg.get("base_url", "")
                        or os.environ.get("OPENAI_API_BASE", "")
                        or os.environ.get("OPENAI_BASE_URL", ""))
            api_key = (oai_cfg.get("api_key", "")
                       or os.environ.get("OPENAI_API_KEY", ""))
            if base_url or api_key:
                try:
                    import openai  # noqa: F401
                    self._status[BackendType.OPENAI_COMPAT] = BackendStatus.AVAILABLE
                except ImportError:
                    try:
                        import requests  # noqa: F401
                        # Can use requests for raw OpenAI-compat calls
                        self._status[BackendType.OPENAI_COMPAT] = BackendStatus.AVAILABLE
                    except ImportError:
                        self._status[BackendType.OPENAI_COMPAT] = BackendStatus.UNAVAILABLE
            else:
                self._status[BackendType.OPENAI_COMPAT] = BackendStatus.UNAVAILABLE

            # ── 4. HuggingFace transformers ────────────────────────────────
            try:
                import transformers  # noqa: F401
                self._status[BackendType.TRANSFORMERS] = BackendStatus.AVAILABLE
            except ImportError:
                self._status[BackendType.TRANSFORMERS] = BackendStatus.UNAVAILABLE

            # ── 5. NumPy sim — always available ───────────────────────────
            self._status[BackendType.NUMPY_SIM] = BackendStatus.AVAILABLE

            self._last_probe = time.time()
            # Select best
            for bt in _BACKEND_PRIORITY:
                if self._status[bt] == BackendStatus.AVAILABLE:
                    self._active = bt
                    break

            logger.info("BackendOrchestrator: active backend = %s",
                        self._active.value if self._active else "none")
            return dict(self._status)

    def best_backend(self, cfg: dict = None) -> BackendType:
        """Return the highest-priority available backend, probing if needed."""
        self.probe_all(cfg)
        return self._active or BackendType.NUMPY_SIM

    def recommend_provider(self, model_cfg: dict = None, cfg: dict = None) -> str:
        """
        Return a provider string compatible with ModelFactory.

        If local-llm-cloud is up, return 'local_llm_cloud'.
        Otherwise map to the existing provider names (llama_cpp, hf, openai…).
        """
        best = self.best_backend(cfg)
        _MAP = {
            BackendType.LOCAL_LLM_CLOUD: "local_llm_cloud",
            BackendType.LLAMA_CPP_DIRECT: "llama_cpp",
            BackendType.OPENAI_COMPAT: "openai",
            BackendType.TRANSFORMERS: "hf",
            BackendType.NUMPY_SIM: "dummy",
        }
        return _MAP.get(best, "llama_cpp")

    # ── Hardware-profile integration ─────────────────────────────────────────

    def apply_hw_profile_to_manager(self) -> None:
        """
        If local-llm-cloud returned a hardware profile, feed its
        context_length and target_tps into AdaptiveModelManager so that
        token budgets and timeouts are calibrated to actual hardware.
        """
        if not self._hw_profile:
            return
        try:
            mgr = get_model_manager()
            profile = self._hw_profile
            # Override context window with hardware-optimal value
            with mgr._context_lock:
                if profile.context_length > 0:
                    mgr.context_window = profile.context_length
                    mgr._initial_context = profile.context_length
                if profile.target_tps > 0:
                    mgr.system_metrics.estimated_cpu_tokens_per_sec = profile.target_tps
            logger.info(
                "BackendOrchestrator: applied HW profile to AdaptiveModelManager "
                "(ctx=%d tps=%.1f kv=%s force_no_think=%s)",
                profile.context_length, profile.target_tps,
                profile.kv_cache_quant, profile.force_no_think,
            )
        except Exception as exc:
            logger.debug("apply_hw_profile_to_manager failed: %s", exc)

    def get_hw_profile(self):
        """Return cached HardwareProfile (None if not yet probed or cloud unavailable)."""
        return self._hw_profile

    # ── Status reporting ─────────────────────────────────────────────────────

    def status_report(self) -> dict:
        return {
            "active_backend": self._active.value if self._active else None,
            "backends": {b.value: s.value for b, s in self._status.items()},
            "hw_profile": (vars(self._hw_profile)
                           if self._hw_profile else None),
            "last_probe": self._last_probe,
        }

    def invalidate_cache(self) -> None:
        """Force re-probe on next call."""
        with self._probe_lock:
            self._last_probe = 0.0


def get_backend_orchestrator(cfg: dict = None) -> BackendOrchestrator:
    """Return the singleton BackendOrchestrator, probing if needed."""
    orch = BackendOrchestrator()
    orch.probe_all(cfg)
    return orch
