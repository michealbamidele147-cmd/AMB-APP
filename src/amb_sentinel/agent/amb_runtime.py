"""
amb_runtime.py — production agent runtime.

Responsibilities
────────────────
  * Compile a config instruction into deterministic, cacheable steps
  * Build a capability profile from the LLM server's own runtime limits
  * Pack a step's prompt into a hard token budget, with priorities and
    shrinkage — the same instruction runs on a 0.6 B nano model and a 70 B
    large model
  * Run the tool-call loop with bounded repair turns, downgrade on repeated
    failure, deduplication, handle resolution, and per-step checkpointing

Design
──────
  * Pure Python; no I/O of its own. The LLM is a callable, tools are callables
  * Deterministic: same instruction + same tool set + same tier → same plan
  * Interruptible: `AgentRun.cancel()` sets an event the loop polls
  * Resumable: on init, if a matching checkpoint exists, the loop skips past
    completed steps
  * Observable: every log line carries run_id / step_id / llm_call_n; the
    final RunResult carries a metrics dict
"""
from __future__ import annotations

import threading
import asyncio
import hashlib
import inspect
import json
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Awaitable, Callable, ClassVar, Dict, List, Optional, Tuple,
)

from amb_sentinel.agent.amb_actions import (
    Action,
    ToolSpec,
    compile_tool_cards,
    parse_action,
    parse_choice,
)

import contextvars

from amb_sentinel.agent.step_gate import (
    StepContract, contract_for, extract_rows, judge_final,
)
from amb_sentinel.observability import NullRecorder

log = logging.getLogger(__name__)

TIERS = ["nano", "small", "medium", "large"]

# ── Prompt-budget tuning ──────────────────────────────────────────────
# Fraction of usable context the prompt budget should target, per tier.
# Hardware-independent: an 8B model on an APU and an 8B model on a GPU
# have the same *ideal* prompt size; only the wall-clock cost differs.
_TIER_CTX_FRACTION: Dict[str, float] = {
    "nano":   0.45,   # ≤1.5B — long-context attention degrades sharply
    "small":  0.55,   # ≤4.5B — moderate long-context handling
    "medium": 0.70,   # ≤14B  — handles most prompts comfortably
    "large":  0.85,   # >14B  — strong long-context handling
}

# Absolute per-tier ceiling on prompt_budget. Prevents a tiny model with
# a huge context window (e.g. a 0.5B model with 32k ctx) from being fed
# a 14k-token prompt it cannot attend to.
_TIER_MAX_PROMPT: Dict[str, int] = {
    "nano":   1024,
    "small":  2048,
    "medium": 6144,
    "large":  16384,
}

# Floor: system prompt + a handful of tool cards + the current step.
# Below this the agent cannot function; better to overrun a latency
# target than to starve the loop.
_MIN_PROMPT_TOKENS: int = 768

# Chat-template / BOS+EOS overhead reserved on top of the prompt budget.
_CTX_OVERHEAD_TOKENS: int = 128

# ═════════════════════════════════════════════════════════════════════════════
# Token estimator — self-calibrating from the server's usage numbers
# ═════════════════════════════════════════════════════════════════════════════
class TokenEstimator:
    """
    Chars-per-token estimate that self-calibrates from `usage.prompt_tokens`.

    No tokenizer dependency: the server tells us the real count after every
    call, so we correct our estimate with an EMA. Between calibrations the
    estimate is conservative (default 3.2 chars/token for English).
    """
    def __init__(self, chars_per_token: float = 3.2):
        self.cpt = float(chars_per_token)

    def est(self, text: str) -> int:
        if not text:
            return 0
        return int(len(text) / self.cpt) + 1

    def observe(self, chars: int, tokens: int) -> None:
        if chars > 0 and tokens > 0:
            measured = chars / tokens
            # Reject absurd samples (empty prompt, single-token reply).
            if 1.0 < measured < 8.0:
                self.cpt = min(6.0, max(1.5, 0.8 * self.cpt + 0.2 * measured))


# ═════════════════════════════════════════════════════════════════════════════
# Capability profile — auto-scaled per hardware + model
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class CapabilityProfile:
    """
    Everything the loop needs to know about its model and hardware.

    Build with `CapabilityProfile.build(...)` or `CapabilityProfile.from_limits(...)`
    rather than instantiating directly — the factory handles size/context
    tiering, budget calculation, and policy lookup.
    """
    tier: str
    n_ctx: int
    decode_tps: float
    prefill_tps: float
    prompt_budget: int          # tokens the prompt may use
    reserve_out: int            # tokens kept free for the answer
    max_tools: int              # tools shown to the model per step
    ledger_items: int           # recent ledger lines included in the prompt
    examples: int               # tool examples shown
    multi_action: bool          # model can emit more than one tool per turn
    allow_thinking: bool        # allow the model's <think> blocks
    autoadvance: bool           # advance a step after the first successful tool
    forced_choice: bool = False  # tools numbered; model answers a digit
    # per-step budgets
    max_attempts_per_step: int = 6
    max_tool_calls_per_step: int = 12

    # ClassVar so the dataclass machinery does NOT treat this as an
    # instance field. A plain `dict` default on a dataclass field raises
    # `ValueError: mutable default <class 'dict'> for field _POLICY is
    # not allowed: use default_factory`. ClassVar removes it from
    # __init__ / __repr__ / __eq__ entirely, which is what we want.
    _POLICY: ClassVar[Dict[str, Dict[str, Any]]] = {
        "nano": dict(
            reserve_out=192, max_tools=4, ledger_items=3, examples=1,
            multi_action=False, allow_thinking=False, autoadvance=True,
            max_attempts_per_step=3, max_tool_calls_per_step=3,
        ),
        "small": dict(
            reserve_out=320, max_tools=6, ledger_items=5, examples=1,
            multi_action=False, allow_thinking=False, autoadvance=True,
            max_attempts_per_step=5, max_tool_calls_per_step=6,
        ),
        "medium": dict(
            reserve_out=640, max_tools=10, ledger_items=10, examples=2,
            multi_action=False, allow_thinking=False, autoadvance=False,
            max_attempts_per_step=6, max_tool_calls_per_step=12,
        ),
        "large": dict(
            reserve_out=1536, max_tools=20, ledger_items=24, examples=2,
            multi_action=True, allow_thinking=True, autoadvance=False,
            max_attempts_per_step=8, max_tool_calls_per_step=24,
        ),
    }

    @classmethod
    def build(
            cls,
            *,
            n_ctx: int,
            decode_tps: float = 5.0,
            model_hint: str = "",
            hosted: bool = False,
            prefill_tps: Optional[float] = None,
            target_step_latency_s: Optional[float] = None,
            force_tier: Optional[str] = None,
            reserve_out_override: Optional[int] = None,
            max_prompt_override: Optional[int] = None,
            ctx_fraction_override: Optional[float] = None,
    ) -> "CapabilityProfile":
        """Derive a profile from model name + context window.

        The prompt budget is a *fraction of the model's usable context*,
        scaled by tier. It is deliberately NOT derived from wall-clock
        latency: on slow hardware, decode dominates and shrinking the
        prompt saves little time while badly damaging the agent's ability
        to see its tools and step. `target_step_latency_s` is an optional
        soft hint that trims the budget only when the caller supplies one
        and the preferred budget would clearly exceed it — never below
        ``_MIN_PROMPT_TOKENS``.
        """
        # ── 1. Model size → size_tier ────────────────────────────────────
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*b\b",
            model_hint.lower().replace("_", "-").replace("-", " ") + " ",
        )
        size_b = float(m.group(1)) if m else None
        if size_b is None:
            size_tier = 3 if hosted else 2
        elif size_b <= 1.5:
            size_tier = 0
        elif size_b <= 4.5:
            size_tier = 1
        elif size_b <= 14:
            size_tier = 2
        else:
            size_tier = 3

        # ── 2. Context window → ctx_tier ─────────────────────────────────
        ctx_tier = (
            0 if n_ctx <= 2048
            else 1 if n_ctx <= 4096
            else 2 if n_ctx <= 16384
            else 3
        )

        # ── 3. Final tier ────────────────────────────────────────────────
        if force_tier and force_tier not in cls._POLICY:
            log.warning(
                "CapabilityProfile.build: unknown force_tier=%r; ignoring. "
                "Known tiers: %s", force_tier, list(cls._POLICY),
            )
            force_tier = None
        tier = force_tier or TIERS[min(size_tier, ctx_tier)]
        pol = dict(cls._POLICY[tier])

        # ── 4. Reserve_out, bounded by context ───────────────────────────
        # The reply must leave room for the prompt. Never let reserve_out
        # swallow more than half the window, and never let it push the
        # prompt below its floor.
        reserve_out = int(pol["reserve_out"])
        if reserve_out_override is not None:
            reserve_out = int(reserve_out_override)
        max_reserve_for_ctx = max(
            64, n_ctx - _MIN_PROMPT_TOKENS - _CTX_OVERHEAD_TOKENS,
        )
        reserve_out = min(reserve_out, n_ctx // 2, max_reserve_for_ctx)
        pol["reserve_out"] = reserve_out

        # ── 5. Context-derived target budget ─────────────────────────────
        # Tier tables are only DEFAULTS. llm.overrides.* / AMB_LLM_* replace
        # them, so no ctx-related number is welded into the source.
        usable_ctx = max(
            _MIN_PROMPT_TOKENS + 64,
            n_ctx - reserve_out - _CTX_OVERHEAD_TOKENS,
        )
        frac = float(ctx_fraction_override) if ctx_fraction_override else _TIER_CTX_FRACTION[tier]
        cap = int(max_prompt_override) if max_prompt_override else _TIER_MAX_PROMPT[tier]
        preferred = min(int(usable_ctx * min(max(frac, 0.05), 0.95)), cap)

        # ── 6. Rates ─────────────────────────────────────────────────────
        dt = max(float(decode_tps or 0.0), 0.2)
        pf = float(prefill_tps or 0.0)
        if pf <= 0.0:
            # Prefill is almost always 1.5–3× decode on CPU-class hardware;
            # 2× is the conservative midpoint when the server doesn't say.
            pf = max(dt * 2.0, 0.5)

        # ── 7. Optional latency hint (soft trim only, never below floor) ─
        decode_time_s = reserve_out / dt
        est_latency_s = preferred / pf + decode_time_s

        if target_step_latency_s and target_step_latency_s > 0:
            if target_step_latency_s <= decode_time_s:
                log.warning(
                    "CapabilityProfile: target_step_latency_s=%.0fs is smaller "
                    "than the estimated decode time for reserve_out=%d tokens "
                    "(%.0fs at %.2f tok/s). Using the %d-token prompt floor; "
                    "increase the target or accept longer LLM turns.",
                    target_step_latency_s, reserve_out, decode_time_s, dt,
                    _MIN_PROMPT_TOKENS,
                )
                budget = _MIN_PROMPT_TOKENS
            else:
                prefill_s = target_step_latency_s - decode_time_s
                latency_cap = max(_MIN_PROMPT_TOKENS, int(prefill_s * pf))
                budget = min(preferred, latency_cap)
                if budget < preferred:
                    log.info(
                        "CapabilityProfile: prompt budget trimmed %d -> %d "
                        "tokens (target=%.0fs, preferred_est=%.0fs, "
                        "prefill_tps=%.2f, decode_tps=%.2f)",
                        preferred, budget, target_step_latency_s,
                        est_latency_s, pf, dt,
                    )
        else:
            budget = preferred

        # ── 8. Final clamp ───────────────────────────────────────────────
        budget = max(_MIN_PROMPT_TOKENS, min(usable_ctx, budget))

        # ── 9. Construct ─────────────────────────────────────────────────
        profile = cls(
            tier=tier,
            n_ctx=n_ctx,
            decode_tps=dt,
            prefill_tps=pf,
            prompt_budget=budget,
            **pol,
        )
        log.info(
            "CapabilityProfile: size_tier=%s n_ctx=%d reserve_out=%d "
            "prompt_budget=%d usable_ctx=%d decode_tps=%.2f prefill_tps=%.2f "
            "est_call_latency=%.0fs",
            tier, n_ctx, reserve_out, budget, usable_ctx, dt, pf,
            profile.est_call_latency_s,
        )
        return profile

    @classmethod
    def from_limits(
            cls,
            limits: Any,
            *,
            model_hint: str = "",
            target_step_latency_s: Optional[float] = None,
            force_tier: Optional[str] = None,
            reserve_out_override: Optional[int] = None,
            max_prompt_override: Optional[int] = None,
            ctx_fraction_override: Optional[float] = None,
    ) -> "CapabilityProfile":
        """Build from a CapabilityLimits-like object, defensively.

        Accepts any object exposing context_length / effective_tps /
        prefill_tps / tier / model_id. Missing or non-numeric fields fall
        back to safe defaults — this method must never raise on a partial
        limits payload.
        """

        def _num(name: str, default: float) -> float:
            v = getattr(limits, name, None)
            if v is None:
                return float(default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(default)

        def _int(name: str, default: int) -> int:
            v = getattr(limits, name, None)
            if v is None:
                return int(default)
            try:
                return int(v)
            except (TypeError, ValueError):
                return int(default)

        n_ctx = _int("context_length", 4096)
        if n_ctx <= 0:
            n_ctx = 4096

        return cls.build(
            n_ctx=n_ctx,
            decode_tps=_num("effective_tps", 3.0),
            prefill_tps=_num("prefill_tps", 0.0),  # 0 → build() picks 2×decode
            model_hint=model_hint or str(getattr(limits, "model_id", "") or ""),
            target_step_latency_s=target_step_latency_s,
            # The server's tier is a *throughput* label — it crosses
            # boundaries between runs (2.2 tps → 3.5 tps flips
            # "medium" → "small") and it splits checkpoint keys.
            # Derive the tier from model size + context instead, which
            # are stable across restarts on the same hardware.
            force_tier=force_tier,
            reserve_out_override=reserve_out_override,
            max_prompt_override=max_prompt_override,
            ctx_fraction_override=ctx_fraction_override,
        )

    def downgraded(self) -> "CapabilityProfile":
        """One tier down. Smaller prompt, fewer tools, forced-choice at nano."""
        i = TIERS.index(self.tier)
        new_tier = TIERS[max(0, i - 1)]
        base = CapabilityProfile.build(
            n_ctx=self.n_ctx,
            decode_tps=self.decode_tps,
            prefill_tps=self.prefill_tps,
            force_tier=new_tier,
        )
        base.prompt_budget = max(
            _MIN_PROMPT_TOKENS,
            min(base.prompt_budget, int(self.prompt_budget * 0.75)),
        )
        base.forced_choice = new_tier == "nano" or self.forced_choice
        return base

    def downgrade_for(self, reason: str) -> "CapabilityProfile":
        """Context-overflow shrinks the budget; format failures drop a tier;
        transport errors do not downgrade at all."""
        r = (reason or "").lower()
        if "context" in r or "overflow" in r:
            clone = CapabilityProfile.build(
                n_ctx=self.n_ctx,
                decode_tps=self.decode_tps,
                prefill_tps=self.prefill_tps,
                force_tier=self.tier,
            )
            # The server *rejected* the prompt, so trust its signal over
            # our floor. Never drop below 256 (system prompt alone).
            clone.prompt_budget = max(256, int(self.prompt_budget * 0.6))
            return clone
        if "parse" in r or "invalid" in r or "format" in r:
            return self.downgraded()
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "n_ctx": self.n_ctx,
            "decode_tps": round(self.decode_tps, 2),
            "prefill_tps": round(self.prefill_tps, 2),
            "prompt_budget": self.prompt_budget,
            "reserve_out": self.reserve_out,
            "max_tools": self.max_tools,
            "ledger_items": self.ledger_items,
            "autoadvance": self.autoadvance,
            "forced_choice": self.forced_choice,
            "est_prompt_latency_s": round(self.est_prompt_latency_s, 1),
            "est_call_latency_s": round(self.est_call_latency_s, 1),
        }

    @property
    def est_prompt_latency_s(self) -> float:
        """Estimated seconds to prefill `prompt_budget` tokens."""
        return self.prompt_budget / max(self.prefill_tps, 0.1)

    @property
    def est_call_latency_s(self) -> float:
        """Estimated seconds for one full LLM turn (prefill + decode).

        Callers should size their per-run `max_llm_calls` and wall clock
        against this, not against a hard-coded constant.
        """
        return (
                self.prompt_budget / max(self.prefill_tps, 0.1)
                + self.reserve_out / max(self.decode_tps, 0.1)
        )


# ═════════════════════════════════════════════════════════════════════════════
# Prompt packing
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Section:
    name: str
    text: str
    priority: int = 5                    # higher = keep longer
    required: bool = False
    min_tokens: int = 0
    shrink: Optional[Callable[[str, int], str]] = None  # (text, max_chars) -> text


def tail_chars(text: str, n: int) -> str:
    return text if len(text) <= n else "…" + text[-(n - 1):]


def head_chars(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


class PromptTooLarge(Exception):
    """Raised when even required sections can't fit inside the budget."""


def pack(
    sections: List[Section],
    budget: int,
    est: TokenEstimator,
) -> Tuple[List[Section], Dict[str, int]]:
    """
    Fit `sections` into `budget` tokens.

    Order of sacrifice:
      1. Shrink non-required sections, lowest priority first
      2. Drop non-required sections entirely, lowest priority first
      3. Shrink required sections (never below min_tokens)
      4. Raise PromptTooLarge
    """
    secs = [Section(s.name, s.text, s.priority, s.required, s.min_tokens, s.shrink) for s in sections]

    def total() -> int:
        return sum(est.est(s.text) for s in secs if s.text)

    def shrink(s: Section, floor_tokens: int) -> None:
        cur = est.est(s.text)
        if cur <= floor_tokens or not s.text:
            return
        overflow = max(1, total() - budget)
        target_tokens = max(floor_tokens, cur - overflow)
        target_chars = max(40, int(target_tokens * est.cpt))
        fn = s.shrink or head_chars
        s.text = fn(s.text, target_chars)

    for s in sorted([x for x in secs if not x.required], key=lambda x: x.priority):
        if total() <= budget:
            break
        shrink(s, s.min_tokens)
    for s in sorted([x for x in secs if not x.required], key=lambda x: x.priority):
        if total() <= budget:
            break
        s.text = ""
    for s in sorted([x for x in secs if x.required], key=lambda x: x.priority):
        if total() <= budget:
            break
        shrink(s, s.min_tokens)

    if total() > budget:
        raise PromptTooLarge(
            f"prompt needs {total()} tokens but budget is {budget}"
        )
    return secs, {s.name: est.est(s.text) for s in secs}


# ═════════════════════════════════════════════════════════════════════════════
# Instruction compiler
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Step:
    id: str
    title: str
    text: str
    tools: List[str] = field(default_factory=list)
    part: Optional[Tuple[int, int]] = None       # (index, total) when segmented
    max_attempts: int = 6


_HDR = re.compile(
    r"^[ \t]*(?:STEP|PHASE|STAGE)\s*(\d+)\s*[—–:.)\-]*\s*(.*)$",
    re.I | re.M,
)
_NUM = re.compile(r"^[ \t]*(\d+)[.)]\s+(.*)$", re.M)


def compile_instruction(text: str, tool_names: List[str]) -> Tuple[str, List[Step]]:
    """
    Parse `text` into (global_rules, steps).

    Recognises STEP N / PHASE N / STAGE N headers, then numbered lists, then
    treats the whole text as one step. Tool names appearing anywhere in a
    step's text become that step's allowlist.
    """
    if not text or not text.strip():
        return "", [Step("1", "Task", "")]
    marks = list(_HDR.finditer(text)) or list(_NUM.finditer(text))
    if not marks:
        return "", [Step("1", "Task", text.strip())]

    pre = text[: marks[0].start()].strip()
    steps: List[Step] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        title = (m.group(2) or "").strip().split("\n", 1)[0][:80] or f"Step {m.group(1)}"
        full = ((m.group(2) or "").strip() + "\n" + body).strip()
        used = [t for t in tool_names
                if re.search(rf"(?<![\w]){re.escape(t)}(?![\w])", full)]
        steps.append(Step(m.group(1), title, full, used))
    return pre, steps


def segment_step(step: Step, max_tokens: int, est: TokenEstimator) -> List[Step]:
    """
    Split an oversized step on blank lines → bullets → sentence boundaries.
    Preserves the step's tool allowlist for each part.
    """
    if est.est(step.text) <= max_tokens:
        return [step]
    pieces = [
        p.strip()
        for p in re.split(
            r"\n\s*\n|\n(?=\s*[-•*]\s)|(?<=[.!?])\s+(?=[A-Z])",
            step.text,
        )
        if p.strip()
    ]
    parts: List[str] = []
    cur = ""
    for p in pieces:
        if cur and est.est(cur + "\n" + p) > max_tokens:
            parts.append(cur)
            cur = p
        else:
            cur = (cur + "\n" + p).strip()
    if cur:
        parts.append(cur)
    if len(parts) <= 1:
        return [step]
    out: List[Step] = []
    for i, body in enumerate(parts, 1):
        out.append(Step(
            id=f"{step.id}{chr(96 + i)}",
            title=f"{step.title} ({i}/{len(parts)})",
            text=body,
            tools=[t for t in step.tools if t in body] or list(step.tools),
            part=(i, len(parts)),
            max_attempts=step.max_attempts,
        ))
    return out


def plan_key(instruction: str, tier: str, tools: List[str]) -> str:
    return hashlib.sha256(
        json.dumps([instruction, tier, sorted(tools)]).encode()
    ).hexdigest()[:16]


# ═════════════════════════════════════════════════════════════════════════════
# Blackboard — the runtime's value-handle store
# ═════════════════════════════════════════════════════════════════════════════
class Blackboard:
    """
    Big tool outputs live here; the prompt only carries a one-line summary and
    a handle ("r1", "r2"). The model passes handles ($r1.links.0) instead of
    copying long values, so no copied-value drift and constant prompt size.
    """
    def __init__(self):
        self.refs: Dict[str, Any] = {}
        self.facts: Dict[str, str] = {}
        self.n = 0

    def put(self, value: Any) -> str:
        self.n += 1
        k = f"r{self.n}"
        self.refs[k] = value
        return k

    def resolve(self, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: self.resolve(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self.resolve(x) for x in v]
        if isinstance(v, str):
            m = re.fullmatch(r"\$(r\d+)((?:\.[\w-]+)*)", v.strip())
            if m:
                cur = self.refs.get(m.group(1), v)
                for part in [p for p in m.group(2).split(".") if p]:
                    try:
                        if isinstance(cur, (list, tuple)):
                            cur = cur[int(part)]
                        else:
                            cur = cur[part]
                    except (KeyError, IndexError, ValueError, TypeError):
                        return v
                return cur
        return v

    def render_facts(self, max_chars: int = 500) -> str:
        if not self.facts:
            return "(none yet)"
        lines = [f"{k}: {v}" for k, v in self.facts.items()]
        s = "\n".join(lines)
        return tail_chars(s, max_chars) if len(s) > max_chars else s

    def snapshot(self) -> Dict[str, Any]:
        # Only the refs are worth persisting (facts are strings, cheap too).
        return {"n": self.n, "refs": self.refs, "facts": self.facts}

    def restore(self, data: Dict[str, Any]) -> None:
        self.n = int(data.get("n", 0))
        self.refs = dict(data.get("refs") or {})
        self.facts = dict(data.get("facts") or {})


# ═════════════════════════════════════════════════════════════════════════════
# Result summarization
# ═════════════════════════════════════════════════════════════════════════════
# Per-tool one-line summarizers. Signature: (result, ref) -> str.
# Fall back to `summarize()` when the tool is not in this map.
TOOL_SUMMARIZERS: Dict[str, Callable[[Any, str], str]] = {}


def register_summarizer(name: str):
    """Decorator: @register_summarizer("parse_feed") def f(result, ref): ..."""
    def deco(fn: Callable[[Any, str], str]):
        TOOL_SUMMARIZERS[name] = fn
        return fn
    return deco


def summarize(result: Any, ref: str, max_chars: int = 220) -> str:
    """Generic one-line summarizer. Prefer a TOOL_SUMMARIZERS entry when one exists."""
    if isinstance(result, dict):
        bits: List[str] = []
        for k, v in result.items():
            if isinstance(v, list):
                first = json.dumps(v[0], default=str)[:60] if v else "-"
                bits.append(f"{k}=[{len(v)} items; first: {first}]")
            elif isinstance(v, (str, int, float, bool)) or v is None:
                bits.append(f"{k}={json.dumps(v, default=str)[:60]}")
            if len(" ".join(bits)) > max_chars:
                break
        text = "{" + ", ".join(bits) + "}"
    elif isinstance(result, list):
        first = json.dumps(result[0], default=str)[:80] if result else "-"
        text = f"[{len(result)} items; first: {first}]"
    elif isinstance(result, str):
        text = result[:max_chars] + ("…" if len(result) > max_chars else "")
    else:
        text = json.dumps(result, default=str)[:max_chars]
    return f"{text[:max_chars]} → ${ref}"


def summarize_for(tool: str, result: Any, ref: str) -> str:
    fn = TOOL_SUMMARIZERS.get(tool)
    if fn is None:
        return summarize(result, ref)
    try:
        return fn(result, ref)
    except Exception as exc:
        log.warning("summarizer %s failed: %s", tool, exc)
        return summarize(result, ref)


_DATA_KEYS = (
    "links", "urls", "results", "items", "rows", "records",
    "data", "text", "content", "json_data", "entries", "files",
    "chunks", "transcript", "extracted_text", "fetched_content",
)

def _failed_item(r: Any) -> bool:
    return isinstance(r, dict) and (
        bool(r.get("error"))
        or r.get("ok") is False
        or r.get("success") is False
    )

_NETWORK_TOOLS = frozenset({
    "search_links", "search_with_dorks", "find_file_links",
    "find_hidden_resources", "extract_links_from_page", "expand_links",
    "http_fetch", "http_fetch_batch", "download_file", "download_multiple",
    "scrape_page", "scrape_batch", "scrape_async", "scrape_batch_async",
    "parse_feed", "browser_scrape", "video_info", "download_video",
    "download_multiple_videos", "transcribe_url_to_vectorstore",
})

def _is_offline_result(res: Any) -> bool:
    if isinstance(res, dict):
        return bool(res.get("offline")) or res.get("error_type") == "OfflineError"
    if isinstance(res, list):
        return bool(res) and all(_is_offline_result(r) for r in res)
    return False

def result_ok(result: Any) -> bool:
    """True only if the tool actually achieved something useful.

    The prior version never checked result.get("ok"), so a dict like
    {"ok": False, "links": [], "count": 0} scored as success — which is
    exactly what the uploaded checkpoints show: steps with zero links
    and zero rows recorded as auto_done. That is the single biggest
    source of false "success" in the agent logs.
    """
    if result is None:
        return False
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        # Explicit failure markers.
        if result.get("error") or result.get("success") is False or result.get("ok") is False:
            return False
        # Everything that came back from a batch failed, and nothing came
        # back from any data key — treat as a failure even if the outer
        # dict has no "error" key.
        errs = result.get("errors")
        if errs and not any(result.get(k) for k in _DATA_KEYS):
            return False
        # count=0 alongside errors is a failure even if the caller didn't
        # set "ok" explicitly.
        if result.get("count") == 0 and errs:
            return False
        # Known data keys all empty AND no positive count → failure.
        count = result.get("count") or result.get("total_count") or result.get("total")
        if count is not None and int(count) == 0:
            return False
    elif isinstance(result, list):
        if not result:
            return False
        if all(_failed_item(r) for r in result):
            return False
    elif isinstance(result, str):
        return bool(result.strip())
    return True

def extract_facts(tool: str, result: Any, facts: Dict[str, str]) -> None:
    """Merge short scalars and short string lists from `result` into `facts`."""
    if not isinstance(result, dict):
        return
    for k, v in result.items():
        if isinstance(v, (int, float, bool)) or (isinstance(v, str) and 0 < len(v) <= 120):
            facts[f"{tool}.{k}"] = str(v)
        elif isinstance(v, list) and v and all(isinstance(x, str) for x in v[:3]):
            head = ", ".join(x[:80] for x in v[:3])
            more = f" (+{len(v) - 3} more)" if len(v) > 3 else ""
            facts[f"{tool}.{k}"] = head + more


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint store
# ═════════════════════════════════════════════════════════════════════════════
class CheckpointStore:
    """
    Persist a run's state so a crash or graceful shutdown resumes cleanly.

    Format: one JSON file per (instruction_hash, tier). Overwritten atomically
    after every step. Contains:
      * ledger (list of {n, step, tool, args, status, obs})
      * board (refs + facts + n)
      * step_status (step_id -> "done" | "auto_done" | "failed" | "skipped")
      * answers (step_id -> final_answer text)
      * completed_at (unix time)
    """
    def __init__(self, directory: Optional[os.PathLike | str]):
        self.dir = Path(directory) if directory else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Optional[Path]:
        if self.dir is None:
            return None
        return self.dir / f"{key}.json"

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        p = self.path_for(key)
        if p is None or not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            log.warning("checkpoint load %s failed: %s", p, exc)
            return None

    def save(self, key: str, data: Dict[str, Any]) -> None:
        p = self.path_for(key)
        if p is None:
            return
        try:
            tmp = p.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, default=str)
            os.replace(tmp, p)
        except Exception as exc:
            log.warning("checkpoint save %s failed: %s", p, exc)

    def clear(self, key: str) -> None:
        p = self.path_for(key)
        if p is not None and p.exists():
            try:
                p.unlink()
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# Run result
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class RunResult:
    success: bool
    answers: Dict[str, str]
    ledger: List[Dict[str, Any]]
    step_status: Dict[str, str]
    llm_calls: int
    prompt_tokens_peak: int
    notes: List[str]
    run_id: str = ""
    cancelled: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# The loop
# ═════════════════════════════════════════════════════════════════════════════
class AgentRun:
    """
    Step-wise agent loop.

    Parameters
    ----------
    chat           Async callable `chat(messages, **kw) -> LLMResponse`. The
                   response must expose `.text`, `.prompt_tokens`,
                   `.completion_tokens`, `.finish_reason`. A ChatClient
                   instance works directly. If it also exposes
                   `chat_stream_iter`, the loop will use streaming with a
                   non-streaming fallback.
    tools          {name: callable}. Callables may be sync or async; the loop
                   handles both.
    profile        CapabilityProfile built via `.build()` or `.from_limits()`.
    instruction    The config instruction. STEP / PHASE headers become steps.
    specs          Optional {name: ToolSpec}. Built from `tools` when omitted.
    aliases        Optional {name: (alias, ...)} applied when specs are built.
    checkpoint_dir Directory for per-step checkpoint files. Optional.
    max_llm_calls  Run-wide ceiling on LLM calls.
    wall_clock_s   Run-wide wall-clock ceiling.
    memory_check   Optional callable returning free-RAM fraction (0..1). When
                   it drops below 0.15 the loop sleeps and retries once before
                   aborting cleanly.
    run_id         Optional; a random id is used when not supplied.
    """

    SYSTEM_CORE = (
        "You are an autonomous agent. Work on ONE step at a time and call ONE tool per reply.\n"
        "Reply with exactly one JSON object and nothing else:\n"
        '  call a tool : {"tool": "<name>", "args": {...}}\n'
        '  step is done: {"tool": null, "final_answer": "<short result>"}\n'
        'Put every parameter inside "args". Use only the tools listed. '
        'To reuse a big earlier result, pass its handle such as "$r1" or '
        '"$r1.links.0" instead of copying it.'
    )

    def __init__(
        self,
        chat: Callable[..., Awaitable[Any]],
        tools: Dict[str, Callable],
        profile: CapabilityProfile,
        instruction: str,
        *,
        specs: Optional[Dict[str, ToolSpec]] = None,
        aliases: Optional[Dict[str, Tuple[str, ...]]] = None,
        est: Optional[TokenEstimator] = None,
        tool_timeout_s: float = 180.0,
        per_call_cap_s: float = 900.0,
        max_llm_calls: int = 60,
        wall_clock_s: float = 3600.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        checkpoint_dir: Optional[os.PathLike | str] = None,
        checkpoint_ttl_s: float = 6 * 3600,
        memory_check: Optional[Callable[[], float]] = None,
        run_id: Optional[str] = None,
        recorder: Any = None,
        contracts: Optional[Dict[str, Any]] = None,
        max_gate_rejects: int = 2,
    ):
        self.chat = chat
        self.tools = tools
        self.profile = profile
        self.offline_seen = False
        self.est = est or TokenEstimator()
        self.rules, steps = compile_instruction(instruction, list(tools))
        self.rec = recorder if recorder is not None else NullRecorder()
        self.contract_overrides = contracts or {}
        self.evidence: Dict[str, Dict[str, Any]] = {}
        self.max_gate_rejects = int(max_gate_rejects)
        # Segment any oversized step down to a third of the prompt budget.
        max_step_tokens = max(120, profile.prompt_budget // 3)
        self.steps: List[Step] = []
        for s in steps:
            self.steps.extend(segment_step(s, max_step_tokens, self.est))
        for s in self.steps:
            s.max_attempts = max(1, int(profile.max_attempts_per_step))

        # Specs
        if specs is None:
            aliases = aliases or {}
            self.specs = {
                n: ToolSpec.from_callable(n, f, aliases=aliases.get(n, ()))
                for n, f in tools.items()
            }
        else:
            self.specs = specs

        # Stable, step-independent tool list so the system prompt is
        # byte-identical on every call. The old _tools_for() rebuilt the
        # list per step, killing the server's prompt-prefix cache after
        # the first ~688 tokens (see llm_server.log: reused=688,
        # evaluated=319…428 — 50–108 s of re-prefill per call).
        _used = [t for s in self.steps for t in s.tools if t in tools]
        _essential = [
            "validate_url", "http_fetch", "scrape_page",
            "db_execute", "db_query", "db_bulk_insert",
        ]
        _base: List[str] = []
        for t in _used + _essential:
            if t in tools and t not in _base:
                _base.append(t)
        self._base_tools = _base[: profile.max_tools]

        # Runtime state
        self.board = Blackboard()
        self.ledger: List[Dict[str, Any]] = []
        self.answers: Dict[str, str] = {}
        self.step_status: Dict[str, str] = {}
        self.memo: Dict[Tuple[str, str], str] = {}
        self.tool_call_count = 0
        self.llm_calls = 0
        self.retries = 0
        self.transport_errors = 0
        self.prompt_tokens_peak = 0
        self.completion_tokens_total = 0
        self.notes: List[str] = []
        self.step_times: Dict[str, float] = {}

        self.tool_timeout = float(tool_timeout_s)
        self.per_call_cap_s = float(per_call_cap_s)
        self.max_llm_calls = int(max_llm_calls)
        self.wall_clock_s = float(wall_clock_s)
        self._deadline = time.monotonic() + self.wall_clock_s
        self.sleep = sleep
        self.memory_check = memory_check

        self.run_id = run_id or uuid.uuid4().hex[:8]
        # The checkpoint key must not include the throughput tier, or a
        # restart with a slightly different measured tps forks progress
        # into two separate files. tools + instruction are the actual
        # inputs that determine the plan.
        self.checkpoint_key = plan_key(instruction, "", list(tools))
        self.checkpoints = CheckpointStore(checkpoint_dir)
        self.checkpoint_ttl_s = float(checkpoint_ttl_s)

        self.cancel_event = asyncio.Event()

    # ── public lifecycle ────────────────────────────────────────────────────
    def cancel(self) -> None:
        """Request a graceful stop at the next loop boundary or token."""
        self.cancel_event.set()

    def metrics(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tier": self.profile.tier,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_call_count,
            "retries": self.retries,
            "transport_errors": self.transport_errors,
            "prompt_tokens_peak": self.prompt_tokens_peak,
            "completion_tokens_total": self.completion_tokens_total,
            "ledger_size": len(self.ledger),
            "steps_total": len(self.steps),
            "steps_done": sum(1 for v in self.step_status.values() if v in ("done", "auto_done")),
            "steps_failed": sum(1 for v in self.step_status.values() if v in ("failed", "skipped")),
            "step_times_s": {k: round(v, 3) for k, v in self.step_times.items()},
            "notes": self.notes[:20],
        }

    # ── tools list for a step ───────────────────────────────────────────────
    def _tools_for(self, step: Step) -> List[str]:
        """Stable base + step extras, in a fixed order.

        _base_tools is built once in __init__ and never changes across
        steps, so the tool-card block at the start of the system prompt
        is byte-identical every call — which is what lets the server's
        prompt-prefix cache reuse it.
        """
        extras = [
            t for t in step.tools
            if t in self.tools and t not in self._base_tools
        ]
        return (self._base_tools + extras)[
            : max(self.profile.max_tools, len(self._base_tools))
        ]

    # ── prompt assembly ─────────────────────────────────────────────────────
    def _messages(
        self,
        step: Step,
        idx: int,
        repair: str = "",
        last_obs: str = "",
    ) -> List[Dict[str, str]]:
        names = self._tools_for(step)
        if self.profile.forced_choice:
            cards = compile_tool_cards(
                self.specs, names, budget_chars=self.profile.prompt_budget * 3,
                numbered=True, include_description=False,
            )
        else:
            cards = compile_tool_cards(
                self.specs, names,
                budget_chars=max(300, int(self.profile.prompt_budget * 3)),
                numbered=False, include_description=True,
                include_example=self.profile.examples,
            )

        recent_ledger = [
            e for e in self.ledger[-self.profile.ledger_items:] if e["step"] == step.id
        ]
        led_lines = []
        for e in recent_ledger:
            args_preview = json.dumps(e["args"], ensure_ascii=False)[:90]
            led_lines.append(
                f"{e['n']}. {e['tool']}({args_preview}) → {e['status']}: {e['obs'][:160]}"
            )
        led = "\n".join(led_lines) or "(none yet)"

        secs: List[Section] = [
            Section("core", self.SYSTEM_CORE, 10, True, 60),
            Section("rules", self.rules, 4, False, 0, head_chars),
            Section("tools", "TOOLS (* = required):\n" + cards, 9, True, 80, head_chars),
            Section("step", f"STEP {idx + 1}/{len(self.steps)} — {step.text}", 8, True, 80, head_chars),
            Section("facts", "KNOWN FACTS:\n" + self.board.render_facts(500), 6, False, 0, tail_chars),
            Section("ledger", "DONE SO FAR IN THIS STEP:\n" + led, 5, False, 0, tail_chars),
            Section("obs", f"LAST OBSERVATION: {last_obs}" if last_obs else "", 3, False, 0, head_chars),
            Section("repair", f"YOUR PREVIOUS REPLY WAS REJECTED: {repair}" if repair else "", 11, False, 0, head_chars),
        ]
        packed, sizes = pack(secs, self.profile.prompt_budget, self.est)
        self.prompt_tokens_peak = max(self.prompt_tokens_peak, sum(sizes.values()))
        t = {s.name: s.text for s in packed}
        system = "\n\n".join(x for x in (t["core"], t["rules"], t["tools"]) if x)
        ask = (
            "Choose the number of the tool to use and nothing else."
            if self.profile.forced_choice
            else "Next action (one JSON object):"
        )
        user = "\n\n".join(x for x in (
            t["step"], t["facts"], t["ledger"], t["obs"], t["repair"], ask,
        ) if x)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ── one LLM turn, with typed-error handling, retries, streaming ─────────
    async def _ask(
        self,
        msgs: List[Dict[str, str]],
        step: Step,
    ) -> str:
        # Imported lazily so amb_runtime.py stays importable without amb_llm_core.
        try:
            from amb_sentinel.llm.amb_llm_core import (
                LLMContextOverflow,
                LLMError,
                LLMStall,
            )
        except Exception:
            LLMError = Exception       # type: ignore
            LLMContextOverflow = Exception  # type: ignore
            LLMStall = Exception       # type: ignore

        # Memory guard
        if self.memory_check is not None:
            try:
                free = float(self.memory_check())
            except Exception:
                free = 1.0
            if free < 0.15:
                self.notes.append(f"low memory {free:.2f}; sleeping 2s")
                await self.sleep(2.0)
                try:
                    free = float(self.memory_check())
                except Exception:
                    free = 1.0
                if free < 0.15:
                    raise RuntimeError("low memory persists; aborting run")

        delay = 2.0
        est_prompt = sum(self.est.est(m["content"]) for m in msgs)
        for attempt in range(4):
            if self.cancel_event.is_set():
                raise asyncio.CancelledError("run cancelled")
            if self.llm_calls >= self.max_llm_calls:
                raise TimeoutError("run llm-call budget exhausted")
            if time.monotonic() > self._deadline:
                raise TimeoutError("run wall-clock budget exhausted")

            self.llm_calls += 1
            call_n = self.llm_calls
            remaining = min(
                max(5.0, self._deadline - time.monotonic()),
                self.per_call_cap_s,
            )

            try:
                self.rec.emit(
                    "llm_request", getattr(step, "id", None), call=call_n,
                    est_prompt_tokens=est_prompt,
                    max_tokens=self.profile.reserve_out,
                    n_ctx=self.profile.n_ctx,
                    system_preview=msgs[0]["content"][:160],
                    user_preview=msgs[-1]["content"][-300:],
                )
                resp = await asyncio.wait_for(
                    self._do_chat(msgs, step, est_prompt, deadline_s=remaining),
                    timeout=remaining + 5.0,
                )
                # Calibrate token estimator from observed prompt_tokens.
                if getattr(resp, "prompt_tokens", 0):
                    self.est.observe(
                        sum(len(m["content"]) for m in msgs),
                        resp.prompt_tokens,
                    )
                self.completion_tokens_total += int(getattr(resp, "completion_tokens", 0) or 0)
                self.rec.emit(
                    "llm_response", step.id, call=call_n,
                    prompt_tokens=getattr(resp, "prompt_tokens", 0),
                    completion_tokens=getattr(resp, "completion_tokens", 0),
                    elapsed_s=round(float(getattr(resp, "elapsed_s", 0) or 0), 2),
                    finish=getattr(resp, "finish_reason", ""),
                    text_preview=(getattr(resp, "text", "") or "")[:300],
                )
                log.warning(
                    "llm reply run=%s step=%s call=%d len=%d",
                    self.run_id, step.id, call_n, len(getattr(resp, "text", "")),
                )
                return getattr(resp, "text", "") or ""


            except asyncio.TimeoutError:

                if time.monotonic() >= self._deadline:
                    raise TimeoutError("run wall-clock budget exhausted") from None
                # Still have budget; treat as a normal retryable failure
                self.transport_errors += 1
                if attempt >= 3:
                    raise
                await self.sleep(2.0 * (2 ** attempt))
                continue

            except LLMContextOverflow:
                # The prompt was too big for the model's context. Shrink and retry.
                before = self.profile.prompt_budget
                self.profile = self.profile.downgrade_for("context overflow")
                self.notes.append(
                    f"context overflow: prompt_budget {before} -> {self.profile.prompt_budget}"
                )
                # Rebuild messages with the smaller budget
                idx = self.steps.index(step)
                msgs = self._messages(step, idx)
                est_prompt = sum(self.est.est(m["content"]) for m in msgs)
                continue

            except LLMError as e:
                self.transport_errors += 1
                if not getattr(e, "retryable", False):
                    raise
                self.notes.append(f"{type(e).__name__}: {str(e)[:80]}")
                if attempt >= 3:
                    raise
                wait = delay * (0.75 + random.random() / 2.0)
                self.retries += 1
                await self.sleep(wait)
                delay = min(delay * 2.0, 30.0)
                continue

            except Exception as e:
                log.exception(
                    "llm call attempt %d/%d failed (run=%s step=%s): %s",
                    attempt + 1, 4, self.run_id, step.id, e,
                )

        raise RuntimeError("LLM unavailable after retries")

    async def _do_chat(
            self,
            msgs: List[Dict[str, str]],
            step: Step,
            est_prompt_tokens: int,
            deadline_s: Optional[float] = None,
    ) -> Any:
        """
        Single non-streaming call. `self.chat` is expected to be an
        async callable `(messages, **kw) -> LLMResponse`.

        `deadline_s` is the caller's cap on how long this call may
        consume from the wall-clock budget. We take the tighter of
        (a) whatever remains on the run's wall clock and (b) the
        caller's cap, so an outer wait_for short enough to preserve
        the run budget cannot be masked by an inner call that runs
        to its own, longer, default.
        """
        _remaining_wall = max(30.0, self._deadline - time.monotonic())
        if deadline_s is not None and deadline_s > 0:
            _remaining_wall = min(_remaining_wall, float(deadline_s))

        return await self.chat(
            msgs,
            max_tokens=self.profile.reserve_out,
            temperature=0.1,
            no_think=not self.profile.allow_thinking,
            est_prompt_tokens=est_prompt_tokens,
            deadline_s=_remaining_wall,
            decode_tps=self.profile.decode_tps,
            prefill_tps=self.profile.prefill_tps,
        )

    # ── tool execution ──────────────────────────────────────────────────────
    async def _exec(self, name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        fn = self.tools[name]

        # Daemon-thread runner for sync tools. Unlike asyncio.to_thread,
        # which schedules on the default ThreadPoolExecutor (whose workers
        # are non-daemon and joined at interpreter exit), this runner uses
        # a raw daemon thread. A hung tool mid-request therefore cannot
        # block Ctrl-C exit — the process can exit even if the tool is
        # still inside a C call.
        async def _run_in_daemon(fn, **kw):
            loop = asyncio.get_running_loop()
            fut = loop.create_future()

            _ctx = contextvars.copy_context()   # keeps CURRENT_RUN visible inside the tool

            def _w():
                try:
                    r = _ctx.run(fn, **kw)
                    loop.call_soon_threadsafe(
                        lambda: fut.done() or fut.set_result(r)
                    )
                except BaseException as exc:
                    loop.call_soon_threadsafe(
                        lambda exc=exc: fut.done() or fut.set_exception(exc)
                    )

            threading.Thread(
                target=_w, daemon=True,
                name=f"tool-{getattr(fn, '__name__', 'fn')}",
            ).start()
            return await fut

        try:
            if inspect.iscoroutinefunction(fn):
                res = await asyncio.wait_for(fn(**args), self.tool_timeout)
            else:
                res = await asyncio.wait_for(
                    _run_in_daemon(fn, **args), self.tool_timeout,
                )
            return result_ok(res), res
        except asyncio.TimeoutError:
            return False, {"error": f"tool timed out after {self.tool_timeout:.0f}s"}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception(
                "tool %s failed (run=%s step-owner=%s): %s",
                name, getattr(self, "run_id", "?"), getattr(self, "_current_step", "?"), e,
            )
            return False, {"error": f"{type(e).__name__}: {str(e)[:160]}"}

    # ── top-level run ───────────────────────────────────────────────────────
    async def run(self) -> RunResult:
        t_start = time.monotonic()
        cancelled = False

        # Resume from checkpoint if one exists.
        existing = self.checkpoints.load(self.checkpoint_key)
        if existing:
            _age = time.time() - float(existing.get("completed_at") or 0)
            if _age > self.checkpoint_ttl_s:
                self.notes.append(
                    f"checkpoint older than TTL ({_age:.0f}s > {self.checkpoint_ttl_s:.0f}s) — starting fresh"
                )
                self.checkpoints.clear(self.checkpoint_key)
                existing = None

        resumed_steps: List[str] = []
        if existing:
            self.ledger = list(existing.get("ledger") or [])
            self.answers = dict(existing.get("answers") or {})
            self.evidence = dict(existing.get("evidence") or {})
            self.board.restore(existing.get("board") or {})
            # Only trust a previously "done" step if it carries evidence
            # (checkpoint v2) or genuinely needs none. v1 checkpoints (no
            # evidence) are re-run.
            for sid, st in dict(existing.get("step_status") or {}).items():
                step_obj = next((x for x in self.steps if x.id == sid), None)
                needs = bool(
                    step_obj
                    and contract_for(
                        step_obj, self.contract_overrides, self.tools,
                    ).require_any
                )
                if st in ("done", "auto_done") and (not needs or sid in self.evidence):
                    self.step_status[sid] = st
                    resumed_steps.append(sid)
            msg = (
                f"RESUMED from checkpoint: trusting steps "
                f"{resumed_steps or 'none'}; re-running "
                f"{[x.id for x in self.steps if x.id not in resumed_steps]}"
            )
            self.notes.append(msg)
            log.warning("run=%s %s", self.run_id, msg)

        self.rec.emit(
            "run_start",
            model=getattr(self.chat, "model", ""),
            n_ctx=self.profile.n_ctx,
            size_tier=self.profile.tier,
            steps=[x.id for x in self.steps],
            resumed=resumed_steps,
            prompt_budget=self.profile.prompt_budget,
            reserve_out=self.profile.reserve_out,
        )

        try:
            for idx, step in enumerate(self.steps):
                if self.cancel_event.is_set():
                    cancelled = True
                    break
                if self.step_status.get(step.id) in ("done", "auto_done"):
                    continue
                t_step = time.monotonic()
                if self.offline_seen and any(t in _NETWORK_TOOLS for t in step.tools):
                    self.step_status[step.id] = "blocked_offline"
                    continue
                _llm0, _tool0 = self.llm_calls, self.tool_call_count
                self.rec.emit(
                    "step_start", step.id,
                    title=step.title, tools=step.tools,
                    contract=contract_for(
                        step, self.contract_overrides, self.tools,
                    ).describe(),
                )
                status = await self._run_step(step, idx)
                self.step_status[step.id] = status
                self.step_times[step.id] = time.monotonic() - t_step
                self.rec.emit(
                    "step_end", step.id,
                    status=status,
                    llm_calls=self.llm_calls - _llm0,
                    tool_calls=self.tool_call_count - _tool0,
                    duration_s=round(self.step_times[step.id], 2),
                )
                self._save_checkpoint()
        except asyncio.CancelledError:
            cancelled = True
            self.notes.append("run cancelled")
            self._save_checkpoint()
        except (TimeoutError, RuntimeError) as e:
            self.notes.append(f"run aborted: {e}")
            for s in self.steps:
                self.step_status.setdefault(s.id, "not_run")
            self._save_checkpoint()

        ok = (
            len(self.step_status) == len(self.steps)
            and all(v in ("done", "auto_done") for v in self.step_status.values())
            and not cancelled
        )
        if ok:
            self.checkpoints.clear(self.checkpoint_key)

        self.rec.emit(
            "run_end",
            success=ok,
            verified=ok and all(
                (not contract_for(
                    x, self.contract_overrides, self.tools,
                ).require_any) or x.id in self.evidence
                for x in self.steps
            ),
            llm_calls=self.llm_calls,
            tool_calls=self.tool_call_count,
            rows_written=getattr(self.rec, "rows_written", 0),
            step_status=self.step_status,
            tools_used=sorted({
                e.get("tool") for e in self.ledger
                if e.get("status") == "ok" and e.get("tool")
            }),
        )

        return RunResult(
            success=ok,
            answers=self.answers,
            ledger=self.ledger,
            step_status=self.step_status,
            llm_calls=self.llm_calls,
            prompt_tokens_peak=self.prompt_tokens_peak,
            notes=self.notes,
            run_id=self.run_id,
            cancelled=cancelled,
            metrics=self.metrics(),
        )

    def _save_checkpoint(self) -> None:
        self.checkpoints.save(self.checkpoint_key, {
            "run_id": self.run_id,
            "tier": self.profile.tier,
            "ledger": self.ledger,
            "answers": self.answers,
            "step_status": self.step_status,
            "board": self.board.snapshot(),
            "evidence": self.evidence,
            "version": 2,
            "completed_at": time.time(),
        })

    # ── per-step loop ───────────────────────────────────────────────────────
    async def _run_step(self, step: Step, idx: int) -> str:
        repair = ""
        last_obs = ""
        dup = 0
        offline_hits = 0
        bad = 0
        tool_calls_this_step = 0
        max_tools_step = self.profile.max_tool_calls_per_step

        # Reset the per-step duplicate memo. Without this, a step that
        # legitimately needs to call (e.g.) db_execute with the same args
        # as a *previous* step is silently skipped as a "duplicate",
        # which broke STEP 5 of the config instruction. The blackboard
        # ($rN handles) is unaffected — only the dedup cache resets.
        self.memo = {}

        contract = contract_for(step, self.contract_overrides, self.tools)
        rejects = 0

        for attempt in range(step.max_attempts):
            if self.cancel_event.is_set():
                return "cancelled"
            if tool_calls_this_step >= max_tools_step:
                self.notes.append(f"step {step.id}: tool-call budget exhausted")
                return "exhausted"

            try:
                text = await self._ask(self._messages(step, idx, repair, last_obs), step)
            except asyncio.CancelledError:
                return "cancelled"
            except (TimeoutError, RuntimeError) as exc:
                self.notes.append(f"step {step.id}: {type(exc).__name__}: {exc}")
                return "failed"  # or "aborted" if you want to distinguish

            names = self._tools_for(step)

            # Forced-choice mode for nano: the model picks a digit.
            if self.profile.forced_choice and names:
                choice = parse_choice(text, len(names))
                if choice is not None:
                    name = names[choice]
                    spec = self.specs.get(name)
                    act = Action("tool", tool=name)
                    if spec is not None:
                        # Pre-fill optional defaults so the LLM is not
                        # asked for them on a repair turn. Without this,
                        # every nano-mode call is rejected for missing
                        # required args that had sensible defaults.
                        for p in spec.params.values():
                            if (not p.required and p.has_default
                                    and p.default is not None):
                                act.args[p.name] = p.default
                        missing = [
                            p.name for p in spec.params.values()
                            if p.required
                        ]
                        if missing:
                            act.errors = [
                                f"missing required argument(s): "
                                f"{', '.join(missing)}"
                            ]
                    # Resolve $rN handles against the blackboard, same as
                    # the parse_action path does.
                    act.args = self.board.resolve(act.args)
                else:
                    act = parse_action(
                        text, self.specs,
                        resolver=self.board.resolve,
                        inject_defaults=False,
                    )
            else:
                act = parse_action(
                    text, self.specs,
                    resolver=self.board.resolve,
                    inject_defaults=False,
                )

            # Final answer from the model.
            self.rec.emit(
                "action", step.id,
                action=act.kind, tool=act.tool, args=act.args,
                errors=act.errors or None,
                notes=getattr(act, "notes", None) or None,
            )

            # Final answer from the model — accepted ONLY with evidence.
            if act.kind == "final" and not act.errors:
                verdict = judge_final(
                    contract,
                    [e for e in self.ledger if e["step"] == step.id],
                )
                if verdict.accepted:
                    self.rec.emit(
                        "gate_accept", step.id,
                        reason=verdict.reason, evidence=verdict.evidence,
                    )
                    self.evidence[step.id] = dict(verdict.evidence)
                    self.answers[step.id] = act.final_answer
                    return "done"

                rejects += 1
                self.rec.emit(
                    "gate_reject", step.id,
                    reason=verdict.reason, evidence=verdict.evidence,
                    claimed=str(act.final_answer)[:200], rejects=rejects,
                )
                self.ledger.append(dict(
                    n=len(self.ledger) + 1, step=step.id,
                    tool="(final_answer)", args={},
                    status="rejected", obs=verdict.reason,
                ))
                repair = verdict.repair_hint(contract)
                if rejects > self.max_gate_rejects:
                    self.notes.append(
                        f"step {step.id}: model claimed done {rejects}x "
                        f"without evidence ({verdict.reason})"
                    )
                    return "failed_no_evidence"
                continue

            # Invalid action — either repair or downgrade.
            if not act.ok:
                bad += 1
                repair = act.repair_hint(self.specs)
                log.info(
                    "step %s attempt %d: repair=%s",
                    step.id, attempt, repair[:120],
                    extra={"run_id": self.run_id, "step_id": step.id},
                )
                if bad >= 2 and not self.profile.forced_choice:
                    self.profile = self.profile.downgrade_for("parse invalid")
                    bad = 0
                    self.notes.append(
                        f"downgraded to tier={self.profile.tier} forced_choice={self.profile.forced_choice}"
                    )
                continue

            repair = ""
            args = dict(act.args)

            # Dedup — same (tool, args) is not re-executed.
            key = (act.tool or "", json.dumps(args, sort_keys=True, default=str))
            n_line = len(self.ledger) + 1
            if key in self.memo:
                dup += 1
                last_obs = (
                    f"You already ran {act.tool} with these exact arguments "
                    f"(result ${self.memo[key]}). Use it or choose a different action."
                )
                self.ledger.append(dict(
                    n=n_line, step=step.id, tool=act.tool, args=args,
                    status="duplicate", obs=last_obs,
                ))
                if dup >= 2:
                    return "skipped_duplicate"
                continue

            # Execute.
            _timer = self.rec.tool_timer(step.id, act.tool, args)
            ok, res = await self._exec(act.tool, args)
            tool_calls_this_step += 1
            self.tool_call_count += 1
            ref = self.board.put(res)
            self.memo[key] = ref
            obs = summarize_for(act.tool, res, ref)

            if ok:
                extract_facts(act.tool, res, self.board.facts)
            _timer.done(ok, res, obs[:200])
            self.ledger.append(dict(
                n=n_line, step=step.id, tool=act.tool, args=args,
                status="ok" if ok else "failed", obs=obs,
                rows=extract_rows(res),
            ))

            if not ok and _is_offline_result(res):
                offline_hits += 1
                self.offline_seen = True
                if offline_hits >= 2:
                    self.notes.append(
                        f"step {step.id}: network offline — blocked, no further LLM calls"
                    )
                    return "blocked_offline"
            last_obs = obs

            # Small-model shortcut: if the successful tool was explicitly named
            # in the step text, advance immediately.
            if ok and self.profile.autoadvance and act.tool in step.tools:
                _v = judge_final(
                    contract,
                    [e for e in self.ledger if e["step"] == step.id],
                )
                if _v.accepted:
                    self.rec.emit(
                        "gate_accept", step.id,
                        reason="autoadvance: " + _v.reason, evidence=_v.evidence,
                    )
                    self.evidence[step.id] = dict(_v.evidence)
                    self.answers[step.id] = obs
                    return "auto_done"

        return "exhausted"
