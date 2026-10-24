"""Single source of truth for every LLM number AMB-APP uses.

The LLM server knows its own context size, speed, and limits — AMB-APP
asks, it does not guess. Precedence per field (first hit wins):

    1. environment      AMB_LLM_<FIELD>
    2. config.yaml      llm.overrides.<field>
    3. LLM server       GET /v1/runtime/limits
    4. built-in default ONLY when the server is unreachable; WARNING logged

Every resolved field remembers its source, so the log shows e.g.
``context_length = 40960 [server:/v1/runtime/limits]`` and
``max_prompt_tokens = 8000 [env:AMB_LLM_MAX_PROMPT_TOKENS]``.

``amb-llm-config`` prints the table and flags settings that contradict
each other (a 90 s analyzer timeout when one call takes 120 s, a context
override above what the server loaded, a legacy ``max_total_tokens``…).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

FIELDS = (
    "context_length", "decode_tps", "prefill_tps", "max_output_tokens",
    "max_prompt_tokens", "prompt_ctx_fraction", "target_step_latency_s",
    "tier",
)

# Used ONLY when the server is unreachable.
_DEFAULTS = {"context_length": 4096, "decode_tps": 3.0, "prefill_tps": 0.0}


# ═════════════════════════════════════════════════════════════════════════════
# Resolved
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Resolved:
    value: Any
    source: str
    note: str = ""


@dataclass
class ResolvedLLMConfig:
    fields: Dict[str, Resolved] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    server_ok: bool = False
    server_error: str = ""
    model_id: str = ""
    backend: str = ""

    def __getitem__(self, k: str) -> Any:
        return self.fields[k].value

    def get(self, k: str, default: Any = None) -> Any:
        r = self.fields.get(k)
        return default if r is None or r.value is None else r.value

    def profile_kwargs(self) -> Dict[str, Any]:
        """Arguments for ``CapabilityProfile.build()``/``from_limits()``."""
        g = self.get
        return {
            "n_ctx": int(g("context_length", _DEFAULTS["context_length"])),
            "decode_tps": float(g("decode_tps", _DEFAULTS["decode_tps"])),
            "prefill_tps": float(g("prefill_tps", 0.0) or 0.0),
            "reserve_out_override": g("max_output_tokens"),
            "max_prompt_override": g("max_prompt_tokens"),
            "ctx_fraction_override": g("prompt_ctx_fraction"),
            "target_step_latency_s": g("target_step_latency_s"),
            "force_tier": g("tier"),
        }

    def table(self) -> str:
        w = max(len(k) for k in self.fields) if self.fields else 10
        lines = [
            f"LLM config  server="
            f"{'OK' if self.server_ok else 'UNREACHABLE ' + self.server_error}  "
            f"model={self.model_id or '?'}  backend={self.backend or '?'}"
        ]
        for k in FIELDS:
            r = self.fields.get(k)
            if r is None:
                continue
            v = "(derived by tier policy)" if r.value is None else r.value
            line = f"  {k:<{w}} = {v!s:<26} [{r.source}]"
            if r.note:
                line += f"  {r.note}"
            lines.append(line)
        for m in self.warnings:
            lines.append(f"  ⚠ {m}")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "server_ok": self.server_ok,
            "server_error": self.server_error,
            "model_id": self.model_id,
            "backend": self.backend,
            "fields": {
                k: {"value": r.value, "source": r.source, "note": r.note}
                for k, r in self.fields.items()
            },
            "warnings": self.warnings,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _num(v: Any, typ=float) -> Optional[Any]:
    if v in (None, "", "null", "None"):
        return None
    try:
        return typ(float(v)) if typ is int else typ(v)
    except (TypeError, ValueError):
        return None


def _limits_dict(limits: Any) -> Dict[str, Any]:
    """Accept the raw /v1/runtime/limits dict or a CapabilityLimits-like object."""
    if limits is None:
        return {}
    if isinstance(limits, Mapping):
        return dict(limits)
    raw = getattr(limits, "raw", None)
    if isinstance(raw, Mapping) and raw:
        return dict(raw)
    # CapabilityLimits built from defaults has empty raw → treat as "server did not answer".
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# Resolve
# ═════════════════════════════════════════════════════════════════════════════
def resolve(
    cfg: Optional[Mapping[str, Any]],
    limits: Any = None,
    env: Optional[Mapping[str, str]] = None,
    server_error: str = "",
) -> ResolvedLLMConfig:
    env = os.environ if env is None else env
    ov = ((cfg or {}).get("llm") or {}).get("overrides") or {}
    srv = _limits_dict(limits)

    out = ResolvedLLMConfig(
        server_ok=bool(srv),
        server_error=server_error,
        model_id=str(srv.get("model_id", "") or ""),
        backend=str(srv.get("backend", "") or ""),
    )

    server_vals = {
        "context_length": _num(srv.get("context_length"), int),
        "decode_tps": _num(srv.get("effective_tps") or srv.get("measured_tps")),
        "prefill_tps": _num(srv.get("prefill_tps")),
        "max_output_tokens": _num(srv.get("recommended_response_tokens"), int) or None,
    }

    typ = {
        "context_length": int,
        "max_output_tokens": int,
        "max_prompt_tokens": int,
        "tier": str,
        "decode_tps": float,
        "prefill_tps": float,
        "prompt_ctx_fraction": float,
        "target_step_latency_s": float,
    }

    for f in FIELDS:
        t = typ[f]
        e_name = f"AMB_LLM_{f.upper()}"
        e_val = _num(env.get(e_name), t) if t is not str else (env.get(e_name) or None)
        c_val = _num(ov.get(f), t) if t is not str else (ov.get(f) or None)
        s_val = server_vals.get(f)

        if e_val is not None:
            out.fields[f] = Resolved(e_val, f"env:{e_name}")
        elif c_val is not None:
            out.fields[f] = Resolved(c_val, f"config:llm.overrides.{f}")
        elif s_val is not None:
            out.fields[f] = Resolved(s_val, "server:/v1/runtime/limits")
        elif f in _DEFAULTS and not srv:
            out.fields[f] = Resolved(
                _DEFAULTS[f],
                "FALLBACK (server unreachable)",
                "set llm_cloud base URL/API key or fix the server",
            )
        else:
            out.fields[f] = Resolved(None, "derived")

    # An override may never exceed what the server actually loaded.
    sctx = server_vals["context_length"]
    cur = out.fields["context_length"]
    if sctx and cur.value and cur.value > sctx:
        out.warnings.append(
            f"context_length override {cur.value} > server n_ctx {sctx}: "
            f"clamped to {sctx}"
        )
        out.fields["context_length"] = Resolved(sctx, "server (override clamped)")

    if not srv:
        out.warnings.append(
            "LLM server limits unavailable — using conservative fallbacks; "
            "prompts will be tiny and timeouts wrong until this is fixed"
        )
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Conflict detection
# ═════════════════════════════════════════════════════════════════════════════
def est_call_latency_s(
    prompt_tokens: int,
    out_tokens: int,
    decode_tps: float,
    prefill_tps: float,
) -> float:
    """Estimate seconds for one full LLM call."""
    dt = max(decode_tps, 0.2)
    pf = prefill_tps if prefill_tps > 0 else dt * 2
    return prompt_tokens / pf + out_tokens / dt


def find_conflicts(
    cfg: Mapping[str, Any],
    rc: ResolvedLLMConfig,
    prompt_tokens: Optional[int] = None,
    out_tokens: Optional[int] = None,
) -> List[str]:
    """Settings that cannot all be true at once. Never raises."""
    w: List[str] = []
    cfg = cfg or {}
    ai = cfg.get("ai_service") or {}
    ctx = rc.get("context_length", _DEFAULTS["context_length"])
    dt = float(rc.get("decode_tps", _DEFAULTS["decode_tps"]))
    pf = float(rc.get("prefill_tps", 0.0) or 0.0)
    pt = int(prompt_tokens or rc.get("max_prompt_tokens") or min(ctx // 2, 2048))
    ot = int(out_tokens or rc.get("max_output_tokens") or 512)
    lat = est_call_latency_s(pt, ot, dt, pf)

    mtt = _num(ai.get("max_total_tokens"), int)
    if mtt and mtt < ctx:
        w.append(
            f"ai_service.max_total_tokens={mtt} but the server context is "
            f"{ctx}: this legacy key only feeds timeout maths. Remove it."
        )

    # Only warn on timeouts that ARE the effective timeout on the hot path.
    # llm_cloud.timeout_s is a *fallback* for direct LLMCloudClient calls;
    # LocalLLMCloudAdapter.chat() passes deadline_s= explicitly, so the
    # configured value is never used by the agent or the analyzer.
    checks = [
        ("data_analyzer.timeout", (cfg.get("data_analyzer") or {}).get("timeout")),
    ]
    for m in ai.get("models") or []:
        if isinstance(m, dict) and m.get("provider") == "local_llm_cloud":
            checks.append((
                "ai_service.models[local_llm_cloud].timeout",
                m.get("timeout"),
            ))
    for name, val in checks:
        v = _num(val)
        if v and v < lat:
            w.append(
                f"{name}={v:g}s is shorter than the estimated {lat:.0f}s "
                f"for one call (prompt≈{pt} tok @ "
                f"{pf or dt * 2:.1f} tok/s prefill, {ot} out @ "
                f"{dt:.1f} tok/s): it will time out"
            )

    da = cfg.get("data_analyzer") or {}
    dmt = _num(da.get("max_tokens"), int)
    if dmt is not None and dmt and dmt < 256:
        w.append(
            f"data_analyzer.max_tokens={dmt} is too small for structured "
            f"JSON analysis"
        )

    return w


# ═════════════════════════════════════════════════════════════════════════════
# Server fetch
# ═════════════════════════════════════════════════════════════════════════════
def server_endpoint(
    cfg: Mapping[str, Any],
    env: Optional[Mapping[str, str]] = None,
) -> tuple[str, str]:
    env = os.environ if env is None else env
    lc = (cfg or {}).get("llm_cloud") or {}
    base = env.get("AMB_LLM_BASE_URL") or lc.get("base_url") or ""
    if not base:
        for m in ((cfg or {}).get("ai_service") or {}).get("models") or []:
            if (
                isinstance(m, dict)
                and m.get("provider") == "local_llm_cloud"
                and m.get("base_url")
            ):
                base = m["base_url"]
                break
    if not base:
        base = f"http://{lc.get('host', 'localhost')}:{lc.get('port', 8080)}"
    key = (
        env.get("LLM_CLOUD_API_KEY")
        or env.get("AMB_LLM_API_KEY")
        or lc.get("api_key")
        or ""
    )
    return base.rstrip("/"), key


def fetch_limits(
    base_url: str,
    api_key: str = "",
    timeout: float = 5.0,
) -> tuple[Dict[str, Any], str]:
    """GET /v1/runtime/limits. Returns (body, error). Never raises."""
    import httpx
    hdr = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = httpx.get(
            base_url + "/v1/runtime/limits", headers=hdr, timeout=timeout,
        )
        if r.status_code in (401, 403):
            return {}, (
                f"HTTP {r.status_code}: server requires an API key "
                f"(set LLM_CLOUD_API_KEY)"
            )
        r.raise_for_status()
        return r.json(), ""
    except Exception as e:      # noqa: BLE001
        return {}, f"{type(e).__name__}: {str(e)[:120]}"


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="amb-llm-config",
        description="Show every LLM setting AMB-APP will use, and where it came from.",
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    from amb_sentinel.dbsetup import load_config, load_env_files
    load_env_files()
    cfg, path = load_config(a.config)
    base, key = server_endpoint(cfg)
    body, err = fetch_limits(base, key)
    rc = resolve(cfg, body, server_error=err)
    rc.warnings.extend(find_conflicts(cfg, rc))

    if a.json:
        print(json.dumps(rc.as_dict(), indent=2, default=str))
    else:
        print(f"config: {path}")
        print(f"server: {base}")
        print(rc.table())

    return 0 if rc.server_ok and not rc.warnings else 1


if __name__ == "__main__":
    raise SystemExit(main())
