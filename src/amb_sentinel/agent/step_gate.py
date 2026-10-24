"""Evidence-based step completion for the AMB-Sentinel adaptive agent.

The bug this fixes: on 2026-10-22 the agent replied
``{"tool": null, "final_answer": "Stored 5 items"}`` without calling any
tool. The loop marked the step done and returned success=True while the
raw_data table was empty. A *claim* is not *evidence*.

Rule
----
A step may only be closed with ``final_answer`` (or autoadvance) after at
least one SUCCESSFUL call to a tool listed in the step's contract. Steps
that write (db_execute, db_bulk_insert, store_raw_item) must additionally
report ``rows >= min_rows``.

Steps that name no tools (e.g. "REPORT: summarise with final_answer") need
no evidence — their contract is empty and every final_answer is accepted.

Contracts can be overridden per step in config.yaml::

    ai_service:
      step_contracts:
        "5": {require_any: [store_raw_item], min_rows: 1}
        "6": {require_any: []}
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

VERSION = 2
log = logging.getLogger(__name__)

# Tools whose success must be backed by a positive row count.
WRITE_TOOLS = frozenset({"db_execute", "db_bulk_insert", "store_raw_item"})

_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


# ═════════════════════════════════════════════════════════════════════════════
# Row-count extraction
# ═════════════════════════════════════════════════════════════════════════════
def extract_rows(result: Any) -> Optional[int]:
    """Rows affected/returned, read from the common tool-result shapes.

    Handles:
      * ``{"inserted": N}``        — store_raw_item, db_bulk_insert
      * ``{"rows": N}``            — generic
      * ``{"count": N}``           — generic
      * ``{"rowcount": N}``        — top level
      * ``{"result": {"rowcount": N}}`` — db_execute
      * ``{"result": [..] }``      — reads

    Returns ``None`` when the shape is unknown (not zero — unknown).
    """
    if not isinstance(result, dict):
        return None
    inner = result.get("result")
    for src in (result, inner if isinstance(inner, dict) else {}):
        for key in ("rowcount", "rows", "inserted", "count"):
            v = src.get(key)
            if isinstance(v, bool):
                continue
            if isinstance(v, int):
                return v
    if isinstance(inner, list):
        return len(inner)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Contract
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class StepContract:
    """What evidence a step needs before it may close with ``final_answer``."""
    require_any: List[str] = field(default_factory=list)
    min_rows: int = 0
    source: str = "inferred"      # "config" | "config-unreachable" | "inferred"

    def describe(self) -> str:
        if not self.require_any:
            return "no evidence required"
        rows = f", >= {self.min_rows} row(s)" if self.min_rows else ""
        return f"one of {self.require_any}{rows}"


# ═════════════════════════════════════════════════════════════════════════════
# Verdict
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Verdict:
    accepted: bool
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def repair_hint(self, contract: StepContract) -> str:
        """Prompt fragment fed back to the model when the verdict is rejected."""
        if self.accepted:
            return ""
        names = ", ".join(contract.require_any) or "a listed tool"
        extra = (
            f" It must report at least {contract.min_rows} row(s) written."
            if contract.min_rows else ""
        )
        return (
            f"You said the step is done but no tool ran ({self.reason}). "
            f"Call one of: {names}.{extra} Reply with a tool call, not final_answer."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Contract resolution
# ═════════════════════════════════════════════════════════════════════════════
def contract_for(
    step: Any,
    overrides: Optional[Dict[str, Any]] = None,
    available: Optional[Iterable[str]] = None,
) -> StepContract:
    """Resolve the evidence contract for *step*.

    Priority: explicit config override > inferred from step.tools.

    Safety: if the override names tools that are **not** registered (e.g.
    ``register_db_tools`` failed at startup), we log a warning and degrade
    to the inferred contract. A misconfigured override must NOT silently
    auto-accept every ``final_answer`` — that is exactly the F1 bug we are
    fixing, re-entering through the back door.
    """
    step_id = str(getattr(step, "id", ""))
    ov = (overrides or {}).get(step_id)
    avail = set(available) if available is not None else None

    if ov is not None:
        wanted = list(ov.get("require_any") or [])
        available_wanted = [t for t in wanted if avail is None or t in avail]
        min_rows = int(ov.get("min_rows", 0) or 0)

        if wanted and not available_wanted:
            log.error(
                "step %s: contract requires %s but none are registered; "
                "keeping the requirement — step will fail if none of them can be called",
                step_id, wanted,
            )
            # Keep 'wanted' — don't degrade to empty. judge_final will reject.
            return StepContract(wanted, min_rows, "config-unreachable")

        return StepContract(available_wanted, min_rows, "config")

    # Inferred: step.tools, filtered by what's actually registered.
    named = [
        t for t in getattr(step, "tools", [])
        if avail is None or t in avail
    ]
    min_rows = 1 if any(t in WRITE_TOOLS for t in named) else 0
    return StepContract(named, min_rows, "inferred")


# ═════════════════════════════════════════════════════════════════════════════
# Judging
# ═════════════════════════════════════════════════════════════════════════════
def judge_final(
    contract: StepContract,
    step_ledger: List[Dict[str, Any]],
) -> Verdict:
    """Decide whether a ``final_answer`` may close the step."""
    if not contract.require_any:
        return Verdict(True, "no tool evidence required", {"tool_ok": 0})

    ok_calls = [
        e for e in step_ledger
        if e.get("status") == "ok" and e.get("tool") in contract.require_any
    ]
    if not ok_calls:
        attempted = [
            e.get("tool") for e in step_ledger
            if e.get("tool") in contract.require_any
        ]
        why = (
            "tools attempted but all failed" if attempted
            else "no tool called in this step"
        )
        return Verdict(False, why, {"tool_ok": 0, "attempted": attempted})

    rows = sum(int(e.get("rows") or 0) for e in ok_calls)
    if contract.min_rows and rows < contract.min_rows:
        return Verdict(
            False,
            f"rows written {rows} < required {contract.min_rows}",
            {"tool_ok": len(ok_calls), "rows": rows},
        )
    return Verdict(
        True,
        "evidence satisfied",
        {"tool_ok": len(ok_calls), "rows": rows},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Independent DB post-condition
# ═════════════════════════════════════════════════════════════════════════════
def verify_rows_in_db(
    db: Any,
    table: str,
    run_id: str,
    conn: str = "primary",
    col: str = "agent_execution_id",
) -> Optional[int]:
    """Count rows the run really left in the database.

    Unlike ``judge_final``, this does not trust the tool, the model, or the
    ledger. It runs a fresh ``SELECT count(*)``. Returns ``None`` on any
    failure — the caller logs ``None`` as "unverified", never as "0 rows".
    """
    if not (isinstance(table, str) and _IDENT_RE.match(table)):
        return None
    if not (isinstance(col, str) and _IDENT_RE.match(col)):
        return None
    if not (isinstance(run_id, str) and run_id):
        return None
    try:
        r = db.execute(
            conn,
            f'SELECT count(*) AS n FROM "{table}" WHERE "{col}" = %s',
            (run_id,),
            fetch="one",
        )
        if isinstance(r, dict) and r.get("ok") and isinstance(r.get("result"), dict):
            return int(r["result"].get("n", 0))
    except Exception as exc:      # noqa: BLE001
        log.warning("verify_rows_in_db(%s, %s) failed: %s", table, run_id, exc)
        return None
    return None
