"""One-shot import-rewrite for the AMB-APP packaging refactor."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "src" / "amb_sentinel"

# Order matters: longer / more specific first
SUBS = [
    # Specific module imports (with dot)
    (r"\bfrom\s+agent\.model_manager\b",   "from amb_sentinel.llm.model_manager"),
    (r"\bfrom\s+agent\.amb_llm_core\b",    "from amb_sentinel.llm.amb_llm_core"),
    (r"\bfrom\s+agent\.amb_actions\b",     "from amb_sentinel.agent.amb_actions"),
    (r"\bfrom\s+agent\.amb_runtime\b",     "from amb_sentinel.agent.amb_runtime"),
    (r"\bfrom\s+agent\.llm_cloud_adapter\b", "from amb_sentinel.llm.llm_cloud_adapter"),
    (r"\bfrom\s+agent\.ai_agent\b",        "from amb_sentinel.agent.ai_agent"),
    (r"\bimport\s+agent\.",                "import amb_sentinel."),

    # Bare (relying on sys.path) — the fragile ones
    (r"\bfrom\s+amb_llm_core\b",           "from amb_sentinel.llm.amb_llm_core"),
    (r"\bfrom\s+amb_runtime\b",            "from amb_sentinel.agent.amb_runtime"),
    (r"\bfrom\s+amb_actions\b",            "from amb_sentinel.agent.amb_actions"),
    (r"\bfrom\s+llm_cloud_adapter\b",      "from amb_sentinel.llm.llm_cloud_adapter"),
    (r"\bfrom\s+model_manager\b",          "from amb_sentinel.llm.model_manager"),

    # Tools
    (r"\bfrom\s+tools\s+import\b",         "from amb_sentinel.tools import"),
    (r"\bfrom\s+tools\.",                  "from amb_sentinel.tools."),
    (r"\bimport\s+tools\.",                "import amb_sentinel.tools."),

    # String-literal module names
    (r"'tools\.extract'",  "'amb_sentinel.tools.extract'"),
    (r"\"tools\.extract\"", "\"amb_sentinel.tools.extract\""),
]

def main():
    changed = []
    for p in ROOT.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        new = text
        for pattern, repl in SUBS:
            new = re.sub(pattern, repl, new)
        if new != text:
            p.write_text(new, encoding="utf-8")
            changed.append(p.relative_to(ROOT))
    for p in changed:
        print(f"  rewrote {p}")
    print(f"\n{len(changed)} file(s) modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
