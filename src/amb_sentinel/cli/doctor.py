"""`sentinel doctor` — diagnostics. Prints install commands, does not run sudo."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys

import yaml


def _check_python() -> tuple[bool, str]:
    # requires-python in pyproject.toml already refuses to install on an
    # older interpreter, so this is just a report — no branch needed.
    return True, f"Python {sys.version.split()[0]}"

def _check_config(config_path: str | None):
    from amb_sentinel import paths
    p = paths.find_config(config_path)
    if not p:
        return False, "no config.yaml (run `sentinel init`)"
    try:
        yaml.safe_load(p.read_text())
        return True, str(p)
    except Exception as e:
        return False, f"invalid YAML: {e}"


def _check_dirs():
    from amb_sentinel import paths
    ok = True
    messages = []
    for fn in (paths.config_dir, paths.data_dir, paths.log_dir, paths.cache_dir):
        d = fn()
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".probe"
            probe.write_text("x")
            probe.unlink()
            messages.append(f"✓ {d}")
        except Exception as e:
            ok = False
            messages.append(f"✗ {d}: {e}")
    return ok, "\n  ".join(messages)


def _check_extras():
    required = ["psycopg2", "redis", "yaml", "httpx", "requests"]
    optional = {"playwright": "web", "yt_dlp": "media", "faster_whisper": "media",
                "torch": "ml", "sentence_transformers": "ml",
                "llama_cpp": "llama", "openai": "openai"}
    present = {}
    for mod in required:
        present[mod] = importlib.util.find_spec(mod) is not None
    opt_present = {k: importlib.util.find_spec(k) is not None for k in optional}
    missing_required = [m for m, ok in present.items() if not ok]
    if missing_required:
        return False, f"missing: {', '.join(missing_required)}"
    note = ", ".join(f"{k}={'✓' if ok else '✗'}" for k, ok in opt_present.items())
    return True, f"core ok; optional: {note}"


def _check_llm_server(cfg: dict):
    try:
        import httpx
    except ImportError:
        return False, "httpx not installed"
    llc = cfg.get("llm_cloud", {}) or {}
    host = llc.get("host", "localhost")
    port = int(llc.get("port", 8080))
    try:
        r = httpx.get(f"http://{host}:{port}/health", timeout=3.0)
        if r.status_code == 200:
            return True, f"http://{host}:{port}/health → 200"
        return False, f"http://{host}:{port}/health → {r.status_code}"
    except Exception as e:
        return False, f"http://{host}:{port} unreachable: {type(e).__name__}"


def _check_binaries():
    missing = []
    for b in ("ffmpeg", "tesseract"):
        if not shutil.which(b):
            missing.append(b)
    if missing:
        cmd = {
            "Linux":   "apt install ffmpeg tesseract-ocr",
            "Darwin":  "brew install ffmpeg tesseract",
            "Windows": "winget install ffmpeg Gyan.FFmpeg; winget install UB-Mannheim.TesseractOCR",
        }.get(platform.system(), "install manually")
        return False, f"missing {missing}. Install: {cmd}"
    return True, "ffmpeg, tesseract present"

def _cmd_doctor(args) -> int:
    from amb_sentinel import paths

    if args.fix_perms:
        return _fix_perms()
    if args.fetch_models:
        return _fetch_models(args.config)

    paths.ensure_dirs()

    checks = [
        ("Python",  _check_python),
        ("Config",  lambda: _check_config(args.config)),
        ("Dirs",    _check_dirs),
        ("Extras",  _check_extras),
        ("Binaries", _check_binaries),
    ]

    all_ok = True
    print("=" * 60)
    for label, fn in checks:
        ok, msg = fn()
        marker = "✓" if ok else "✗"
        print(f"  [{marker}] {label:10s} — {msg}")
        all_ok = all_ok and ok

    # Config-dependent checks
    p = paths.find_config(args.config)
    if p:
        try:
            cfg = yaml.safe_load(p.read_text()) or {}
            ok, msg = _check_llm_server(cfg)
            print(f"  [{'✓' if ok else '✗'}] LLM server — {msg}")
            all_ok = all_ok and ok
        except Exception as e:
            print(f"  [✗] LLM server — {e}")
            all_ok = False

    print("=" * 60)
    print("Healthy" if all_ok else "Issues found — see above")
    return 0 if all_ok else 1

def _fix_perms() -> int:
    """chmod 600 config.yaml, chmod 700 log/data/cache dirs. POSIX only."""
    from amb_sentinel import paths
    if os.name == "nt":
        print("--fix-perms: not implemented on Windows (use icacls manually)")
        return 1
    issues = 0
    cfg = paths.find_config()
    if cfg:
        try:
            os.chmod(cfg, 0o600)
            print(f"  chmod 600 {cfg}")
        except Exception as e:
            print(f"  ✗ {cfg}: {e}")
            issues += 1
    for d in (paths.log_dir(), paths.data_dir(), paths.cache_dir()):
        try:
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o700)
            print(f"  chmod 700 {d}")
        except Exception as e:
            print(f"  ✗ {d}: {e}")
            issues += 1
    return 0 if issues == 0 else 1

def _fetch_models(config_path: str | None) -> int:
    """Download the configured embedding model to paths.model_dir()."""
    from amb_sentinel import paths
    paths.ensure_dirs()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed: pip install 'amb-sentinel[ml]'")
        return 1

    cfg_path = paths.find_config(config_path)
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path else {}
    model = (cfg.get("ai_service", {}).get("embeddings", {}) or {}).get(
        "model", "all-MiniLM-L6-v2"
    )
    dest = paths.model_dir()
    try:
        m = SentenceTransformer(model, cache_folder=str(dest))
        dim = len(m.encode(["probe"])[0])
        print(f"fetched {model} → {dest} (dim={dim})")
        return 0
    except Exception as e:
        print(f"fetch failed: {e}")
        return 1

def add_subparser(sub) -> None:
    p = sub.add_parser("doctor", help="Diagnose install and connectivity")
    p.add_argument("--config", default=None)
    p.add_argument("--fix-perms", action="store_true",
                   help="chmod 600 config.yaml and 700 log/data/cache dirs")
    p.add_argument("--fetch-models", action="store_true",
                   help="Download the configured embedding model")
    p.set_defaults(func=_cmd_doctor)

