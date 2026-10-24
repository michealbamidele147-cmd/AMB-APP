"""Every filesystem location used by AMB-Sentinel.

No other module may construct paths from __file__, os.getcwd(), or a literal
'logs/' / 'config.yaml' / 'tmp/'. All paths flow through this file.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from platformdirs import PlatformDirs

APP = "amb-sentinel"
_pd = PlatformDirs(APP, appauthor=False)

# Written by `sentinel service install --system` so that user-scope CLI
# invocations (sentinel-logs, sentinel db status) can find the system
# install's directories without requiring AMB_HOME in the caller's shell.
_SYSTEM_POINTER = Path("/etc/amb-sentinel/paths.json")
_USER_POINTER   = Path.home() / ".config" / "amb-sentinel" / "paths.json"

def _p(v) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(v)))).resolve()


def _warn(msg: str) -> None:
    """Surface install-pointer problems without breaking CLI commands.

    paths.py is imported at module load time by every `sentinel` subcommand,
    before logging is configured. A warnings.warn here lands on stderr via
    Python's default warning filter — visible, non-fatal, and honest about
    the fact that a fallback is happening.
    """
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _read_pointer(ptr: Path) -> Path | None:
    """Read one install pointer. Return its amb_home if valid and the
    pointed-to directory still exists; None if the file is missing.
    Warn and return None if the file exists but is unusable — so a
    corrupted pointer silently falling back to platform defaults is
    never a mystery.
    """
    try:
        with ptr.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (PermissionError, json.JSONDecodeError, OSError) as exc:
        _warn(f"install pointer {ptr} unreadable ({exc}); falling back to defaults")
        return None

    home = data.get("amb_home") if isinstance(data, dict) else None
    if not isinstance(home, str) or not home.strip():
        _warn(f"install pointer {ptr} has no valid 'amb_home'; falling back to defaults")
        return None

    p = _p(home)  # expands ~ and ${VAR}, resolves symlinks — same as every other path
    if not p.is_dir():
        _warn(f"install pointer {ptr} points at {p}, which no longer exists; "
              f"falling back to defaults")
        return None
    return p

def _system_root() -> Path | None:
    """Return the AMB_HOME recorded by a --user or --system install.

    Precedence: the invoking user's own pointer wins over the shared
    system pointer. This lets a user who ran `service install --user`
    see their own logs even if a `--system` install also exists on the
    box, while a user with no user pointer transparently falls back to
    the system install if they can read /etc/amb-sentinel/paths.json.

    To force the system install when both exist, uninstall the user one
    (`sentinel service uninstall --user`) or override the base directly
    with AMB_HOME=/path/to/home on the CLI invocation.

    A corrupt, malformed, stale, or missing pointer never raises —
    paths.py is imported at module load time by every CLI command, and
    a broken pointer must not turn a working `sentinel-logs` into a
    traceback. Every problem is warned to stderr and skipped.
    """
    for ptr in (_USER_POINTER, _SYSTEM_POINTER):
        root = _read_pointer(ptr)
        if root is not None:
            return root
    return None

def _pick(env: str, sub: str, default: str) -> Path:
    if v := os.environ.get(env):
        return _p(v)
    if root := os.environ.get("AMB_HOME"):
        return _p(root) / sub
    if root := _system_root():
        return root / sub
    return Path(default)


def config_dir() -> Path:
    """Directory that contains config.yaml and .env.

    For --home / systemd installs, config.yaml and .env live at the
    root of AMB_HOME — matching the systemd unit's
    AMB_CONFIG=<home>/config.yaml and EnvironmentFile=<home>/.env.
    For legacy user-scope installs with no AMB_HOME, falls back to the
    platform config directory.
    """
    if v := os.environ.get("AMB_CONFIG_DIR"):
        return _p(v)
    if v := os.environ.get("AMB_HOME"):
        return _p(v)
    if root := _system_root():
        return root
    return Path(_pd.user_config_dir)

def data_dir()    -> Path: return _pick("AMB_DATA_DIR",   "data",   _pd.user_data_dir)
def log_dir()     -> Path: return _pick("AMB_LOG_DIR",    "logs",   _pd.user_log_dir)
def cache_dir()   -> Path: return _pick("AMB_CACHE_DIR",  "cache",  _pd.user_cache_dir)


def download_dir()   -> Path: return data_dir() / "downloads"
def model_dir()      -> Path: return cache_dir() / "models"
def backup_dir()     -> Path: return data_dir() / "execution_backups"
def checkpoint_dir() -> Path: return data_dir() / "agent_checkpoints"


def find_config(explicit: str | None = None) -> Path | None:
    """--config > $AMB_CONFIG > ./config.yaml > <config_dir>/config.yaml"""
    for c in (explicit, os.environ.get("AMB_CONFIG"), "config.yaml", config_dir() / "config.yaml"):
        if c and (p := _p(c)).is_file():
            return p
    return None


def ensure_dirs() -> None:
    for d in (config_dir(), data_dir(), log_dir(), cache_dir(),
              download_dir(), model_dir(), backup_dir(), checkpoint_dir()):
        d.mkdir(parents=True, exist_ok=True)
