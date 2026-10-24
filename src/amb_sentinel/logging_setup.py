"""One logging configuration module. No other module may call basicConfig or
create FileHandler('logs/...') directly.

Logger name → file name is fixed in LOG_FILE_MAP. Every logger listed there
gets a RotatingFileHandler when `configure()` runs, plus a shared console
handler when stderr is a TTY. The root logger also gets a catch-all
`sentinel.log` file so unmapped loggers still reach a file.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock
from typing import Dict, Set

from amb_sentinel import paths

LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Logger name -> file name. Fully-qualified names for modules that use
# getLogger(__name__), so their records land in the right file.
LOG_FILE_MAP: Dict[str, str] = {
    # Core service
    "ai_agent":                              "ai_agent.log",
    "amb_sentinel.agent.ai_agent":           "ai_agent.log",
    "Main":                                  "main.log",
    "AgentExecutor":                         "agent_executor.log",
    "DataAnalyzer":                          "data_analyzer.log",
    "System_Metric":                         "system_metric.log",
    "BackupRetrier":                         "agent_executor.log",
    "MemoryMonitor":                         "system_metric.log",
    "ModelManager":                          "modelmanager.log",
    "Registrar":                             "registrar.log",
    "Initials":                              "initials.log",
    "CircuitBreaker":                        "agent_executor.log",
    "InstructionValidator":                  "agent_executor.log",
    # Tool modules
    "Video":                                 "video.log",
    "Scrape":                                "scrape.log",
    "Utils":                                 "utils.log",
    "Transcribe":                            "transcribe.log",
    "Download":                              "download.log",
    "Extract":                               "extract.log",
    "Pipeline":                              "pipeline.log",
    "HTTP":                                  "http.log",
    "FindLink":                              "findlink.log",
    "DBTools":                               "dbtools.log",
    # Runtime modules (getLogger(__name__) — full dotted name required)
    "amb_sentinel.agent.amb_runtime":        "runtime.log",
    "amb_sentinel.agent.amb_actions":        "runtime.log",
    "amb_sentinel.llm.amb_llm_core":         "llm_transport.log",
    "amb_sentinel.llm.llm_cloud_adapter":    "llm_transport.log",
    # Scheduler
    "apscheduler":                           "apscheduler.log",
    "apscheduler.scheduler":                 "apscheduler.log",
    "apscheduler.executors":                 "apscheduler.log",
    # Catch-all
    "sentinel":                              "sentinel.log",
}

_NOISY = (
    "botocore", "urllib3", "matplotlib", "PIL",
    "transformers", "httpx", "aiohttp", "sentence_transformers",
)

_lock = RLock()
_handlers: Dict[Path, logging.Handler] = {}
_attached: Set[str] = set()
_console_handler: logging.Handler | None = None
_root_file_attached = False


def _file_formatter() -> logging.Formatter:
    return logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)


def _console_formatter() -> logging.Formatter:
    try:
        import colorama
        colorama.init(autoreset=True)
        from colorama import Fore, Style

        class _C(logging.Formatter):
            _COLORS = {
                "DEBUG": Fore.CYAN, "INFO": Fore.GREEN,
                "WARNING": Fore.YELLOW, "ERROR": Fore.RED,
                "CRITICAL": Fore.RED + Style.BRIGHT,
            }

            def format(self, r):
                color = self._COLORS.get(r.levelname, "")
                return f"{color}{super().format(r)}{Style.RESET_ALL}"

        return _C(LOG_FORMAT, datefmt=DATE_FORMAT)
    except Exception:
        return _file_formatter()


def _handler_for(path: Path, level: int = logging.NOTSET) -> logging.Handler:
    """Shared RotatingFileHandler per absolute path. delay=True so files are
    only created on first log, keeping empty logs = tool never ran."""
    key = path.resolve()
    with _lock:
        if key in _handlers:
            return _handlers[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        h = RotatingFileHandler(
            path, maxBytes=20 * 1024 * 1024, backupCount=5,
            encoding="utf-8", delay=True,
        )
        h.setFormatter(_file_formatter())
        h.setLevel(level)
        _handlers[key] = h
        return h


def _get_console_handler() -> logging.Handler:
    global _console_handler
    with _lock:
        if _console_handler is None:
            _console_handler = logging.StreamHandler(sys.stderr)
            _console_handler.setFormatter(_console_formatter())
            _console_handler.setLevel(logging.NOTSET)
        return _console_handler


def _has_file_handler(lg: logging.Logger, path: Path) -> bool:
    target = path.resolve()
    for h in lg.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename).resolve() == target:
                    return True
            except Exception:
                pass
    return False


def _ensure_root_file_handler() -> None:
    global _root_file_attached
    with _lock:
        root = logging.getLogger()
        target = (paths.log_dir() / "sentinel.log").resolve()
        if _root_file_attached:
            return
        if _has_file_handler(root, target):
            _root_file_attached = True
            return
        root.addHandler(_handler_for(paths.log_dir() / "sentinel.log", logging.NOTSET))
        _root_file_attached = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with its file handler attached. Idempotent; safe to
    call from anywhere, including after importlib.reload."""
    with _lock:
        lg = logging.getLogger(name)
        if name in _attached:
            return lg
        filename = LOG_FILE_MAP.get(name, f"{name.lower()}.log")
        path = paths.log_dir() / filename
        if not _has_file_handler(lg, path):
            lg.addHandler(_handler_for(path, logging.NOTSET))
        lg.setLevel(logging.INFO)
        lg.propagate = False
        _attached.add(name)
        return lg


def configure(level: str = "INFO", console: bool = True) -> None:
    """Apply a global level everywhere and attach console + file handlers.

    Re-entrant: calling `configure("DEBUG")` after `configure("INFO")` upgrades
    the level on every mapped logger and adds the console handler if it is
    newly enabled. Duplicate handlers are not created.
    """
    with _lock:
        paths.ensure_dirs()
        lvl = level.upper()

        root = logging.getLogger()
        root.setLevel(lvl)

        # Strip stale console handlers from previous configure() calls on root.
        for h in list(root.handlers):
            if not isinstance(h, logging.FileHandler):
                root.removeHandler(h)

        use_console = bool(console and sys.stderr.isatty())
        if use_console:
            root.addHandler(_get_console_handler())

        _ensure_root_file_handler()

        for name in LOG_FILE_MAP:
            lg = get_logger(name)
            lg.setLevel(lvl)
            # Reset console handlers on named loggers (respect new use_console).
            for h in list(lg.handlers):
                if not isinstance(h, logging.FileHandler):
                    lg.removeHandler(h)
            if use_console:
                lg.addHandler(_get_console_handler())

        for noisy in _NOISY:
            logging.getLogger(noisy).setLevel(logging.WARNING)
