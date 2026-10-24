"""Lazy, guarded imports for optional heavy dependencies."""
from __future__ import annotations

import importlib


def require(module: str, extra: str):
    """Import *module* or raise ImportError pointing at `pip install 'amb-sentinel[extra]'`."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"'{module}' is required for this feature. "
            f"Install it with: pip install 'amb-sentinel[{extra}]'"
        ) from e


class _LazyModule:
    """Proxy that imports on first attribute access.

    Used to keep torch and other heavy deps out of the core-install import
    path. Attribute access triggers `require()`, so the error names the exact
    extra to install if the module is missing.
    """
    def __init__(self, name: str, extra: str):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_extra", extra)
        object.__setattr__(self, "_mod", None)

    def __getattr__(self, attr):
        if object.__getattribute__(self, "_mod") is None:
            object.__setattr__(
                self, "_mod",
                require(object.__getattribute__(self, "_name"),
                        object.__getattribute__(self, "_extra"))
            )
        return getattr(object.__getattribute__(self, "_mod"), attr)


def lazy(name: str, extra: str) -> "_LazyModule":
    return _LazyModule(name, extra)
