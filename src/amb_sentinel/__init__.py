"""AMB-Sentinel package — light, only exposes __version__."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("amb-sentinel")
except PackageNotFoundError:      # running from a raw source tree
    __version__ = "1.0.0"

__all__ = ["__version__"]
