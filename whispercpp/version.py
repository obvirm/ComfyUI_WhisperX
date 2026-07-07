"""
Version information — single source of truth.
Reads from pyproject.toml. All other modules import from here.
"""
import tomllib
from pathlib import Path

_VERSION = None
_FALLBACK = "v2.1.5"


def get_version() -> str:
    """Read version from pyproject.toml. Cached after first read."""
    global _VERSION
    if _VERSION is not None:
        return _VERSION
    try:
        pyproject = Path(__file__).resolve().parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        _VERSION = "v" + data["project"]["version"]
    except Exception:
        _VERSION = _FALLBACK
    return _VERSION


if __name__ == "__main__":
    print(get_version())
