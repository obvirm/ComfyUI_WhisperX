"""Pytest configuration for the ``tests`` package.

The repository root ships a ComfyUI ``__init__.py`` (required so ComfyUI can
register the node). That file does ``from .whispercpp_node import ...`` which
needs ComfyUI on ``sys.path`` and must NOT be imported while pytest collects
the unit tests — otherwise collection fails with an ``ImportError``.

Because this conftest lives in ``tests/`` (which is NOT a package — no
``__init__.py`` of its own) it is imported as a plain module and does not
trigger the root package ``__init__.py``. At configure time we temporarily
hide the root ``__init__.py`` so pytest does not treat the repo root as a
package during collection, then restore it at unconfigure time. The repo root
is still added to ``sys.path`` so the ``whispercpp`` subpackage is importable.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_ROOT_INIT = os.path.join(_ROOT, "__init__.py")
_ROOT_INIT_HIDDEN = os.path.join(_ROOT, "__init__.py.pytest_hidden")


def pytest_configure(config):
    # Temporarily hide the ComfyUI node package __init__.py so pytest does not
    # treat the repo root as a package (which would import it on collection).
    if os.path.exists(_ROOT_INIT) and not os.path.exists(_ROOT_INIT_HIDDEN):
        try:
            os.rename(_ROOT_INIT, _ROOT_INIT_HIDDEN)
        except OSError:
            pass


def pytest_unconfigure(config):
    # Restore the ComfyUI node package __init__.py.
    if os.path.exists(_ROOT_INIT_HIDDEN):
        try:
            os.rename(_ROOT_INIT_HIDDEN, _ROOT_INIT)
        except OSError:
            pass
