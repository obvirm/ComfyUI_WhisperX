"""Tests for whisper_lib module.

The repo root has a ComfyUI ``__init__.py`` (required to register the node),
so collection uses ``--import-mode=importlib`` (see pyproject.toml) and the
root ``conftest.py`` injects the repo root onto ``sys.path``. That makes the
``whispercpp`` subpackage importable without pulling in the node package.
"""
import threading

import pytest

from whispercpp.whisper_lib import WhisperCPP


def test_whispercpp_creation():
    """Test WhisperCPP instance creation."""
    wcpp = WhisperCPP()
    assert wcpp._lib is None
    assert wcpp._ctx is None
    # Lock must be a usable reentrant-free lock object.
    assert hasattr(wcpp._lock, "acquire") and hasattr(wcpp._lock, "release")


def test_whispercpp_singleton():
    """Test that WhisperCPP instances share the same null-lib behaviour."""
    wcpp1 = WhisperCPP()
    wcpp2 = WhisperCPP()
    # Different instances but same behaviour
    assert wcpp1._lib is None
    assert wcpp2._lib is None


def test_thread_safety():
    """Test that the lock is usable (acquire/release)."""
    wcpp = WhisperCPP()

    # Should be able to acquire and release
    assert wcpp._lock.acquire(blocking=False)
    wcpp._lock.release()


def test_empty_audio_validation():
    """Test empty/short audio validation."""
    import numpy as np

    wcpp = WhisperCPP()

    # This path needs a loaded model context; skip if running without one.
    if wcpp._ctx is None:
        pytest.skip("requires a loaded whisper model (no GGML model in env)")

    # Empty audio should raise ValueError
    with pytest.raises(ValueError, match="empty"):
        wcpp.transcribe(np.array([], dtype=np.float32))

    # Short audio should raise ValueError
    with pytest.raises(ValueError, match="too short"):
        wcpp.transcribe(np.array([0.0] * 100, dtype=np.float32))
