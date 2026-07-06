"""Tests for whisper_lib module."""
import os
import sys
import threading
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_whispercpp_creation():
    """Test WhisperCPP instance creation."""
    from whispercpp.whisper_lib import WhisperCPP
    wcpp = WhisperCPP()
    assert wcpp._lib is None
    assert wcpp._ctx is None
    assert isinstance(wcpp._lock, type(threading.Lock()))


def test_whispercpp_singleton():
    """Test that whispercpp_node uses singleton pattern."""
    from whispercpp.whisper_lib import WhisperCPP
    wcpp1 = WhisperCPP()
    wcpp2 = WhisperCPP()
    # Different instances but same behavior
    assert wcpp1._lib is None
    assert wcpp2._lib is None


def test_thread_safety():
    """Test that lock is thread-safe."""
    from whispercpp.whisper_lib import WhisperCPP
    wcpp = WhisperCPP()
    
    # Lock should be a threading.Lock
    assert isinstance(wcpp._lock, threading.Lock)
    
    # Should be able to acquire and release
    assert wcpp._lock.acquire(blocking=False)
    wcpp._lock.release()


def test_empty_audio_validation():
    """Test empty audio validation."""
    import numpy as np
    from whispercpp.whisper_lib import WhisperCPP
    wcpp = WhisperCPP()
    
    # Empty audio should raise ValueError
    with pytest.raises(ValueError, match="empty"):
        wcpp.transcribe(np.array([], dtype=np.float32))
    
    # Short audio should raise ValueError
    with pytest.raises(ValueError, match="too short"):
        wcpp.transcribe(np.array([0.0] * 100, dtype=np.float32))
