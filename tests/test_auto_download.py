"""Tests for auto_download module."""
import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_version_detection():
    """Test version detection from pyproject.toml."""
    from whispercpp.auto_download import _get_version
    version = _get_version()
    assert version.startswith("v")
    assert len(version) > 2


def test_gpu_detection():
    """Test GPU detection."""
    from whispercpp.gpu_detect import detect_gpu
    gpu = detect_gpu()
    assert gpu["platform"] in ["Windows", "Linux", "Darwin"]
    assert gpu["backend"] in ["cuda", "vulkan", "opencl", "cpu", "metal"]
    assert isinstance(gpu["has_nvidia"], bool)


def test_check_module_files():
    """Test module file checking."""
    from whispercpp.auto_download import check_module_files
    # Non-existent directory should return False
    assert check_module_files("whisper", "/tmp/nonexistent") == False


def test_safety_checks():
    """Test safety check functions."""
    from whispercpp.auto_download import _check_disk_space, _verify_checksum, _get_file_lock
    
    # Disk space check
    assert _check_disk_space("/tmp", min_mb=1) == True
    
    # Checksum with no hash = True
    assert _verify_checksum("/tmp/test", expected_sha256=None) == True
    
    # File lock
    import threading
    lock = _get_file_lock("/tmp/test")
    assert isinstance(lock, type(threading.Lock()))
