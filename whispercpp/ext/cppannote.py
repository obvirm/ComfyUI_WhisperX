"""
cpp-annote VAD + diarization via ctypes binding.
Konsisten sama pattern: bs_roformer_lib.py, whisper_lib.py.
"""

import logging
from pathlib import Path

logger = logging.getLogger("WhisperCPP.CppAnnote")

CPPANNOTE_AVAILABLE = False
try:
    from .cpp_annote_lib import cpp_annote_init as _init, is_available as _avail
    CPPANNOTE_AVAILABLE = _avail()
except ImportError:
    pass
except Exception as e:
    logger.warning(f"cpp-annote lib unavailable: {e}")

_ctx = None

def ensure_loaded():
    global _ctx
    if _ctx is None and CPPANNOTE_AVAILABLE:
        _ctx = _init()
        if _ctx is None:
            CPPANNOTE_AVAILABLE = False
    return _ctx


def segment(audio, sr=16000):
    """
    VAD — return list of (start_sec, end_sec) speech segments.
    audio: float32 numpy array (mono) at sr Hz.
    """
    ctx = ensure_loaded()
    if ctx is None:
        logger.warning("cpp-annote not available, treating entire audio as speech")
        if audio is None or len(audio) == 0:
            return []
        return [(0.0, len(audio) / sr)]

    try:
        result = ctx.vad(audio, sr)
        return [(s["start"], s["end"]) for s in result]
    except Exception as e:
        logger.error(f"VAD failed: {e}")
        return [(0.0, len(audio) / sr)]


def diarize(audio, sr=16000):
    """
    Diarization — return list of {"start", "end", "speaker"}.
    audio: float32 numpy array (mono) at sr Hz.
    """
    ctx = ensure_loaded()
    if ctx is None:
        logger.warning("cpp-annote not available")
        if audio is None or len(audio) == 0:
            return []
        return [{"start": 0.0, "end": len(audio) / sr, "speaker": 0}]

    try:
        return ctx.diarize(audio, sr)
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        if audio is not None and len(audio) > 0:
            return [{"start": 0.0, "end": len(audio) / sr, "speaker": 0}]
        return []
