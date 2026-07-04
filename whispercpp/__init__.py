import ctypes, logging, os, platform
from pathlib import Path

# Pre-load correct onnxruntime.dll BEFORE any other module loads a different version
_NODE_DIR = Path(__file__).resolve().parent.parent
_ORT_CANDIDATES = [
    _NODE_DIR / "onnxruntime.dll",
    _NODE_DIR / "cpp-annote" / "build" / "Release" / "onnxruntime.dll",
]
for _p in _ORT_CANDIDATES:
    if _p.exists():
        try:
            ctypes.CDLL(str(_p))
            logging.getLogger("WhisperCPP").info(f"Pre-loaded onnxruntime: {_p}")
            if platform.system() == "Windows":
                os.add_dll_directory(str(_p.parent.resolve()))
        except Exception as _e:
            logging.getLogger("WhisperCPP").warning(f"onnxruntime pre-load failed: {_e}")
        break

from .whisper_lib import WhisperCPP
from .audio import AudioProcessor

__all__ = ["WhisperCPP", "AudioProcessor"]
