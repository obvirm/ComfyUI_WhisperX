"""
ctypes binding untuk cpp_annote.dll — C API dari cpp-annote-c.h.
VAD + diarization via ONNX Runtime DLL. Pola sama kayak whisper_lib.py + bs_roformer_lib.py.
"""

import ctypes, json, logging, os, platform
from pathlib import Path

logger = logging.getLogger("WhisperCPP.CPPAnnoteLib")

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"
IS_MACOS   = platform.system() == "Darwin"

if IS_WINDOWS: LIB_NAMES = ["cpp_annote.dll", "libcpp_annote.dll"]
elif IS_LINUX: LIB_NAMES = ["libcpp_annote.so"]
elif IS_MACOS: LIB_NAMES = ["libcpp_annote.dylib"]
else:          LIB_NAMES = ["libcpp_annote.so"]

NODE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = NODE_DIR / "cpp-annote" / "artifacts"

_available = False
_lib = None


def cleanup():
    """Unload cpp-annote library and reset global state."""
    global _lib, _available
    _lib = None
    _available = False


def _find_library():
    base = NODE_DIR
    search = []
    for n in LIB_NAMES:
        search.append(str(base / n))
        search.append(str(base / "cpp-annote" / n))
        search.append(str(base / "cpp-annote" / "build" / n))
        search.append(str(base / "cpp-annote" / "build" / "src" / n))
        search.append(str(base / "cpp-annote" / "build" / "lib" / n))
        search.append(str(base / "cpp-annote" / "build" / "bin" / n))
        if IS_WINDOWS:
            search.append(str(base / "cpp-annote" / "build" / "Release" / n))
        if not IS_WINDOWS:
            import glob
            for p in ["build/*.so*", "build/src/*.so*", "build/lib/*.so*",
                       "build/*.dylib", "build/src/*.dylib", "build/lib/*.dylib"]:
                for f in glob.glob(str(base / "cpp-annote" / p)):
                    search.append(f)
        stem = n.replace(".dll","").replace(".so","").replace(".dylib","")
        f = ctypes.util.find_library(stem)
        if f: search.append(str(f))
    seen = set()
    for p in search:
        if p and os.path.isfile(p) and p not in seen:
            return p
        seen.add(p)
    return None


def _auto_download():
    """Download cpp_annote from GitHub Releases."""
    try:
        from ..auto_download import download_module, check_module_files
        from ..gpu_detect import detect_gpu
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gpu = detect_gpu()
        has_gpu = gpu["backend"] != "cpu"
        # Skip if already present
        if check_module_files("cpp_annote", base_dir, has_gpu=has_gpu):
            return
        download_module("cpp_annote", base_dir, has_gpu=has_gpu)
    except Exception as e:
        logger.warning(f"Auto-download failed: {e}")


def _load():
    global _lib, _available
    if _lib is not None: return _lib
    
    # Check version & update if outdated
    try:
        from ..auto_download import check_version_and_update
        from ..gpu_detect import detect_gpu
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gpu = detect_gpu()
        check_version_and_update(base_dir, has_gpu=gpu["backend"] != "cpu")
    except Exception as e:
        logger.debug(f"Version check skipped: {e}")
    
    dll_path = _find_library()
    if dll_path is None:
        # Try auto-download
        _auto_download()
        dll_path = _find_library()
    if dll_path is None:
        logger.error("cpp_annote not found. Build: build_cpp_annote.py")
        _available = False
        return None
    if IS_WINDOWS:
        deps_dir = Path(dll_path).parent.resolve()
        # Pre-load onnxruntime.dll 1.27.0 BEFORE cpp_annote.dll
        # so Windows doesn't use the already-loaded system version
        ort_dll = deps_dir / "onnxruntime.dll"
        if ort_dll.exists():
            try:
                ctypes.CDLL(str(ort_dll))
                logger.info(f"Pre-loaded onnxruntime: {ort_dll}")
            except Exception as e:
                logger.warning(f"Failed to pre-load onnxruntime: {e}")
        os.add_dll_directory(str(deps_dir))
    logger.info(f"Loading cpp_annote: {dll_path}")
    try:
        _lib = ctypes.CDLL(str(dll_path))
    except Exception as e:
        logger.error(f"Failed: {e}")
        _available = False
        return None
    _setup_functions(_lib)
    _available = True
    return _lib


def _setup_functions(lib):
    f = lib.cpp_annote_init
    f.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    f.restype = ctypes.c_void_p

    f = lib.cpp_annote_free
    f.argtypes = [ctypes.c_void_p]
    f.restype = None

    f = lib.cpp_annote_diarize
    f.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                  ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    f.restype = ctypes.c_int

    f = lib.cpp_annote_vad
    f.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                  ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    f.restype = ctypes.c_int

    f = lib.cpp_annote_free_string
    f.argtypes = [ctypes.c_char_p]
    f.restype = None


class CppAnnoteContext:
    def __init__(self, ctx_ptr):
        self._ctx = ctx_ptr

    def free(self):
        if self._ctx:
            _lib.cpp_annote_free(self._ctx)
            self._ctx = None

    def __del__(self):
        self.free()

    def diarize(self, audio, sr=16000):
        """Returns list of {start, end, speaker} dicts."""
        import numpy as np
        audio_f = np.ascontiguousarray(audio.ravel(), dtype=np.float32)
        n = len(audio_f)
        c_json = ctypes.c_char_p()
        ret = _lib.cpp_annote_diarize(
            self._ctx,
            audio_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, sr,
            ctypes.byref(c_json)
        )
        if ret != 0:
            raise RuntimeError(f"cpp_annote_diarize failed: {ret}")
        try:
            return json.loads(c_json.value.decode("utf-8"))
        finally:
            _lib.cpp_annote_free_string(c_json)

    def vad(self, audio, sr=16000):
        """Returns list of {start, end} dicts."""
        import numpy as np
        audio_f = np.ascontiguousarray(audio.ravel(), dtype=np.float32)
        n = len(audio_f)
        c_json = ctypes.c_char_p()
        ret = _lib.cpp_annote_vad(
            self._ctx,
            audio_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, sr,
            ctypes.byref(c_json)
        )
        if ret != 0:
            raise RuntimeError(f"cpp_annote_vad failed: {ret}")
        try:
            return json.loads(c_json.value.decode("utf-8"))
        finally:
            _lib.cpp_annote_free_string(c_json)


def _auto_download_models():
    """Download ONNX models if missing."""
    try:
        from ..auto_download import download_module, check_module_files
        from ..gpu_detect import detect_gpu
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gpu = detect_gpu()
        has_gpu = gpu["backend"] != "cpu"
        # Skip if already present
        if check_module_files("cpp_annote_models", base_dir, has_gpu=has_gpu):
            return True
        return download_module("cpp_annote_models", base_dir, has_gpu=has_gpu)
    except Exception as e:
        logger.warning(f"Auto-download models failed: {e}")
        return False


def cpp_annote_init(seg_path=None, emb_path=None):
    """
    Load VAD/diarization engine. Paths default ke artifacts/.
    Returns CppAnnoteContext or None.
    """
    lib = _load()
    if lib is None: return None

    if seg_path is None:
        seg_path = str(ARTIFACTS_DIR / "community1-segmentation.onnx")
    if emb_path is None:
        emb_path = str(ARTIFACTS_DIR / "community1-embedding.onnx")

    # Auto-download ONNX models if missing
    if not os.path.isfile(seg_path) or not os.path.isfile(emb_path):
        logger.info("ONNX models missing, downloading...")
        _auto_download_models()

    ctx_ptr = lib.cpp_annote_init(seg_path.encode("utf-8"), emb_path.encode("utf-8"))
    if ctx_ptr is None:
        logger.error(f"cpp_annote_init failed. Check paths:\n  {seg_path}\n  {emb_path}")
        return None
    return CppAnnoteContext(ctx_ptr)


def is_available():
    if _lib is None: _load()
    return _available
