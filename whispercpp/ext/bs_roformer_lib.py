"""
ctypes binding untuk bs_roformer — konsisten sama whisper_lib.py.
Cross-platform: .dll (Windows), .so (Linux), .dylib (macOS).
"""

import ctypes, logging, os, platform
from pathlib import Path

logger = logging.getLogger("WhisperCPP.BSRoformerLib")


def _ensure_so_copies(deps_dir):
    """Copy .so to versioned names (e.g., libggml.so -> libggml.so.0)."""
    import shutil
    for src, dst in [("libggml.so", "libggml.so.0"), ("libggml-base.so", "libggml-base.so.0"), ("libggml-cpu.so", "libggml-cpu.so.0")]:
        s = os.path.join(deps_dir, src)
        d = os.path.join(deps_dir, dst)
        if os.path.isfile(s) and not os.path.exists(d):
            try:
                shutil.copy2(s, d)
                logger.info(f"  Created: {dst}")
            except OSError:
                pass

def _ensure_dylib_copies(deps_dir):
    """Copy .dylib to versioned names (e.g., libggml.dylib -> libggml.0.dylib)."""
    import shutil
    for src, dst in [("libggml.dylib", "libggml.0.dylib"), ("libggml-base.dylib", "libggml-base.0.dylib"), ("libggml-cpu.dylib", "libggml-cpu.0.dylib")]:
        s = os.path.join(deps_dir, src)
        d = os.path.join(deps_dir, dst)
        if os.path.isfile(s) and not os.path.exists(d):
            try:
                shutil.copy2(s, d)
                logger.info(f"  Created: {dst}")
            except OSError:
                pass


# ── Platform detection ──
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"
IS_MACOS   = platform.system() == "Darwin"

if IS_WINDOWS: LIB_NAMES = ["bs_roformer.dll", "libbs_roformer.dll"]
elif IS_LINUX: LIB_NAMES = ["libbs_roformer.so"]
elif IS_MACOS: LIB_NAMES = ["libbs_roformer.dylib"]
else:          LIB_NAMES = ["libbs_roformer.so"]

NODE_DIR = Path(__file__).resolve().parent.parent.parent

# ── Module-level state ──
_available = False
_lib = None


def cleanup():
    """Unload bs_roformer library and reset global state."""
    global _lib, _available
    _lib = None
    _available = False


def _find_library():
    """Cari bs_roformer library di semua platform."""
    base = NODE_DIR
    search = []

    for n in LIB_NAMES:
        # Root
        search.append(str(base / n))
        # bs_roformer.cpp build output
        search.append(str(base / "bs_roformer.cpp" / n))
        search.append(str(base / "bs_roformer.cpp" / "build" / n))
        search.append(str(base / "bs_roformer.cpp" / "build" / "src" / n))
        search.append(str(base / "bs_roformer.cpp" / "build" / "lib" / n))
        search.append(str(base / "bs_roformer.cpp" / "build" / "bin" / n))
        # Windows: build/Release/
        search.append(str(base / "bs_roformer.cpp" / "build" / "Release" / n))
        # macOS/Linux with version suffix
        if not IS_WINDOWS:
            import glob
            for p in ["build/*.so*", "build/src/*.so*", "build/lib/*.so*",
                       "build/*.dylib", "build/src/*.dylib", "build/lib/*.dylib"]:
                for f in glob.glob(str(base / "bs_roformer.cpp" / p)):
                    search.append(f)
        # ctypes.util fallback
        stem = n.replace(".dll","").replace(".so","").replace(".dylib","")
        f = ctypes.util.find_library(stem)
        if f:
            search.append(str(f))

    seen = set()
    for p in search:
        if p and os.path.isfile(p) and p not in seen:
            return p
        seen.add(p)
    return None


def _auto_download():
    """Download bs_roformer from GitHub Releases."""
    try:
        from ..auto_download import download_module, check_module_files
        from ..gpu_detect import detect_gpu
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gpu = detect_gpu()
        has_gpu = gpu["backend"] != "cpu"
        # Skip if already present
        if check_module_files("bs_roformer", base_dir, has_gpu=has_gpu):
            return
        download_module("bs_roformer", base_dir, has_gpu=has_gpu)
    except Exception as e:
        logger.warning(f"Auto-download failed: {e}")


def _load():
    global _lib, _available
    if _lib is not None:
        return _lib

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
        logger.error("bs_roformer not found. Build: python build_bs_roformer.py")
        _available = False
        return None

    # Load dependency DLLs
    deps_dir = str(Path(dll_path).parent.resolve())
    if IS_WINDOWS:
        os.add_dll_directory(deps_dir)
    else:
        # Pre-load versioned filenames (LEAF to ROOT order)
        # Ensure versioned copies exist (releases only have base names)
        _ensure_dylib_copies(deps_dir) if IS_MACOS else _ensure_so_copies(deps_dir)
        deps_unix = []
        if IS_MACOS:
            deps_unix = ["libggml-base.0.dylib", "libggml-cpu.0.dylib", "libggml.0.dylib"]
        else:
            deps_unix = ["libggml-base.so.0", "libggml-cpu.so.0", "libggml.so.0"]
        for dep in deps_unix:
            dep_path = os.path.join(deps_dir, dep)
            if os.path.isfile(dep_path):
                try:
                    ctypes.CDLL(dep_path, mode=ctypes.RTLD_GLOBAL)
                    logger.info(f"  Pre-loaded: {dep}")
                except BaseException as e:
                    logger.warning(f"  Pre-load failed: {dep}")
        # If files not found, try to download to deps_dir directly
        if not any(os.path.isfile(os.path.join(deps_dir, dep)) for dep in deps_unix):
            try:
                from ..auto_download import download_module
                download_module("bs_roformer", deps_dir)
                # Retry preload after download
                for dep in deps_unix:
                    dep_path = os.path.join(deps_dir, dep)
                    if os.path.isfile(dep_path):
                        try:
                            ctypes.CDLL(dep_path, mode=ctypes.RTLD_GLOBAL)
                            logger.info(f"  Pre-loaded: {dep}")
                        except BaseException as e:
                            logger.warning(f"  Pre-load failed: {dep}")
            except Exception as e:
                logger.warning(f"  Download to deps_dir failed: {e}")

    logger.info(f"Loading bs_roformer: {dll_path}")
    try:
        _lib = ctypes.CDLL(str(dll_path))
    except Exception as e:
        logger.error(f"Failed to load bs_roformer: {e}")
        _available = False
        return None

    _setup_functions(_lib)
    _available = True
    return _lib


def _setup_functions(lib):
    """Setup argtypes/restype untuk semua C API functions."""

    # ── bs_roformer_init_from_file ──
    f = lib.bs_roformer_init_from_file
    f.argtypes = [ctypes.c_char_p]
    f.restype = ctypes.c_void_p

    # ── bs_roformer_free ──
    f = lib.bs_roformer_free
    f.argtypes = [ctypes.c_void_p]
    f.restype = None

    # ── bs_roformer_sample_rate ──
    f = lib.bs_roformer_sample_rate
    f.argtypes = [ctypes.c_void_p]
    f.restype = ctypes.c_int

    # ── bs_roformer_num_stems ──
    f = lib.bs_roformer_num_stems
    f.argtypes = [ctypes.c_void_p]
    f.restype = ctypes.c_int

    # ── bs_roformer_default_chunk_size ──
    f = lib.bs_roformer_default_chunk_size
    f.argtypes = [ctypes.c_void_p]
    f.restype = ctypes.c_int

    # ── bs_roformer_default_num_overlap ──
    f = lib.bs_roformer_default_num_overlap
    f.argtypes = [ctypes.c_void_p]
    f.restype = ctypes.c_int

    # ── bs_roformer_process ──
    f = lib.bs_roformer_process
    f.argtypes = [
        ctypes.c_void_p,                    # ctx
        ctypes.POINTER(ctypes.c_float),     # input
        ctypes.c_int,                       # n_samples
        ctypes.POINTER(ctypes.c_float),     # output
        ctypes.c_int,                       # chunk_size
        ctypes.c_int,                       # num_overlap
    ]
    f.restype = ctypes.c_int


class BSRoformerContext:
    """Wrapper untuk bs_roformer_context* — otomatis free."""

    def __init__(self, ctx_ptr):
        self._ctx = ctx_ptr

    def free(self):
        if self._ctx:
            _lib.bs_roformer_free(self._ctx)
            self._ctx = None

    def __del__(self):
        self.free()

    @property
    def sample_rate(self) -> int:
        return _lib.bs_roformer_sample_rate(self._ctx)

    @property
    def num_stems(self) -> int:
        return _lib.bs_roformer_num_stems(self._ctx)

    @property
    def default_chunk_size(self) -> int:
        return _lib.bs_roformer_default_chunk_size(self._ctx)

    @property
    def default_num_overlap(self) -> int:
        return _lib.bs_roformer_default_num_overlap(self._ctx)

    def process(self, audio, chunk_size=-1, num_overlap=-1):
        """
        Process interleaved stereo float32 audio.

        Args:
            audio: np.ndarray (n_samples,) float32, interleaved stereo
            chunk_size: -1 = default
            num_overlap: -1 = default

        Returns:
            np.ndarray [num_stems, n_samples] float32
        """
        import numpy as np
        n_samples = len(audio)
        num_stems = self.num_stems
        cs = chunk_size if chunk_size > 0 else self.default_chunk_size
        nol = num_overlap if num_overlap >= 0 else self.default_num_overlap

        audio_f = np.ascontiguousarray(audio, dtype=np.float32)
        output = np.zeros(num_stems * n_samples, dtype=np.float32)

        ret = _lib.bs_roformer_process(
            self._ctx,
            audio_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_samples,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cs, nol,
        )

        if ret != 0:
            raise RuntimeError(f"bs_roformer_process failed: {ret}")

        return output.reshape(num_stems, n_samples)


# ── Public API ──

def bs_roformer_init(model_path):
    """
    Load model GGUF dan return BSRoformerContext.

    Args:
        model_path: path ke file .gguf

    Returns:
        BSRoformerContext atau None kalo gagal
    """
    lib = _load()
    if lib is None:
        return None

    ctx_ptr = lib.bs_roformer_init_from_file(model_path.encode("utf-8"))
    if ctx_ptr is None:
        logger.error(f"Gagal load model: {model_path}")
        return None

    return BSRoformerContext(ctx_ptr)


def is_available():
    """Cek apakah library bs_roformer tersedia."""
    if _lib is None:
        _load()
    return _available
