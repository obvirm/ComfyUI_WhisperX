"""
ctypes binding untuk bs_roformer.dll — C API dari bs_roformer.h.
Pola sama kayak whisper_lib.py.
"""

import ctypes, pathlib, sys, os, logging, numpy as np

logger = logging.getLogger("WhisperCPP.BSRoformerLib")

# ── C API ────────────────────────────────────────────────────────────
# Lihat bs_roformer.cpp/include/bs_roformer/bs_roformer.h

NODE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

# Cari DLL
_DLL_PATH = None
_search_paths = [
    NODE_DIR / "bs_roformer.cpp" / "build" / "Release" / "bs_roformer.dll",
    NODE_DIR / "bs_roformer.dll",
]

# Cari dependensi DLL (ggml*.dll) di build atau root
_DEPS_DIRS = [
    NODE_DIR / "bs_roformer.cpp" / "build" / "Release",
    NODE_DIR,
]

_available = False
_lib = None

def _load_deps():
    """Load ggml dependency DLLs sebelum bs_roformer.dll."""
    # Tambah PATH biar Windows bisa nemuin dependency
    for d in _DEPS_DIRS:
        if d.is_dir():
            os.add_dll_directory(str(d.resolve()))

def _load():
    global _lib, _available
    if _lib is not None:
        return _lib

    _load_deps()

    for p in _search_paths:
        if p.is_file():
            _DLL_PATH = str(p.resolve())
            logger.info(f"Loading bs_roformer.dll: {_DLL_PATH}")
            _lib = ctypes.CDLL(_DLL_PATH)
            _setup_functions(_lib)
            _available = True
            return _lib

    logger.error("bs_roformer.dll tidak ditemukan! Build dulu lewat build_bs_roformer_dll.bat")
    _available = False
    return None


def _setup_functions(lib):
    # ── bs_roformer_init_from_file ──
    lib.bs_roformer_init_from_file.argtypes = [ctypes.c_char_p]
    lib.bs_roformer_init_from_file.restype = ctypes.c_void_p

    # ── bs_roformer_free ──
    lib.bs_roformer_free.argtypes = [ctypes.c_void_p]
    lib.bs_roformer_free.restype = None

    # ── bs_roformer_sample_rate ──
    lib.bs_roformer_sample_rate.argtypes = [ctypes.c_void_p]
    lib.bs_roformer_sample_rate.restype = ctypes.c_int

    # ── bs_roformer_num_stems ──
    lib.bs_roformer_num_stems.argtypes = [ctypes.c_void_p]
    lib.bs_roformer_num_stems.restype = ctypes.c_int

    # ── bs_roformer_default_chunk_size ──
    lib.bs_roformer_default_chunk_size.argtypes = [ctypes.c_void_p]
    lib.bs_roformer_default_chunk_size.restype = ctypes.c_int

    # ── bs_roformer_default_num_overlap ──
    lib.bs_roformer_default_num_overlap.argtypes = [ctypes.c_void_p]
    lib.bs_roformer_default_num_overlap.restype = ctypes.c_int

    # ── bs_roformer_process ──
    # int bs_roformer_process(ctx, input, n_samples, output, chunk_size, num_overlap)
    lib.bs_roformer_process.argtypes = [
        ctypes.c_void_p,      # ctx
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.c_int,         # n_samples
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,         # chunk_size
        ctypes.c_int,         # num_overlap
    ]
    lib.bs_roformer_process.restype = ctypes.c_int


class BSRoformerContext:
    """Wrapper untuk bs_roformer_context* — otomatis free pas cleanup."""

    def __init__(self, ctx_ptr):
        self._ctx = ctx_ptr
        self._lib = _lib

    def free(self):
        if self._ctx:
            self._lib.bs_roformer_free(self._ctx)
            self._ctx = None

    def __del__(self):
        self.free()

    # ── properties ──
    @property
    def sample_rate(self) -> int:
        return self._lib.bs_roformer_sample_rate(self._ctx)

    @property
    def num_stems(self) -> int:
        return self._lib.bs_roformer_num_stems(self._ctx)

    @property
    def default_chunk_size(self) -> int:
        return self._lib.bs_roformer_default_chunk_size(self._ctx)

    @property
    def default_num_overlap(self) -> int:
        return self._lib.bs_roformer_default_num_overlap(self._ctx)

    # ── process ──
    def process(self, audio: np.ndarray, chunk_size: int = -1, num_overlap: int = -1) -> np.ndarray:
        """
        Process interleaved stereo float32 audio.

        Args:
            audio: np.ndarray (n_samples,) interleaved stereo float32
            chunk_size: -1 = default
            num_overlap: -1 = default

        Returns:
            np.ndarray [num_stems, n_samples] float32 — stem 0 = vocals
        """
        n_samples = len(audio)
        num_stems = self.num_stems
        cs = chunk_size if chunk_size > 0 else self.default_chunk_size
        nol = num_overlap if num_overlap >= 0 else self.default_num_overlap

        # Prepare
        audio_f = np.ascontiguousarray(audio, dtype=np.float32)
        output = np.zeros(num_stems * n_samples, dtype=np.float32)

        input_ptr = audio_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = self._lib.bs_roformer_process(
            self._ctx, input_ptr, n_samples, output_ptr, cs, nol
        )

        if ret != 0:
            raise RuntimeError(f"bs_roformer_process gagal: return code {ret}")

        # Reshape: [num_stems, n_samples]
        return output.reshape(num_stems, n_samples)


# ── Public API ──

def bs_roformer_init(model_path: str) -> BSRoformerContext:
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

    mp = model_path.encode("utf-8")
    ctx_ptr = lib.bs_roformer_init_from_file(mp)
    if ctx_ptr is None:
        logger.error(f"Gagal load model: {model_path}")
        return None

    return BSRoformerContext(ctx_ptr)


# Call _load() saat import biar is_available() akurat
_load()

def is_available() -> bool:
    return _available
