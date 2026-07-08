import ctypes, ctypes.util, logging, os, platform
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("WhisperCPP")

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"

WHISPER_SAMPLE_RATE = 16000
WHISPER_SAMPLING_GREEDY = 0
WHISPER_SAMPLING_BEAM_SEARCH = 1

if IS_WINDOWS: LIB_NAMES = ["whisper.dll", "libwhisper.dll"]
elif IS_LINUX: LIB_NAMES = ["libwhisper.so"]
elif IS_MACOS: LIB_NAMES = ["libwhisper.dylib"]
else: LIB_NAMES = ["libwhisper.so"]

class WhisperAhead(ctypes.Structure):
    """whisper_ahead: {n_head, n_layer}"""
    _fields_ = [("n_head", ctypes.c_int), ("n_layer", ctypes.c_int)]

class WhisperAheads(ctypes.Structure):
    """whisper_aheads: {n_heads, heads}"""
    _fields_ = [("n_heads", ctypes.c_int), ("heads", ctypes.POINTER(WhisperAhead))]

class WhisperContextParams(ctypes.Structure):
    _fields_ = [
        ("use_gpu", ctypes.c_bool), ("flash_attn", ctypes.c_bool), ("gpu_device", ctypes.c_int),
        ("dtw_token_timestamps", ctypes.c_bool), ("dtw_aheads_preset", ctypes.c_int), ("dtw_n_top", ctypes.c_int),
        ("dtw_aheads", WhisperAheads),
        ("dtw_mem_size", ctypes.c_size_t),
    ]

class WhisperTokenData(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int32), ("tid", ctypes.c_int32), ("p", ctypes.c_float), ("plog", ctypes.c_float),
        ("pt", ctypes.c_float), ("ptsum", ctypes.c_float), ("t0", ctypes.c_int64), ("t1", ctypes.c_int64),
        ("t_dtw", ctypes.c_int64), ("vlen", ctypes.c_float),
    ]

class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int32), ("n_max_text_ctx", ctypes.c_int32), ("offset_ms", ctypes.c_int32), ("duration_ms", ctypes.c_int32),
        ("translate", ctypes.c_bool), ("no_context", ctypes.c_bool), ("no_timestamps", ctypes.c_bool),
        ("single_segment", ctypes.c_bool), ("print_special", ctypes.c_bool), ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool), ("print_timestamps", ctypes.c_bool),
        ("token_timestamps", ctypes.c_bool), ("thold_pt", ctypes.c_float), ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int32), ("split_on_word", ctypes.c_bool), ("max_tokens", ctypes.c_int32),
        ("debug_mode", ctypes.c_bool), ("audio_ctx", ctypes.c_int32), ("tdrz_enable", ctypes.c_bool),
        ("suppress_regex", ctypes.c_char_p),
        ("initial_prompt", ctypes.c_char_p), ("carry_initial_prompt", ctypes.c_bool),
        ("prompt_tokens", ctypes.POINTER(ctypes.c_int32)), ("prompt_n_tokens", ctypes.c_int32),
        ("language", ctypes.c_char_p), ("detect_language", ctypes.c_bool),
        ("suppress_blank", ctypes.c_bool), ("suppress_nst", ctypes.c_bool),
        ("temperature", ctypes.c_float), ("max_initial_ts", ctypes.c_float), ("length_penalty", ctypes.c_float),
        ("temperature_inc", ctypes.c_float), ("entropy_thold", ctypes.c_float), ("logprob_thold", ctypes.c_float),
        ("no_speech_thold", ctypes.c_float),
        ("best_of", ctypes.c_int32), ("beam_size", ctypes.c_int32), ("patience", ctypes.c_float),
        ("new_segment_callback", ctypes.c_void_p), ("new_segment_callback_user_data", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p), ("progress_callback_user_data", ctypes.c_void_p),
        ("encoder_begin_callback", ctypes.c_void_p), ("encoder_begin_callback_user_data", ctypes.c_void_p),
        ("abort_callback", ctypes.c_void_p), ("abort_callback_user_data", ctypes.c_void_p),
        ("logits_filter_callback", ctypes.c_void_p), ("logits_filter_callback_user_data", ctypes.c_void_p),
        ("grammar_rules", ctypes.c_void_p), ("n_grammar_rules", ctypes.c_size_t), ("i_start_rule", ctypes.c_size_t), ("grammar_penalty", ctypes.c_float),
        ("vad", ctypes.c_bool), ("vad_model_path", ctypes.c_char_p),
        ("vad_threshold", ctypes.c_float), ("vad_min_speech_duration_ms", ctypes.c_int), ("vad_min_silence_duration_ms", ctypes.c_int),
        ("vad_max_speech_duration_s", ctypes.c_float), ("vad_speech_pad_ms", ctypes.c_int), ("vad_samples_overlap", ctypes.c_float),
    ]

class WhisperCPP:
    def __init__(self, lib_path=None):
        import threading
        self._lib = None; self._ctx = None; self._lib_path = lib_path; self._model_path = None
        self._lock = threading.Lock()  # Thread safety for transcribe()

    def _find_library(self):
        if self._lib_path and os.path.isfile(self._lib_path):
            return self._lib_path
        base_dir = Path(__file__).resolve().parent.parent
        search = []
        if self._lib_path: search.append(self._lib_path)
        for n in LIB_NAMES:
            # Root
            search.append(str(base_dir / n))
            # whisper.cpp build output (all platforms)
            search.append(str(base_dir / "whisper.cpp" / n))
            search.append(str(base_dir / "whisper.cpp" / "build" / n))
            search.append(str(base_dir / "whisper.cpp" / "build" / "src" / n))
            search.append(str(base_dir / "whisper.cpp" / "build" / "lib" / n))
            search.append(str(base_dir / "whisper.cpp" / "build" / "bin" / n))
            # Windows: build/bin/Release/
            search.append(str(base_dir / "whisper.cpp" / "build" / "bin" / "Release" / n))
            search.append(str(base_dir / "whisper.cpp" / "build" / "src" / "Release" / n))
            # macOS: .dylib with version suffix
            if IS_MACOS:
                import glob
                for p in ["build/bin/*.dylib", "build/src/*.dylib", "build/lib/*.dylib"]:
                    for f in glob.glob(str(base_dir / "whisper.cpp" / p)):
                        search.append(f)
            # Linux: .so with version suffix
            if IS_LINUX:
                import glob
                for p in ["build/bin/libwhisper.so.*", "build/src/libwhisper.so.*", "build/lib/libwhisper.so.*"]:
                    for f in glob.glob(str(base_dir / "whisper.cpp" / p)):
                        search.append(f)
            # ctypes.util fallback
            stem = n.replace(".dll","").replace(".so","").replace(".dylib","")
            f = ctypes.util.find_library(stem)
            if f: search.append(str(f))
        seen = set()
        for p in search:
            if p and os.path.isfile(p) and p not in seen:
                return p
            seen.add(p)
        raise RuntimeError(f"Cannot find whisper lib. Build first: python build_whisper_cpp.py")

    def load_library(self, lib_path=None):
        if self._lib is not None: return
        if lib_path: self._lib_path = lib_path
        
        # Check version & update if outdated
        try:
            from .auto_download import check_version_and_update
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            from .gpu_detect import detect_gpu
            gpu = detect_gpu()
            check_version_and_update(base_dir, has_gpu=gpu["backend"] != "cpu")
        except Exception as e:
            logger.debug(f"Version check skipped: {e}")
        
        logger.info("Step 7: Finding whisper library")
        try:
            lib = self._find_library()
        except RuntimeError:
            logger.info("Step 7a: Library not found, trying auto-download")
            # Step 1: Try auto-download from GitHub Releases
            if self._auto_download():
                try:
                    lib = self._find_library()
                except RuntimeError:
                    logger.warning("Download succeeded but library not found")
                    lib = None
            else:
                # Step 2: Try auto-build (requires CMake + compiler)
                logger.info("Step 7b: Trying auto-build")
                self._auto_build()
                try:
                    lib = self._find_library()
                except RuntimeError:
                    lib = None
        if lib is None:
            raise RuntimeError("whisper library not found. Build: python build_whisper_cpp.py")
        logger.info(f"Step 8: Loading whisper library: {lib}")
        
        # Preload ggml deps on macOS/Linux (same as bs_roformer)
        if not IS_WINDOWS:
            deps_dir = str(Path(lib).parent.resolve())
            # Ensure versioned copies exist
            if IS_MACOS:
                for src, dst in [("libggml.dylib", "libggml.0.dylib"), ("libggml-base.dylib", "libggml-base.0.dylib"), ("libggml-cpu.dylib", "libggml-cpu.0.dylib"), ("libggml-blas.dylib", "libggml-blas.0.dylib")]:
                    s = os.path.join(deps_dir, src); d = os.path.join(deps_dir, dst)
                    if os.path.isfile(s) and not os.path.exists(d):
                        try:
                            import shutil; shutil.copy2(s, d)
                            logger.info(f"  Created: {dst}")
                            # Fix install_name so preload with RTLD_GLOBAL works
                            import subprocess
                            subprocess.run(["install_name_tool", "-id", f"@loader_path/{dst}", d],
                                           capture_output=True, timeout=10)
                        except:
                            pass
            ggml_deps = ["libggml.0.dylib", "libggml-base.0.dylib", "libggml-cpu.0.dylib", "libggml-blas.0.dylib"] if IS_MACOS else ["libggml.so.0", "libggml-base.so.0", "libggml-cpu.so.0"]
            for dep in ggml_deps:
                dep_path = os.path.join(deps_dir, dep)
                if os.path.isfile(dep_path):
                    try:
                        ctypes.CDLL(dep_path, mode=ctypes.RTLD_GLOBAL)
                        logger.info(f"  Pre-loaded: {dep}")
                    except BaseException:
                        pass
        
        # macOS: fix @rpath in libwhisper.dylib to use @loader_path
        if IS_MACOS:
            try:
                import subprocess
                for ref in ["@rpath/libggml.0.dylib", "@rpath/libggml-base.0.dylib", "@rpath/libggml-cpu.0.dylib", "@rpath/libggml-blas.0.dylib"]:
                    loader_ref = ref.replace("@rpath/", "@loader_path/")
                    subprocess.run(["install_name_tool", "-change", ref, loader_ref, lib],
                                   capture_output=True, timeout=10)
            except Exception:
                pass
        
        self._lib = ctypes.cdll.LoadLibrary(lib)
        logger.info("Step 9: Setting up functions")
        self._setup_functions()
        # Log version
        try:
            ver = self._lib.whisper_version().decode()
            logger.info(f"Step 9b: whisper version = {ver}")
        except Exception:
            logger.debug("Could not get whisper version")

    def _auto_download(self) -> bool:
        """Download DLLs from GitHub Releases based on detected GPU."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            from .gpu_detect import detect_gpu
            from .auto_download import download_module, check_module_files, get_latest_version
            gpu = detect_gpu()
            has_gpu = gpu["backend"] != "cpu"
            # Skip if already present
            if check_module_files("whisper", base_dir, has_gpu=has_gpu):
                return True
            ver = get_latest_version()
            logger.info(f"Downloading DLLs from {ver}...")
            return download_module("whisper", base_dir, version=ver, has_gpu=has_gpu)
        except Exception as e:
            logger.warning(f"Auto-download failed: {e}")
            return False

    def _auto_build(self):
        """Try auto-build (may fail without CMake/compiler)."""
        import subprocess, sys
        build_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build_whisper_cpp.py")
        if not os.path.isfile(build_script):
            return
        logger.info("Attempting auto-build (requires CMake + compiler)...")
        try:
            r = subprocess.run([sys.executable, build_script], cwd=os.path.dirname(build_script),
                timeout=600, capture_output=True, text=True)
            if r.returncode == 0:
                logger.info("Auto-build succeeded")
            else:
                logger.warning(f"Auto-build failed: {r.stderr[:200]}")
        except Exception as e:
            logger.warning(f"Auto-build error: {e}")

    def _setup_functions(self):
        L = self._lib
        L.whisper_version.restype = ctypes.c_char_p
        L.whisper_init_from_file_with_params.argtypes = [ctypes.c_char_p, WhisperContextParams]
        L.whisper_init_from_file_with_params.restype = ctypes.c_void_p
        L.whisper_free.argtypes = [ctypes.c_void_p]; L.whisper_free.restype = None
        L.whisper_context_default_params.restype = WhisperContextParams
        L.whisper_full_default_params.argtypes = [ctypes.c_int]
        L.whisper_full_default_params.restype = WhisperFullParams
        L.whisper_full.argtypes = [ctypes.c_void_p, WhisperFullParams, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        L.whisper_full.restype = ctypes.c_int
        L.whisper_full_n_segments.argtypes = [ctypes.c_void_p]; L.whisper_full_n_segments.restype = ctypes.c_int
        L.whisper_full_lang_id.argtypes = [ctypes.c_void_p]; L.whisper_full_lang_id.restype = ctypes.c_int
        L.whisper_full_get_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int]; L.whisper_full_get_segment_text.restype = ctypes.c_char_p
        for f in ["whisper_full_get_segment_t0","whisper_full_get_segment_t1"]:
            getattr(L, f).argtypes = [ctypes.c_void_p, ctypes.c_int]; getattr(L, f).restype = ctypes.c_int64
        L.whisper_full_get_segment_speaker_turn_next.argtypes = [ctypes.c_void_p, ctypes.c_int]; L.whisper_full_get_segment_speaker_turn_next.restype = ctypes.c_bool
        L.whisper_full_n_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int]; L.whisper_full_n_tokens.restype = ctypes.c_int
        for f in ["whisper_full_get_token_text","whisper_full_get_token_id","whisper_full_get_token_p","whisper_full_get_token_t0","whisper_full_get_token_t1"]:
            getattr(L, f).argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]; getattr(L, f).restype = ctypes.c_char_p if "text" in f else ctypes.c_float if "p" in f else ctypes.c_int64 if "t" in f else ctypes.c_int32
        L.whisper_full_get_token_data.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]; L.whisper_full_get_token_data.restype = WhisperTokenData
        L.whisper_full_get_segment_no_speech_prob.argtypes = [ctypes.c_void_p, ctypes.c_int]; L.whisper_full_get_segment_no_speech_prob.restype = ctypes.c_float
        L.whisper_lang_id.argtypes = [ctypes.c_char_p]; L.whisper_lang_id.restype = ctypes.c_int
        L.whisper_lang_str.argtypes = [ctypes.c_int]; L.whisper_lang_str.restype = ctypes.c_char_p
        L.whisper_print_system_info.restype = ctypes.c_char_p
        L.whisper_print_timings.argtypes = [ctypes.c_void_p]; L.whisper_print_timings.restype = None
        L.whisper_model_type_readable.argtypes = [ctypes.c_void_p]; L.whisper_model_type_readable.restype = ctypes.c_char_p

    def load_model(self, model_path, use_gpu=True, gpu_device=0, flash_attn=False, dtw_token_timestamps=False, dtw_aheads_preset=0, dtw_n_top=-1):
        if self._lib is None: self.load_library()
        logger.info("Step 10a: Acquiring model lock")
        with self._lock:
            logger.info("Step 10b: Freeing old model")
            self.free_model()
            if not os.path.isfile(model_path): raise FileNotFoundError(f"Model not found: {model_path}")
            logger.info("Step 10c: Setting context params")
            cparams = self._lib.whisper_context_default_params()
            cparams.use_gpu = use_gpu; cparams.gpu_device = gpu_device; cparams.flash_attn = flash_attn
            cparams.dtw_token_timestamps = dtw_token_timestamps; cparams.dtw_aheads_preset = dtw_aheads_preset; cparams.dtw_n_top = dtw_n_top
            logger.info("Step 10d: Calling whisper_init_from_file_with_params (heavy operation)...")
            ctx = self._lib.whisper_init_from_file_with_params(model_path.encode("utf-8"), cparams)
            logger.info("Step 10e: whisper_init completed")
            if not ctx: raise RuntimeError(f"Failed to init whisper from {model_path}")
            self._ctx = ctx; self._model_path = model_path

    def free_model(self):
        if self._ctx is not None:
            self._lib.whisper_free(self._ctx); self._ctx = None; self._model_path = None

    def __del__(self): self.free_model()

    def is_loaded(self): return self._ctx is not None

    def _build_full_params(self, **kwargs):
        strategy = kwargs.get("strategy", WHISPER_SAMPLING_GREEDY)
        params = self._lib.whisper_full_default_params(strategy)
        field_map = {
            "n_threads": ("n_threads", int), "n_max_text_ctx": ("n_max_text_ctx", int),
            "offset_ms": ("offset_ms", int), "duration_ms": ("duration_ms", int),
            "translate": ("translate", bool), "no_context": ("no_context", bool),
            "no_timestamps": ("no_timestamps", bool), "single_segment": ("single_segment", bool),
            "print_special": ("print_special", bool), "print_progress": ("print_progress", bool),
            "print_realtime": ("print_realtime", bool), "print_timestamps": ("print_timestamps", bool),
            "token_timestamps": ("token_timestamps", bool),
            "thold_pt": ("thold_pt", float), "thold_ptsum": ("thold_ptsum", float),
            "max_len": ("max_len", int), "split_on_word": ("split_on_word", bool), "max_tokens": ("max_tokens", int),
            "debug_mode": ("debug_mode", bool), "audio_ctx": ("audio_ctx", int), "tdrz_enable": ("tdrz_enable", bool),
            "suppress_regex": ("suppress_regex", lambda v: v.encode() if v else None),
            "initial_prompt": ("initial_prompt", lambda v: v.encode() if v else None),
            "carry_initial_prompt": ("carry_initial_prompt", bool),
            "language": ("language", lambda v: v.encode() if v else None),
            "detect_language": ("detect_language", bool),
            "suppress_blank": ("suppress_blank", bool), "suppress_nst": ("suppress_nst", bool),
            "temperature": ("temperature", float), "max_initial_ts": ("max_initial_ts", float),
            "length_penalty": ("length_penalty", float), "temperature_inc": ("temperature_inc", float),
            "entropy_thold": ("entropy_thold", float), "logprob_thold": ("logprob_thold", float),
            "no_speech_thold": ("no_speech_thold", float),
            "best_of": ("best_of", int), "beam_size": ("beam_size", int), "patience": ("patience", float),
            "vad": ("vad", bool), "vad_model_path": ("vad_model_path", lambda v: v.encode() if v else None),
            "vad_threshold": ("vad_threshold", float), "vad_min_speech_duration_ms": ("vad_min_speech_duration_ms", int),
            "vad_min_silence_duration_ms": ("vad_min_silence_duration_ms", int), "vad_max_speech_duration_s": ("vad_max_speech_duration_s", float),
            "vad_speech_pad_ms": ("vad_speech_pad_ms", int), "vad_samples_overlap": ("vad_samples_overlap", float),
            "grammar_penalty": ("grammar_penalty", float),
        }
        for kw, (field, conv) in field_map.items():
            if kw in kwargs and kwargs[kw] is not None:
                try: setattr(params, field, conv(kwargs[kw]))
                except: pass
        return params

    def transcribe(self, audio_data, **kwargs):
        import numpy as np
        if self._lib is None: self.load_library()
        if self._ctx is None: raise RuntimeError("No model loaded")
        
        # Validate audio input
        audio = np.asarray(audio_data, dtype=np.float32).ravel()
        if len(audio) == 0:
            raise ValueError("Audio input is empty")
        if len(audio) < 1600:  # Less than 0.1s at 16kHz
            raise ValueError(f"Audio too short: {len(audio)} samples ({len(audio)/16000:.2f}s)")
        
        task = kwargs.pop("task", None)
        if task == "translate": kwargs["translate"] = True
        elif task == "transcribe": kwargs["translate"] = False
        if kwargs.pop("word_timestamps", None): kwargs["token_timestamps"] = True; kwargs["split_on_word"] = True
        if "n_threads" not in kwargs or kwargs["n_threads"] is None: kwargs["n_threads"] = max(1, os.cpu_count() or 4)
        params = self._build_full_params(**kwargs)
        audio_ptr = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Thread safety: whisper_full() is NOT thread-safe
        with self._lock:
            ret = self._lib.whisper_full(self._ctx, params, audio_ptr, len(audio))
        if ret != 0: raise RuntimeError(f"whisper_full failed: {ret}")
        n_seg = self._lib.whisper_full_n_segments(self._ctx)
        lang_id = self._lib.whisper_full_lang_id(self._ctx)
        detected_lang = (self._lib.whisper_lang_str(lang_id) or b"").decode() if lang_id >= 0 else "unknown"
        segments, full_text = [], []
        for i in range(n_seg):
            text = (self._lib.whisper_full_get_segment_text(self._ctx, i) or b"").decode(errors="replace")
            t0 = self._lib.whisper_full_get_segment_t0(self._ctx, i) / 100.0
            t1 = self._lib.whisper_full_get_segment_t1(self._ctx, i) / 100.0
            st = bool(self._lib.whisper_full_get_segment_speaker_turn_next(self._ctx, i))
            nsp = float(self._lib.whisper_full_get_segment_no_speech_prob(self._ctx, i))
            n_tok = self._lib.whisper_full_n_tokens(self._ctx, i)
            tokens, words = [], []
            for j in range(n_tok):
                tt = (self._lib.whisper_full_get_token_text(self._ctx, i, j) or b"").decode(errors="replace")
                tp = float(self._lib.whisper_full_get_token_p(self._ctx, i, j))
                tt0 = self._lib.whisper_full_get_token_t0(self._ctx, i, j) / 100.0
                tt1 = self._lib.whisper_full_get_token_t1(self._ctx, i, j) / 100.0
                tid = self._lib.whisper_full_get_token_id(self._ctx, i, j)
                tokens.append({"id": tid, "text": tt, "probability": tp, "start": tt0, "end": tt1})
                if tt.strip() and not tt.startswith("[") and not tt.startswith("<"):
                    words.append({"word": tt.strip(), "start": tt0, "end": tt1, "probability": tp})
            segments.append({"start": t0, "end": t1, "text": text, "tokens": tokens, "words": words, "speaker_turn_next": st, "no_speech_prob": nsp})
            full_text.append(text)
        vad_segs = []
        if hasattr(self._lib, "whisper_full_n_vad_segments") and kwargs.get("vad", False):
            n_vad = self._lib.whisper_full_n_vad_segments(self._ctx)
            for i in range(n_vad):
                vt0 = self._lib.whisper_full_get_vad_segment_t0(self._ctx, i) / 100.0
                vt1 = self._lib.whisper_full_get_vad_segment_t1(self._ctx, i) / 100.0
                vad_segs.append({"start": vt0, "end": vt1})
        return {"text": " ".join(full_text).strip(), "segments": segments, "language": detected_lang, "n_segments": n_seg, "vad_segments": vad_segs, "model_type": self._lib.whisper_model_type_readable(self._ctx).decode() if self._ctx else "unknown"}

    def version(self): return (self._lib.whisper_version() or b"").decode()
    def print_timings(self): self._lib.whisper_print_timings(self._ctx)
