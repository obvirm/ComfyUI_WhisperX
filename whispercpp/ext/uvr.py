"""
Vocal separation via BSRoformer.cpp — DLL + ctypes (no subprocess/CLI).
Konsisten sama whisper_lib.py pattern.
"""

import logging, numpy as np
from pathlib import Path

UVR_AVAILABLE = False
logger = logging.getLogger("WhisperCPP.UVR")

NODE_DIR = Path(__file__).resolve().parent.parent.parent

# Default model path
try:
    import folder_paths
    MODEL_DIR = Path(folder_paths.models_dir) / "uvr"
except ImportError:
    MODEL_DIR = NODE_DIR / "models" / "uvr"

DEFAULT_MODEL = "voc_fv6-Q8_0.gguf"
MODEL_URLS = {
    "voc_fv6-Q8_0.gguf": "https://huggingface.co/chenmozhijin/BSRoformer-GGUF/resolve/main/GaboxR67/MelBandRoformers/melbandroformers/vocals/voc_fv6-Q8_0.gguf",
    "BSRoformer-anvuew-Q8_0.gguf": "https://huggingface.co/chenmozhijin/BSRoformer-GGUF/resolve/main/anvuew/BS-RoFormer/BSRoformer-anvuew-Q8_0.gguf",
    "becruily_deux-Q8_0.gguf": "https://huggingface.co/chenmozhijin/BSRoformer-GGUF/resolve/main/edw1n09/becruily_deux-Q8_0.gguf",
    "voc_fv6-FP16.gguf": "https://huggingface.co/chenmozhijin/BSRoformer-GGUF/resolve/main/GaboxR67/MelBandRoformers/melbandroformers/vocals/voc_fv6-FP16.gguf",
}

# Try loading the ctypes binding
try:
    from .bs_roformer_lib import bs_roformer_init, is_available as _dll_available
    if _dll_available():
        UVR_AVAILABLE = True
        logger.info("BSRoformer DLL loaded via ctypes")
    else:
        logger.warning("BSRoformer DLL not found/available")
except ImportError as e:
    logger.warning(f"BSRoformer ctypes binding not available: {e}")
except Exception as e:
    logger.warning(f"BSRoformer DLL load failed: {e}")


def ensure_model(model_name=DEFAULT_MODEL):
    """Download model if not present."""
    model_path = MODEL_DIR / model_name
    if model_path.exists():
        return str(model_path)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = MODEL_URLS.get(model_name)
    if not url:
        raise FileNotFoundError(f"Unknown model: {model_name}")
    logger.info(f"Downloading {model_name}...")
    import urllib.request
    urllib.request.urlretrieve(url, str(model_path))
    size_mb = model_path.stat().st_size / 1e6
    logger.info(f"Downloaded {model_name}: {size_mb:.1f} MB")
    return str(model_path)


# Cache model context — init sekali, reuse
_ctx_cache = {}

def cleanup():
    """Free all cached BSRoformer contexts."""
    global _ctx_cache
    for path, ctx in _ctx_cache.items():
        try:
            ctx.free()
        except Exception:
            pass
    _ctx_cache.clear()
    logger.info("BSRoformer cache cleaned up")


def separate_vocals(audio_data, sr=44100, model_name=DEFAULT_MODEL, chunk_size=-1, overlap=-1, use_gpu=True):
    """
    Run BS-Roformer vocal separation via DLL (no temp files, no subprocess).

    Args:
        audio_data: numpy array (float32, mono)
        sr: sample rate
        model_name: GGUF model filename
        chunk_size: samples per chunk (-1 = model default)
        overlap: overlap count (-1 = model default)

    Returns:
        numpy array (float32, mono) of vocal audio
    """
    if not UVR_AVAILABLE:
        logger.warning("BSRoformer DLL not available. Build dulu lewat build_bs_roformer_dll.bat")
        return audio_data

    audio_1d = audio_data if audio_data.ndim == 1 else audio_data.mean(axis=1)

    if sr != 44100:
        # Resample to 44100 secara sederhana
        import scipy.signal
        target_len = int(len(audio_1d) * 44100 / sr)
        audio_1d = scipy.signal.resample(audio_1d, target_len).astype(np.float32)

    model_path = ensure_model(model_name)

    # Get or create cached context
    ctx = _ctx_cache.get(model_path)
    if ctx is None:
        logger.info(f"Loading BSRoformer model: {model_name}")
        ctx = bs_roformer_init(model_path)
        if ctx is None:
            logger.error(f"Gagal load model: {model_path}")
            return audio_data
        _ctx_cache[model_path] = ctx
        logger.info(f"Model loaded: {ctx.num_stems} stem(s), sample rate={ctx.sample_rate}")

    try:
        # Convert mono to interleaved stereo (BSRoformer expects stereo)
        n_samples = len(audio_1d)
        stereo = np.empty(n_samples * 2, dtype=np.float32)
        stereo[0::2] = audio_1d  # L
        stereo[1::2] = audio_1d  # R

        # Process via DLL
        result = ctx.process(stereo, chunk_size=chunk_size, num_overlap=overlap)
        # result shape: [num_stems, n_samples_interleaved_stereo]
        # stem 0 = vocals (interleaved stereo) -> convert ke mono
        vocals_st = result[0]
        vocals = (vocals_st[0::2] + vocals_st[1::2]) * 0.5

        # Normalisasi: BSRoformer output sangat kecil (~3e-5)
        peak = np.abs(vocals).max()
        if peak > 1e-10:
            vocals = vocals / peak
        vocals = np.clip(vocals, -1.0, 1.0)

        logger.info(f"Vocal separation done via DLL: {len(vocals)} samples, peak={peak:.6f}")
        return vocals.astype(np.float32)

    except Exception as e:
        logger.error(f"BSRoformer DLL error: {e}", exc_info=True)
        return audio_data
