"""
Vocal separation via BSRoformer.cpp (GGML-based, C++ inference).
Much faster than audio-separator — uses CPU/GPU with GGML.
"""

import os, logging, tempfile, subprocess, numpy as np
from pathlib import Path

UVR_AVAILABLE = False
logger = logging.getLogger("WhisperCPP.UVR")

# Built binary path
NODE_DIR = Path(__file__).resolve().parent.parent.parent  # ComfyUI-WhisperXX
BS_ROFORMER_BIN = NODE_DIR / "bs_roformer-cli.exe"

# Default model path (downloaded on first use)
try:
    import folder_paths
    MODEL_DIR = Path(folder_paths.models_dir) / "uvr"
except ImportError:
    MODEL_DIR = NODE_DIR / "models" / "uvr"

DEFAULT_MODEL = "voc_fv6-Q8_0.gguf"
MODEL_URLS = {
    "voc_fv6-Q8_0.gguf": "https://huggingface.co/chenmozhijin/BSRoformer-GGUF/resolve/main/GaboxR67/MelBandRoformers/melbandroformers/vocals/voc_fv6-Q8_0.gguf",
    "BSRoformer-anvuew-Q8_0.gguf": "https://huggingface.co/chenmozhijin/BSRoformer-GGUF/resolve/main/anvuew/BS-RoFormer/BSRoformer-anvuew-Q8_0.gguf",
}

if BS_ROFORMER_BIN.exists():
    UVR_AVAILABLE = True
    logger.info(f"BSRoformer CLI found: {BS_ROFORMER_BIN}")
else:
    logger.warning(f"BSRoformer CLI not found at {BS_ROFORMER_BIN}")


def ensure_model(model_name=DEFAULT_MODEL):
    """Download model if not present."""
    model_path = MODEL_DIR / model_name
    if model_path.exists():
        return str(model_path)
    os.makedirs(MODEL_DIR, exist_ok=True)
    url = MODEL_URLS.get(model_name)
    if not url:
        raise FileNotFoundError(f"Unknown model: {model_name}")
    logger.info(f"Downloading {model_name} ({url})...")
    import urllib.request
    urllib.request.urlretrieve(url, str(model_path))
    size_mb = os.path.getsize(model_path) / 1e6
    logger.info(f"Downloaded {model_name}: {size_mb:.1f} MB")
    return str(model_path)


def separate_vocals(audio_data, sr=44100, model_name=DEFAULT_MODEL, denoise=0.5, use_gpu=True):
    """
    Run BS-Roformer vocal separation via command-line tool.

    Args:
        audio_data: numpy array (float32, mono)
        sr: sample rate
        model_name: GGUF model filename
        denoise: not used by BSRoformer (kept for API compat)
        use_gpu: hint for GPU (BSRoformer auto-detects)

    Returns:
        numpy array (float32, mono) of vocal audio
    """
    if not UVR_AVAILABLE:
        logger.warning("BSRoformer CLI not built. Run build_bs_roformer.bat first.")
        return audio_data

    # Ensure input is 1D mono
    audio_1d = audio_data if audio_data.ndim == 1 else audio_data.mean(axis=1)

    model_path = ensure_model(model_name)
    tmpdir = tempfile.mkdtemp(prefix="bsr_")
    try:
        input_wav = os.path.join(tmpdir, "input.wav")
        output_wav = os.path.join(tmpdir, "output.wav")

        # Write input WAV
        import scipy.io.wavfile as wavfile
        audio_int16 = np.clip(audio_1d * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(input_wav, sr, audio_int16)

        # Run BS-Roformer
        logger.info(f"Running BSRoformer: {model_name}")
        cmd = [
            str(BS_ROFORMER_BIN),
            str(model_path),
            input_wav,
            output_wav,
        ]
        if use_gpu:
            cmd.extend(["--gpu", "auto"])

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            logger.error(f"BSRoformer failed: {r.stderr[:200]}")
            return audio_1d

        # Read output WAV
        if not os.path.isfile(output_wav):
            logger.warning("BSRoformer produced no output file, using original")
            return audio_1d

        sr_out, vocal_data = wavfile.read(output_wav)
        vocal_float = vocal_data.astype(np.float32) / 32767.0
        if vocal_float.ndim > 1:
            vocal_float = vocal_float.mean(axis=1)

        logger.info(f"Vocal separation done: {len(vocal_float)} samples @ {sr_out}Hz")
        return vocal_float

    except subprocess.TimeoutExpired:
        logger.error("BSRoformer timed out (>10 min)")
        return audio_1d
    except Exception as e:
        logger.error(f"BSRoformer error: {e}")
        return audio_1d
    finally:
        import shutil
        try: shutil.rmtree(tmpdir)
        except: pass
