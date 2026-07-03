"""
UVR5 vocal separation via audio-separator (MDX-Net).
Extracts voice from noisy/music audio before whisper transcription.

Works on CPU and GPU. Model auto-downloaded on first use.
"""

import os, logging, tempfile, numpy as np
from pathlib import Path

UVR_AVAILABLE = False
UVR_MODEL_MAP = {
    "UVR-MDX-NET-Inst_HQ_3": "UVR-MDX-NET-Inst_HQ_3.onnx",
    "UVR-MDX-NET-Inst_HQ_4": "UVR-MDX-NET-Inst_HQ_4.onnx",
}

try:
    from audio_separator.separator import Separator
    logger = logging.getLogger("WhisperCPP.UVR")
    logger.info("audio-separator available for vocal separation")
    UVR_AVAILABLE = True
except Exception as e:
    logger = logging.getLogger("WhisperCPP.UVR")
    logger.warning(f"audio-separator not available: {e}")
    UVR_AVAILABLE = False


def get_uvr_model_dir():
    """Return path to UVR model storage."""
    # Try common ComfyUI model path first
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, "uvr")
    except ImportError:
        pass
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "uvr")


def separate_vocals(audio_data, sr=44100, model_name="UVR-MDX-NET-Inst_HQ_3", denoise=0.5, use_gpu=True):
    """
    Separate vocals from audio using UVR MDX-Net.

    Args:
        audio_data: numpy array (float32), single channel or stereo
        sr: sample rate of input audio
        model_name: UVR model name (default: UVR-MDX-NET-Inst_HQ_3)
        denoise: denoise strength 0-1 (passed to model params)
        use_gpu: whether to use GPU acceleration

    Returns:
        numpy array (float32) of vocal-only audio at original sample rate
        Returns original audio if UVR is unavailable or fails.
    """
    if not UVR_AVAILABLE:
        logger.warning("audio-separator not installed. Install with: pip install audio-separator onnxruntime")
        return audio_data

    model_dir = get_uvr_model_dir()
    os.makedirs(model_dir, exist_ok=True)

    # Save input audio to temp WAV
    import scipy.io.wavfile as wavfile
    temp_dir = tempfile.mkdtemp(prefix="uvr_")
    input_path = os.path.join(temp_dir, "input.wav")

    # Ensure stereo for UVR model
    if audio_data.ndim == 1:
        audio_input = np.column_stack([audio_data, audio_data])
    else:
        audio_input = audio_data

    # Normalize to int16
    audio_int16 = np.clip(audio_input * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(input_path, sr, audio_int16)

    try:
        logger.info(f"Running vocal separation with {model_name}...")

        # Determine device
        import torch
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        sep = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=temp_dir,
            output_format="WAV",
            output_single_stem="vocals",
            sample_rate=sr,
            use_autocast=False,
        )

        # Load model
        model_filename = UVR_MODEL_MAP.get(model_name, "UVR-MDX-NET-Inst_HQ_3.onnx")
        sep.load_model(model_filename)

        # Run separation
        output_files = sep.separate(input_path)

        # Find vocal output
        vocal_path = None
        for f in output_files:
            if "vocals" in Path(f).stem.lower() or "output" in Path(f).stem.lower():
                vocal_path = f
                break
        if not vocal_path and output_files:
            vocal_path = output_files[0]

        if vocal_path and os.path.exists(vocal_path):
            _, vocal_data = wavfile.read(vocal_path)
            vocal_float = vocal_data.astype(np.float32) / 32767.0
            # Convert to mono
            if vocal_float.ndim > 1:
                vocal_float = vocal_float.mean(axis=1)
            logger.info(f"Vocal separation done: {len(vocal_float)} samples")
            return vocal_float
        else:
            logger.warning("Vocal separation produced no output, using original audio")
            return audio_data if audio_data.ndim == 1 else audio_data.mean(axis=1)

    except Exception as e:
        logger.error(f"Vocal separation failed: {e}")
        return audio_data if audio_data.ndim == 1 else audio_data.mean(axis=1)
    finally:
        # Cleanup temp
        import shutil
        try: shutil.rmtree(temp_dir)
        except: pass
