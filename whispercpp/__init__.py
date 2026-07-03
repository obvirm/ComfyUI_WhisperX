# Lazy imports: ModelManager hanya di-import saat dipakai
from .whisper_lib import WhisperCPP
from .audio import AudioProcessor

# ModelManager di-import di model.py langsung oleh whispercpp_node.py
# Tidak perlu re-export di sini karena ModelManager butuh tqdm & requests
# yang mungkin belum terinstall di ComfyUI environment

__all__ = ["WhisperCPP", "AudioProcessor"]
