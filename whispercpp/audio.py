import logging, os
import numpy as np

logger = logging.getLogger("WhisperCPP")
WHISPER_SAMPLE_RATE = 16000

class AudioProcessor:
    @staticmethod
    def load_audio(file_path, sr=WHISPER_SAMPLE_RATE):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio not found: {file_path}")
        try:
            import torch, torchaudio
            audio, orig_sr = torchaudio.load(file_path)
            if audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
            if orig_sr != sr: audio = torchaudio.transforms.Resample(orig_sr, sr)(audio)
            return audio.squeeze().numpy().astype(np.float32)
        except ImportError:
            import librosa
            audio, _ = librosa.load(file_path, sr=sr, mono=True)
            return audio.astype(np.float32)

    @staticmethod
    def process_comfy_audio(audio_input):
        if audio_input is None: raise ValueError("No audio input")
        if isinstance(audio_input, dict):
            wf = audio_input.get("waveform")
            sr = audio_input.get("sample_rate", WHISPER_SAMPLE_RATE)
            if hasattr(wf, "shape"):
                arr = wf.cpu().numpy() if hasattr(wf, "cpu") else np.array(wf)
                if arr.ndim == 3: arr = arr[0]
                if arr.ndim == 2 and arr.shape[0] > 1: arr = arr.mean(axis=0)
                elif arr.ndim == 2: arr = arr[0]
                arr = arr.ravel().astype(np.float32)
                if sr != WHISPER_SAMPLE_RATE:
                    import torch, torchaudio
                    t = torch.from_numpy(arr).float().unsqueeze(0)
                    arr = torchaudio.transforms.Resample(sr, WHISPER_SAMPLE_RATE)(t).squeeze().numpy().astype(np.float32)
                return arr
            if isinstance(wf, (list, tuple)): return np.array(wf, dtype=np.float32).ravel()
        if isinstance(audio_input, str): return AudioProcessor.load_audio(audio_input)
        if isinstance(audio_input, np.ndarray): return audio_input.ravel().astype(np.float32)
        raise ValueError(f"Unknown audio type: {type(audio_input)}")
