import logging
import os
import numpy as np

logger = logging.getLogger("WhisperCPP")
WHISPER_SAMPLE_RATE = 16000

class AudioProcessor:
    @staticmethod
    def load_audio(file_path: str, sr: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        try:
            import torch
            import torchaudio
            waveform, orig_sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if orig_sr != sr:
                waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
            return waveform.squeeze().numpy().astype(np.float32)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"torchaudio failed: {e}")
        try:
            import librosa
            audio, _ = librosa.load(file_path, sr=sr, mono=True)
            return audio.astype(np.float32)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"librosa failed: {e}")
        import subprocess
        cmd = ["ffmpeg", "-i", file_path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-loglevel", "error", "-"]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    @staticmethod
    def process_comfy_audio(audio: dict) -> np.ndarray:
        import torch
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.is_cuda:
            waveform = waveform.cpu()
        if waveform.shape[1] > 1:
            waveform = torch.mean(waveform, dim=1, keepdim=True).squeeze(1)
        if waveform.dim() > 1:
            waveform = waveform[0]
        if sample_rate != WHISPER_SAMPLE_RATE:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sample_rate, WHISPER_SAMPLE_RATE)
        return waveform.numpy().astype(np.float32)

    @staticmethod
    def trim_silence(audio: np.ndarray, sr: int = WHISPER_SAMPLE_RATE, threshold: float = 0.01, padding_ms: int = 300) -> np.ndarray:
        padding = int(sr * padding_ms / 1000)
        abs_audio = np.abs(audio)
        mask = abs_audio > threshold
        if not np.any(mask):
            return audio
        start = max(0, np.argmax(mask) - padding)
        end = min(len(audio), len(audio) - np.argmax(mask[::-1]) + padding)
        return audio[start:end]
