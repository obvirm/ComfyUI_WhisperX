"""
cpp-annote VAD & diarization via Python onnxruntime.
Uses community1-segmentation + community1-embedding ONNX models.
"""

import logging, numpy as np
from pathlib import Path

logger = logging.getLogger("WhisperCPP.CppAnnote")

CPPANNOTE_AVAILABLE = False
try:
    import onnxruntime
    CPPANNOTE_AVAILABLE = True
except ImportError:
    pass

NODE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = NODE_DIR / "cpp-annote" / "artifacts"

SR = 16000
WINDOW = 160000  # 10s
HOP = 80000      # 5s


class CppAnnote:
    def __init__(self, device="cpu"):
        self.device = device
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if device != "cpu" else ["CPUExecutionProvider"]

        seg_path = str(MODEL_DIR / "community1-segmentation.onnx")
        emb_path = str(MODEL_DIR / "community1-embedding.onnx")

        logger.info(f"Loading segmentation model: {seg_path}")
        self.seg_session = onnxruntime.InferenceSession(seg_path, providers=providers)
        logger.info(f"Loading embedding model: {emb_path}")
        self.emb_session = onnxruntime.InferenceSession(emb_path, providers=providers)

    def segment(self, audio):
        """Return list of (start_sec, end_sec) speech segments."""
        audio_16k = self._to_16k(audio)
        if len(audio_16k) < SR:
            return []

        speech = []
        window = WINDOW
        hop = HOP
        for start in range(0, len(audio_16k), hop):
            chunk = audio_16k[start:start + window]
            if len(chunk) < window:
                chunk = np.pad(chunk, (0, window - len(chunk)))
            inp = chunk.reshape(1, 1, -1).astype(np.float32)
            out = self.seg_session.run(None, {"waveforms": inp})
            seg = out[0][0]  # (589, 3): non-speech, speech, overlap
            speech_prob = seg[:, 1]  # ch1 = speech activity
            frame_sec = 10.0 / len(speech_prob)

            in_speech = False
            s_start = 0.0
            for i, p in enumerate(speech_prob):
                if p > 0.5 and not in_speech:
                    in_speech = True
                    s_start = start / SR + i * frame_sec
                elif p <= 0.3 and in_speech:
                    in_speech = False
                    speech.append((s_start, start / SR + i * frame_sec))
            if in_speech:
                speech.append((s_start, (start + window) / SR))

        return self._merge(speech, gap=0.3)

    def diarize(self, audio, sample_rate=SR):
        """Full diarization: VAD → embedding → clustering."""
        segments = self.segment(audio)
        if not segments:
            return [{"start": 0.0, "end": len(audio) / SR, "speaker": 0}] if len(audio) > 0 else []

        aud_16k = self._to_16k(audio)
        embs = []
        for st, en in segments:
            s = int(st * SR)
            e = int(en * SR)
            chunk = aud_16k[s:e]
            if len(chunk) < SR * 0.4:
                continue
            emb = self._embed(chunk)
            if emb is not None:
                embs.append((emb, st, en))

        if not embs:
            return [{"start": round(s, 3), "end": round(e, 3), "speaker": 0}
                    for s, e in segments]

        # Cosine-similarity clustering
        mat = np.array([e[0] for e in embs])
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        n = len(embs)
        spk = [-1] * n
        next_spk = 0
        for i in range(n):
            if spk[i] >= 0:
                continue
            spk[i] = next_spk
            for j in range(i + 1, n):
                if spk[j] < 0 and np.dot(mat[i], mat[j]) > 0.7:
                    spk[j] = next_spk
            next_spk += 1

        return [{"start": round(embs[i][1], 3), "end": round(embs[i][2], 3),
                 "speaker": spk[i]} for i in range(n)]

    def _embed(self, audio_16k):
        """Extract 256-dim speaker embedding."""
        try:
            import torch, torchaudio
            t = torch.from_numpy(audio_16k).float().unsqueeze(0)
            fbank = torchaudio.compliance.kaldi.fbank(
                t, num_mel_bins=80, sample_frequency=SR,
                frame_length=25, frame_shift=10
            ).numpy().astype(np.float32)
        except ImportError:
            try:
                import librosa
                mel = librosa.feature.melspectrogram(y=audio_16k, sr=SR, n_mels=80,
                    n_fft=400, hop_length=160).T
                fbank = librosa.amplitude_to_db(mel).astype(np.float32)
            except ImportError:
                return None

        if fbank.ndim == 2:
            fbank = fbank[np.newaxis, :, :80]
        T = fbank.shape[1]
        w = np.ones((1, T), dtype=np.float32)
        try:
            emb = self.emb_session.run(None, {"fbank": fbank, "weights": w})
            return emb[0][0]
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _to_16k(self, audio):
        if audio is None or len(audio) == 0:
            return np.array([], dtype=np.float32)
        a = audio.ravel().astype(np.float64)
        # Assume input is SR=16000 (whisper standard)
        # If audio length doesn't match expected, simple resample
        return a.astype(np.float32)

    @staticmethod
    def _merge(segs, gap=0.3):
        if not segs:
            return []
        m = [list(segs[0])]
        for s, e in segs[1:]:
            if s - m[-1][1] <= gap:
                m[-1][1] = max(m[-1][1], e)
            else:
                m.append([s, e])
        return [(s, e) for s, e in m]
