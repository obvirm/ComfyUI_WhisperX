#!/usr/bin/env python3
"""Add first_chunk_waveform.npz to an existing golden bundle (same logic as dump_diarization_golden)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.core import Segment


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="pyannote/speaker-diarization-community-1")
    p.add_argument("--token", default=None)
    p.add_argument("bundle_dir", type=Path, help="Utterance folder (contains segmentations.npz)")
    p.add_argument("wav", type=Path, help="Same WAV used for the original dump")
    args = p.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    pipeline = Pipeline.from_pretrained(args.checkpoint, token=token)
    uri = args.wav.stem
    file = {"audio": str(args.wav.resolve()), "uri": uri}

    inf = pipeline._segmentation
    model = inf.model
    audio = Audio(sample_rate=model.audio.sample_rate, mono="downmix")
    chunk = Segment(0.0, float(inf.duration))
    w, sr = audio.crop(file, chunk, mode="pad")
    wave_np = np.asarray(w.cpu().numpy(), dtype=np.float32)
    if wave_np.ndim == 2:
        wave_np = wave_np[np.newaxis, ...]
    out = args.bundle_dir / "first_chunk_waveform.npz"
    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        waveforms=wave_np,
        sample_rate=np.int32(sr),
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
