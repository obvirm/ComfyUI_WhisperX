#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026- pyannote.audio contributors
#
# Download one utterance from Hugging Face `diarizers-community/callhome` for golden dumps.
#
# Note: this dataset revision exposes a single split named ``data`` (not ``train``).
# The English subset is typical for diarization smoke tests.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio


def main() -> None:
    p = argparse.ArgumentParser(
        description="Save one Callhome clip from diarizers-community/callhome as WAV."
    )
    p.add_argument(
        "--subset",
        default="eng",
        choices=("eng", "zho", "deu", "jpn", "spa"),
        help="Callhome language subset (HF config name)",
    )
    p.add_argument(
        "--split",
        default="data",
        help="HF split name (current dataset revision uses ``data`` for all rows)",
    )
    p.add_argument("--index", type=int, default=0, help="Row index in the split")
    p.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="Truncate to this many seconds from the start (full calls are ~5 min)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .wav path (parent dirs created)",
    )
    args = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset(
        "diarizers-community/callhome",
        args.subset,
        split=args.split,
        trust_remote_code=True,
    )
    if args.index < 0 or args.index >= len(ds):
        raise SystemExit(f"index {args.index} out of range [0, {len(ds) - 1}]")

    row = ds[args.index]
    audio = row["audio"]
    arr = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    max_samples = int(args.max_seconds * sr)
    if arr.shape[0] > max_samples:
        arr = arr[:max_samples]

    wav = torch.from_numpy(arr).unsqueeze(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), wav, sr)
    print(
        f"Wrote {args.output} ({wav.shape[1] / sr:.1f}s, {sr} Hz) "
        f"from diarizers-community/callhome[{args.subset!r}][{args.split!r}][{args.index}]"
    )


if __name__ == "__main__":
    main()
