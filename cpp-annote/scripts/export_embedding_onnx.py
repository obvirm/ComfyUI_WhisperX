#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026- pyannote.audio contributors
#
# Export WeSpeaker embedding ResNet to ONNX (log-fbank + frame weights; fbank is host-side).

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import onnx
import torch

from pyannote.audio import Model

from onnx_wrappers import EmbeddingFbankOnnxWrapper


def _pick_token(explicit: str | None) -> str | bool | None:
    if explicit:
        return explicit
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pyannote embedding model to ONNX.")
    parser.add_argument(
        "--checkpoint",
        default="pyannote/speaker-diarization-community-1",
        help="HF repo id or local checkpoint directory",
    )
    parser.add_argument("--subfolder", default="embedding", help="HF subfolder for weights")
    parser.add_argument("--revision", default=None, help="HF revision (optional)")
    parser.add_argument("--token", default=None, help="HF token (else HF_TOKEN / HUGGING_FACE_HUB_TOKEN)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .onnx path (metadata written alongside as .json)",
    )
    args = parser.parse_args()

    token = _pick_token(args.token)

    model = Model.from_pretrained(
        args.checkpoint,
        subfolder=args.subfolder,
        revision=args.revision,
        token=token,
        strict=False,
    )
    model.eval()
    if not hasattr(model, "compute_fbank"):
        raise SystemExit("Embedding model has no compute_fbank; ONNX export is not supported.")
    wrapped = EmbeddingFbankOnnxWrapper(model)
    wrapped.eval()

    sample_rate = int(model.audio.sample_rate)
    num_channels = int(model.hparams.num_channels)
    num_mel_bins = int(model.hparams.num_mel_bins)
    # Long enough for a stable fbank length (matches typical diarization chunk scale).
    num_samples = int(round(5.0 * float(sample_rate)))
    waveforms = torch.zeros(1, num_channels, num_samples, dtype=torch.float32)

    with torch.inference_mode():
        fbank_ref = model.compute_fbank(waveforms)

    num_frames = int(fbank_ref.shape[1])
    fbank = torch.zeros(1, num_frames, num_mel_bins, dtype=torch.float32)
    weights = torch.ones(1, num_frames, dtype=torch.float32)

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # dynamo=False: see export_segmentation_onnx.py (ParamSincFB / torch.clamp).
    # Inputs are log-fbank after ``compute_fbank`` (Kaldi fbank + mean subtraction); not raw audio.
    torch.onnx.export(
        wrapped,
        (fbank, weights),
        str(out_path),
        input_names=["fbank", "weights"],
        output_names=["embedding"],
        dynamic_axes={
            "fbank": {0: "batch", 1: "num_frames"},
            "weights": {0: "batch", 1: "num_frames"},
            "embedding": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    meta = {
        "model_type": "embedding",
        "checkpoint": args.checkpoint,
        "subfolder": args.subfolder,
        "revision": args.revision,
        "sample_rate": sample_rate,
        "num_channels": num_channels,
        "num_mel_bins": num_mel_bins,
        "frame_length_ms": float(model.hparams.frame_length),
        "frame_shift_ms": float(model.hparams.frame_shift),
        "fbank_centering_span": model.hparams.fbank_centering_span,
        "embedding_dim": int(model.dimension),
        "example_num_samples_for_fbank_ref": num_samples,
        "example_num_frames": num_frames,
        "input_names": ["fbank", "weights"],
        "output_names": ["embedding"],
        "opset": args.opset,
        "notes": "ONNX inputs are Kaldi log-fbank features matching pyannote "
        "``BaseWeSpeakerResNet.compute_fbank`` output: shape (batch, num_frames, num_mel_bins). "
        "Compute fbank outside ONNX (torchaudio kaldi.fbank + scaling/mean per checkpoint). "
        "weights (batch, num_frames): 1.0 keep frame, 0.0 drop; must match num_frames for fbank.",
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {meta_path}")

    repo = _SCRIPT_DIR.parent.parent
    regen = [
        sys.executable,
        str(_SCRIPT_DIR / "export_cpp_annote_embedded.py"),
        "--repo-root",
        str(repo),
        "--checkpoint",
        str(args.checkpoint),
    ]
    if args.revision:
        regen.extend(["--revision", str(args.revision)])
    if args.token:
        regen.extend(["--token", str(args.token)])
    subprocess.run(regen, check=True)


if __name__ == "__main__":
    main()
