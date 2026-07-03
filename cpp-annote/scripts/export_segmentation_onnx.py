#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026- pyannote.audio contributors
#
# Export segmentation PyTorch weights to ONNX for the C++ port.
# Extends https://github.com/pengzhendong/pyannote-onnx (adds powerset + metadata).

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

from onnx_wrappers import SegmentationInferenceWrapper


def _pick_token(explicit: str | None) -> str | bool | None:
    if explicit:
        return explicit
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pyannote segmentation model to ONNX.")
    parser.add_argument(
        "--checkpoint",
        default="pyannote/speaker-diarization-community-1",
        help="HF repo id or local checkpoint directory",
    )
    parser.add_argument("--subfolder", default="segmentation", help="HF subfolder for weights")
    parser.add_argument("--revision", default=None, help="HF revision (optional)")
    parser.add_argument("--token", default=None, help="HF token (else HF_TOKEN / HUGGING_FACE_HUB_TOKEN)")
    parser.add_argument(
        "--skip-powerset-conversion",
        action="store_true",
        help="Export raw logits (Identity); use only if you will decode powerset in C++",
    )
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
    wrapped = SegmentationInferenceWrapper(model, skip_conversion=args.skip_powerset_conversion)
    wrapped.eval()

    specs = model.specifications
    if isinstance(specs, tuple):
        if len(specs) != 1:
            raise SystemExit("Multi-task segmentation is not supported for export.")
        spec = specs[0]
    else:
        spec = specs

    duration = spec.duration
    num_samples = int(model.audio.get_num_samples(duration))
    sample_rate = int(model.audio.sample_rate)
    num_channels = int(model.hparams.num_channels)
    dummy = torch.zeros(1, num_channels, num_samples, dtype=torch.float32)

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # dynamo=False: torch.export path fails on ParamSincFB (asteroid_filterbanks)
    # torch.clamp(..., int_min, float_max) after type promotion (ONNX FX pass).
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        input_names=["waveforms"],
        output_names=["segmentation"],
        dynamic_axes={"waveforms": {0: "batch"}, "segmentation": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    # Default pipeline step matches SpeakerDiarization (segmentation_step=0.1 × chunk duration).
    chunk_step_sec = float(0.1 * float(duration))

    meta = {
        "model_type": "segmentation",
        "checkpoint": args.checkpoint,
        "subfolder": args.subfolder,
        "revision": args.revision,
        "sample_rate": sample_rate,
        "num_channels": num_channels,
        "chunk_duration_sec": float(duration),
        "chunk_step_sec": chunk_step_sec,
        "chunk_num_samples": num_samples,
        "powerset": bool(spec.powerset),
        "export_includes_powerset_to_multilabel": not args.skip_powerset_conversion,
        "input_names": ["waveforms"],
        "output_names": ["segmentation"],
        "opset": args.opset,
        "notes": "Input layout (batch, channel, samples) matches Inference.infer. "
        "Output matches post-conversion multilabel when powerset and skip_powerset_conversion=False.",
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
