#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026- pyannote.audio contributors
#
# Compare ONNX Runtime outputs to PyTorch for exported community-1 models.

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import onnxruntime as ort
import torch

from pyannote.audio import Model

from onnx_wrappers import EmbeddingFbankOnnxWrapper, SegmentationInferenceWrapper


def _pick_token(explicit: str | None) -> str | bool | None:
    if explicit:
        return explicit
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    return None


def _load_meta(onnx_path: Path) -> dict:
    meta_path = onnx_path.with_suffix(".json")
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def verify_segmentation(
    onnx_path: Path,
    *,
    checkpoint: str,
    subfolder: str,
    revision: str | None,
    token: str | bool | None,
    rtol: float,
    atol: float,
) -> None:
    meta = _load_meta(onnx_path)
    model = Model.from_pretrained(
        checkpoint,
        subfolder=subfolder,
        revision=revision,
        token=token,
        strict=False,
    )
    model.eval()
    skip_powerset_conversion = not meta.get("export_includes_powerset_to_multilabel", True)
    wrapped = SegmentationInferenceWrapper(model, skip_conversion=skip_powerset_conversion)
    wrapped.eval()

    num_samples = meta.get("chunk_num_samples")
    if num_samples is None:
        specs = model.specifications
        spec = specs[0] if isinstance(specs, tuple) else specs
        num_samples = int(model.audio.get_num_samples(spec.duration))
    num_channels = int(meta.get("num_channels", model.hparams.num_channels))

    torch.manual_seed(0)
    waveforms = torch.randn(2, num_channels, num_samples, dtype=torch.float32) * 0.01

    with torch.inference_mode():
        expected = wrapped(waveforms).cpu().numpy()

    sess = _session(onnx_path)
    feeds = {"waveforms": waveforms.numpy().astype(np.float32)}
    got = sess.run(None, feeds)[0]

    if got.shape != expected.shape:
        raise SystemExit(
            f"segmentation shape mismatch: torch {expected.shape} vs onnx {got.shape}"
        )
    if not np.allclose(expected, got, rtol=rtol, atol=atol):
        diff = np.max(np.abs(expected.astype(np.float64) - got.astype(np.float64)))
        raise SystemExit(
            f"segmentation parity failed (max abs diff={diff:g}, rtol={rtol}, atol={atol})"
        )
    print(f"OK segmentation: max abs diff {np.max(np.abs(expected - got)):.3e}")


def verify_embedding(
    onnx_path: Path,
    *,
    checkpoint: str,
    subfolder: str,
    revision: str | None,
    token: str | bool | None,
    rtol: float,
    atol: float,
) -> None:
    meta = _load_meta(onnx_path)
    model = Model.from_pretrained(
        checkpoint,
        subfolder=subfolder,
        revision=revision,
        token=token,
        strict=False,
    )
    model.eval()
    if not hasattr(model, "compute_fbank"):
        raise SystemExit("embedding model has no compute_fbank; skip or fix checkpoint")

    wrapped = EmbeddingFbankOnnxWrapper(model)
    wrapped.eval()

    sample_rate = int(model.audio.sample_rate)
    num_channels = int(meta.get("num_channels", model.hparams.num_channels))
    num_samples = int(meta.get("example_num_samples_for_fbank_ref", round(5.0 * float(sample_rate))))

    torch.manual_seed(1)
    waveforms = torch.randn(2, num_channels, num_samples, dtype=torch.float32) * 0.01

    with torch.inference_mode():
        fbank = model.compute_fbank(waveforms)
        _b, num_frames, _m = fbank.shape
        weights = torch.rand(2, num_frames, dtype=torch.float32)
        expected = wrapped(fbank, weights).cpu().numpy()

    sess = _session(onnx_path)
    feeds = {
        "fbank": fbank.numpy().astype(np.float32),
        "weights": weights.numpy().astype(np.float32),
    }
    got = sess.run(None, feeds)[0]

    if got.shape != expected.shape:
        raise SystemExit(f"embedding shape mismatch: torch {expected.shape} vs onnx {got.shape}")
    if not np.allclose(expected, got, rtol=rtol, atol=atol):
        diff = np.max(np.abs(expected.astype(np.float64) - got.astype(np.float64)))
        raise SystemExit(
            f"embedding parity failed (max abs diff={diff:g}, rtol={rtol}, atol={atol})"
        )
    print(f"OK embedding: max abs diff {np.max(np.abs(expected - got)):.3e}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    default_seg = root / "cpp" / "artifacts" / "community1-segmentation.onnx"
    default_emb = root / "cpp" / "artifacts" / "community1-embedding.onnx"

    p = argparse.ArgumentParser(description="Verify ONNX matches PyTorch for exported models.")
    p.add_argument("--checkpoint", default="pyannote/speaker-diarization-community-1")
    p.add_argument("--segmentation-subfolder", default="segmentation")
    p.add_argument("--embedding-subfolder", default="embedding")
    p.add_argument("--revision", default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--segmentation-onnx", type=Path, default=default_seg)
    p.add_argument("--embedding-onnx", type=Path, default=default_emb)
    p.add_argument("--skip-segmentation", action="store_true")
    p.add_argument("--skip-embedding", action="store_true")
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-4)
    args = p.parse_args()
    token = _pick_token(args.token)

    if not args.skip_segmentation:
        if not args.segmentation_onnx.is_file():
            raise SystemExit(f"missing segmentation ONNX: {args.segmentation_onnx}")
        verify_segmentation(
            args.segmentation_onnx,
            checkpoint=args.checkpoint,
            subfolder=args.segmentation_subfolder,
            revision=args.revision,
            token=token,
            rtol=args.rtol,
            atol=args.atol,
        )

    if not args.skip_embedding:
        if not args.embedding_onnx.is_file():
            raise SystemExit(f"missing embedding ONNX: {args.embedding_onnx}")
        verify_embedding(
            args.embedding_onnx,
            checkpoint=args.checkpoint,
            subfolder=args.embedding_subfolder,
            revision=args.revision,
            token=token,
            rtol=args.rtol,
            atol=args.atol,
        )

    print("All requested ONNX parity checks passed.")


if __name__ == "__main__":
    main()
