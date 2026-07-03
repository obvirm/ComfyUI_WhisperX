#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026- pyannote.audio contributors
#
# Run SpeakerDiarization (same control flow as apply()) and save NPZ/JSON artifacts
# for C++ parity tests. See cpp/porting-plan.md §11.

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.audio.pipelines.utils.diarization import set_num_speakers
from pyannote.audio.utils.signal import binarize
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from onnx_wrappers import EmbeddingFbankOnnxWrapper


def _pick_token(explicit: str | None) -> str | bool | None:
    if explicit:
        return explicit
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    return None


def _slug_uri(uri: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", uri).strip("_")
    return s or "utterance"


def _sliding_window_to_dict(sw: SlidingWindow) -> dict[str, float]:
    return {
        "start": float(sw.start),
        "duration": float(sw.duration),
        "step": float(sw.step),
    }


def _write_golden_speaker_bounds(
    out_sub: Path,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: Any,
) -> None:
    """Sidecar for C++ tests: resolved bounds after ``set_num_speakers`` (``max_speakers`` null = no cap / inf)."""
    max_out: int | None
    if max_speakers is None:
        max_out = None
    else:
        mf = float(max_speakers)
        max_out = None if np.isposinf(mf) else int(max_speakers)

    payload = {
        "num_speakers": None if num_speakers is None else int(num_speakers),
        "min_speakers": int(min_speakers),
        "max_speakers": max_out,
    }
    with (out_sub / "golden_speaker_bounds.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_sliding_window_feature(path: Path, swf: SlidingWindowFeature) -> None:
    sw = swf.sliding_window
    np.savez_compressed(
        path,
        data=np.asarray(swf.data),
        sliding_window_start=np.float64(sw.start),
        sliding_window_duration=np.float64(sw.duration),
        sliding_window_step=np.float64(sw.step),
    )


def _save_embedding_ort_reference(
    pipeline: Any,
    file: dict[str, Any],
    binarized_segmentations: SlidingWindowFeature,
    embeddings: np.ndarray,
    out_sub: Path,
) -> None:
    """fbank + segmentation-frame weights + expected embedding for chunk 0 / local speaker 0 (ORT inputs)."""
    num_chunks, num_frames, num_speakers = binarized_segmentations.data.shape
    if num_speakers < 1 or num_chunks < 1:
        return

    exclude_overlap = pipeline.embedding_exclude_overlap
    min_num_samples = pipeline._embedding.min_num_samples
    duration = binarized_segmentations.sliding_window.duration
    num_samples = duration * pipeline._embedding.sample_rate
    min_num_frames = math.ceil(num_frames * min_num_samples / num_samples) if exclude_overlap else -1

    if exclude_overlap:
        clean_frames = 1.0 * (np.sum(binarized_segmentations.data, axis=2, keepdims=True) < 2)
        clean_seg = SlidingWindowFeature(
            binarized_segmentations.data * clean_frames,
            binarized_segmentations.sliding_window,
        )
    else:
        clean_seg = SlidingWindowFeature(
            binarized_segmentations.data, binarized_segmentations.sliding_window
        )

    (chunk, masks), (_, clean_masks) = next(zip(binarized_segmentations, clean_seg))
    masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
    clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)
    mask0 = masks[:, 0]
    clean0 = clean_masks[:, 0]
    if np.sum(clean0) > min_num_frames:
        used_mask = clean0
    else:
        used_mask = mask0

    waveform, _ = pipeline._audio.crop(file, chunk, mode="pad")
    wav = waveform.unsqueeze(0).float()

    em = pipeline._embedding
    model = em.model_
    if not hasattr(model, "compute_fbank"):
        return

    mask_t = torch.from_numpy(used_mask).float().unsqueeze(0)
    with torch.inference_mode():
        fbank = model.compute_fbank(wav.to(model.device))
        wrap = EmbeddingFbankOnnxWrapper(model)
        emb_onnx = wrap(fbank, mask_t.to(fbank.device)).float().cpu().numpy().reshape(-1)

    ref = np.asarray(embeddings[0, 0], dtype=np.float32)
    if not np.allclose(emb_onnx, ref, rtol=1e-4, atol=1e-4):
        raise RuntimeError(
            "EmbeddingFbankOnnxWrapper vs pipeline embedding mismatch for chunk0/spk0"
        )

    np.savez_compressed(
        out_sub / "embedding_chunk0_spk0_ort.npz",
        fbank=np.asarray(fbank.detach().cpu().numpy(), dtype=np.float32),
        weights=np.asarray(mask_t.cpu().numpy(), dtype=np.float32),
        expected_embedding=ref,
    )


def _save_first_chunk_waveform(pipeline: Any, file: dict[str, Any], out_sub: Path) -> None:
    """First sliding-window chunk (t=0) with same padding as Inference — input for ORT parity tests."""
    inf = pipeline._segmentation
    model = inf.model
    audio = Audio(sample_rate=model.audio.sample_rate, mono="downmix")
    chunk = Segment(0.0, float(inf.duration))
    w, sr = audio.crop(file, chunk, mode="pad")
    # Audio.crop returns (channel, time); ONNX / Inference expect (batch, channel, time).
    wave_np = np.asarray(w.cpu().numpy(), dtype=np.float32)
    if wave_np.ndim == 2:
        wave_np = wave_np[np.newaxis, ...]
    np.savez_compressed(
        out_sub / "first_chunk_waveform.npz",
        waveforms=wave_np,
        sample_rate=np.int32(sr),
    )


def _receptive_field_dict(pipeline: Any) -> dict[str, Any]:
    rf = pipeline._segmentation.model.receptive_field
    if isinstance(rf, SlidingWindow):
        return _sliding_window_to_dict(rf)
    if isinstance(rf, tuple):
        return {"tuple": [_sliding_window_to_dict(x) for x in rf]}
    return {"repr": repr(rf)}


def _annotation_to_json(ann: Annotation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn, _, label in ann.itertracks(yield_label=True):
        rows.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(label),
            }
        )
    return rows


def _try_float(x: Any) -> Any:
    try:
        return float(x)
    except Exception:
        return repr(x)


def _pipeline_snapshot(pipeline: Any) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "class": type(pipeline).__name__,
        "segmentation_step": float(pipeline.segmentation_step),
        "embedding_exclude_overlap": bool(pipeline.embedding_exclude_overlap),
        "klustering": str(pipeline.klustering),
    }
    if hasattr(pipeline, "segmentation"):
        snap["segmentation.threshold"] = _try_float(getattr(pipeline.segmentation, "threshold", None))
        snap["segmentation.min_duration_off"] = _try_float(
            getattr(pipeline.segmentation, "min_duration_off", None)
        )
    if hasattr(pipeline, "clustering"):
        for name in ("threshold", "Fa", "Fb"):
            if hasattr(pipeline.clustering, name):
                snap[f"clustering.{name}"] = _try_float(getattr(pipeline.clustering, name))
    return snap


def _run_one(
    pipeline: Any,
    audio_path: Path,
    out_sub: Path,
    *,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> dict[str, Any]:
    out_sub.mkdir(parents=True, exist_ok=True)
    uri = audio_path.stem
    file: dict[str, Any] = {"audio": str(audio_path.resolve()), "uri": uri}

    num_speakers, min_speakers, max_speakers = set_num_speakers(
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    _write_golden_speaker_bounds(out_sub, num_speakers, min_speakers, max_speakers)

    if pipeline._expects_num_speakers and num_speakers is None:
        raise ValueError(
            f"num_speakers must be provided when using {pipeline.klustering} clustering"
        )

    segmentations = pipeline.get_segmentations(file)
    _save_sliding_window_feature(out_sub / "segmentations.npz", segmentations)
    _save_first_chunk_waveform(pipeline, file, out_sub)

    if pipeline._segmentation.model.specifications.powerset:
        binarized_segmentations = segmentations
    else:
        binarized_segmentations = binarize(
            segmentations,
            onset=pipeline.segmentation.threshold,
            initial_state=False,
        )
    _save_sliding_window_feature(out_sub / "binarized_segmentations.npz", binarized_segmentations)

    count = pipeline.speaker_count(
        binarized_segmentations,
        pipeline._segmentation.model.receptive_field,
        warm_up=(0.0, 0.0),
    )
    _save_sliding_window_feature(out_sub / "speaker_count_initial.npz", count)

    if np.nanmax(count.data) == 0.0:
        np.savez_compressed(out_sub / "early_exit.npz", no_speech=np.array(True))
        with (out_sub / "diarization.json").open("w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        with (out_sub / "exclusive_diarization.json").open("w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return {"uri": uri, "early_exit": True}

    embeddings = pipeline.get_embeddings(
        file,
        binarized_segmentations,
        exclude_overlap=pipeline.embedding_exclude_overlap,
    )
    np.savez_compressed(
        out_sub / "embeddings.npz",
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
    _save_embedding_ort_reference(
        pipeline, file, binarized_segmentations, embeddings, out_sub
    )

    hard_clusters, soft_clusters, centroids = pipeline.clustering(
        embeddings=embeddings,
        segmentations=binarized_segmentations,
        num_clusters=num_speakers,
        min_clusters=min_speakers,
        max_clusters=max_speakers,
        file=file,
        frames=pipeline._segmentation.model.receptive_field,
    )
    clu_kw: dict[str, Any] = {
        "hard_clusters": np.asarray(hard_clusters, dtype=np.int8),
        "soft_clusters": np.asarray(soft_clusters, dtype=np.float32),
    }
    if centroids is not None:
        clu_kw["centroids"] = np.asarray(centroids, dtype=np.float32)
    np.savez_compressed(out_sub / "clustering.npz", **clu_kw)

    count.data = np.minimum(count.data, max_speakers).astype(np.int8)
    _save_sliding_window_feature(out_sub / "speaker_count_capped.npz", count)

    inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
    hard_clusters_final = np.array(hard_clusters, dtype=np.int8, copy=True)
    hard_clusters_final[inactive_speakers] = -2
    np.savez_compressed(
        out_sub / "hard_clusters_final.npz",
        hard_clusters=np.asarray(hard_clusters_final, dtype=np.int8),
        inactive_speakers=np.asarray(inactive_speakers, dtype=np.bool_),
    )

    discrete_diarization = pipeline.reconstruct(segmentations, hard_clusters_final, count)
    _save_sliding_window_feature(out_sub / "discrete_diarization_overlap.npz", discrete_diarization)

    diarization = pipeline.to_annotation(
        discrete_diarization,
        min_duration_on=0.0,
        min_duration_off=pipeline.segmentation.min_duration_off,
    )
    diarization.uri = uri

    count_exclusive = count
    count_exclusive.data = np.minimum(count_exclusive.data, 1).astype(np.int8)
    exclusive_discrete = pipeline.reconstruct(segmentations, hard_clusters_final, count_exclusive)
    _save_sliding_window_feature(out_sub / "discrete_diarization_exclusive.npz", exclusive_discrete)

    exclusive_diarization = pipeline.to_annotation(
        exclusive_discrete,
        min_duration_on=0.0,
        min_duration_off=pipeline.segmentation.min_duration_off,
    )
    exclusive_diarization.uri = uri

    if "annotation" in file and file["annotation"]:
        _, mapping = pipeline.optimal_mapping(file["annotation"], diarization, return_mapping=True)
        mapping = {key: mapping.get(key, key) for key in diarization.labels()}
    else:
        mapping = {
            label: expected_label
            for label, expected_label in zip(diarization.labels(), pipeline.classes())
        }

    diarization = diarization.rename_labels(mapping=mapping)
    exclusive_diarization = exclusive_diarization.rename_labels(mapping=mapping)

    if centroids is not None:
        if len(diarization.labels()) > centroids.shape[0]:
            centroids = np.pad(
                centroids,
                ((0, len(diarization.labels()) - centroids.shape[0]), (0, 0)),
            )
        inverse_mapping = {label: index for index, label in mapping.items()}
        centroids = centroids[[inverse_mapping[label] for label in diarization.labels()]]
        np.savez_compressed(
            out_sub / "speaker_embeddings_output.npz",
            centroids=np.asarray(centroids, dtype=np.float32),
        )

    with (out_sub / "diarization.json").open("w", encoding="utf-8") as f:
        json.dump(_annotation_to_json(diarization), f, indent=2)
    with (out_sub / "exclusive_diarization.json").open("w", encoding="utf-8") as f:
        json.dump(_annotation_to_json(exclusive_diarization), f, indent=2)

    with (out_sub / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): str(v) for k, v in mapping.items()}, f, indent=2)

    return {"uri": uri, "early_exit": False}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dump SpeakerDiarization golden tensors (NPZ) + JSON for C++ tests."
    )
    ap.add_argument(
        "--checkpoint",
        default="pyannote/speaker-diarization-community-1",
        help="HF pipeline id or local config directory",
    )
    ap.add_argument("--revision", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to create (one subfolder per input uri)",
    )
    ap.add_argument("audio", nargs="+", type=Path, help="One or more WAV (or supported) audio files")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    args = ap.parse_args()

    token = _pick_token(args.token)

    pipeline = Pipeline.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        token=token,
    )
    if pipeline is None:
        raise SystemExit("Pipeline.from_pretrained returned None (download / token error).")

    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        import pyannote.audio as pa

        pa_version = pa.__version__
    except Exception:
        pa_version = "unknown"

    manifest: dict[str, Any] = {
        "pyannote_audio_version": pa_version,
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "receptive_field": _receptive_field_dict(pipeline),
        "pipeline_snapshot": _pipeline_snapshot(pipeline),
        "utterances": [],
    }
    with (out_root / "receptive_field.json").open("w", encoding="utf-8") as f:
        json.dump(manifest["receptive_field"], f, indent=2)
    with (out_root / "pipeline_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(manifest["pipeline_snapshot"], f, indent=2)

    for audio_path in args.audio:
        if not audio_path.is_file():
            raise SystemExit(f"not a file: {audio_path}")
        sub = out_root / _slug_uri(audio_path.stem)
        info = _run_one(
            pipeline,
            audio_path,
            sub,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        info["path"] = str(sub.relative_to(out_root))
        manifest["utterances"].append(info)

    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote golden bundle under {out_root.resolve()}")


if __name__ == "__main__":
    main()
