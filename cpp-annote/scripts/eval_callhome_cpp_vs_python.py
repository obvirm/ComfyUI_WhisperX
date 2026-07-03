#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "huggingface_hub",
#     "numpy",
#     "pyannote.audio",
#     "pyannote.core",
#     "pyannote.metrics",
#     "torch",
#     "torchaudio",
# ]
# ///
# MIT License
#
# Copyright (c) 2026- pyannote.audio contributors
#
# Evaluate Diarization Error Rate (DER) on the first n clips from Hugging Face
# ``diarizers-community/callhome`` (English subset by default): reference labels
# from dataset timestamps vs (1) full ``SpeakerDiarization`` in Python and
# (2) C++ ``cpp-annote-cli`` (segmentation ORT + ORT embedding + VBx in C++).
# By default we evaluate the **NIST-style Part 2** slice (see ``--callhome-part``):
# for 140-row configs that is HF rows 80–139 (Part 1 is 0–79); for 120-row configs,
# Part 2 is rows 60–119 (Part 1 is 0–59).

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

_SCRIPT_DIR = Path(__file__).resolve().parent


def _repo_root() -> Path:
    for d in _SCRIPT_DIR.parents:
        if (d / "scripts" / "dump_diarization_golden.py").is_file():
            return d
    raise RuntimeError(
        "Could not locate pyannote-audio repository root "
        "(expected scripts/dump_diarization_golden.py in a parent of this script)."
    )


_REPO_ROOT = _repo_root()


def _pick_token(explicit: str | None) -> str | bool | None:
    if explicit:
        return explicit
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(key)
        if v:
            return v
    return None


def _utterance_stem(subset: str, index: int, max_seconds: float) -> str:
    return f"callhome_{subset}_data_idx{index}_head{int(max_seconds)}s"


def _callhome_hf_part_start_row(n_rows: int, part: str) -> int:
    """First HF row index for NIST-style Part 1 / Part 2 on ``diarizers-community/callhome``."""
    if part == "1":
        return 0
    if n_rows == 140:
        return 80
    if n_rows == 120:
        return 60
    return 0


def _row_to_reference(row: dict[str, Any], crop_end: float, uri: str) -> Any:
    from pyannote.core import Annotation, Segment

    ann: Annotation = Annotation(uri=uri)
    for t0, t1, spk in zip(
        row["timestamps_start"],
        row["timestamps_end"],
        row["speakers"],
        strict=True,
    ):
        t0f = float(t0)
        t1f = float(t1)
        if t0f >= crop_end:
            continue
        t1f = min(t1f, crop_end)
        if t1f <= t0f:
            continue
        ann[Segment(t0f, t1f)] = str(spk)
    return ann


def _wav_duration_sec(path: Path) -> float:
    info = torchaudio.info(str(path))
    return float(info.num_frames) / float(info.sample_rate)


def _save_wav_row(row: dict[str, Any], path: Path, max_seconds: float) -> float:
    audio = row["audio"]
    arr = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    max_samples = int(max_seconds * sr)
    if arr.shape[0] > max_samples:
        arr = arr[:max_samples]
    wav = torch.from_numpy(arr).unsqueeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav, sr)
    return float(arr.shape[0]) / float(sr)


def _hf_embedding_plda_paths(
    checkpoint: str,
    revision: str | None,
    token: str | bool | None,
) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    kw: dict[str, Any] = {"repo_id": checkpoint, "token": token}
    if revision is not None:
        kw["revision"] = revision
    xvec = Path(hf_hub_download(filename="plda/xvec_transform.npz", **kw))
    plda = Path(hf_hub_download(filename="plda/plda.npz", **kw))
    return xvec, plda


def _json_diarization_to_annotation(path: Path, uri: str) -> Any:
    from pyannote.core import Annotation, Segment

    ann: Annotation = Annotation(uri=uri)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    turns = data["turns"] if isinstance(data, dict) and "turns" in data else data
    for t in turns:
        ann[Segment(float(t["start"]), float(t["end"]))] = str(t["speaker"])
    return ann


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "DER on first n CallHome (HF) clips: Python community-1 pipeline vs C++ cpp-annote-cli "
            "(ORT segmentation + ORT embedding + VBx). "
            "Defaults to NIST-style Part 2 rows (``--callhome-part 2``); use ``--callhome-part 1`` or "
            "``--start-index`` to change the slice."
        )
    )
    ap.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=20,
        help="Number of consecutive dataset rows starting at --start-index (default: 20)",
    )
    ap.add_argument(
        "--callhome-part",
        choices=("1", "2"),
        default="2",
        help=(
            "NIST-style CallHome partition on the HF ``data`` split: "
            "``1`` = first block (rows 0–79 for 140-row configs, 0–59 for 120-row); "
            "``2`` = second / evaluation block (80–139 or 60–119). "
            "Ignored when --start-index is set."
        ),
    )
    ap.add_argument(
        "--start-index",
        type=int,
        default=None,
        help=(
            "First row index in the HF split. "
            "When omitted, derived from ``--callhome-part`` (default part: 2)."
        ),
    )
    ap.add_argument(
        "--subset",
        default="eng",
        choices=("eng", "zho", "deu", "jpn", "spa"),
        help="Callhome language subset (HF config name)",
    )
    ap.add_argument("--split", default="data", help="HF split name (default: data)")
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="Truncate each call to this many seconds from the start (default: 120)",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=_REPO_ROOT / ".callhome_eval_work",
        help="Scratch directory for wavs, golden bundle, and C++ JSON output",
    )
    ap.add_argument(
        "--checkpoint",
        default="pyannote/speaker-diarization-community-1",
        help="Pipeline id for Python + golden dump (default: community-1)",
    )
    ap.add_argument("--revision", default=None)
    ap.add_argument("--token", default=None, help="HF token (else HF_TOKEN / HUGGING_FACE_HUB_TOKEN)")
    ap.add_argument(
        "--device",
        default="auto",
        help="Torch device for pipeline: auto | cpu | cuda | cuda:0 | mps | …",
    )
    ap.add_argument(
        "--cpp-binary",
        type=Path,
        default=_REPO_ROOT / "build" / "cpp-annote-cli",
        help="Path to cpp-annote-cli executable",
    )
    ap.add_argument("--skip-cpp", action="store_true", help="Skip C++ binary; only Python DER + golden dump")
    ap.add_argument("--skip-pyannote", action="store_true", help="Skip pyannote Python pipeline; only C++ DER")
    ap.add_argument(
        "--dump-golden",
        action="store_true",
        help="Run golden dump (required for --cpp-clustering-check; skipped by default)",
    )
    ap.add_argument(
        "--cluster-cadence",
        type=float,
        default=None,
        help="Streaming: re-cluster every N seconds of new audio (passed to cpp-annote-cli)",
    )
    ap.add_argument(
        "--analyze-cadence",
        type=float,
        default=None,
        help="Streaming: step between seg+emb model runs in seconds (>0, <=10; passed to cpp-annote-cli)",
    )
    ap.add_argument(
        "--segmentation-onnx",
        type=Path,
        default=None,
        help="Path to external segmentation .onnx file (passed to cpp-annote-cli; default: use compiled-in model)",
    )
    ap.add_argument(
        "--embedding-onnx",
        type=Path,
        default=None,
        help="Path to external embedding .onnx file (passed to cpp-annote-cli; default: use compiled-in model)",
    )
    ap.add_argument(
        "--cpp-clustering-check",
        action="store_true",
        help=(
            "After C++ batch, run clustering_golden_test on the first utterance only "
            "(requires build/clustering_golden_test and golden NPZ from the dump)."
        ),
    )
    args = ap.parse_args()

    if args.num_files < 1:
        raise SystemExit("--num-files must be >= 1")
    if args.cpp_clustering_check:
        args.dump_golden = True

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install datasets: pip install datasets") from e

    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    if not args.skip_pyannote:
        from pyannote.audio import Pipeline

    work: Path = args.work_dir.resolve()
    wav_dir = work / "wavs"
    golden_root = work / "golden"
    cpp_out = work / "cpp_out"
    cpp_out.mkdir(parents=True, exist_ok=True)

    token = _pick_token(args.token)
    ds = load_dataset("diarizers-community/callhome", args.subset, split=args.split)

    start_index = (
        args.start_index
        if args.start_index is not None
        else _callhome_hf_part_start_row(len(ds), args.callhome_part)
    )
    if args.start_index is None:
        print(
            f"Using CallHome HF part {args.callhome_part} → --start-index {start_index} "
            f"(split len={len(ds)})",
            flush=True,
        )

    hi = start_index + args.num_files
    if start_index < 0 or hi > len(ds):
        raise SystemExit(
            f"row range [{start_index}, {hi}) out of range for split (len={len(ds)})"
        )

    stems: list[str] = []
    wav_paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    durations: list[float] = []

    wav_dir.mkdir(parents=True, exist_ok=True)
    for i in range(start_index, hi):
        stem = _utterance_stem(args.subset, i, args.max_seconds)
        wav_path = wav_dir / f"{stem}.wav"
        row = ds[i]
        dur = _save_wav_row(row, wav_path, args.max_seconds)
        durations.append(dur)
        stems.append(stem)
        wav_paths.append(wav_path.resolve())
        rows.append(row)

    if args.dump_golden:
        golden_root.mkdir(parents=True, exist_ok=True)
        dump_py = _REPO_ROOT / "scripts" / "dump_diarization_golden.py"
        cmd = [
            sys.executable,
            str(dump_py),
            "-o",
            str(golden_root),
            "--checkpoint",
            args.checkpoint,
            *[str(p) for p in wav_paths],
        ]
        if args.revision:
            cmd.extend(["--revision", args.revision])
        if token:
            cmd.extend(["--token", str(token)])
        print(
            f"Running golden dump: {dump_py.name} -o {golden_root} ({len(wav_paths)} wav(s))",
            flush=True,
        )
        subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)

    if not args.skip_cpp:
        cpp_bin = args.cpp_binary.resolve()
        if not cpp_bin.is_file():
            raise SystemExit(f"C++ binary not found: {cpp_bin} (build with scripts/build_cpp.sh)")

        manifest_path = work / "cpp_diar_manifest.tsv"
        man_lines: list[str] = []
        for wav_path, stem in zip(wav_paths, stems, strict=True):
            out_json = cpp_out / f"{stem}.json"
            man_lines.append(f"{wav_path}\t{out_json}")
        manifest_path.write_text("\n".join(man_lines) + "\n", encoding="utf-8")

        cmd_cpp = [
            str(cpp_bin),
            "--manifest",
            str(manifest_path),
        ]
        if args.cluster_cadence is not None:
            cmd_cpp += ["--cluster-cadence", str(args.cluster_cadence)]
        if args.analyze_cadence is not None:
            cmd_cpp += ["--analyze-cadence", str(args.analyze_cadence)]
        if args.segmentation_onnx is not None:
            cmd_cpp += ["--segmentation-onnx", str(args.segmentation_onnx.resolve())]
        if args.embedding_onnx is not None:
            cmd_cpp += ["--embedding-onnx", str(args.embedding_onnx.resolve())]
        print("Running C++ (streaming):", cpp_bin.name, flush=True)
        subprocess.run(cmd_cpp, cwd=str(_REPO_ROOT), check=True)

        if args.cpp_clustering_check and stems:
            clu_bin = cpp_bin.parent / "clustering_golden_test"
            utter0 = golden_root / stems[0]
            if clu_bin.is_file() and utter0.is_dir():
                print(
                    f"Running clustering_golden_test on first utterance: {utter0.name}",
                    flush=True,
                )
                subprocess.run(
                    [str(clu_bin), str(utter0), str(xvec_resolved), str(plda_resolved)],
                    cwd=str(_REPO_ROOT),
                    check=False,
                )

    pipeline = None
    if not args.skip_pyannote:
        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)

        pipeline = Pipeline.from_pretrained(
            args.checkpoint,
            revision=args.revision,
            token=token,
        )
        if pipeline is None:
            raise SystemExit("Pipeline.from_pretrained returned None (token / download error).")
        pipeline.to(device)

    py_ders: list[float] = []
    cpp_ders: list[float] = []

    print()
    print(f"{'idx':>4}  {'uri':<42}  {'DER py':>8}  {'DER cpp':>8}")
    print("-" * 72)

    for j, (i, stem, wav_path, row, dur) in enumerate(
        zip(
            range(start_index, hi),
            stems,
            wav_paths,
            rows,
            durations,
            strict=True,
        )
    ):
        uri = stem
        ref = _row_to_reference(row, crop_end=dur, uri=uri)
        uem = Timeline([Segment(0.0, dur)])

        der_py = float("nan")
        if pipeline is not None:
            file = {"audio": str(wav_path), "uri": uri}
            with torch.inference_mode():
                out = pipeline(file)
            hyp_py = out.speaker_diarization
            der_py = float(DiarizationErrorRate()(ref, hyp_py, uem=uem))
        py_ders.append(der_py)

        der_cpp = float("nan")
        if not args.skip_cpp:
            cpp_json = cpp_out / f"{stem}.json"
            if cpp_json.is_file():
                hyp_cpp = _json_diarization_to_annotation(cpp_json, uri=uri)
                der_cpp = float(DiarizationErrorRate()(ref, hyp_cpp, uem=uem))
        cpp_ders.append(der_cpp)

        print(f"{i:4d}  {uri:<42}  {100.0 * der_py:7.2f}%  {100.0 * der_cpp:7.2f}%")

    def _mean(xs: list[float]) -> float:
        vals = [x for x in xs if not np.isnan(x)]
        return float(np.mean(vals)) if vals else float("nan")

    print("-" * 72)
    print(
        f"{'mean':>4}  {'(over files with finite DER)':<42}  "
        f"{100.0 * _mean(py_ders):7.2f}%  {100.0 * _mean(cpp_ders):7.2f}%"
    )
    print()
    print("Reference: HF ``timestamps_*`` / ``speakers`` clipped to truncated WAV length.")
    print("Python: full SpeakerDiarization output on each WAV.")
    print(
        "C++: cpp-annote-cli (segmentation ORT + ORT embedding + VBx; clusters computed in C++, not from NPZ)."
    )


if __name__ == "__main__":
    main()
