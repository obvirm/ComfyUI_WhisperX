#!/usr/bin/env python3
# MIT License
#
# Run staged C++ parity checks plus optional DER between golden and C++ diarization JSON.
# Use this to localize why C++ DER is worse than Python (oracle path vs full VBx).
#
# Typical oracle-path diagnosis:
#   1) full_segmentation_window_parity_test — if FAIL, ORT segmentations ≠ Torch dump;
#      oracle hard_clusters are inconsistent with C++ inputs → large DER.
#   2) speaker_count_golden_test — if FAIL, count path drift (usually follows (1)).
#   3) reconstruct_golden_test — if FAIL, reconstruct / discrete tensor mismatch.
#   4) DER(golden diarization.json, cpp.json) — if (1)-(3) PASS, diff is mostly
#      to_annotation / label mapping / exclusive vs overlap vs eval reference.

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for d in here.parents:
        if (d / "cpp" / "scripts" / "dump_diarization_golden.py").is_file():
            return d
    raise RuntimeError("Could not find repo root (dump_diarization_golden.py).")


def _run(name: str, argv: list[str], cwd: Path | None) -> int:
    print()
    print("---", name, "---")
    print(shlex.join(argv))
    r = subprocess.run(argv, cwd=str(cwd) if cwd else None)
    print("exit_code:", r.returncode)
    return int(r.returncode)


def _json_to_annotation(path: Path, uri: str):
    import json

    from pyannote.core import Annotation, Segment

    ann: Annotation = Annotation(uri=uri)
    with path.open(encoding="utf-8") as f:
        turns = json.load(f)
    for t in turns:
        ann[Segment(float(t["start"]), float(t["end"]))] = str(t["speaker"])
    return ann


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Staged parity checks for C++ diarization vs golden dump.")
    ap.add_argument(
        "--golden-utterance-dir",
        type=Path,
        required=True,
        help="Per-utterance folder (contains binarized_segmentations.npz, diarization.json, …)",
    )
    ap.add_argument("--wav", type=Path, required=True, help="Same WAV used for the golden dump / eval")
    ap.add_argument("--build-dir", type=Path, default=root / "cpp" / "build", help="CMake build directory")
    ap.add_argument(
        "--segmentation-onnx",
        type=Path,
        default=root / "cpp" / "artifacts" / "community1-segmentation.onnx",
    )
    ap.add_argument(
        "--receptive-field",
        type=Path,
        default=None,
        help="Defaults to <golden>/../receptive_field.json",
    )
    ap.add_argument(
        "--cpp-diarization-json",
        type=Path,
        default=None,
        help="If set, print DER between golden diarization.json and this file (same reference as eval).",
    )
    ap.add_argument("--uri", default=None, help="URI for annotation (default: utterance folder name)")
    ap.add_argument(
        "--skip-clustering-golden",
        action="store_true",
        help="Do not run clustering_golden_test (VBx numeric drift vs SciPy is known).",
    )
    args = ap.parse_args()

    golden: Path = args.golden_utterance_dir.resolve()
    wav: Path = args.wav.resolve()
    bdir: Path = args.build_dir.resolve()
    onnx: Path = args.segmentation_onnx.resolve()

    rf: Path = args.receptive_field if args.receptive_field is not None else golden.parent / "receptive_field.json"
    rf = rf.resolve()

    uri = args.uri or golden.name

    codes: list[tuple[str, int]] = []

    full_seg = bdir / "full_segmentation_window_parity_test"
    if not full_seg.is_file():
        print(f"Missing {full_seg} — reconfigure and build cpp (full_segmentation_window_parity_test).", file=sys.stderr)
        raise SystemExit(1)

    codes.append(
        (
            "full_segmentation_window_parity_test",
            _run(
                "full_segmentation_window_parity_test",
                [str(full_seg), str(onnx), str(golden), str(wav)],
                cwd=root,
            ),
        )
    )

    sct = bdir / "speaker_count_golden_test"
    if sct.is_file():
        codes.append(
            (
                "speaker_count_golden_test",
                _run("speaker_count_golden_test", [str(sct), str(golden), str(rf)], cwd=root),
            )
        )

    rgt = bdir / "reconstruct_golden_test"
    if rgt.is_file():
        codes.append(("reconstruct_golden_test", _run("reconstruct_golden_test", [str(rgt), str(golden)], cwd=root)))

    emb_onnx = root / "cpp" / "artifacts" / "community1-embedding.onnx"
    emb_npz = golden / "embedding_chunk0_spk0_ort.npz"
    emb_test = bdir / "embedding_golden_test"
    if emb_test.is_file() and emb_onnx.is_file() and emb_npz.is_file():
        codes.append(
            ("embedding_golden_test", _run("embedding_golden_test", [str(emb_test), str(emb_onnx), str(golden)], cwd=root))
        )

    clu_test = bdir / "clustering_golden_test"
    if clu_test.is_file() and not args.skip_clustering_golden:
        xvec = os.environ.get("PYANNOTE_CPP_XVEC")
        plda = os.environ.get("PYANNOTE_CPP_PLDA")
        if xvec and plda and Path(xvec).is_file() and Path(plda).is_file():
            env = os.environ.copy()
            codes.append(
                (
                    "clustering_golden_test",
                    _run(
                        "clustering_golden_test",
                        [str(clu_test), str(golden), xvec, plda],
                        cwd=root,
                    ),
                )
            )
        else:
            print()
            print("--- clustering_golden_test ---")
            print("SKIP (set PYANNOTE_CPP_XVEC and PYANNOTE_CPP_PLDA to plda/xvec_transform.npz paths)")

    gold_json = golden / "diarization.json"
    if args.cpp_diarization_json is not None and gold_json.is_file():
        cpp_json = args.cpp_diarization_json.resolve()
        print()
        print("--- DER(golden diarization.json vs C++ json) ---")
        try:
            from pyannote.core import Segment, Timeline
            from pyannote.metrics.diarization import DiarizationErrorRate
        except ImportError as e:
            print("SKIP:", e)
        else:
            ref_py = _json_to_annotation(gold_json, uri=uri)
            hyp_cpp = _json_to_annotation(cpp_json, uri=uri)
            t_end = max(ref_py.get_timeline().extent().end, hyp_cpp.get_timeline().extent().end)
            uem = Timeline([Segment(0.0, t_end)])
            # reference = Python pipeline dump; hypothesis = C++ JSON (optimal speaker mapping inside metric).
            der = float(DiarizationErrorRate()(ref_py, hyp_cpp, uem=uem))
            print(f"DER(python_dump diarization.json, cpp_json) = {100.0 * der:.2f}%")

    print()
    print("=== summary ===")
    for name, c in codes:
        print(f"  {name}: {'PASS' if c == 0 else 'FAIL'} ({c})")
    snap = golden.parent / "pipeline_snapshot.json"
    if snap.is_file():
        print()
        print("--- pipeline_snapshot.json (keys used by C++) ---")
        d = json.loads(snap.read_text(encoding="utf-8"))
        for key in sorted(d):
            if key.startswith("segmentation.") or key.startswith("clustering.") or key in (
                "embedding_exclude_overlap",
            ):
                print(f"  {key}: {d[key]!r}")

    if any(c != 0 for _, c in codes):
        print()
        print(
            "If full_segmentation_window_parity_test failed, fix ORT/Torch segmentation parity first; "
            "oracle clustering cannot match C++ segmentations otherwise."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
