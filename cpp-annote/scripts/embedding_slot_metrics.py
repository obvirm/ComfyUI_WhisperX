#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cross-check ``embedding_golden_test`` CSV rows vs golden ``binarized_segmentations.npz``.

Recomputes ``sum_clean``, ``prefer_clean``, ``n_active_seg`` using the same rules as C++
(``cpp-annote.cpp`` / ``embedding_golden_test --all``): overlap mask, ``> 0.5`` counts on the
used segmentation mask, and ``min_nf_seg`` from the CSV column.

Usage:
  python cpp/scripts/embedding_slot_metrics.py \\
    --golden-dir cpp/golden/.../callhome_eng_data_idx0_head120s \\
    --from-cpp-csv /tmp/embedding_nan_mismatch.csv

Exit 1 if any row's ``n_active_seg`` (or legacy ``n_keep``) differs from the CSV (sanity check on the dump).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def slot_metrics(
    binarized: np.ndarray,
    c: int,
    s: int,
    min_nf_seg: int,
) -> tuple[float, bool, int]:
    """Returns (sum_clean, prefer_clean, n_active_seg)."""
    c_int, s_int, f_int = int(c), int(s), int(binarized.shape[1])
    clean_col = np.zeros(f_int, dtype=np.float32)
    full_col = np.zeros(f_int, dtype=np.float32)
    for f in range(f_int):
        rowsum = float(binarized[c_int, f, :].sum())
        overlap_ok = 1.0 if rowsum < 2.0 - 1e-5 else 0.0
        v = float(binarized[c_int, f, s_int])
        full_col[f] = v
        clean_col[f] = v * overlap_ok
    sum_clean = float((clean_col > 0.5).sum())
    prefer_clean = True if min_nf_seg < 0 else (sum_clean > float(min_nf_seg))
    src = clean_col if prefer_clean else full_col
    n_active_seg = int((src > 0.5).sum())
    return sum_clean, prefer_clean, n_active_seg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden-dir", type=Path, required=True)
    ap.add_argument("--from-cpp-csv", type=Path, required=True)
    args = ap.parse_args()

    bin_npz = args.golden_dir / "binarized_segmentations.npz"
    if not bin_npz.is_file():
        print("missing", bin_npz, file=sys.stderr)
        return 2
    binarized = np.load(bin_npz)["data"].astype(np.float32)

    mism = 0
    with args.from_cpp_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            c = int(row["c"])
            s = int(row["s"])
            min_nf_seg = int(row["min_nf_seg"])
            nk_csv = int(row["n_active_seg"]) if "n_active_seg" in row else int(row["n_keep"])
            sc, pref, nk = slot_metrics(binarized, c, s, min_nf_seg)
            if nk != nk_csv or abs(sc - float(row["sum_clean"])) > 1e-4:
                print(
                    f"MISMATCH recompute c={c} s={s}: csv n_active_seg={nk_csv} py={nk} "
                    f"csv sum_clean={row['sum_clean']} py={sc}",
                    file=sys.stderr,
                )
                mism += 1
    if mism:
        print(f"FAIL {mism} rows", file=sys.stderr)
        return 1
    print("PASS: Python recomputed n_active_seg / sum_clean match C++ CSV for all rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
