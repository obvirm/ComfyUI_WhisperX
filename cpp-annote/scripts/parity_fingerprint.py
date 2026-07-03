#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Float32 fingerprint matching ``pyannote::parity::fingerprint_float32`` (``parity_log.cpp``).

Used to diff stderr lines from C++ ``PYANNOTE_CPP_PARITY=1`` against Python-side arrays
(``embeddings.npz``, PLDA ``fea``, etc.) without shipping full tensors.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


def fingerprint_float32(a: np.ndarray, stride: int = 409) -> str:
    x = np.asarray(a, dtype=np.float32).ravel()
    n = int(x.size)
    if stride < 1:
        stride = 1
    h = FNV_OFFSET
    for i in range(0, n, stride):
        bits = struct.unpack("<I", struct.pack("<f", float(x[i])))[0]
        h = (h ^ bits) * FNV_PRIME
        h &= 0xFFFFFFFFFFFFFFFF
    h ^= n
    return f"{h:016x}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", type=Path, help="path to .npz")
    ap.add_argument("key", help="array key, e.g. embeddings or fea")
    ap.add_argument("--stride", type=int, default=409, help="must match C++ default")
    args = ap.parse_args()
    z = np.load(args.npz)
    if args.key not in z:
        print(f"missing key {args.key!r} in {args.npz}", file=sys.stderr)
        return 2
    fp = fingerprint_float32(z[args.key], stride=args.stride)
    arr = z[args.key]
    nan_ct = int(np.isnan(np.asarray(arr, dtype=np.float32)).sum())
    print(f"key={args.key} shape={arr.shape} dtype={arr.dtype} nan_count={nan_ct} fp={fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
