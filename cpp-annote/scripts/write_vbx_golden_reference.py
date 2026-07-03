#!/usr/bin/env python3
"""Build ``vbx_reference.npz`` for ``vbx_parity_test`` (fixed VBx iterations, no ELBO early stop).

Loads the same train / x0 / fea / AHC / Phi path as ``VBxClustering`` + ``cluster_vbx``, then runs
exactly ``n_vbx_iters`` VBx iterations (matching ``vbx.py`` math) and writes traces for C++.

Example:
  python cpp/scripts/write_vbx_golden_reference.py \\
    --golden-utterance cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s \\
    --xvec ~/.cache/huggingface/hub/.../xvec_transform.npz \\
    --plda ~/.cache/huggingface/hub/.../plda.npz \\
    --out cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s/vbx_reference.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.special import logsumexp, softmax


def load_vbx():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "src", "pyannote", "audio", "utils", "vbx.py")
    spec = importlib.util.spec_from_file_location("vbx", os.path.abspath(root))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.vbx_setup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-utterance", required=True, help="utterance dir with embeddings.npz + binarized_segmentations.npz")
    ap.add_argument("--xvec", required=True)
    ap.add_argument("--plda", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--Fa", type=float, default=0.07)
    ap.add_argument("--Fb", type=float, default=0.8)
    ap.add_argument("--lda-dim", type=int, default=128)
    ap.add_argument("--min-active-ratio", type=float, default=0.2)
    ap.add_argument("--init-smoothing", type=float, default=7.0)
    ap.add_argument("--n-vbx-iters", type=int, default=20, help="fixed iterations (no ELBO early stop in reference)")
    args = ap.parse_args()

    vbx_setup = load_vbx()
    golden = args.golden_utterance
    emb = np.load(os.path.join(golden, "embeddings.npz"))["embeddings"]
    binar = np.load(os.path.join(golden, "binarized_segmentations.npz"))["data"]
    c, s, dim = emb.shape
    f = binar.shape[1]
    single_active_mask = (np.sum(binar, axis=2, keepdims=True) == 1)
    num_clean_frames = np.sum(binar * single_active_mask, axis=1)
    active = num_clean_frames >= args.min_active_ratio * f
    valid = ~np.isnan(emb).any(axis=2)
    chunk_idx, spk_idx = np.where(active * valid)
    chunk_idx = chunk_idx.astype(np.int32)
    spk_idx = spk_idx.astype(np.int32)
    train = emb[chunk_idx, spk_idx].astype(np.float64)
    train_n = train / np.linalg.norm(train, axis=1, keepdims=True)
    z = linkage(train_n, method="centroid", metric="euclidean")
    pd = pdist(train_n, metric="euclidean")
    ahc = fcluster(z, args.threshold, criterion="distance") - 1
    _, ahc = np.unique(ahc, return_inverse=True)

    xvec_tf, plda_tf, plda_psi = vbx_setup(args.xvec, args.plda)
    # C-contiguous row-major so C++ ``Map`` / row-major indexing matches ``(i, j)``.
    x0 = np.ascontiguousarray(xvec_tf(train).astype(np.float64))
    fea = plda_tf(x0, lda_dim=args.lda_dim).astype(np.float64)
    phi = plda_psi[: args.lda_dim].astype(np.float64)

    qinit = np.zeros((len(ahc), int(ahc.max()) + 1), dtype=np.float64)
    qinit[np.arange(len(ahc)), ahc.astype(int)] = 1.0
    gamma = softmax(qinit * args.init_smoothing, axis=1)
    s_spk = gamma.shape[1]
    pi = np.ones(s_spk, dtype=np.float64) / s_spk
    d = fea.shape[1]
    g = -0.5 * (np.sum(fea**2, axis=1, keepdims=True) + d * np.log(2 * np.pi))
    v = np.sqrt(phi)
    rho = fea * v

    tr_g = []
    tr_p = []
    for _ in range(args.n_vbx_iters):
        inv_l = 1.0 / (1 + args.Fa / args.Fb * gamma.sum(axis=0, keepdims=True).T * phi)
        alpha = args.Fa / args.Fb * inv_l * gamma.T.dot(rho)
        log_p_ = args.Fa * (rho.dot(alpha.T) - 0.5 * (inv_l + alpha**2).dot(phi) + g)
        eps = 1e-8
        lpi = np.log(pi + eps)
        log_p_x = logsumexp(log_p_ + lpi, axis=-1)
        gamma = np.exp(log_p_ + lpi - log_p_x[:, None])
        pi = gamma.sum(axis=0)
        pi /= pi.sum()
        tr_g.append(gamma.copy())
        tr_p.append(pi.copy())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(
        args.out,
        x0=x0,
        fea=fea,
        Phi=phi,
        ahc=ahc.astype(np.int32),
        train=train.astype(np.float64),
        train_n=train_n.astype(np.float64),
        train_chunk_idx=chunk_idx,
        train_speaker_idx=spk_idx,
        fcluster_threshold=np.float64(args.threshold),
        pdist_condensed=pd.astype(np.float64),
        linkage_Z=z.astype(np.float64).reshape(-1),
        Fa=np.float64(args.Fa),
        Fb=np.float64(args.Fb),
        init_smoothing=np.float64(args.init_smoothing),
        n_vbx_iters=np.int32(args.n_vbx_iters),
        gamma_trace=np.stack(tr_g, axis=0),
        pi_trace=np.stack(tr_p, axis=0),
    )
    print("Wrote", args.out, "T,S,D=", fea.shape[0], s_spk, fea.shape[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
