#!/usr/bin/env python3
"""
DTXQCD Pfaffian-sign reweighting analysis.

Reads hmc_diagnostics.<traj>.h5 files produced by Test_dtxqcd_2pt_gencfgs
(after commit 9fe39fbb) and:
  - derives per-traj sign s_n = (-1)^{n_neg(g5M_evals[n])}
  - reports <s> ± σ, effective sample size N_eff = N·<s>²
  - histograms |λ|_min and flags trajs near the sign boundary
  - emits an optional `signs.<traj>.h5` sidecar with the per-traj sign
    array, ready to be multiplied into any downstream observable

For a measured observable O_n stored elsewhere (e.g. meas_conn h5):
  <O>_phys = mean(s*O) / mean(s)
with errors via jackknife over the joint (s, O) ensemble.

Usage:
  ./analyze_sign_reweighting.py <ensemble_dir> [--emit-signs]
  ./analyze_sign_reweighting.py configs_2pt_dtxqcd_v2_lam10_g5M
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np


def jackknife_mean_err(x):
    n = len(x)
    if n < 2:
        return float(np.mean(x)) if n else 0.0, 0.0
    s = np.sum(x)
    jk = (s - x) / (n - 1)
    m = np.mean(jk)
    err = np.sqrt((n - 1) * np.mean((jk - m) ** 2))
    return float(m), float(err)


def per_traj_sign(evals_row):
    """parity_lowK from a single traj's lowest-K signed eigenvalues."""
    arr = np.asarray(evals_row, dtype=float)
    n_neg = int(np.sum(arr < 0))
    return 1 if (n_neg & 1) == 0 else -1, n_neg, float(np.min(np.abs(arr)))


def load_diagnostics(ensemble_dir):
    """Concatenate all hmc_diagnostics.*.h5 in `ensemble_dir` in traj
    order. Returns dict of arrays keyed by traj."""
    files = sorted(
        glob.glob(os.path.join(ensemble_dir, "hmc_diagnostics.*.h5")),
        key=lambda p: int(os.path.basename(p).split(".")[1]),
    )
    if not files:
        sys.exit(f"no hmc_diagnostics.*.h5 in {ensemble_dir}")
    rows = []
    for f in files:
        with h5py.File(f, "r") as h:
            if "g5M_evals" not in h:
                print(f"  {os.path.basename(f)}: no g5M_evals key, skipping")
                continue
            trajs = h["traj"][...]
            evals = h["g5M_evals"][...]  # (Ntraj, K)
            plaq = h["plaq"][...]
            for i, t in enumerate(trajs):
                rows.append((int(t), evals[i], float(plaq[i])))
    rows.sort(key=lambda r: r[0])
    if not rows:
        sys.exit("no g5M_evals data found")
    trajs = np.array([r[0] for r in rows])
    evals = np.stack([r[1] for r in rows])
    plaq = np.array([r[2] for r in rows])
    return trajs, evals, plaq


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ensemble_dir")
    p.add_argument("--emit-signs", action="store_true",
                   help="write signs.<last_traj>.h5 sidecar")
    p.add_argument("--min-lam-warn", type=float, default=0.5,
                   help="warn on |λ|_min below this (default 0.5)")
    args = p.parse_args()

    trajs, evals, plaq = load_diagnostics(args.ensemble_dir)
    N = len(trajs)
    K = evals.shape[1]

    signs = np.empty(N, dtype=np.int8)
    n_neg = np.empty(N, dtype=np.int32)
    abs_min = np.empty(N, dtype=np.float64)
    for i in range(N):
        s, nn, am = per_traj_sign(evals[i])
        signs[i] = s
        n_neg[i] = nn
        abs_min[i] = am

    n_pos = int(np.sum(signs > 0))
    n_neg_traj = int(np.sum(signs < 0))
    s_mean, s_err = jackknife_mean_err(signs.astype(np.float64))
    N_eff = N * (s_mean ** 2) if abs(s_mean) > 0 else 0.0

    print(f"Ensemble: {args.ensemble_dir}")
    print(f"  Trajs:    {N}  (range {trajs[0]}..{trajs[-1]}, "
          f"step {int(trajs[1]-trajs[0]) if N>1 else '-'})")
    print(f"  K=lowest: {K}")
    print(f"  Sign breakdown: {n_pos}/{N} (+), {n_neg_traj}/{N} (−)")
    print(f"  ⟨s⟩  = {s_mean:+.5f} ± {s_err:.5f}")
    print(f"  N_eff = N·⟨s⟩² = {N_eff:.1f}    (penalty {N/max(N_eff,1e-12):.2f}×)")
    print(f"  ⟨|λ|_min⟩ = {abs_min.mean():.4f}, "
          f"min over trajs = {abs_min.min():.4f}")

    near = int(np.sum(abs_min < args.min_lam_warn))
    if near:
        print(f"  ⚠  {near}/{N} trajs have |λ|_min < {args.min_lam_warn}")
        bad_idx = np.where(abs_min < args.min_lam_warn)[0]
        for idx in bad_idx[:10]:
            print(f"    traj {trajs[idx]}: |λ|_min={abs_min[idx]:.4f} "
                  f"sign={int(signs[idx]):+d}")
        if len(bad_idx) > 10:
            print(f"    ...and {len(bad_idx)-10} more")

    # Sign-flip events (per-step change in sign)
    flips = int(np.sum(signs[1:] != signs[:-1]))
    print(f"  Sign flips between adjacent trajs: {flips}")

    # Reweighting demonstration on plaq (sign-blind, so should match)
    p_mean, p_err = jackknife_mean_err(plaq)
    p_rw_num, _ = jackknife_mean_err(signs * plaq)
    p_rw = p_rw_num / s_mean if abs(s_mean) > 1e-12 else float("nan")
    print(f"  ⟨plaq⟩_HMC  = {p_mean:.6f} ± {p_err:.6f}")
    print(f"  ⟨plaq⟩_rw   = {p_rw:.6f}    (sign-blind obs; should match HMC)")

    if args.emit_signs:
        out = os.path.join(args.ensemble_dir,
                           f"signs.{int(trajs[-1])}.h5")
        with h5py.File(out, "w") as h:
            h["traj"] = trajs
            h["sign"] = signs
            h["n_neg_lowK"] = n_neg
            h["abs_min"] = abs_min
            h["g5M_evals"] = evals
            h.attrs["K_lowest"] = K
            h.attrs["s_mean"] = s_mean
            h.attrs["s_err"] = s_err
            h.attrs["N_eff"] = N_eff
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
