#!/usr/bin/env python3
"""Cross-lambda comparison from 64-source TXQCD measurements.

Reads pion_per_src and nucleon_per_src from each h5, shifts each source's
correlator so source-time → 0, then computes:
  - mean(C(t)) and SEM(C(t)) across sources (with proper t-shift)
  - effective mass at each t

Usage: ./compare_64src_lambdas.py
       (paths hard-coded for the 2026-05-09 batch)
"""
import h5py
import numpy as np
import os

T = 48
ST = 8       # MEAS_TIME_SRC
SX = 2       # MEAS_SPACE_SRC per dim
NSRC = SX**3 * ST
TLATT = T

SPECS = [
    # md20 cfg.20 (early thermalization, more sources of λ to compare)
    ("5",   20, "λ=5 md20",  "txqcd", "_fromchroma_md20"),
    ("6",   20, "λ=6 md20",  "txqcd", "_fromchroma_md20"),
    ("6.5", 20, "λ=6.5 md20","txqcd", "_fromchroma_md20"),
    ("7",   20, "λ=7 md20",  "txqcd", "_fromchroma_md20"),
    ("7.5", 20, "λ=7.5 md20","txqcd", "_fromchroma_md20"),
    ("8",   20, "λ=8 md20",  "txqcd", "_fromchroma_md20"),
    # md10 latest cfg per stream (deeper thermalization)
    ("6",   50, "λ=6 cfg.50",  "txqcd", "_fromchroma_md10"),
    ("7",   40, "λ=7 cfg.40",  "txqcd", "_fromchroma_md10"),
    ("12",  60, "λ=12 cfg.60", "txqcd", "_fromchroma_md10"),
    ("16",  50, "λ=16 cfg.50", "txqcd", "_fromchroma_md10"),
    # QCD reference (chroma cfg 11100)
    ("qcd", 11100, "QCD chroma 11100", "qcd", ""),
]
DATA_DIR_FMT_TXQCD = "../../meas_2pt/txqcd_lam{lam}.0000{suf}"
DATA_DIR_QCD       = "../../meas_2pt/qcd_chroma_64src"


def load_correlators(h5path):
    """Return shifted, per-source pion + nucleon correlators (NSRC × T)."""
    with h5py.File(h5path, "r") as f:
        pi_raw = np.array(f["pion_per_src"])
        nu_raw = np.array(f["nucleon_per_src"])
        if nu_raw.dtype.names == ("re", "im"):
            nu_raw = nu_raw["re"] + 1j * nu_raw["im"]
    # Each row is one source's correlator on absolute time.  Source s has
    # t_s = (s mod ST) * (T/ST).  Shift to relative time.
    pi_shift = np.zeros((NSRC, T))
    nu_shift = np.zeros((NSRC, T), dtype=complex)
    for s in range(NSRC):
        ts = (s % ST) * (T // ST)
        for t in range(T):
            rel = (t - ts) % T
            pi_shift[s, rel] = pi_raw[s, t]
            nu_shift[s, rel] = nu_raw[s, t]
    return pi_shift, nu_shift


def stats(corr):
    """Mean and SEM (over source axis)."""
    mean = corr.mean(axis=0)
    sem = corr.std(axis=0, ddof=1) / np.sqrt(corr.shape[0])
    return mean.real, sem.real


def effective_mass(c):
    """Effective mass: m_eff(t) = log(C(t)/C(t+1)). NaN where ratio invalid."""
    em = np.full(T, np.nan)
    for t in range(T - 1):
        if c[t] > 0 and c[t + 1] > 0:
            em[t] = np.log(c[t] / c[t + 1])
    return em


def fmt(x, prec=4):
    if np.isnan(x): return f"{'-':>{prec+5}}"
    return f"{x:>{prec+5}.{prec}g}"


def main():
    results = {}
    for lam_tag, traj, label, kind, suf in SPECS:
        if kind == "qcd":
            h5 = os.path.join(DATA_DIR_QCD, f"conn_qcd_{traj}.h5")
        else:
            # params.h formats lambda with %.4f → "6.5" must become "6.5000".
            lam_fmt = f"{float(lam_tag):.4f}"
            h5 = os.path.join(
                DATA_DIR_FMT_TXQCD.format(lam=lam_fmt, suf=suf),
                f"conn_txqcd_{traj}.h5",
            )
        if not os.path.exists(h5):
            print(f"MISSING: {h5}")
            continue
        pi_shift, nu_shift = load_correlators(h5)
        pi_mean, pi_sem = stats(pi_shift)
        nu_mean, nu_sem = stats(nu_shift)
        results[label] = dict(
            pi_mean=pi_mean, pi_sem=pi_sem,
            nu_mean=nu_mean, nu_sem=nu_sem,
            pi_em=effective_mass(pi_mean),
        )

    if not results:
        print("No results loaded.")
        return

    # Pion mean ± SEM
    print("\n=== Pion correlator mean ± SEM (after t-shift, NSRC=64) ===")
    print("  t  " + "  ".join(f"{k:>22s}" for k in results))
    for t in range(T // 2 + 1):
        row = f"  {t:2d} "
        for k, r in results.items():
            row += f"  {r['pi_mean'][t]:10.4e} ± {r['pi_sem'][t]:8.2e}"
        print(row)

    # Pion SEM/|mean| (relative noise)
    print("\n=== Pion SEM / |mean| (lower = better signal) ===")
    print("  t  " + "  ".join(f"{k:>10s}" for k in results))
    for t in range(T // 2 + 1):
        row = f"  {t:2d} "
        for k, r in results.items():
            rel = r["pi_sem"][t] / abs(r["pi_mean"][t]) if r["pi_mean"][t] != 0 else np.nan
            row += f"  {fmt(rel, 3):>10s}"
        print(row)

    # Pion effective mass
    print("\n=== Pion effective mass (log C(t)/C(t+1)) ===")
    print("  t  " + "  ".join(f"{k:>10s}" for k in results))
    for t in range(T // 2):
        row = f"  {t:2d} "
        for k, r in results.items():
            row += f"  {fmt(r['pi_em'][t], 3):>10s}"
        print(row)

    # Nucleon SEM/|mean|
    print("\n=== Nucleon SEM / |Re(mean)| (lower = better signal) ===")
    print("  t  " + "  ".join(f"{k:>10s}" for k in results))
    for t in range(T // 2 + 1):
        row = f"  {t:2d} "
        for k, r in results.items():
            m = r["nu_mean"][t]
            s = r["nu_sem"][t]
            rel = s / abs(m) if m != 0 else np.nan
            row += f"  {fmt(rel, 3):>10s}"
        print(row)

    # Nucleon effective mass (use real part of correlator)
    print("\n=== Nucleon effective mass (log Re(C(t))/Re(C(t+1))) ===")
    print("  t  " + "  ".join(f"{k:>10s}" for k in results))
    for k, r in results.items():
        em = np.full(T, np.nan)
        for t in range(T - 1):
            if r["nu_mean"][t] > 0 and r["nu_mean"][t + 1] > 0:
                em[t] = np.log(r["nu_mean"][t] / r["nu_mean"][t + 1])
        r["nu_em"] = em
    for t in range(T // 2):
        row = f"  {t:2d} "
        for k, r in results.items():
            row += f"  {fmt(r['nu_em'][t], 3):>10s}"
        print(row)

    # Cross-lambda summary at fixed plateau region
    print("\n=== Plateau summary (t = 8-15 average) ===")
    print(f"{'λ':>6s} {'m_π':>10s} {'m_N':>10s} {'pi_SEM/|C|':>12s} {'N_SEM/|ReC|':>14s}")
    for k, r in results.items():
        mpi = np.nanmean(r["pi_em"][8:16])
        mN = np.nanmean(r["nu_em"][8:16])
        pi_rel = np.nanmean(r["pi_sem"][8:16] / np.abs(r["pi_mean"][8:16]))
        nu_rel = np.nanmean(r["nu_sem"][8:16] / np.abs(r["nu_mean"][8:16]))
        print(f"{k:>6s} {mpi:>10.4f} {mN:>10.4f} {pi_rel:>12.4f} {nu_rel:>14.4f}")


if __name__ == "__main__":
    main()
