#!/usr/bin/env python3
"""Ensemble significance of the undeformed aux-baryon (baryon8) correlator:
baryon_nucleon_avg (P+ fwd nucleon channel, source/component averaged) per
config -> C(t) mean, error (tau_int~0.5 at 10-traj spacing), significance
and m_eff at small t.  Usage: analyze_baryon8_signal.py <lam> [<lam> ...]"""
import glob
import sys
import numpy as np
import h5py

P = "/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt"


def getc(v):
    v = np.array(v)
    if v.dtype.names and "re" in v.dtype.names:
        v = v["re"] + 1j * v["im"]
    return v


for lam in sys.argv[1:]:
    files = sorted(glob.glob(f"{P}/dtxqcd_lam{lam}_baryon8/baryon_dtxqcd_*.h5"),
                   key=lambda s: int(s.split("_")[-1].split(".")[0]))
    C = []
    nfall, maxdev = 0, 0.0
    FWD = ["baryon_uu_p_up_fwd", "baryon_uu_p_dn_fwd",
           "baryon_uu_n_up_fwd", "baryon_uu_n_dn_fwd"]
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                # nucleon channel = mean of the 4 P+ fwd uu components; the
                # early post-fix writer (2026-07-11) lacks the precomputed
                # baryon_nucleon_avg, so fall back to the components.
                cf = np.mean([np.real(getc(h[k])).ravel() for k in FWD], axis=0)
                if "baryon_nucleon_avg" in h:
                    c = np.real(getc(h["baryon_nucleon_avg"])).ravel()
                    maxdev = max(maxdev, np.abs(c - cf).max() /
                                 max(np.abs(c).max(), 1e-300))
                else:
                    nfall += 1
                C.append(cf)
        except Exception as e:
            print(f"skip {f}: {e}")
    print(f"   ({nfall} files via fwd-component fallback; nucleon_avg vs "
          f"component-mean max rel dev {maxdev:.1e})")
    C = np.array(C)
    n = len(C)
    m = C.mean(axis=0)
    e = C.std(axis=0, ddof=1) / np.sqrt(n)
    print(f"== lambda={lam}: {n} configs (baryon_nucleon_avg) ==")
    print(f"{'t':>3} {'C(t)':>12} {'err':>10} {'sig':>6} {'m_eff':>7}")
    for t in range(0, 7):
        meff = (np.log(m[t] / m[t + 1]) if (m[t] > 0 and m[t + 1] > 0)
                else float("nan"))
        print(f"{t:3d} {m[t]:+12.3e} {e[t]:10.2e} {m[t]/e[t]:6.1f} {meff:7.3f}")
