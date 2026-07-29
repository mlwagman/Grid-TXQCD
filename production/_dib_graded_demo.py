#!/usr/bin/env python3
"""Graded-down dibaryon estimator demo (sm60 site operator).

Estimator per (ts, dt), tent centered at the d*^3 end (ts):
  O_eta = e^{-3 eta p(dt)} * Re[ (S(ts+dt) + S(ts-dt))/2 * conj(S(ts)) ]
Part 1 (all 175 cfgs, DETERMINISTIC): graded per-sample std vs undeformed --
  the floor reduction is exact bookkeeping; shows the S/N arithmetic.
Part 2 (pilot cfgs with tent DlogPf): reweighted mean <O w>/<w> vs the
  undeformed 175-cfg C(dt) (fixed t0=0 contour; n=3-4 -- pipeline demo)."""
import glob
import numpy as np
import h5py

P = "/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt/dtxqcd_lam5_dibaryon_aux_sm2"
T = 48
p = np.array([min(t, T - t) for t in range(T)], float)

# pilot tent DlogPf (multi-seed averaged high-stat smokes; re, im)
W = {
    0.4: {300: (0.333 - 0.360, 0.280 + 0.086), 400: (0.665 + 0.295, 0.726 + 0.337),
          500: (1.032 + 0.215, 0.363 + 0.311), 600: (0.869 - 0.091, 1.967 + 1.326)},
    0.6: {300: (25.615 - 0.027, 0.604 + 0.366), 400: (25.183 + 0.960, 0.006 + 1.063),
          500: (23.509 + 1.247, 1.162 + 0.674)},
}


def getc(v):
    v = np.array(v)
    return (v["re"] + 1j * v["im"]) if (v.dtype.names and "re" in v.dtype.names) \
        else v.astype(complex)


files = sorted(glob.glob(f"{P}/dibaryon_aux_dtxqcd_*.h5"),
               key=lambda s: int(s.split("_")[-1].split(".")[0]))
trajs = [int(f.split("_")[-1].split(".")[0]) for f in files]
S = np.array([getc(h5py.File(f)["dib_I1_sm60_S_0"]) for f in files])
N = len(S)

# undeformed all-TSRC reference
print(f"{N} configs, sm60 site operator")
print("\nPart 1 -- deterministic floor reduction (all cfgs, all TSRC):")
print(f"{'dt':>3} | {'C_undef':>10} {'sig0':>5} | {'floor':>9} | "
      f"{'graded floor e=0.6':>18} {'C/gfloor':>9}")
for dt in (1, 2, 3, 4, 5):
    X = np.mean(0.5 * (np.roll(S, -dt, axis=1) + np.roll(S, dt, axis=1)) *
                np.conj(S), axis=1).real
    m, e = X.mean(), X.std(ddof=1) / np.sqrt(N)
    g = np.exp(-3 * 0.6 * p[dt])
    print(f"{dt:3d} | {m:+10.2e} {abs(m)/e:5.1f} | {X.std(ddof=1):9.2e} | "
          f"{g*X.std(ddof=1):18.2e} | {abs(m)/(g*X.std(ddof=1)):9.2f}")

print("\nPart 2 -- reweighted graded mean, fixed t0=0 contour, pilot cfgs:")
for eta, wd in W.items():
    sel = [i for i, t in enumerate(trajs) if t in wd]
    w = np.array([np.exp(wd[trajs[i]][0] + 1j * wd[trajs[i]][1]) for i in sel])
    w = w / np.abs(w).mean()          # common-scale normalization
    print(f"  eta={eta} ({len(sel)} cfgs):")
    for dt in (1, 2, 3):
        g = np.exp(-3 * eta * p[dt])
        O = np.array([g * (0.5 * (S[i, dt] + S[i, -dt]) *
                           np.conj(S[i, 0])).real for i in sel])
        rw = (O * w.real).mean() / w.real.mean() / g   # ungraded units
        X = np.mean(0.5 * (np.roll(S, -dt, axis=1) + np.roll(S, dt, axis=1)) *
                    np.conj(S), axis=1).real
        print(f"    dt={dt}: reweighted {rw:+9.2e}   "
              f"(175-cfg undeformed {X.mean():+9.2e})")
