#!/usr/bin/env python3
"""Aux-baryon graded estimator table (lam5 pilot configs), three estimators:
  (A) eta=0 unreweighted        : <C_undef>          [physical units]
  (B) graded, NO w              : <C_def>            [GRADED units -- grading
                                   is baked into the deformed solve; smaller
                                   mean AND smaller std, S/N ~ unchanged]
  (C) graded + w                : <w C_def>/<w>       [physical units; the
                                   only estimator that restores the mean into
                                   the graded-noise world]
per dt at eta = 0.4, 0.6 (pilot w's from the high-stat smokes)."""
import glob
import numpy as np
import h5py

T, V3 = 48, 16 ** 3
P = "/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt"
COMPS = [f"uu_{f}_{s}" for f in ("p", "n") for s in ("up", "dn")]
TSRC = [0, 12, 24, 36]
# cumulative [0,eta] log-weights per pilot config (re,im), high-stat smokes
W = {
    0.4: {300: (-0.027, 0.366), 400: (0.960, 0.343), 500: (1.247, 0.674),
          600: (0.778, 3.293)},
    0.6: {300: (25.588, 0.970), 400: (26.143, 1.069), 500: (24.756, 1.836)},
}
TAG = {0.4: "eta0p4_s48", 0.6: "eta0p6_s48"}


def getc(v):
    v = np.array(v)
    return (v["re"] + 1j * v["im"]) if (v.dtype.names and "re" in v.dtype.names) \
        else v.astype(complex)


def fwd4(path):
    with h5py.File(path) as h:
        acc = np.zeros(T, complex)
        for c in COMPS:
            SB = getc(h[f"baryon_SB_{c}_2d"]).reshape(T, T)
            for dt in range(T):
                for ts in TSRC:
                    tl = (dt + ts) % T
                    sgn = -1.0 if (dt + ts) >= T else 1.0
                    acc[dt] += sgn * SB[ts][tl]
        return acc / (4.0 * V3 * V3 * len(TSRC)) * 256.0


for eta in (0.4, 0.6):
    trs = sorted(W[eta])
    trs = [t for t in trs
           if glob.glob(f"{P}/dtxqcd_lam5_deform_{TAG[eta]}/baryon_deform_dtxqcd_{t}.h5")
           and glob.glob(f"{P}/dtxqcd_lam5_baryon8/baryon_dtxqcd_{t}.h5")]
    U = np.array([fwd4(f"{P}/dtxqcd_lam5_baryon8/baryon_dtxqcd_{t}.h5") for t in trs])
    D = np.array([fwd4(f"{P}/dtxqcd_lam5_deform_{TAG[eta]}/baryon_deform_dtxqcd_{t}.h5")
                  for t in trs])
    lw = np.array([complex(*W[eta][t]) for t in trs])
    lw -= complex(lw.real.mean(), 0.0)
    w = np.exp(lw)
    n = len(trs)
    print(f"\n==== eta={eta}, {n} pilot configs {trs}  "
          f"(Re<w>/<|w|> = {w.sum().real/np.abs(w).sum():+.3f}) ====")
    print(f"{'dt':>3} | {'(A) eta=0 unrew':>22} | "
          f"{'(B) graded no-w [graded u]':>28} | {'(C) graded + w':>22}")
    for dt in (1, 2, 3):
        a, ea = U[:, dt].real.mean(), U[:, dt].real.std(ddof=1) / np.sqrt(n)
        b, eb = D[:, dt].real.mean(), D[:, dt].real.std(ddof=1) / np.sqrt(n)
        num = (w * D[:, dt]).sum().real
        den = w.sum().real
        sub = np.array([((w * D[:, dt]).sum() - w[i] * D[i, dt]).real /
                        (den - w[i].real) for i in range(n)])
        cfull = num / den
        ec = np.sqrt((n - 1) / n * ((sub - cfull) ** 2).sum())
        print(f"{dt:3d} | {a:+10.2e}±{ea:8.2e} | {b:+11.2e}±{eb:8.2e} | "
              f"{cfull:+10.2e}±{ec:8.2e}")
