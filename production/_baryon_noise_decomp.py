#!/usr/bin/env python3
"""Baryon grading noise decomposition: is the deformed baryon correlator's
per-config shrink the pure analytic e^{-2 eta p} (noiseless, dibaryon-like),
or does the deformed PROPAGATOR add fluctuations on top?
  std(C_def)/std(C_undef)  vs  e^{-2 eta p(dt)}  ->  excess = solve noise."""
import numpy as np
import h5py

T, V3 = 48, 16 ** 3
P = "/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt"
COMPS = [f"uu_{f}_{s}" for f in ("p", "n") for s in ("up", "dn")]
TSRC = [0, 12, 24, 36]
p = np.array([min(t, T - t) for t in range(T)], float)


def gc(v):
    v = np.array(v)
    return (v["re"] + 1j * v["im"]) if (v.dtype.names and "re" in v.dtype.names) \
        else v.astype(complex)


def fwd4(path):
    h = h5py.File(path)
    acc = np.zeros(T, complex)
    for c in COMPS:
        SB = gc(h[f"baryon_SB_{c}_2d"]).reshape(T, T)
        for dt in range(T):
            for ts in TSRC:
                tl = (dt + ts) % T
                sgn = -1.0 if (dt + ts) >= T else 1.0
                acc[dt] += sgn * SB[ts][tl]
    return acc / (4.0 * V3 * V3 * len(TSRC)) * 256.0


for eta, tag, trs in [(0.6, "eta0p6_s48", [300, 400, 500]),
                      (0.4, "eta0p4_s48", [300, 400, 500, 600])]:
    U = np.array([fwd4(f"{P}/dtxqcd_lam5_baryon8/baryon_dtxqcd_{t}.h5") for t in trs])
    D = np.array([fwd4(f"{P}/dtxqcd_lam5_deform_{tag}/baryon_deform_dtxqcd_{t}.h5")
                  for t in trs])
    print(f"== eta={eta}, n={len(trs)} ==")
    print(" dt | std_def/std_undef | analytic e^-2etap | excess (solve noise)")
    for dt in (1, 2, 3, 4):
        su = np.abs(U[:, dt]).std()
        sd = np.abs(D[:, dt]).std()
        g = np.exp(-2 * eta * p[dt])
        print(f" {dt:2d} |      {sd/su:6.3f}       |     {g:7.4f}       "
              f"|   x{(sd/su)/g:7.1f}")
