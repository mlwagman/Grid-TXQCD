#!/usr/bin/env python3
"""Paired deformed-vs-undeformed aux baryon analysis.

Usage: analyze_baryon_deform.py [--eta 0.2] [--tag eta0p2]

Auto-pairs every deformed h5 in meas_2pt/dtxqcd_lam4_deform_<tag>/ with its
undeformed partner in meas_2pt/dtxqcd_lam4_baryon8/.

The saved deformed SB has the contour grading BAKED IN, so its reweighted
ensemble mean equals the UNDEFORMED <C(dt)> exactly (Cauchy); the win must
appear as reduced fluctuations.  mwagman's signature (CONFIRMED on the
first 5 pairs, 2026-07-13: 4/5 configs at +0.20..0.28 for eta=0.2): the
per-config |C_def| decays with m_eff + eta (the graded magnitude) while the
phase/sign fluctuations shrink to preserve the mean.

Reports per dt: ensemble means (agreement check), per-config std (the S/N
verdict), magnitude m_eff shift (expect +eta), and the phase spread across
configs (expect smaller for deformed at dt where phases are unsaturated).
"""
import glob
import os
import sys
import numpy as np
import h5py

T, V3 = 48, 16 ** 3
ETA = float(sys.argv[sys.argv.index("--eta") + 1]) if "--eta" in sys.argv else 0.2
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else \
    "eta" + f"{ETA}".replace(".", "p")
P = "/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt"
COMPS = [f"uu_{f}_{s}" for f in ("p", "n") for s in ("up", "dn")]

TRAJS = []
for f in sorted(glob.glob(f"{P}/dtxqcd_lam4_deform_{TAG}/baryon_deform_dtxqcd_*.h5"),
                key=lambda s: int(s.split("_")[-1].split(".")[0])):
    t = int(f.split("_")[-1].split(".")[0])
    if os.path.exists(f"{P}/dtxqcd_lam4_baryon8/baryon_dtxqcd_{t}.h5"):
        TRAJS.append(t)
TSRC = [0, 12, 24, 36]


def getc(v):
    v = np.array(v)
    return (v["re"] + 1j * v["im"]) if (v.dtype.names and "re" in v.dtype.names) \
        else v.astype(complex)


def fwd4(path, tsrc):
    with h5py.File(path) as h:
        acc = np.zeros(T, complex)
        for c in COMPS:
            SB = getc(h[f"baryon_SB_{c}_2d"]).reshape(T, T)
            for dt in range(T):
                for ts in tsrc:
                    tl = (dt + ts) % T
                    sgn = -1.0 if (dt + ts) >= T else 1.0
                    acc[dt] += sgn * SB[ts][tl]
        return acc / (4.0 * V3 * V3 * len(tsrc))


D, Uc = [], []
for tr in TRAJS:
    D.append(fwd4(f"{P}/dtxqcd_lam4_deform_{TAG}/baryon_deform_dtxqcd_{tr}.h5",
                  TSRC))
    Uc.append(fwd4(f"{P}/dtxqcd_lam4_baryon8/baryon_dtxqcd_{tr}.h5", TSRC))
D = np.array(D) * 256.0   # lambda^4 (deform files are unscaled; baryon8 SB raw)
Uc = np.array(Uc) * 256.0
n = len(TRAJS)

print(f"paired configs: {n} {TRAJS}   tsrc={TSRC}   eta={ETA} tent "
      "(grading baked in)")

rng = np.random.default_rng(12345)
NBOOT = 2000


def boot_ratio(u, d):
    """Bootstrap error on std(d)/std(u), paired resampling."""
    idx = rng.integers(0, n, size=(NBOOT, n))
    su = np.array([u[i].std(ddof=1) for i in idx])
    sd = np.array([d[i].std(ddof=1) for i in idx])
    r = sd[su > 0] / su[su > 0]
    r = r[np.isfinite(r)]
    return r.std(ddof=1) if len(r) > 2 else float("nan")


print(f"\n{'dt':>3} | {'<C_undef>':>10} {'std':>9} | {'<C_def>':>10} {'std':>9} | "
      f"{'std ratio':>15} | {'sphi_u':>7} {'sphi_d':>7}")
print("-" * 100)
for dt in range(1, 13):
    mu, su = Uc[:, dt].real.mean(), Uc[:, dt].real.std(ddof=1)
    md, sd = D[:, dt].real.mean(), D[:, dt].real.std(ddof=1)
    dr = boot_ratio(Uc[:, dt].real, D[:, dt].real)
    # phase spread across configs (dispersion of arg, mean-centered)
    pu = np.angle(Uc[:, dt] / Uc[:, dt].mean()) if abs(Uc[:, dt].mean()) > 0 else np.zeros(n)
    pd = np.angle(D[:, dt] / D[:, dt].mean()) if abs(D[:, dt].mean()) > 0 else np.zeros(n)
    print(f"{dt:3d} | {mu:+10.2e} {su:9.2e} | {md:+10.2e} {sd:9.2e} | "
          f"{sd/su:7.3f} +-{dr:6.3f} | {pu.std(ddof=1):7.3f} {pd.std(ddof=1):7.3f}")

# mean preservation: grading is baked in, so the (unreweighted, factor ~1)
# <C_def> must be consistent with <C_undef> -- a failure here is a
# grading/bookkeeping bug, not physics.
print("\nmean preservation, paired diff <C_def - C_undef> (expect 0 within err):")
for dt in range(1, 9):
    diff = (D[:, dt] - Uc[:, dt]).real
    err = diff.std(ddof=1) / np.sqrt(n)
    print(f"  dt={dt}: {diff.mean():+10.2e} +- {err:8.2e}"
          f"   ({diff.mean()/err:+5.1f} sigma)")

# ensemble-level magnitude m_eff signature (per-config lines follow)
lu = np.log(np.abs(Uc[:, 1]) / np.abs(Uc[:, 2]))
ld = np.log(np.abs(D[:, 1]) / np.abs(D[:, 2]))
dm = ld - lu
print(f"\nensemble magnitude m_eff shift (dt=1->2): "
      f"{dm.mean():+.3f} +- {dm.std(ddof=1)/np.sqrt(n):.3f}   (expect +{ETA})")

print("\nper-config MAGNITUDE m_eff (|C| slope dt=1->2), expect def = undef + eta:")
for i, tr in enumerate(TRAJS):
    mu_u = np.log(abs(Uc[i, 1]) / abs(Uc[i, 2]))
    mu_d = np.log(abs(D[i, 1]) / abs(D[i, 2]))
    print(f"  traj {tr}: m|C|_undef = {mu_u:+.3f}   m|C|_def = {mu_d:+.3f}   "
          f"diff = {mu_d - mu_u:+.3f}  (eta = {ETA})")

print("\nS/N at fixed dt (|mean|/std), undeformed vs deformed "
      "(NOTE: deformed mean needs the ~unity reweighting factor):")
for dt in (2, 4, 6, 8, 10, 12):
    mu, su = Uc[:, dt].real.mean(), Uc[:, dt].real.std(ddof=1)
    md, sd = D[:, dt].real.mean(), D[:, dt].real.std(ddof=1)
    print(f"  dt={dt:2d}: S/N_undef = {abs(mu)/su:6.2f}   S/N_def = {abs(md)/sd:6.2f}")
