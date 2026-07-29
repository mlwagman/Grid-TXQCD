#!/usr/bin/env python3
"""Pure-aux dibaryon (triple-diquark) correlator analysis.

C_g(dt) = (1/T) sum_ts <S_g(ts+dt) S_g*(ts)>   (periodic: B=2 => no APBC sign)
dt=0 EXCLUDED (Gaussian contact term).

Geometry classes (per family, per L):
  site     : r1=r3=0                      (I=1 only)
  disp1    : one w displaced, one at 0    (I=1 only)
  collinear: (r, -r)
  sameside : (r, r)                       (I=1 only)
  planar   : (r_mu, r_nu), mu != nu
Class-averaged at the correlator level (mwagman: average over orientations).

Reports mean +/- jackknife err and significance at dt=1..4 per class, x lam^6
(SD normalization: aux dibaryon x lam^6 <-> 6-quark correlator).

Usage: analyze_dibaryon_aux.py <h5dir> [--lam 5] [--fam I1|I0] [--dtmax 6]
"""
import glob
import sys
import numpy as np
import h5py

d5 = sys.argv[1] if len(sys.argv) > 1 else \
    "/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt/dtxqcd_lam5_dibaryon_aux"
LAM = float(sys.argv[sys.argv.index("--lam") + 1]) if "--lam" in sys.argv else 5.0
FAM = sys.argv[sys.argv.index("--fam") + 1] if "--fam" in sys.argv else "I1"
DTMAX = int(sys.argv[sys.argv.index("--dtmax") + 1]) if "--dtmax" in sys.argv else 6
L6 = LAM ** 6


def getc(v):
    v = np.array(v)
    return (v["re"] + 1j * v["im"]) if (v.dtype.names and "re" in v.dtype.names) \
        else v.astype(complex)


def classify(r1, r3):
    n1, n3 = np.abs(r1).sum(), np.abs(r3).sum()
    L = max(n1, n3)
    if n1 == 0 and n3 == 0:
        return ("site", 0)
    if n1 == 0 or n3 == 0:
        return ("disp1", L)
    if np.all(r1 == -r3):
        return ("collinear", L)
    if np.all(r1 == r3):
        return ("sameside", L)
    return ("planar", L)


files = sorted(glob.glob(f"{d5}/dibaryon_aux_dtxqcd_*.h5"),
               key=lambda s: int(s.split("_")[-1].split(".")[0]))
S = []          # per config: dict geo -> T-vector
meta = None
for f in files:
    with h5py.File(f) as h:
        ng = sum(1 for k in h.keys() if k.startswith(f"dib_{FAM}_S_"))
        if meta is None:
            g = np.array(h[f"geo_{FAM}"]).reshape(2, ng, 3)
            meta = [(g[0, i], g[1, i]) for i in range(ng)]
        S.append([getc(h[f"dib_{FAM}_S_{i}"]) for i in range(ng)])
T = len(S[0][0])
S = np.array(S)                        # (ncfg, ngeo, T)
ncfg, ng, T = S.shape
print(f"{FAM}: {ncfg} configs, {ng} geometries, T={T}, x lam^6={L6:.0f}")

# per-config translation-averaged correlator, per geometry
X = np.zeros((ncfg, ng, DTMAX + 1), complex)
for dt in range(DTMAX + 1):
    X[:, :, dt] = np.mean(np.roll(S, -dt, axis=2) * np.conj(S), axis=2)
X *= L6

classes = {}
for i, (r1, r3) in enumerate(meta):
    classes.setdefault(classify(np.array(r1), np.array(r3)), []).append(i)

print(f"\n{'class':>10} {'L':>2} | " +
      " | ".join(f"{'dt=%d' % dt:>19}" for dt in range(1, min(DTMAX, 4) + 1)))
for (cls, L), idx in sorted(classes.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    Xc = X[:, idx, :].mean(axis=1).real          # class-avg per config
    row = f"{cls:>10} {L:>2} |"
    for dt in range(1, min(DTMAX, 4) + 1):
        m = Xc[:, dt].mean()
        # jackknife
        jk = np.array([np.delete(Xc[:, dt], i).mean() for i in range(ncfg)])
        e = np.sqrt((ncfg - 1) / ncfg * np.sum((jk - jk.mean()) ** 2))
        sig = abs(m) / e if e > 0 else 0
        row += f" {m:+9.2e}({sig:4.1f}s) |"
    print(row)

print("\nnoise floor <|S|^2> (t-avg, x lam^6) per class, vs |C(1)|:")
for (cls, L), idx in sorted(classes.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    floor = (np.abs(S[:, idx, :]) ** 2).mean() * L6
    Xc = X[:, idx, 1].mean(axis=1).real.mean()
    print(f"  {cls:>10} L={L}: floor {floor:.2e}   |C(1)|/floor "
          f"{abs(Xc)/floor:.3e}")
print("\n(deform projection: graded m_eff shift = -3*eta*p'(dt); at tent "
      "eta=0.7, 3*eta=2.1 vs 2M_N~2.4 -- reweighted grading nearly "
      "flattens the fall-off.  Unreweighted graded significance is "
      "eta-independent; gains require the reweighting factors.)")
