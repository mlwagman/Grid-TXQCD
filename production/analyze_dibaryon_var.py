#!/usr/bin/env python3
"""Variational / global-detection analysis of the pure-aux dibaryon basis.

Per dt, per config: y_c = vector of class-averaged diagonal correlators
(13 classes for I=1, 6 for I=0).  Three statistics, overfitting-safe:

1. HOTELLING T^2 (global test): is the mean vector nonzero in ANY linear
   combination?  T2 = N ybar' Sigma^-1 ybar; (N-p)/(p(N-1)) T2 ~ F(p, N-p)
   under the null.  Exact look-elsewhere over the whole class space.
2. MATCHED FILTER, SPLIT-HALF: w = Sigma^-1 ybar from a random half
   (train), significance of w.y evaluated on the other half (test) --
   unbiased optimized-combination significance.  Averaged over 50 splits.
(A full-basis 82-dim shrunk-covariance test was tried and REMOVED: its
sign-flip permutation calibration is broken at p=82/n=175 -- it fired
p=0.000 on config-SHUFFLED data and at dt=16-20 where any signal is
impossible.  Do not resurrect without a sound null; the class-level
Hotelling + split-half tests below are the trustworthy statistics.)

Usage: analyze_dibaryon_var.py <h5dir> [--fam I1|I0]
"""
import glob
import sys
import numpy as np
import h5py

d5 = sys.argv[1] if len(sys.argv) > 1 else \
    "/lustre2/nplqcd/Grid-DTXQCD/production/meas_2pt/dtxqcd_lam5_dibaryon_aux"
FAM = sys.argv[sys.argv.index("--fam") + 1] if "--fam" in sys.argv else "I1"
rng = np.random.default_rng(17)


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
S, meta = [], None
for f in files:
    with h5py.File(f) as h:
        ng = sum(1 for k in h.keys() if k.startswith(f"dib_{FAM}_S_"))
        if meta is None:
            g = np.array(h[f"geo_{FAM}"]).reshape(2, ng, 3)
            meta = [(g[0, i], g[1, i]) for i in range(ng)]
        S.append([getc(h[f"dib_{FAM}_S_{i}"]) for i in range(ng)])
S = np.array(S)
N, NG, T = S.shape
classes = {}
for i, (r1, r3) in enumerate(meta):
    classes.setdefault(classify(np.array(r1), np.array(r3)), []).append(i)
NC = len(classes)
cidx = sorted(classes, key=lambda k: (k[1], k[0]))
print(f"{FAM}: {N} configs, {NG} geometries -> {NC} classes")

try:
    from scipy.stats import f as fdist
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def hotelling(Y):
    n, p = Y.shape
    yb = Y.mean(axis=0)
    Sg = np.cov(Y.T) if p > 1 else np.array([[Y.var(ddof=1)]])
    T2 = n * yb @ np.linalg.solve(Sg, yb)
    F = (n - p) / (p * (n - 1)) * T2
    pv = 1 - fdist.cdf(F, p, n - p) if HAVE_SCIPY else float("nan")
    return T2, F, pv


def hotelling_shrunk(Y, lam=0.2):
    n, p = Y.shape
    yb = Y.mean(axis=0)
    Sg = np.cov(Y.T)
    Sg = (1 - lam) * Sg + lam * np.trace(Sg) / p * np.eye(p)
    T2 = n * yb @ np.linalg.solve(Sg, yb)
    # calibrate null by sign-flip permutation (shrinkage breaks F-form)
    null = []
    for _ in range(400):
        s = rng.choice([-1, 1], n)[:, None]
        yp = (s * Y).mean(axis=0)
        null.append(n * yp @ np.linalg.solve(Sg, yp))
    pv = float(np.mean(np.array(null) >= T2))
    return T2, pv


print(f"\n{'dt':>3} | {'Hotelling p (class)':>20} | {'split-half sig':>14}")
for dt in (1, 2, 3, 4):
    X = np.mean(np.roll(S, -dt, axis=2) * np.conj(S), axis=2).real  # (N, NG)
    Y = np.stack([X[:, classes[c]].mean(axis=1) for c in cidx], axis=1)
    _, _, pv = hotelling(Y)
    # split-half matched filter
    sigs = []
    for _ in range(50):
        perm = rng.permutation(N)
        a, b = perm[: N // 2], perm[N // 2:]
        yb = Y[a].mean(axis=0)
        Sg = np.cov(Y[a].T)
        w = np.linalg.solve(Sg + 1e-12 * np.eye(NC) * np.trace(Sg), yb)
        z = Y[b] @ w
        sigs.append(z.mean() / (z.std(ddof=1) / np.sqrt(len(b))))
    print(f"{dt:3d} | {pv:20.3f} | {np.mean(sigs):+8.2f}+-{np.std(sigs):4.2f}")

print("\n(null expectations: Hotelling p uniform in [0,1]; split-half sig ~ 0"
      " +- 1; a real signal gives small p at ALL dt and positive coherent"
      " split-half sig.)")
